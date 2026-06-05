"""
V2 — ELO Ratings
Computes a running ELO rating for every team across all 49k matches (1872–present).

K-factors by match type:
  - World Cup / major tournament finals:  40
  - Qualifying / continental tournaments: 30
  - Friendlies / minor tournaments:       20

Output:
  data/processed/elo_ratings.csv   — one row per (date, home_team, away_team)
                                      with pre-match ELO for both teams

Run with:
  python -m src.features.elo
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parents[2]
DATA_RAW  = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"

RESULTS_PATH = DATA_RAW / "international_results" / "results.csv"
ELO_OUT      = DATA_PROC / "elo_ratings.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_ELO = 1500  # Starting rating for any team's first appearance

# K-factor by tournament category
HIGH_K_TOURNAMENTS = {
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "CONCACAF Championship",
}

MED_K_TOURNAMENTS_KEYWORDS = [
    "qualification", "qualifier", "qualifying",
    "nations league", "nations cup",
    "copa america", "gold cup",
    "african cup", "afc asian",
    "concacaf gold", "uefa nations",
]

def get_k_factor(tournament: str) -> float:
    t = tournament.lower()
    if tournament in HIGH_K_TOURNAMENTS:
        return 40.0
    for kw in MED_K_TOURNAMENTS_KEYWORDS:
        if kw in t:
            return 30.0
    return 20.0  # Friendly / minor tournament


# ── ELO update formula ────────────────────────────────────────────────────────
def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that team A beats team B under ELO."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    home_elo: float,
    away_elo: float,
    home_score: int,
    away_score: int,
    k: float,
    home_advantage: float = 100.0,
) -> tuple[float, float]:
    """
    Returns (new_home_elo, new_away_elo).
    home_advantage: additive bonus applied to home team's effective rating.
    Set to 0 for neutral-venue matches.
    """
    # Actual result from home team's perspective
    if home_score > away_score:
        actual = 1.0
    elif home_score == away_score:
        actual = 0.5
    else:
        actual = 0.0

    # Expected score with home advantage
    exp = expected_score(home_elo + home_advantage, away_elo)

    delta = k * (actual - exp)
    return home_elo + delta, away_elo - delta


# ── Main computation ──────────────────────────────────────────────────────────
def compute_elo(results: pd.DataFrame) -> pd.DataFrame:
    """
    Walk through every match chronologically.
    Record pre-match ELO for both teams, then update ratings.
    Returns a DataFrame with columns:
      date, home_team, away_team, home_elo, away_elo, elo_diff
    """
    results = results.dropna(subset=["home_score", "away_score"])
    results = results.sort_values("date").reset_index(drop=True)

    ratings: dict[str, float] = {}  # team → current ELO

    rows = []
    for _, row in results.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        neutral = str(row.get("neutral", "FALSE")).upper() == "TRUE"

        # Initialize unseen teams
        if h not in ratings:
            ratings[h] = DEFAULT_ELO
        if a not in ratings:
            ratings[a] = DEFAULT_ELO

        pre_home = ratings[h]
        pre_away = ratings[a]

        k = get_k_factor(row["tournament"])
        ha = 0.0 if neutral else 100.0

        new_home, new_away = update_elo(
            pre_home, pre_away,
            int(row["home_score"]), int(row["away_score"]),
            k=k, home_advantage=ha,
        )

        rows.append({
            "date":       row["date"],
            "home_team":  h,
            "away_team":  a,
            "home_elo":   round(pre_home, 2),
            "away_elo":   round(pre_away, 2),
            "elo_diff":   round(pre_home - pre_away, 2),
        })

        ratings[h] = new_home
        ratings[a] = new_away

    elo_df = pd.DataFrame(rows)
    return elo_df, ratings


def main():
    print("Loading results...")
    results = pd.read_csv(RESULTS_PATH, parse_dates=["date"])
    print(f"  {len(results):,} matches | {results['date'].min().date()} → {results['date'].max().date()}")

    print("Computing ELO ratings...")
    elo_df, final_ratings = compute_elo(results)

    elo_df.to_csv(ELO_OUT, index=False)
    print(f"\n✅ Saved elo_ratings.csv: {len(elo_df):,} rows")

    # Sanity check — top 20 teams by final ELO
    top = sorted(final_ratings.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\nTop 20 teams by final ELO:")
    for i, (team, elo) in enumerate(top, 1):
        print(f"  {i:>2}. {team:<30} {elo:.0f}")

    # Spot-check a few key teams
    print("\nKey team spot-check:")
    for team in ["France", "Brazil", "Argentina", "United States", "Mexico", "Spain", "England"]:
        val = final_ratings.get(team, None)
        print(f"  {team:<20} {val:.0f}" if val else f"  {team:<20} NOT FOUND")


if __name__ == "__main__":
    main()
