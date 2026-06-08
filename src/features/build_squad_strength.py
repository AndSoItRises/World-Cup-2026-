"""
V4: Build a historical national-team squad STRENGTH + DEPTH table from FIFA
video-game ratings (lbenz730/fifa_model, FIFA editions 2005-2020).

Why FIFA ratings (not Transfermarkt £ value)?
  - The `rating` (FIFA overall) column is populated for EVERY edition 2005-2020;
    the monetary `value` column is missing/NA in the early editions. Rating is a
    consistent, broad (~150 nations) quality unit across the whole window, so it
    yields a clean leakage-safe time series. Transfermarkt history only exists
    for the recent top nations and would inject NaNs (the V3 silent-NaN trap).

Leakage safety:
  - A FIFA edition labelled year=Y ships in late September of (Y-1). We stamp each
    edition with availability_date = (Y-1)-10-01. feature_engineering merges these
    onto matches with merge_asof BACKWARD, so a match only ever sees the most
    recent edition released strictly before it. No future squad info leaks.

Depth is the point (Jake's ask): from each national squad's rating distribution we
derive top-end quality AND bench quality, so the model can tell a top-heavy side
from a deep one:
  squad_strength  : mean rating of the top 23 players (realistic tournament pool)
  squad_top11     : mean rating of the top 11 (first-choice quality)
  squad_depth     : mean rating of players ranked 12..23 (absolute bench quality)
  squad_n_quality : count of players with rating >= 75 (size of the quality pool)

Output: data/processed/squad_strength_by_year.csv  (tracked; raw FIFA CSVs are
cached under data/raw/fifa_ratings/ which is gitignored + re-downloadable).

Run: python -m src.features.build_squad_strength
"""

import sys
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.data_cleaning import standardize_name

BASE = Path(__file__).resolve().parents[2]
RAW_DIR = BASE / "data" / "raw" / "fifa_ratings"
OUT = BASE / "data" / "processed" / "squad_strength_by_year.csv"

# ── Canonical V4 squad feature interface (single source of truth) ─────────────
# The model features fed to XGB/LGBM (squad_top11 + squad_n_quality were dropped
# in train_v4 as collinear/noisy per signal_test). feature_engineering builds the
# training columns; squad_fields() builds them at prediction time.
SQUAD_MODEL_FEATURES = [
    "home_squad_strength", "away_squad_strength", "squad_strength_diff",
    "home_squad_depth", "away_squad_depth", "squad_depth_diff",
    "squad_both_covered",
]
# "Not in FIFA" sentinel — weaker than the weakest rated nation (≈1st pct).
SQUAD_SENTINEL = {"squad_strength": 60.0, "squad_top11": 63.0, "squad_depth": 55.0, "squad_n_quality": 0.0}


@lru_cache(maxsize=1)
def _latest_squad_lookup():
    """Most-recent edition per team (for prediction — leakage-safe: FC26 ships 2025)."""
    df = pd.read_csv(OUT, parse_dates=["availability_date"])
    return df.sort_values("availability_date").groupby("team").tail(1).set_index("team")


def squad_fields(home: str, away: str) -> dict:
    """Build the 7 model squad features for a fixture, with sentinel + coverage flag.
    Mirrors feature_engineering.add_squad_strength so training and prediction agree."""
    sq = _latest_squad_lookup()
    h = sq.loc[home] if home in sq.index else None
    a = sq.loc[away] if away in sq.index else None
    hs = float(h["squad_strength"]) if h is not None else SQUAD_SENTINEL["squad_strength"]
    as_ = float(a["squad_strength"]) if a is not None else SQUAD_SENTINEL["squad_strength"]
    hd = float(h["squad_depth"]) if h is not None else SQUAD_SENTINEL["squad_depth"]
    ad = float(a["squad_depth"]) if a is not None else SQUAD_SENTINEL["squad_depth"]
    return {
        "home_squad_strength": hs, "away_squad_strength": as_, "squad_strength_diff": hs - as_,
        "home_squad_depth": hd, "away_squad_depth": ad, "squad_depth_diff": hd - ad,
        "squad_both_covered": int(h is not None and a is not None),
    }

# Source editions. FIFA overall is the same 0-99 scale across fifaindex (lbenz730)
# and sofifa (stefanoleone992 schema), so they unify on `rating`/`overall`.
#   kind="fifaindex": lbenz730/fifa_model, cols rating, nationality      (2005-2020)
#   kind="sofifa"   : stefanoleone992 schema, cols overall, nationality_name, value_eur
# Each gives anchor points; feature_engineering merge_asof BACKWARD carries the
# most-recent-available edition onto each match (gaps -> stale-but-safe, never leak).
FIFAINDEX_YEARS = list(range(2005, 2021))  # 2005..2020 inclusive
FIFAINDEX_URL = "https://raw.githubusercontent.com/lbenz730/fifa_model/master/stats/player_stats_{year}.csv"
SOFIFA_SOURCES = [
    # (year, url) - sofifa schema (overall, nationality_name, value_eur)
    (2022, "https://raw.githubusercontent.com/abineshta/FIFA-22-complete-player-dataset-EDA/main/players_22.csv"),
    (2026, "https://raw.githubusercontent.com/ismailoksuz/EAFC26-DataHub/main/data/players.csv"),
]

SQUAD_N = 23          # tournament squad size
QUALITY_THRESHOLD = 75  # FIFA overall considered "quality" depth

# FIFA `nationality` spelling  ->  model canonical name (results.csv spelling).
# Canonical names are the results.csv side; data_cleaning.standardize_name maps
# OTHER spellings TO these (e.g. "South Korea" -> "Korea Republic"). We therefore
# first push FIFA spelling through this map, THEN through standardize_name so any
# residual alias is also normalised. Populated from the coverage audit below.
FIFA_NATION_MAP = {
    "Republic of Ireland": "Ireland",          # -> standardize_name -> Ireland
    "China PR": "China PR",
    "IR Iran": "Iran",
    "Korea Republic": "Korea Republic",
    "Korea DPR": "Korea DPR",
    "United States": "USA",
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "DR Congo": "Congo DR",
    "Congo DR": "Congo DR",
    "Cape Verde": "Cape Verde Islands",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Bosnia Herzegovina": "Bosnia-Herzegovina",
    "Czech Republic": "Czechia",
    "Curacao": "Curaçao",
    "Kyrgyzstan": "Kyrgyz Republic",
    "FYR Macedonia": "North Macedonia",
    "Macedonia": "North Macedonia",
    "Swaziland": "Eswatini",
    "Brunei": "Brunei Darussalam",
}


def download(url: str, dest_name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / dest_name
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    print(f"  downloading {dest_name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"{dest.stat().st_size // 1024} KB")
    return dest


def map_nation(name: str) -> str:
    return standardize_name(FIFA_NATION_MAP.get(name, name))


def aggregate_year(path: Path, year: int, kind: str) -> pd.DataFrame:
    if kind == "fifaindex":
        df = pd.read_csv(path, usecols=["rating", "nationality"], low_memory=False)
        df = df.rename(columns={"nationality": "nat"})
    else:  # sofifa
        df = pd.read_csv(path, usecols=["overall", "nationality_name"], low_memory=False)
        df = df.rename(columns={"overall": "rating", "nationality_name": "nat"})
    df = df.dropna(subset=["rating", "nat"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["team"] = df["nat"].astype(str).map(map_nation)

    rows = []
    for team, grp in df.groupby("team"):
        ratings = np.sort(grp["rating"].values)[::-1]  # desc
        top = ratings[:SQUAD_N]
        if len(top) < 11:
            continue  # too few players to form a credible XI
        top11 = top[:11]
        bench = top[11:SQUAD_N]  # players 12..23
        rows.append({
            "team": team,
            "year": year,
            # availability: FIFA year=Y ships ~Sep of Y-1 -> stamp Oct 1 of Y-1
            "availability_date": pd.Timestamp(year - 1, 10, 1),
            "squad_strength": float(np.mean(top)),
            "squad_top11": float(np.mean(top11)),
            "squad_depth": float(np.mean(bench)) if len(bench) else float(np.mean(top11)),
            "squad_n_quality": int((grp["rating"] >= QUALITY_THRESHOLD).sum()),
            "n_players": int(len(grp)),
        })
    return pd.DataFrame(rows)


def coverage_audit(squad: pd.DataFrame):
    print("\n" + "=" * 64)
    print("  COVERAGE AUDIT (run before trusting the join)")
    print("=" * 64)

    # 1. vs the 48 WC2026 teams
    sv = pd.read_csv(BASE / "data" / "raw" / "wc2026_squad_values.csv")
    wc_teams = set(sv["team"].apply(standardize_name))
    covered = set(squad["team"])
    missing_wc = sorted(wc_teams - covered)
    print(f"\n  WC2026 teams covered: {len(wc_teams) - len(missing_wc)}/{len(wc_teams)}")
    if missing_wc:
        print(f"    MISSING: {missing_wc}")

    # 2. vs the training/test match team universe
    try:
        tr = pd.read_csv(BASE / "data" / "processed" / "train.csv", usecols=["home_team", "away_team"])
        te = pd.read_csv(BASE / "data" / "processed" / "test.csv", usecols=["home_team", "away_team"])
        match_teams = set(tr["home_team"]) | set(tr["away_team"]) | set(te["home_team"]) | set(te["away_team"])
        missing_match = sorted(match_teams - covered)
        print(f"\n  Match-universe teams covered: {len(match_teams) - len(missing_match)}/{len(match_teams)}")
        print(f"    (uncovered get NaN -> imputed; fine for minnows, audit the notable ones)")
        notable = [t for t in missing_match if t in wc_teams]
        if notable:
            print(f"    NOTABLE MISSING (WC2026 team not matched!): {notable}")
        print(f"    sample uncovered: {missing_match[:25]}")
    except FileNotFoundError:
        print("\n  (train.csv/test.csv not found - run data_cleaning first for full audit)")

    # 3. unmapped FIFA nations that look like real countries (potential map gaps)
    print(f"\n  Distinct nations in FIFA data: {covered.__len__()}")


def main():
    print("=" * 64)
    print("  V4: Build historical squad strength + depth (FIFA 2005-2020)")
    print("=" * 64)
    frames = []
    for y in FIFAINDEX_YEARS:
        path = download(FIFAINDEX_URL.format(year=y), f"player_stats_{y}.csv")
        agg = aggregate_year(path, y, "fifaindex")
        frames.append(agg)
        print(f"  {y} (fifaindex): {len(agg)} nations")
    for y, url in SOFIFA_SOURCES:
        path = download(url, f"sofifa_{y}.csv")
        agg = aggregate_year(path, y, "sofifa")
        frames.append(agg)
        print(f"  {y} (sofifa):    {len(agg)} nations")
    squad = pd.concat(frames, ignore_index=True).sort_values(["team", "year"]).reset_index(drop=True)

    coverage_audit(squad)

    squad.to_csv(OUT, index=False)
    print(f"\n  Saved {len(squad)} team-year rows -> {OUT}")
    print(f"  Columns: {list(squad.columns)}")


if __name__ == "__main__":
    main()
