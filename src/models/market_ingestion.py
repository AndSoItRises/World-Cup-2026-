"""
V6 Phase 1 — Market odds ingestion + juice stripping (the Vegas layer's data spine).

A bookmaker's quoted odds embed an overround (vig): summed implied probabilities
exceed 1. Before any model-vs-market comparison, the vig must be stripped to get
FAIR implied probabilities. Two methods here:

  proportional — divide each implied prob by the sum (what market_divergence.py
                 does). Simple, but assigns vig evenly, which over-shades favorites.
  shin         — Shin (1992): models the vig as the book protecting itself from
                 insider traders; longshots carry proportionally more vig
                 (favorite-longshot bias). Solves for insider fraction z such that
                 fair probs sum to 1. DEFAULT method.

Inputs (Jake maintains):
  data/raw/wc2026_market_odds.csv  — tournament winner futures: team, american_odds
  data/raw/wc2026_match_odds.csv   — per-match 3-way odds (OPTIONAL; a header
                                     template is auto-created). Keyed by match_id
                                     (matches data/raw/wc2026_fixtures.csv) so
                                     there is no team-name ambiguity.
                                     Columns: match_id, book, snapshot
                                     (opening|current|closing), odds_format
                                     (american|decimal), home_odds, draw_odds,
                                     away_odds, fetched_at
Outputs:
  data/processed/market_implied_probs.csv    — per-match fair 3-way probs.
                                               Matches without real odds use the
                                               model's own probs as fair odds,
                                               flagged market_source="model_estimated".
  data/processed/market_implied_futures.csv  — tournament-winner fair probs (real odds).

Run: python -m src.models.market_ingestion
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_RAW = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"

FUTURES_ODDS_PATH = DATA_RAW / "wc2026_market_odds.csv"
MATCH_ODDS_PATH   = DATA_RAW / "wc2026_match_odds.csv"
PREDICTIONS_PATH  = DATA_PROC / "wc2026_predictions.csv"
PROBS_LIVE_PATH   = DATA_PROC / "tournament_probs_live.csv"

MATCH_IMPLIED_OUT   = DATA_PROC / "market_implied_probs.csv"
FUTURES_IMPLIED_OUT = DATA_PROC / "market_implied_futures.csv"

MATCH_ODDS_COLUMNS = ["match_id", "book", "snapshot", "odds_format",
                      "home_odds", "draw_odds", "away_odds", "fetched_at"]

# When several snapshots exist for a match, bet on the most informed line available.
SNAPSHOT_PRIORITY = {"closing": 3, "current": 2, "opening": 1}

# Market (ESPN/bookmaker) names → model/training names (same map as market_divergence.py)
MARKET_TO_MODEL = {
    "United States":          "USA",
    "South Korea":            "Korea Republic",
    "Ivory Coast":            "Côte d'Ivoire",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Cape Verde":             "Cape Verde Islands",
    "Curacao":                "Curaçao",
}


# ── Odds conversion ───────────────────────────────────────────────────────────
def american_to_decimal(american) -> float:
    a = float(str(american).replace("+", ""))
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def decimal_to_implied(decimal_odds: float) -> float:
    return 1.0 / float(decimal_odds)


def parse_odds_row(values, odds_format: str) -> np.ndarray:
    """[home, draw, away] quotes in the stated format → decimal odds array."""
    fmt = str(odds_format).strip().lower()
    if fmt == "american":
        return np.array([american_to_decimal(v) for v in values], dtype=float)
    if fmt == "decimal":
        return np.array([float(v) for v in values], dtype=float)
    raise ValueError(f"Unknown odds_format '{odds_format}' (use 'american' or 'decimal')")


# ── Juice stripping ───────────────────────────────────────────────────────────
def proportional_fair(implied_raw: np.ndarray) -> np.ndarray:
    pi = np.asarray(implied_raw, dtype=float)
    return pi / pi.sum()


def shin_fair(implied_raw: np.ndarray) -> np.ndarray:
    """Shin (1992) de-vig: solve insider fraction z so fair probs sum to 1.
    Falls back to proportional if no root (e.g. no vig to strip)."""
    pi = np.asarray(implied_raw, dtype=float)
    total = pi.sum()
    if total <= 1.0 + 1e-9:
        return proportional_fair(pi)

    def fair(z):
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / total) - z) / (2.0 * (1.0 - z))

    try:
        z = brentq(lambda z: fair(z).sum() - 1.0, 0.0, 0.999, xtol=1e-12)
        return fair(z)
    except ValueError:
        return proportional_fair(pi)


# ── Futures (tournament winner) ───────────────────────────────────────────────
def build_futures_implied() -> pd.DataFrame:
    """Tournament-winner futures: american odds → decimal → fair probs, joined to
    the model's live p_winner. Prints the team name audit (non-negotiable)."""
    odds = pd.read_csv(FUTURES_ODDS_PATH)
    odds["model_team"] = odds["team"].replace(MARKET_TO_MODEL)
    odds["decimal_odds"] = odds["american_odds"].map(american_to_decimal)
    odds["implied_raw"] = odds["decimal_odds"].map(decimal_to_implied)

    overround = odds["implied_raw"].sum()
    odds["fair_prob_prop"] = proportional_fair(odds["implied_raw"].values)
    odds["fair_prob_shin"] = shin_fair(odds["implied_raw"].values)
    odds["market_prob"] = odds["fair_prob_shin"]   # default method
    odds["overround"] = overround

    live = pd.read_csv(PROBS_LIVE_PATH)
    merged = odds.merge(
        live[["team", "p_winner", "eliminated"]].rename(columns={"team": "model_team"}),
        on="model_team", how="left",
    )

    matched = merged["p_winner"].notna()
    unmatched = merged.loc[~matched, "team"].tolist()
    model_only = sorted(set(live["team"]) - set(merged.loc[matched, "model_team"]))
    print(f"\n  ── Futures name audit ──")
    print(f"  Market teams: {len(odds)} | matched to model: {int(matched.sum())}")
    if unmatched:
        print(f"  ⚠️  In market, not in model: {unmatched}")
    if model_only:
        print(f"  ⚠️  In model, not in market: {model_only}")
    print(f"  Overround: {overround:.3f} ({(overround - 1) * 100:.1f}% vig)")

    cols = ["model_team", "team", "american_odds", "decimal_odds", "implied_raw",
            "fair_prob_prop", "fair_prob_shin", "market_prob", "overround",
            "p_winner", "eliminated"]
    out = merged[cols].rename(columns={"model_team": "team", "team": "market_team"})
    return out.sort_values("market_prob", ascending=False).reset_index(drop=True)


# ── Match-level 3-way ─────────────────────────────────────────────────────────
def ensure_match_odds_template():
    if not MATCH_ODDS_PATH.exists():
        MATCH_ODDS_PATH.write_text(",".join(MATCH_ODDS_COLUMNS) + "\n")
        print(f"  Created odds template: {MATCH_ODDS_PATH.name} (fill with real lines)")


def load_match_odds() -> pd.DataFrame:
    """Real per-match odds, best snapshot per match (closing > current > opening,
    then latest fetched_at). Empty DataFrame if none entered yet."""
    ensure_match_odds_template()
    raw = pd.read_csv(MATCH_ODDS_PATH)
    if len(raw) == 0:
        return raw
    missing = [c for c in MATCH_ODDS_COLUMNS[:7] if c not in raw.columns]
    if missing:
        raise ValueError(f"{MATCH_ODDS_PATH.name} missing columns: {missing}")
    raw["snapshot"] = raw["snapshot"].str.strip().str.lower()
    bad = sorted(set(raw["snapshot"]) - set(SNAPSHOT_PRIORITY))
    if bad:
        raise ValueError(f"Unknown snapshot values {bad} (use opening|current|closing)")
    raw["_prio"] = raw["snapshot"].map(SNAPSHOT_PRIORITY)
    raw = (raw.sort_values(["match_id", "_prio", "fetched_at"])
              .groupby("match_id", as_index=False).last()
              .drop(columns="_prio"))
    return raw


def build_match_implied() -> pd.DataFrame:
    """Fair 3-way implied probs for every upcoming match in the model's
    prediction sheet. Real odds where entered; model-estimated otherwise."""
    preds = pd.read_csv(PREDICTIONS_PATH)
    real = load_match_odds()
    real = real.set_index("match_id") if len(real) else real

    rows = []
    for _, m in preds.iterrows():
        mid = int(m["match_id"])
        row = {
            "match_id": mid, "date": m["date"], "stage": m["stage"], "group": m["group"],
            "home_team": m["home_team"], "away_team": m["away_team"],
        }
        if mid in real.index:
            r = real.loc[mid]
            dec = parse_odds_row([r["home_odds"], r["draw_odds"], r["away_odds"]],
                                 r["odds_format"])
            implied = 1.0 / dec
            fair_shin = shin_fair(implied)
            fair_prop = proportional_fair(implied)
            row.update({
                "market_source": "real", "book": r["book"], "snapshot": r["snapshot"],
                "overround": implied.sum(),
            })
        else:
            # No market line: the model's own probs ARE the fair odds (EV ≡ 0 —
            # placeholder that keeps the pipeline live, never flags fake value)
            fair_shin = fair_prop = np.array(
                [m["p_home_win"], m["p_draw"], m["p_away_win"]], dtype=float)
            dec = 1.0 / fair_shin
            row.update({
                "market_source": "model_estimated", "book": "", "snapshot": "",
                "overround": 1.0,
            })
        for i, side in enumerate(["home", "draw", "away"]):
            row[f"{side}_decimal_odds"] = round(float(dec[i]), 4)
            row[f"{side}_fair_prob"] = round(float(fair_shin[i]), 4)
            row[f"{side}_fair_prob_prop"] = round(float(fair_prop[i]), 4)
        rows.append(row)

    out = pd.DataFrame(rows)
    n_real = int((out["market_source"] == "real").sum())
    print(f"\n  ── Match odds coverage ──")
    print(f"  Matches: {len(out)} | real market lines: {n_real} | "
          f"model_estimated: {len(out) - n_real}")
    if n_real == 0:
        print(f"  ⚠️  No real match odds yet — fill {MATCH_ODDS_PATH.name} to activate match EV")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("═" * 60)
    print("  V6 — Market Odds Ingestion + Juice Stripping")
    print("═" * 60)

    futures = build_futures_implied()
    futures.to_csv(FUTURES_IMPLIED_OUT, index=False)

    matches = build_match_implied()
    matches.to_csv(MATCH_IMPLIED_OUT, index=False)

    for df, name in [(futures, FUTURES_IMPLIED_OUT), (matches, MATCH_IMPLIED_OUT)]:
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            print(f"  ⚠️  NaNs in {name.name}: {nan_cols}")

    print(f"\n✅ Saved: {FUTURES_IMPLIED_OUT}")
    print(f"✅ Saved: {MATCH_IMPLIED_OUT}")

    # Quick read: where do the two de-vig methods disagree most? (longshots)
    f = futures.dropna(subset=["p_winner"]).copy()
    f["shin_vs_prop_pp"] = (f["fair_prob_shin"] - f["fair_prob_prop"]) * 100
    top = f.reindex(f["shin_vs_prop_pp"].abs().sort_values(ascending=False).index).head(5)
    print(f"\n  ── Shin vs proportional (largest gaps, pp) ──")
    for _, r in top.iterrows():
        print(f"  {r['team']:<22}{r['shin_vs_prop_pp']:>+7.2f}pp "
              f"(shin {r['fair_prob_shin']*100:.2f}% / prop {r['fair_prob_prop']*100:.2f}%)")


if __name__ == "__main__":
    main()
