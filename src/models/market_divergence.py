"""
Phase 6 (V2): Market Divergence Analysis.
Compares the model's tournament win probabilities against the betting market.

Steps:
  1. Load market odds (American) → implied probability
  2. De-vig: normalize implied probs to sum to 1 (removes bookmaker overround)
  3. Join to model tournament_probs.csv (p_winner)
  4. Edge = model_prob - market_prob; ratio = model / market
  5. Rank biggest divergences (model value picks vs fades)

This is the key V2 question: does the model see edge the market doesn't?
Outputs:
  data/processed/market_divergence.csv
  models/market_divergence_report.json

Run with:
  python -m src.models.market_divergence
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

BASE      = Path(__file__).resolve().parents[2]
DATA_RAW  = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"
MODELS    = BASE / "models"

ODDS_PATH  = DATA_RAW  / "wc2026_market_odds.csv"
PROBS_PATH = DATA_PROC / "tournament_probs.csv"

# Market (ESPN/bookmaker) names → model/training names used in tournament_probs.csv
MARKET_TO_MODEL = {
    "United States":          "USA",
    "South Korea":            "Korea Republic",
    "Ivory Coast":            "Côte d'Ivoire",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Cape Verde":             "Cape Verde Islands",
    "Curacao":                "Curaçao",
    "Congo DR":               "DR Congo",
}


def american_to_implied(american: str) -> float:
    """American odds string (e.g. '+450', '-120') → implied probability."""
    a = int(str(american).replace("+", ""))
    if a > 0:
        return 100.0 / (a + 100.0)
    else:
        return -a / (-a + 100.0)


def main():
    print("═" * 64)
    print("  Phase 6: Market Divergence Analysis")
    print("═" * 64)

    odds  = pd.read_csv(ODDS_PATH)
    probs = pd.read_csv(PROBS_PATH)

    # 1. implied prob from American odds
    odds["implied_raw"] = odds["american_odds"].apply(american_to_implied)
    overround = odds["implied_raw"].sum()
    print(f"\n  Market teams: {len(odds)} | overround (sum implied): {overround:.3f} "
          f"({(overround-1)*100:.1f}% vig)")

    # 2. de-vig → fair market probability
    odds["market_prob"] = odds["implied_raw"] / overround

    # 3. map names and join to model
    odds["model_team"] = odds["team"].replace(MARKET_TO_MODEL)
    merged = odds.merge(
        probs[["team", "fifa_rank", "p_winner"]],
        left_on="model_team", right_on="team", how="left", suffixes=("_mkt", "_mdl")
    )

    matched   = merged[merged["p_winner"].notna()].copy()
    unmatched = merged[merged["p_winner"].isna()]["team_mkt"].tolist()
    model_only = set(probs["team"]) - set(matched["model_team"])

    print(f"  Matched teams: {len(matched)} / {len(odds)}")
    if unmatched:
        print(f"  In market, not in model (didn't qualify / name gap): {unmatched}")
    if model_only:
        print(f"  In model, not in market: {sorted(model_only)}")

    # 4. edge metrics
    matched["model_prob"] = matched["p_winner"]
    matched["edge"]  = matched["model_prob"] - matched["market_prob"]
    matched["ratio"] = matched["model_prob"] / matched["market_prob"].clip(lower=1e-6)

    # 5. rank divergences
    matched = matched.sort_values("edge", ascending=False)

    out = matched[["model_team", "fifa_rank", "market_prob", "model_prob", "edge", "ratio"]].copy()
    out.columns = ["team", "fifa_rank", "market_prob", "model_prob", "edge", "ratio"]
    out_path = DATA_PROC / "market_divergence.csv"
    out.to_csv(out_path, index=False)

    print(f"\n── Model sees MORE value than market (top 8 positive edge) ──")
    print(f"  {'Team':<22} {'Mkt%':>7} {'Mdl%':>7} {'Edge':>7} {'Ratio':>6}")
    print(f"  {'-'*52}")
    for _, r in out.head(8).iterrows():
        print(f"  {r['team']:<22} {r['market_prob']*100:>6.1f}% {r['model_prob']*100:>6.1f}% "
              f"{r['edge']*100:>+6.1f}% {r['ratio']:>5.2f}x")

    print(f"\n── Model FADES vs market (top 8 negative edge) ──")
    print(f"  {'Team':<22} {'Mkt%':>7} {'Mdl%':>7} {'Edge':>7} {'Ratio':>6}")
    print(f"  {'-'*52}")
    for _, r in out.tail(8).iloc[::-1].iterrows():
        print(f"  {r['team']:<22} {r['market_prob']*100:>6.1f}% {r['model_prob']*100:>6.1f}% "
              f"{r['edge']*100:>+6.1f}% {r['ratio']:>5.2f}x")

    # Calibration summary: correlation + mean abs divergence
    corr = matched["model_prob"].corr(matched["market_prob"])
    mad  = matched["edge"].abs().mean()
    print(f"\n── Agreement ──")
    print(f"  Correlation (model vs market): {corr:.3f}")
    print(f"  Mean abs divergence:           {mad*100:.2f}pp")

    report = {
        "market_overround":      round(float(overround), 4),
        "matched_teams":         int(len(matched)),
        "unmatched_market":      unmatched,
        "model_market_corr":     round(float(corr), 4),
        "mean_abs_divergence_pp": round(float(mad * 100), 3),
        "top_value_picks": [
            {"team": r["team"], "edge_pp": round(r["edge"] * 100, 2),
             "model_pct": round(r["model_prob"] * 100, 2),
             "market_pct": round(r["market_prob"] * 100, 2)}
            for _, r in out.head(5).iterrows()
        ],
        "top_fades": [
            {"team": r["team"], "edge_pp": round(r["edge"] * 100, 2),
             "model_pct": round(r["model_prob"] * 100, 2),
             "market_pct": round(r["market_prob"] * 100, 2)}
            for _, r in out.tail(5).iloc[::-1].iterrows()
        ],
    }
    with open(MODELS / "market_divergence_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Saved: {out_path}")
    print(f"✅ Saved: {MODELS / 'market_divergence_report.json'}")


if __name__ == "__main__":
    main()
