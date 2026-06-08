"""
Learning Path — Step 2: The market as benchmark (you compete with a price).

V3's big lesson: being "right" is worthless; being RIGHTER THAN THE PRICE is
everything. The market already embeds squad value, injuries, everything. Your
edge = your probability − the fair price.

This script teaches the INFORMATION COEFFICIENT (IC): the rank correlation
between a model's predictive score and the realised outcome. Higher IC ⇒ the
model's ordering of matches tracks reality better.

  1. Model IC on the held-out test set: Spearman(model supremacy, realised
     result), where supremacy = P(home win) − P(away win). Compared v3 vs v4 to
     see whether squad value improved the ordering, overall and per confederation.
  2. Market-as-benchmark on the WC2026 cross-section: correlation of model win%
     vs de-vigged market win%, and the biggest divergences (your "edge" — only
     real if the model's IC beats the market's, which needs the tournament to
     play out / historical odds; see note + V4 P4).

Concepts: information coefficient, edge vs hit-rate, market efficiency.

Run: python -m src.models.market_backtest
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.features.data_cleaning import standardize_name
from src.models.validate_v3 import conf_map

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"
MODELS = BASE / "models"


def model_ic(proba, y):
    """Spearman(supremacy, realised result). Supremacy = P(home) − P(away)."""
    supremacy = proba[:, 2] - proba[:, 0]
    return spearmanr(supremacy, y).correlation


def main():
    test = pd.read_csv(DATA_PROC / "test_features.csv", parse_dates=["date"])
    y = test["result"].astype(int).values
    cm = conf_map()
    conf = test["home_team"].map(cm).fillna("OTHER")

    print("═" * 60)
    print("  Information Coefficient — model ordering vs reality")
    print("═" * 60)

    print(f"\n  {'version':<10}{'overall IC':>12}")
    proba = {}
    for ver in ["v3", "v4"]:
        p = np.load(MODELS / f"ensemble_test_proba_{ver}.npy")
        proba[ver] = p
        print(f"  {ver:<10}{model_ic(p, y):>12.4f}")

    print(f"\n  ── IC by confederation (v3 → v4) ──")
    print(f"  {'conf':<10}{'n':>6}{'v3 IC':>9}{'v4 IC':>9}{'Δ':>9}")
    for c in ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC"]:
        m = (conf == c).values
        if m.sum() < 30:
            continue
        ic3, ic4 = model_ic(proba["v3"][m], y[m]), model_ic(proba["v4"][m], y[m])
        print(f"  {c:<10}{m.sum():>6}{ic3:>9.4f}{ic4:>9.4f}{ic4-ic3:>+9.4f}")

    # ── Market-as-benchmark on the WC2026 cross-section ──
    md_path = DATA_PROC / "market_divergence.csv"
    if md_path.exists():
        md = pd.read_csv(md_path)
        corr = md["model_prob"].corr(md["market_prob"])
        md = md.assign(edge_pp=(md["model_prob"] - md["market_prob"]) * 100)
        print(f"\n  ── WC2026 model vs market (tournament win%) ──")
        print(f"  Correlation model↔market: {corr:.3f}  (high ⇒ model mostly agrees with the price)")
        print(f"  Biggest model edges (model − market):")
        for _, r in md.reindex(md['edge_pp'].abs().sort_values(ascending=False).index).head(6).iterrows():
            print(f"    {r['team']:<18}{r['model_prob']*100:>6.1f}% vs mkt {r['market_prob']*100:>5.1f}%  edge {r['edge_pp']:>+6.1f}pp")
        print("\n  NOTE: an 'edge' is only real money if the model's IC beats the market's.")
        print("  Proving that needs realised results for matches WITH odds (historical")
        print("  bookmaker lines) — the V4 P4 past-tournament backtest. We don't launder")
        print("  the market's opinion into the model until that bar is cleared.")


if __name__ == "__main__":
    main()
