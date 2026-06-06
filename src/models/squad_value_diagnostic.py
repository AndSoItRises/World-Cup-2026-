"""
V4 P1: Squad-value diagnostic.
Tests whether squad market value carries signal the model is MISSING — using the
betting market as the best-available proxy for true team quality (it prices talent
the model can't see). NOT a model change; a diagnostic to decide if V4's premise holds.

Key tests:
  1. corr(squad_value, market_prob) vs corr(squad_value, model_prob)
     — if value tracks the market more than the model, the model is blind to it.
  2. Does log(squad_value) predict market_prob AFTER controlling for model_prob?
     — incremental R² > 0 means value carries info the model lacks.
  3. Sign of corr(squad_value, edge) where edge = model − market
     — negative means high-value teams are UNDER-rated by the model (the Euro bias).

Caveat: the market is a proxy, not ground truth. The definitive test is historical
CV, which needs historical squad values we don't have. This is directional evidence.

Run with:
  python -m src.models.squad_value_diagnostic
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_RAW = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"
MODELS = BASE / "models"


def main():
    print("═" * 64)
    print("  V4 P1: Squad-Value Diagnostic")
    print("═" * 64)

    sv = pd.read_csv(DATA_RAW / "wc2026_squad_values.csv")
    md = pd.read_csv(DATA_PROC / "market_divergence.csv")  # team, market_prob, model_prob, edge

    df = sv.merge(md, on="team", how="inner")
    miss = set(sv["team"]) - set(df["team"])
    print(f"\n  Matched {len(df)}/{len(sv)} teams" + (f" | unmatched: {sorted(miss)}" if miss else ""))

    df["log_value"] = np.log(df["squad_value_gbp_m"])

    c_mkt = df["log_value"].corr(df["market_prob"])
    c_mdl = df["log_value"].corr(df["model_prob"])
    c_edge = df["log_value"].corr(df["edge"])
    print(f"\n── Correlations (log squad value vs …) ──")
    print(f"  market win prob : {c_mkt:+.3f}")
    print(f"  model  win prob : {c_mdl:+.3f}")
    print(f"  edge (model−mkt): {c_edge:+.3f}   (negative ⇒ model underrates valuable squads)")

    # Incremental R²: predict market_prob from model_prob, then + log_value
    y = df["market_prob"].values
    X1 = df[["model_prob"]].values
    X2 = df[["model_prob", "log_value"]].values
    r2_1 = LinearRegression().fit(X1, y).score(X1, y)
    reg2 = LinearRegression().fit(X2, y)
    r2_2 = reg2.score(X2, y)
    print(f"\n── Predicting MARKET prob ──")
    print(f"  R² from model_prob alone        : {r2_1:.3f}")
    print(f"  R² from model_prob + log_value  : {r2_2:.3f}")
    print(f"  Incremental R² from squad value : {r2_2 - r2_1:+.3f}")
    print(f"  log_value coefficient           : {reg2.coef_[1]:+.4f} (positive ⇒ adds signal)")

    # Biggest model-vs-value mismatches (overrated = high model%, low value)
    df["value_rank"] = df["squad_value_gbp_m"].rank(ascending=False).astype(int)
    df["model_rank"] = df["model_prob"].rank(ascending=False).astype(int)
    df["rank_gap"] = df["value_rank"] - df["model_rank"]   # +ve ⇒ model rates higher than value
    over = df.sort_values("rank_gap", ascending=False).head(6)
    under = df.sort_values("rank_gap").head(6)
    print(f"\n── Model OVER-rates vs squad value (inflated) ──")
    print(f"  {'team':<20}{'value£m':>9}{'val_rk':>7}{'mdl_rk':>7}{'model%':>8}{'mkt%':>7}")
    for _, r in over.iterrows():
        print(f"  {r['team']:<20}{r['squad_value_gbp_m']:>9.0f}{r['value_rank']:>7}{r['model_rank']:>7}"
              f"{r['model_prob']*100:>7.1f}%{r['market_prob']*100:>6.1f}%")
    print(f"\n── Model UNDER-rates vs squad value ──")
    print(f"  {'team':<20}{'value£m':>9}{'val_rk':>7}{'mdl_rk':>7}{'model%':>8}{'mkt%':>7}")
    for _, r in under.iterrows():
        print(f"  {r['team']:<20}{r['squad_value_gbp_m']:>9.0f}{r['value_rank']:>7}{r['model_rank']:>7}"
              f"{r['model_prob']*100:>7.1f}%{r['market_prob']*100:>6.1f}%")

    verdict = ("SIGNAL CONFIRMED" if (r2_2 - r2_1) > 0.02 and reg2.coef_[1] > 0
               else "weak / redundant with model")
    print(f"\n  → Squad value vs model: {verdict}")

    report = {
        "n_matched": int(len(df)),
        "corr_logvalue_market": round(float(c_mkt), 3),
        "corr_logvalue_model": round(float(c_mdl), 3),
        "corr_logvalue_edge": round(float(c_edge), 3),
        "r2_model_only": round(float(r2_1), 3),
        "r2_model_plus_value": round(float(r2_2), 3),
        "incremental_r2": round(float(r2_2 - r2_1), 3),
        "logvalue_coef": round(float(reg2.coef_[1]), 4),
        "verdict": verdict,
    }
    with open(MODELS / "squad_value_diagnostic.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Saved: {MODELS / 'squad_value_diagnostic.json'}")


if __name__ == "__main__":
    main()
