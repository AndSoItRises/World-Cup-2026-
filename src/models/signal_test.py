"""
V4 — signal_test: does a candidate feature carry signal ORTHOGONAL to what the
model already sees? (Also Learning-Path Step 6: orthogonality / incremental IC.)

The quant idea: a new feature is only worth adding if it adds information the
existing features lack. Raw correlation with winning lies — a feature can be
strongly predictive yet fully redundant with ELO/rank. So we:

  1. Orthogonalise: regress the candidate on the existing feature set, keep the
     RESIDUAL (the part nothing else explains).
  2. Incremental IC: Spearman corr of that residual with the match outcome.
     If raw IC is high but incremental IC ≈ 0, the feature is redundant — cut it.
  3. Incremental log loss: multinomial-logistic temporal CV on base features vs
     base + candidate. Positive ΔLL ⇒ the candidate genuinely helps a LINEAR
     model. (Trees may extract more; this is a fast, conservative screen — the
     definitive test is the full XGB/LGBM v4 retrain + validate harness.)

Reported on ALL matches and on the COVERED subset (both teams have real squad
data, not the sentinel) — the covered read isolates signal from coverage dilution.

Run:
  python -m src.models.signal_test                       # default: squad diffs
  python -m src.models.signal_test squad_strength_diff squad_depth_diff
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

from src.models.train_xgb import TARGET
from src.models.train_v4 import FEATURE_COLS_V4

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"

DEFAULT_CANDIDATES = [
    "squad_strength_diff", "squad_top11_diff", "squad_depth_diff", "squad_n_quality_diff",
]
# Heuristic sentinel match (feature_engineering.SQUAD_SENTINEL) to flag uncovered rows.
SENTINEL_STRENGTH = 60.0


def load(base):
    tr = pd.read_csv(DATA_PROC / "train_features.csv", parse_dates=["date"])
    te = pd.read_csv(DATA_PROC / "test_features.csv", parse_dates=["date"])
    df = pd.concat([tr, te], ignore_index=True).sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(int)
    # Median-impute base feature NaNs (same convention as train_xgb)
    for c in base:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


def cv_logloss(X, y, n_splits=5):
    """Multinomial-logistic temporal CV mean log loss (features standardized per fold)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    lls = []
    for tr_idx, va_idx in tscv.split(X):
        sc = StandardScaler().fit(X[tr_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr_idx]), y[tr_idx])
        p = clf.predict_proba(sc.transform(X[va_idx]))
        lls.append(log_loss(y[va_idx], p, labels=[0, 1, 2]))
    return float(np.mean(lls))


def report_block(df, candidates, base, label):
    print(f"\n{'═'*64}\n  {label}  (n={len(df)})\n{'═'*64}")
    y = df[TARGET].astype(int).values
    Xb = df[base].values

    print(f"\n  ── Orthogonality (raw IC vs incremental IC after removing base) ──")
    print(f"  {'candidate':<24}{'raw IC':>10}{'incr IC':>10}{'redundant?':>13}")
    for c in candidates:
        cv = df[c].values.astype(float)
        raw_ic = spearmanr(cv, y).correlation
        resid = cv - LinearRegression().fit(Xb, cv).predict(Xb)
        inc_ic = spearmanr(resid, y).correlation
        redundant = abs(inc_ic) < 0.02
        print(f"  {c:<24}{raw_ic:>+10.3f}{inc_ic:>+10.3f}{('YES' if redundant else 'no'):>13}")

    print(f"\n  ── Incremental predictive value (multinomial-logistic temporal CV) ──")
    ll_base = cv_logloss(Xb, y)
    ll_full = cv_logloss(df[base + candidates].values, y)
    print(f"  base features            CV log loss : {ll_base:.4f}")
    print(f"  base + candidates        CV log loss : {ll_full:.4f}")
    print(f"  ΔLL (positive = helps)               : {ll_base - ll_full:+.4f}")
    return ll_base - ll_full


def main():
    candidates = sys.argv[1:] or DEFAULT_CANDIDATES
    # Base = the CURRENT model's feature set (v4) minus the candidates, so a new
    # feature is judged for signal BEYOND everything already in the model.
    base = [f for f in FEATURE_COLS_V4 if f not in candidates]
    df = load(base + candidates)
    missing = [c for c in candidates if c not in df.columns]
    if missing:
        raise SystemExit(f"Candidate columns not found: {missing}")

    print("═" * 64)
    print("  signal_test — incremental information of candidate features")
    print("═" * 64)
    print(f"  Base feature set : {len(base)} features (current model minus candidates)")
    print(f"  Candidates       : {candidates}")

    delta_all = report_block(df, candidates, base, "ALL MATCHES")

    covered = df[(df["home_squad_strength"] != SENTINEL_STRENGTH) &
                 (df["away_squad_strength"] != SENTINEL_STRENGTH)].reset_index(drop=True)
    delta_cov = report_block(covered, candidates, base, "COVERED SUBSET (both teams real squad data)")

    print(f"\n{'─'*64}")
    verdict = ("PROMISING — carries orthogonal signal" if (delta_cov > 0.001 or delta_all > 0.0005)
               else "WEAK / REDUNDANT — likely cut")
    print(f"  VERDICT: {verdict}")
    print(f"  (ΔLL all={delta_all:+.4f}, covered={delta_cov:+.4f}; definitive test = full v4 retrain)")


if __name__ == "__main__":
    main()
