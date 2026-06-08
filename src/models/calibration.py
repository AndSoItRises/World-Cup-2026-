"""
Learning Path — Step 1: Calibration & proper scoring rules.

A probability of 30% should be right 30% of the time. Proper scoring rules
(log loss, Brier) are minimised only by HONEST probabilities — which is why this
project never optimised accuracy. This script:

  1. Builds a reliability table per outcome (home/draw/away): predicted prob bins
     vs realised frequency. Perfect calibration → points on the diagonal.
  2. Computes the multiclass Brier score (mean squared error of the prob vector).
  3. Recalibrates with Platt scaling (logistic) and isotonic regression, then
     measures whether recalibration improves test log loss + Brier.

Concepts: sharpness vs calibration; why a well-calibrated 55% beats a
miscalibrated 90%; why we score with log loss, not accuracy.

Run: python -m src.models.calibration            # uses v4 ensemble
     python -m src.models.calibration v3          # compare a different version
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"
MODELS = BASE / "models"
LABELS = ["away_win", "draw", "home_win"]


def brier_multiclass(proba, y):
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def reliability_table(p_class, y_bin, n_bins=10):
    """Per-class reliability: mean predicted vs observed frequency in each bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p_class, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append((f"{bins[b]:.1f}-{bins[b+1]:.1f}", int(m.sum()),
                     float(p_class[m].mean()), float(y_bin[m].mean())))
    return rows


def main():
    ver = sys.argv[1] if len(sys.argv) > 1 else "v4"
    proba = np.load(MODELS / f"ensemble_test_proba_{ver}.npy")
    test = pd.read_csv(DATA_PROC / "test_features.csv")
    y = test["result"].astype(int).values

    print("═" * 60)
    print(f"  Calibration & proper scoring — ensemble {ver}")
    print("═" * 60)
    print(f"  Test matches: {len(y)}")
    print(f"\n  Log loss : {log_loss(y, proba, labels=[0,1,2]):.4f}")
    print(f"  Brier    : {brier_multiclass(proba, y):.4f}")

    print("\n── Reliability per outcome (pred vs observed; want pred≈obs) ──")
    for c, name in enumerate(LABELS):
        print(f"\n  {name}:")
        print(f"    {'bin':<12}{'n':>6}{'pred':>8}{'obs':>8}{'gap':>8}")
        for label, n, pred, obs in reliability_table(proba[:, c], (y == c).astype(int)):
            print(f"    {label:<12}{n:>6}{pred:>8.3f}{obs:>8.3f}{pred-obs:>+8.3f}")

    # ── Recalibration: fit on half the test set, evaluate on the other half ──
    # (A true pipeline calibrates on a held-out fold; here we split test for a
    #  clean before/after read without touching training.)
    print("\n── Recalibration (Platt & isotonic, fit/eval split) ──")
    i_fit, i_ev = train_test_split(np.arange(len(y)), test_size=0.5, random_state=42, stratify=y)
    base_ll = log_loss(y[i_ev], proba[i_ev], labels=[0, 1, 2])

    # Platt: multinomial logistic on the log-probabilities
    platt = LogisticRegression(max_iter=2000)
    platt.fit(np.log(proba[i_fit] + 1e-9), y[i_fit])
    p_platt = platt.predict_proba(np.log(proba[i_ev] + 1e-9))

    # Isotonic per class, then renormalise
    iso_cols = []
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(proba[i_fit, c], (y[i_fit] == c).astype(int))
        iso_cols.append(ir.predict(proba[i_ev, c]))
    p_iso = np.clip(np.vstack(iso_cols).T, 1e-9, None)
    p_iso = p_iso / p_iso.sum(axis=1, keepdims=True)

    print(f"  {'method':<14}{'log loss':>10}{'Brier':>10}")
    print(f"  {'raw':<14}{base_ll:>10.4f}{brier_multiclass(proba[i_ev], y[i_ev]):>10.4f}")
    print(f"  {'Platt':<14}{log_loss(y[i_ev], p_platt, labels=[0,1,2]):>10.4f}{brier_multiclass(p_platt, y[i_ev]):>10.4f}")
    print(f"  {'isotonic':<14}{log_loss(y[i_ev], p_iso, labels=[0,1,2]):>10.4f}{brier_multiclass(p_iso, y[i_ev]):>10.4f}")
    print("\n  Interpretation: if Platt/isotonic barely beat 'raw', the ensemble is")
    print("  already well-calibrated (good). Large gains ⇒ systematic over/under-confidence.")


if __name__ == "__main__":
    main()
