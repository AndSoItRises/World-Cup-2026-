"""
Learning Path — Step 3: Backtesting rigor & the overfitting trap.

The #1 way quant models fail is look-ahead bias and tuning-to-the-test. This
project used a single temporal split; real quant uses WALK-FORWARD (expanding
window) validation: retrain at each season boundary, predict ONLY the next
season, never peeking forward.

This script:
  1. Walk-forward over seasons: for each cutoff year Y, train on all matches
     < Y, predict year Y, record log loss & accuracy. This is the honest
     estimate of out-of-sample performance.
  2. The p-hacking demo: also fit a model that picks its iteration count by
     peeking at each test season (look-ahead). Watch the peeking model report
     a flattering number that the walk-forward number doesn't support — the
     overfitting trap, felt in our own data.

Concepts: look-ahead bias, multiple-testing / p-hacking, why the non-negotiables
("no leakage", "validate out-of-sample") exist.

Run: python -m src.models.backtest
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

from src.models.train_xgb import XGB_PARAMS, DRAW_CLASS_WEIGHT
from src.models.train_v4 import FEATURE_COLS_V4

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"
START_YEAR = 2014   # first season to score (need enough history before it)


def load():
    tr = pd.read_csv(DATA_PROC / "train_features.csv", parse_dates=["date"])
    te = pd.read_csv(DATA_PROC / "test_features.csv", parse_dates=["date"])
    df = pd.concat([tr, te], ignore_index=True).sort_values("date").reset_index(drop=True)
    for c in FEATURE_COLS_V4:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    df["neutral"] = df["neutral"].astype(int)
    df["year"] = df["date"].dt.year
    return df


def fit_predict(train, test, n_estimators=None, early_stop_on_test=False):
    X_tr, y_tr = train[FEATURE_COLS_V4], train["result"]
    X_te, y_te = test[FEATURE_COLS_V4], test["result"]
    w = np.ones(len(train)); w[y_tr.values == 1] *= DRAW_CLASS_WEIGHT
    params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    if n_estimators:
        params["n_estimators"] = n_estimators
    kwargs = {}
    if early_stop_on_test:  # the p-hack: peek at the test season for early stopping
        params["early_stopping_rounds"] = 30
        kwargs = dict(eval_set=[(X_te, y_te)], verbose=False)
    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, y_tr, sample_weight=w, **kwargs)
    p = m.predict_proba(X_te)
    return log_loss(y_te, p, labels=[0, 1, 2]), accuracy_score(y_te, np.argmax(p, axis=1))


def main():
    df = load()
    years = [y for y in sorted(df["year"].unique()) if y >= START_YEAR]

    print("═" * 60)
    print("  Walk-forward backtest (expanding window) — honest OOS")
    print("═" * 60)
    print(f"  {'season':<8}{'n_test':>8}{'WF logloss':>13}{'WF acc':>9}{'  | peek logloss':>18}")
    wf_lls, peek_lls = [], []
    for y in years:
        train = df[df["year"] < y]
        test = df[df["year"] == y]
        if len(test) < 30 or len(train) < 500:
            continue
        wf_ll, wf_acc = fit_predict(train, test)
        peek_ll, _ = fit_predict(train, test, early_stop_on_test=True)
        wf_lls.append(wf_ll); peek_lls.append(peek_ll)
        print(f"  {y:<8}{len(test):>8}{wf_ll:>13.4f}{wf_acc:>9.3f}{peek_ll:>18.4f}")

    print(f"\n  Mean walk-forward log loss : {np.mean(wf_lls):.4f}   (the number to trust)")
    print(f"  Mean peeking   log loss    : {np.mean(peek_lls):.4f}   (flattering — used the test season)")
    print(f"  Optimism gap from peeking  : {np.mean(wf_lls) - np.mean(peek_lls):+.4f}")
    print("\n  Lesson: the peeking model looks better only because it tuned on the")
    print("  season it was scored on. Walk-forward is the honest estimate.")


if __name__ == "__main__":
    main()
