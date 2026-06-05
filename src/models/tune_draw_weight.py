"""
Phase 4 (V2): Draw-weight sweep.
Trains XGB + LGBM in-memory at several DRAW_CLASS_WEIGHT values, blends each
with the saved Dixon-Coles params, and reports ENSEMBLE metrics so we can pick
the weight that lifts draw recall without regressing log loss.

This is a tuning utility — it does NOT save any model. Once a weight is chosen,
set DRAW_CLASS_WEIGHT in train_xgb.py and train_lgbm.py and run the official retrain.

Run with:
  python -m src.models.tune_draw_weight
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import accuracy_score, log_loss, classification_report
import xgboost as xgb
import lightgbm as lgb

from src.models.ensemble import (
    FEATURE_COLS, TARGET, WEIGHTS, get_dc_proba,
    TRAIN_PATH, TEST_PATH,
)
from src.models.train_xgb import XGB_PARAMS
from src.models.train_lgbm import LGBM_PARAMS

warnings.filterwarnings("ignore")

WEIGHT_COL = "sample_weight"
SWEEP = [1.0, 1.25, 1.5, 1.75, 2.0]   # 1.0 = no upweighting (baseline)


def load():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["date"])
    for col in FEATURE_COLS:
        if col in train.columns and train[col].isna().any():
            m = train[col].median()
            train[col] = train[col].fillna(m)
            test[col]  = test[col].fillna(m)
    train["neutral"] = train["neutral"].astype(int)
    test["neutral"]  = test["neutral"].astype(int)
    return train, test


def train_xgb_proba(X_train, y_train, w_train, X_test, y_test):
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, sample_weight=w_train,
              eval_set=[(X_test, y_test)], verbose=False)
    return model.predict_proba(X_test)


def train_lgb_proba(X_train, y_train, w_train, X_test, y_test):
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train, sample_weight=w_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(period=-1)])
    return model.predict_proba(X_test)


def metrics(proba, y_true):
    pred = np.argmax(proba, axis=1)
    acc  = accuracy_score(y_true, pred)
    ll   = log_loss(y_true, proba)
    rep  = classification_report(y_true, pred,
                                 target_names=["away_win", "draw", "home_win"],
                                 output_dict=True, zero_division=0)
    return acc, ll, rep["draw"]["recall"], rep["draw"]["precision"]


def main():
    print("═" * 70)
    print("  Draw-Weight Sweep (ensemble metrics)")
    print(f"  Baseline V2: acc 0.612 | log_loss 0.8477 | draw_recall 0.006")
    print("═" * 70)

    train, test = load()
    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_test,  y_test  = test[FEATURE_COLS],  test[TARGET].values
    w_base = train[WEIGHT_COL].values

    # DC proba is independent of the draw weight — compute once.
    print("\nLoading Dixon-Coles probabilities (once)...")
    dc_proba = get_dc_proba(test)

    print(f"\n  {'weight':>7} {'acc':>8} {'log_loss':>10} {'draw_rec':>9} {'draw_prec':>10}")
    print(f"  {'-'*46}")

    results = []
    for w in SWEEP:
        w_train = w_base.copy()
        w_train[y_train.values == 1] *= w

        xgb_p = train_xgb_proba(X_train, y_train, w_train, X_test, y_test)
        lgb_p = train_lgb_proba(X_train, y_train, w_train, X_test, y_test)

        ens = (WEIGHTS["xgb"] * xgb_p +
               WEIGHTS["lgb"] * lgb_p +
               WEIGHTS["dc"]  * dc_proba)

        acc, ll, dr, dp = metrics(ens, y_test)
        results.append((w, acc, ll, dr, dp))
        print(f"  {w:>7.2f} {acc:>8.4f} {ll:>10.4f} {dr:>9.4f} {dp:>10.4f}")

    print(f"  {'-'*46}")
    print("\n  Pick the highest draw_recall with log_loss <= 0.86 and acc within ~1pp of 0.612.")


if __name__ == "__main__":
    main()
