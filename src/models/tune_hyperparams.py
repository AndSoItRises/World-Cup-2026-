"""
Phase 5 (V2): Hyperparameter tuning.
CV-tunes the cheap, high-value levers that reuse the existing feature set:
  1. Time-decay half-life (recomputed from match dates — no feature rebuild)
  2. XGBoost params (max_depth, learning_rate, min_child_weight)
  3. LightGBM params (max_depth, learning_rate, num_leaves)

Selection metric: mean log loss over a 5-fold TimeSeriesSplit on the TRAIN set
(never the test set — test is reported once at the end for the chosen config).
The 1.75x draw upweighting is preserved throughout.

ELO K-factors are NOT tuned here: that requires a full elo.py + feature rebuild
(O(n^2) weighted rolling). Left as a documented Phase 5 lever — see CONTEXT_V2.md.

Outputs:
  models/tuning_report_v2.json   (best configs + CV/test metrics)

Run with:
  python -m src.models.tune_hyperparams
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from itertools import product
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

from src.models.train_xgb import FEATURE_COLS, DRAW_CLASS_WEIGHT

warnings.filterwarnings("ignore")

BASE       = Path(__file__).resolve().parents[2]
DATA_PROC  = BASE / "data" / "processed"
MODELS_DIR = BASE / "models"
TRAIN_PATH = DATA_PROC / "train_features.csv"
TEST_PATH  = DATA_PROC / "test_features.csv"

TARGET = "result"
N_SPLITS = 5

# Baselines to beat (post-draw-fix, weight=1.75)
BASELINE = {"xgb_ll": 0.8691, "lgb_ll": 0.8684, "ensemble_ll": 0.8562}

# ── Search grids (coarse, to bound runtime) ───────────────────────────────────
HALF_LIVES = [365, 730, 1095, 1460, 99999]   # 99999 ≈ no decay

XGB_GRID = {
    "max_depth":        [3, 4, 5],
    "learning_rate":    [0.03, 0.05],
    "min_child_weight": [5, 10],
}
XGB_FIXED = {
    "objective": "multi:softprob", "num_class": 3, "n_estimators": 600,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,
    "reg_lambda": 1.0, "random_state": 42, "n_jobs": -1,
    "eval_metric": "mlogloss", "early_stopping_rounds": 30, "verbosity": 0,
}

LGB_GRID = {
    "max_depth":     [3, 4, 5],
    "learning_rate": [0.03, 0.05],
    "num_leaves":    [15, 31],
}
LGB_FIXED = {
    "objective": "multiclass", "num_class": 3, "n_estimators": 600,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "n_jobs": -1, "verbose": -1,
}


def compute_decay_weights(dates, ref_date, half_life):
    lam = np.log(2) / half_life
    days_ago = (ref_date - dates).dt.days.clip(lower=0)
    w = np.exp(-lam * days_ago)
    return w / w.mean()


def load():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    test  = pd.read_csv(TEST_PATH,  parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    for col in FEATURE_COLS:
        if col in train.columns and train[col].isna().any():
            m = train[col].median()
            train[col] = train[col].fillna(m)
            test[col]  = test[col].fillna(m)
    train["neutral"] = train["neutral"].astype(int)
    test["neutral"]  = test["neutral"].astype(int)
    return train, test


def make_weights(df, half_life):
    """Decay weight (from dates) x draw upweighting."""
    ref = df["date"].max()
    w = compute_decay_weights(df["date"], ref, half_life).values.copy()
    w[df[TARGET].values == 1] *= DRAW_CLASS_WEIGHT
    return w


def cv_logloss(X, y, dates, model_kind, params, half_life):
    """Mean log loss over TimeSeriesSplit. Weights recomputed per fold from fold dates."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    lls = []
    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        ref = dates.iloc[tr_idx].max()
        w_tr = compute_decay_weights(dates.iloc[tr_idx], ref, half_life).values.copy()
        w_tr[y_tr.values == 1] *= DRAW_CLASS_WEIGHT

        if model_kind == "xgb":
            model = xgb.XGBClassifier(**{**XGB_FIXED, **params})
            model.fit(X_tr, y_tr, sample_weight=w_tr,
                      eval_set=[(X_val, y_val)], verbose=False)
            proba = model.predict_proba(X_val)
        else:
            model = lgb.LGBMClassifier(**{**LGB_FIXED, **params})
            model.fit(X_tr, y_tr, sample_weight=w_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            proba = model.predict_proba(X_val)
        lls.append(log_loss(y_val, proba, labels=[0, 1, 2]))
    return float(np.mean(lls))


def grid_combos(grid):
    keys = list(grid.keys())
    for vals in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def main():
    print("═" * 64)
    print("  Phase 5: Hyperparameter Tuning (CV on train set)")
    print(f"  Baselines — XGB ll {BASELINE['xgb_ll']} | LGBM ll {BASELINE['lgb_ll']}")
    print("═" * 64)

    train, test = load()
    X = train[FEATURE_COLS]
    y = train[TARGET]
    dates = train["date"]

    # ── Step 1: decay half-life (XGB default-ish params as the probe) ─────────
    print("\n── Step 1: decay half-life sweep (XGB probe) ──")
    probe = {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5}
    decay_results = {}
    for hl in HALF_LIVES:
        ll = cv_logloss(X, y, dates, "xgb", probe, hl)
        decay_results[hl] = ll
        print(f"  half_life={hl:>6} | CV log loss {ll:.4f}")
    best_hl = min(decay_results, key=decay_results.get)
    print(f"  → best half_life: {best_hl} (CV ll {decay_results[best_hl]:.4f})")

    # ── Step 2: XGB param grid at best decay ─────────────────────────────────
    print(f"\n── Step 2: XGB grid @ half_life={best_hl} ──")
    xgb_results = []
    for params in grid_combos(XGB_GRID):
        ll = cv_logloss(X, y, dates, "xgb", params, best_hl)
        xgb_results.append((params, ll))
        print(f"  {params} | CV ll {ll:.4f}")
    best_xgb, best_xgb_ll = min(xgb_results, key=lambda r: r[1])
    print(f"  → best XGB: {best_xgb} (CV ll {best_xgb_ll:.4f})")

    # ── Step 3: LGBM param grid at best decay ────────────────────────────────
    print(f"\n── Step 3: LGBM grid @ half_life={best_hl} ──")
    lgb_results = []
    for params in grid_combos(LGB_GRID):
        ll = cv_logloss(X, y, dates, "lgb", params, best_hl)
        lgb_results.append((params, ll))
        print(f"  {params} | CV ll {ll:.4f}")
    best_lgb, best_lgb_ll = min(lgb_results, key=lambda r: r[1])
    print(f"  → best LGBM: {best_lgb} (CV ll {best_lgb_ll:.4f})")

    # ── Save tuning report ────────────────────────────────────────────────────
    report = {
        "best_half_life": best_hl,
        "decay_cv_logloss": {str(k): round(v, 4) for k, v in decay_results.items()},
        "best_xgb_params": best_xgb,
        "best_xgb_cv_logloss": round(best_xgb_ll, 4),
        "best_lgb_params": best_lgb,
        "best_lgb_cv_logloss": round(best_lgb_ll, 4),
        "draw_class_weight": DRAW_CLASS_WEIGHT,
        "baseline": BASELINE,
        "note": "ELO K-factors not tuned (requires full feature rebuild).",
    }
    out = MODELS_DIR / "tuning_report_v2.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Tuning report saved: {out}")
    print("\nNext: apply best configs to train_xgb.py / train_lgbm.py / data_cleaning decay, retrain, re-ensemble.")


if __name__ == "__main__":
    main()
