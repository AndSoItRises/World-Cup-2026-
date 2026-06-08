"""
V4: train XGB + LGBM with the squad strength/depth features and produce the v4
ensemble test probabilities — apples-to-apples vs v3 (same params, same train/test
split, same draw upweight, same ensemble weights; Dixon-Coles is unchanged because
it does not use squad data).

Saves to v4 paths only (v1/v2/v3 untouched, per the non-negotiables):
  models/xgb_v4.json, models/lgbm_v4.txt, models/ensemble_test_proba_v4.npy
  models/training_report_v4.json

The validate-or-cut decision is made by validate_v4.py (v4 vs v3 on the 4 metrics)
plus a WC2026 bracket inspection. This script just reports the CV log loss delta.

Run: python -m src.models.train_v4
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

from src.models.train_xgb import FEATURE_COLS, XGB_PARAMS, DRAW_CLASS_WEIGHT
from src.models.train_lgbm import LGBM_PARAMS
from src.models.ensemble import get_dc_proba, WEIGHTS, blend
from src.features.build_squad_strength import SQUAD_MODEL_FEATURES as SQUAD_FEATURES

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"
MODELS = BASE / "models"
TARGET = "result"

# V4 squad features (squad_top11 dropped: collinear with strength; squad_n_quality
# dropped: redundant + noisy — both per signal_test). Defined canonically in
# build_squad_strength.SQUAD_MODEL_FEATURES.
FEATURE_COLS_V4 = FEATURE_COLS + SQUAD_FEATURES


def load():
    train = pd.read_csv(DATA_PROC / "train_features.csv", parse_dates=["date"])
    test = pd.read_csv(DATA_PROC / "test_features.csv", parse_dates=["date"])
    for c in FEATURE_COLS_V4:
        if c in train.columns and train[c].isna().any():
            m = train[c].median()
            train[c] = train[c].fillna(m)
            test[c] = test[c].fillna(m)
    train["neutral"] = train["neutral"].astype(int)
    test["neutral"] = test["neutral"].astype(int)
    return train, test


def cv_logloss_xgb(train, feats):
    """Same TimeSeriesSplit(5) CV as train_xgb, for a given feature set."""
    X, y = train[feats], train[TARGET]
    w = np.ones(len(train)); w[y.values == 1] *= DRAW_CLASS_WEIGHT
    tscv = TimeSeriesSplit(n_splits=5)
    lls = []
    for tr, va in tscv.split(X):
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(X.iloc[tr], y.iloc[tr], sample_weight=w[tr],
              eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        lls.append(log_loss(y.iloc[va], m.predict_proba(X.iloc[va]), labels=[0, 1, 2]))
    return float(np.mean(lls))


def main():
    print("═" * 60)
    print("  V4: train squad-augmented models (vs v3)")
    print("═" * 60)
    train, test = load()
    y_test = test[TARGET].values
    print(f"  Train {len(train):,} | Test {len(test):,}")
    print(f"  Features: {len(FEATURE_COLS)} (v3) → {len(FEATURE_COLS_V4)} (v4, +{len(SQUAD_FEATURES)} squad)")

    # ── CV log loss: v3 features vs v4 features (the validate-or-cut bar) ──
    print("\n── XGB CV log loss (TimeSeriesSplit 5) ──")
    cv_v3 = cv_logloss_xgb(train, FEATURE_COLS)
    cv_v4 = cv_logloss_xgb(train, FEATURE_COLS_V4)
    print(f"  v3 features : {cv_v3:.4f}")
    print(f"  v4 features : {cv_v4:.4f}")
    print(f"  Δ (v3−v4, positive = v4 better): {cv_v3 - cv_v4:+.4f}")

    # ── Train final v4 models on train split (same method as v3) ──
    Xtr, ytr = train[FEATURE_COLS_V4], train[TARGET]
    w = np.ones(len(train)); w[ytr.values == 1] *= DRAW_CLASS_WEIGHT
    Xte = test[FEATURE_COLS_V4]

    xgb_m = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_m.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xte, y_test)], verbose=False)
    xgb_m.save_model(str(MODELS / "xgb_v4.json"))
    xgb_p = xgb_m.predict_proba(Xte)

    lgb_params = {k: v for k, v in LGBM_PARAMS.items()}
    lgb_m = lgb.LGBMClassifier(**lgb_params)
    lgb_m.fit(Xtr, ytr, sample_weight=w)
    lgb_m.booster_.save_model(str(MODELS / "lgbm_v4.txt"))
    lgb_p = lgb_m.booster_.predict(Xte.values)

    dc_p = get_dc_proba(test)  # unchanged Dixon-Coles (v3 params)
    ens_p = blend(xgb_p, lgb_p, dc_p)
    np.save(str(MODELS / "ensemble_test_proba_v4.npy"), ens_p)

    # ── Headline test metrics (full comparison done in validate_v4) ──
    v3_ens = np.load(MODELS / "ensemble_test_proba_v3.npy")
    print("\n── Test-set ensemble (v3 vs v4) ──")
    for name, p in [("v3", v3_ens), ("v4", ens_p)]:
        acc = accuracy_score(y_test, np.argmax(p, axis=1))
        ll = log_loss(y_test, p, labels=[0, 1, 2])
        print(f"  {name}: acc {acc:.4f} | log loss {ll:.4f}")

    # ── XGB feature importance: where do squad features rank? ──
    imp = xgb_m.get_booster().get_score(importance_type="gain")
    imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    rank = {f: i + 1 for i, (f, _) in enumerate(imp)}
    print("\n── Squad feature importance rank (of {} used) ──".format(len(imp)))
    for f in SQUAD_FEATURES:
        print(f"  {f:<24} rank {rank.get(f, '—')}  gain {dict(imp).get(f, 0):.1f}")

    report = {
        "model": "v4_squad",
        "features": FEATURE_COLS_V4,
        "n_features": len(FEATURE_COLS_V4),
        "squad_features": SQUAD_FEATURES,
        "cv_logloss_v3_features": round(cv_v3, 4),
        "cv_logloss_v4_features": round(cv_v4, 4),
        "cv_delta": round(cv_v3 - cv_v4, 4),
        "ensemble_weights": WEIGHTS,
        "squad_importance_rank": {f: rank.get(f) for f in SQUAD_FEATURES},
    }
    with open(MODELS / "training_report_v4.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Saved xgb_v4.json, lgbm_v4.txt, ensemble_test_proba_v4.npy, training_report_v4.json")
    print(f"   Next: python -m src.models.validate_v4")


if __name__ == "__main__":
    main()
