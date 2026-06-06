"""
Phase 7 (V2): Stacking Meta-Learner.
Replaces the hand-set ensemble weights (0.275 / 0.275 / 0.45) with a trained
logistic-regression meta-learner over the three base models' probabilities.

Methodology (leakage-safe):
  - Out-of-fold (OOF) base predictions on TRAIN via 5-fold TimeSeriesSplit.
    All three base models (XGB, LGBM, Dixon-Coles) are refit on each fold's
    train block and predict its val block. Decay + 1.75x draw weights applied.
  - Meta-learner (multinomial LogisticRegression) trains on the 9 OOF features
    (3 models x 3 classes) → 3-class output.
  - TEST: base models refit on full train, predict test; meta-learner blends.
  - Compared head-to-head against the fixed-weight ensemble on the test set.

Outputs:
  models/stacking_meta.json     (meta-learner coefficients + metrics)
  models/stacking_report_v2.json

Run with:
  python -m src.models.stacking
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

from src.models.train_xgb import FEATURE_COLS, XGB_PARAMS, DRAW_CLASS_WEIGHT
from src.models.train_lgbm import LGBM_PARAMS
from src.models.dixon_coles import fit_dixon_coles, predict_dataset
from src.models.ensemble import WEIGHTS as FIXED_WEIGHTS

warnings.filterwarnings("ignore")

BASE       = Path(__file__).resolve().parents[2]
DATA_PROC  = BASE / "data" / "processed"
MODELS_DIR = BASE / "models"
TRAIN_PATH = DATA_PROC / "train_features.csv"
TEST_PATH  = DATA_PROC / "test_features.csv"

TARGET = "result"
N_SPLITS = 5
DECAY_HALF_LIFE = 1460   # V3 P3: 4yr partial decay (CV-min on P1-corrected features)


def compute_decay(dates, ref, half_life=DECAY_HALF_LIFE):
    lam = np.log(2) / half_life
    days_ago = (ref - dates).dt.days.clip(lower=0)
    w = np.exp(-lam * days_ago)
    return w / w.mean()


def weights_for(df, y):
    ref = df["date"].max()
    w = compute_decay(df["date"], ref).values.copy()
    w[y.values == 1] *= DRAW_CLASS_WEIGHT
    return w


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


def base_preds(train_df, predict_df):
    """Fit XGB, LGBM, DC on train_df; return stacked [away,draw,home] x3 for predict_df."""
    X_tr = train_df[FEATURE_COLS]
    y_tr = train_df[TARGET]
    w_tr = weights_for(train_df, y_tr)
    X_pred = predict_df[FEATURE_COLS]

    xgb_m = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_pred, predict_df[TARGET])], verbose=False)
    xgb_p = xgb_m.predict_proba(X_pred)             # [away, draw, home]

    lgb_m = lgb.LGBMClassifier(**LGBM_PARAMS)
    lgb_m.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_pred, predict_df[TARGET])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    lgb_p = lgb_m.predict_proba(X_pred)             # [away, draw, home]

    # Dixon-Coles: fit on train_df, predict predict_df. predict_dataset returns
    # [p_home, p_draw, p_away] — reorder to [away, draw, home].
    teams, attack, defense, home_adv, rho = fit_dixon_coles(train_df)
    dc_raw = predict_dataset(predict_df, teams, attack, defense, home_adv, rho)
    dc_p = np.column_stack([dc_raw[:, 2], dc_raw[:, 1], dc_raw[:, 0]])

    return xgb_p, lgb_p, dc_p


def stack_features(xgb_p, lgb_p, dc_p):
    return np.hstack([xgb_p, lgb_p, dc_p])   # (n, 9)


def evaluate(proba, y, label):
    pred = np.argmax(proba, axis=1)
    acc = accuracy_score(y, pred)
    ll  = log_loss(y, proba, labels=[0, 1, 2])
    rep = classification_report(y, pred, target_names=["away_win", "draw", "home_win"],
                                output_dict=True, zero_division=0)
    print(f"  {label:<22} acc {acc:.4f} | log loss {ll:.4f} | draw recall {rep['draw']['recall']:.4f}")
    return {"label": label, "accuracy": round(acc, 4), "log_loss": round(ll, 4),
            "draw_recall": round(rep["draw"]["recall"], 4)}


def main():
    print("═" * 64)
    print("  Phase 7: Stacking Meta-Learner")
    print("═" * 64)

    train, test = load()
    y_train = train[TARGET]
    y_test  = test[TARGET].values

    # ── OOF base predictions on train ─────────────────────────────────────────
    print(f"\n── Generating OOF base predictions ({N_SPLITS}-fold TimeSeriesSplit) ──")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof = np.full((len(train), 9), np.nan)

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(train), 1):
        print(f"  Fold {fold}: train {len(tr_idx)} → val {len(val_idx)}")
        tr_df  = train.iloc[tr_idx]
        val_df = train.iloc[val_idx]
        xgb_p, lgb_p, dc_p = base_preds(tr_df, val_df)
        oof[val_idx] = stack_features(xgb_p, lgb_p, dc_p)

    oof_mask = ~np.isnan(oof).any(axis=1)
    X_meta = oof[oof_mask]
    y_meta = y_train.values[oof_mask]
    print(f"  OOF rows for meta-training: {oof_mask.sum()} / {len(train)}")

    # ── Train meta-learner ────────────────────────────────────────────────────
    print("\n── Training meta-learner (multinomial logistic regression) ──")
    meta = LogisticRegression(max_iter=2000, C=1.0)
    meta.fit(X_meta, y_meta)

    # ── TEST: base models on full train, then blend ──────────────────────────
    print("\n── Refitting base models on full train, predicting test ──")
    xgb_te, lgb_te, dc_te = base_preds(train, test)
    X_meta_test = stack_features(xgb_te, lgb_te, dc_te)
    stacked_proba = meta.predict_proba(X_meta_test)

    # Fixed-weight ensemble on the SAME test base predictions (apples to apples)
    fixed_proba = (FIXED_WEIGHTS["xgb"] * xgb_te +
                   FIXED_WEIGHTS["lgb"] * lgb_te +
                   FIXED_WEIGHTS["dc"]  * dc_te)

    print("\n── Test comparison ──")
    res_xgb   = evaluate(xgb_te,        y_test, "XGBoost")
    res_lgb   = evaluate(lgb_te,        y_test, "LightGBM")
    res_dc    = evaluate(dc_te,         y_test, "Dixon-Coles")
    res_fixed = evaluate(fixed_proba,   y_test, "Fixed-weight ensemble")
    res_stack = evaluate(stacked_proba, y_test, "Stacked meta-learner")

    improvement = res_fixed["log_loss"] - res_stack["log_loss"]
    verdict = "STACKING WINS" if improvement > 0 else "fixed weights win"
    print(f"\n  Δ log loss (fixed − stacked): {improvement:+.4f}  →  {verdict}")

    # ── Save ──────────────────────────────────────────────────────────────────
    meta_out = {
        "classes": meta.classes_.tolist(),
        "coef":    meta.coef_.tolist(),
        "intercept": meta.intercept_.tolist(),
        "feature_order": "[xgb_away,xgb_draw,xgb_home, lgb_away,lgb_draw,lgb_home, dc_away,dc_draw,dc_home]",
        "decay_half_life": DECAY_HALF_LIFE,
    }
    with open(MODELS_DIR / "stacking_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    report = {
        "models": [res_xgb, res_lgb, res_dc, res_fixed, res_stack],
        "stacked_minus_fixed_logloss": round(-improvement, 4),
        "verdict": verdict,
    }
    with open(MODELS_DIR / "stacking_report_v2.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Saved: {MODELS_DIR / 'stacking_meta.json'}")
    print(f"✅ Saved: {MODELS_DIR / 'stacking_report_v2.json'}")


if __name__ == "__main__":
    main()
