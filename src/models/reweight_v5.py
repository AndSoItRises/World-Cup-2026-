"""
V5 Lever 3: ensemble reweight. XGB improved in V4 (squad features), so the fixed
0.275/0.275/0.45 (XGB/LGBM/DC) blend may no longer be optimal. Re-tune — honestly.

Anti-overfit protocol: a grid search that tunes weights on the test set and reports
the test log loss would just overfit the test set. Instead we split the test set in
half (stratified), tune weights on half A → evaluate on held-out half B, and tune on
B → evaluate on A. The averaged HELD-OUT log loss is what we compare to the fixed
weights. Adopt the new weights only if they beat fixed out-of-sample by a real margin
(V3 found fixed weights near-optimal vs a stacked meta-learner — so the prior is "keep").

Run: python -m src.models.reweight_v5
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from src.models.train_v4 import FEATURE_COLS_V4
from src.models.ensemble import get_dc_proba, WEIGHTS as FIXED

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"
MODELS = BASE / "models"


def grid():
    """All (xgb,lgb,dc) weights on a 0.05 grid summing to 1."""
    out = []
    for a in range(0, 21):
        for b in range(0, 21 - a):
            c = 20 - a - b
            out.append((a / 20, b / 20, c / 20))
    return out


def best_weights(p_xgb, p_lgb, p_dc, y):
    best, best_ll = None, 1e9
    for w in grid():
        ll = log_loss(y, w[0] * p_xgb + w[1] * p_lgb + w[2] * p_dc, labels=[0, 1, 2])
        if ll < best_ll:
            best_ll, best = ll, w
    return best, best_ll


def main():
    test = pd.read_csv(DATA_PROC / "test_features.csv", parse_dates=["date"])
    for c in FEATURE_COLS_V4:
        if test[c].isna().any():
            test[c] = test[c].fillna(test[c].median())
    test["neutral"] = test["neutral"].astype(int)
    y = test["result"].astype(int).values

    xm = xgb.XGBClassifier(); xm.load_model(str(MODELS / "xgb_v4.json"))
    lm = lgb.Booster(model_file=str(MODELS / "lgbm_v4.txt"))
    p_xgb = xm.predict_proba(test[FEATURE_COLS_V4])
    p_lgb = lm.predict(test[FEATURE_COLS_V4].values)
    p_dc = get_dc_proba(test)

    fixed_w = (FIXED["xgb"], FIXED["lgb"], FIXED["dc"])
    print("═" * 60)
    print("  V5 ensemble reweight — honest held-out comparison")
    print("═" * 60)
    print(f"  Fixed weights (current): XGB {fixed_w[0]} / LGBM {fixed_w[1]} / DC {fixed_w[2]}")

    # Two-fold held-out: tune on one half, score the other
    iA, iB = train_test_split(np.arange(len(y)), test_size=0.5, random_state=42, stratify=y)
    held_fixed, held_tuned, tuned_ws = [], [], []
    for tune_idx, ev_idx in [(iA, iB), (iB, iA)]:
        w, _ = best_weights(p_xgb[tune_idx], p_lgb[tune_idx], p_dc[tune_idx], y[tune_idx])
        tuned_ws.append(w)
        ev = lambda ww: log_loss(y[ev_idx], ww[0]*p_xgb[ev_idx] + ww[1]*p_lgb[ev_idx] + ww[2]*p_dc[ev_idx], labels=[0,1,2])
        held_fixed.append(ev(fixed_w))
        held_tuned.append(ev(w))
        print(f"\n  tuned weights: XGB {w[0]:.2f} / LGBM {w[1]:.2f} / DC {w[2]:.2f}")
        print(f"    held-out LL  fixed {ev(fixed_w):.4f}  vs tuned {ev(w):.4f}")

    mf, mt = np.mean(held_fixed), np.mean(held_tuned)
    print(f"\n  Mean held-out log loss — fixed {mf:.4f} | tuned {mt:.4f} | Δ {mf-mt:+.4f}")
    adopt = (mf - mt) > 0.002 and tuned_ws[0] == tuned_ws[1]  # material + stable across folds
    print(f"  → {'ADOPT new weights' if adopt else 'KEEP fixed weights (tuning gain not robust)'}")
    if not adopt:
        print("    (consistent with V3: the fixed blend is near-optimal; reweighting overfits.)")


if __name__ == "__main__":
    main()
