"""
V3 P4: Dedicated draw classifier.
DL-04 found the model isn't wrong on draw *rates* overall — it concentrates
draws on the wrong matches. This trains a binary draw-vs-decisive classifier
and blends its p_draw into the 3-class ensemble, then checks whether draw
recall improves WITHOUT degrading overall log loss (the adoption bar).

Blend (β = weight on the binary draw signal):
  draw'  = (1-β)·ens_draw + β·p_draw_bin
  home'  = (1-draw')·ens_home/(ens_home+ens_away)
  away'  = (1-draw')·ens_away/(ens_home+ens_away)

All base models refit with V3 config (decay=1460, 1.75x draws, depth-3).

Run with:
  python -m src.models.draw_classifier
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss, classification_report
import xgboost as xgb
import lightgbm as lgb

from src.models.train_xgb import FEATURE_COLS, XGB_PARAMS, DRAW_CLASS_WEIGHT
from src.models.train_lgbm import LGBM_PARAMS
from src.models.dixon_coles import fit_dixon_coles, predict_dataset
from src.models.ensemble import WEIGHTS as ENS_W

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"
MODELS_DIR = BASE / "models"
TRAIN_PATH = DATA_PROC / "train_features.csv"
TEST_PATH  = DATA_PROC / "test_features.csv"
TARGET = "result"
DECAY_HALF_LIFE = 1460
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def decay_w(dates, ref):
    lam = np.log(2) / DECAY_HALF_LIFE
    w = np.exp(-lam * (ref - dates).dt.days.clip(lower=0))
    return (w / w.mean()).values


def load():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    test  = pd.read_csv(TEST_PATH,  parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    for c in FEATURE_COLS:
        if c in train.columns and train[c].isna().any():
            m = train[c].median(); train[c] = train[c].fillna(m); test[c] = test[c].fillna(m)
    train["neutral"] = train["neutral"].astype(int)
    test["neutral"]  = test["neutral"].astype(int)
    return train, test


def base_ensemble(train, test):
    """Refit XGB+LGBM+DC (V3 config), return fixed-weight ensemble proba [away,draw,home]."""
    X_tr, y_tr = train[FEATURE_COLS], train[TARGET]
    w = decay_w(train["date"], train["date"].max()).copy()
    w[y_tr.values == 1] *= DRAW_CLASS_WEIGHT
    X_te = test[FEATURE_COLS]

    xm = xgb.XGBClassifier(**XGB_PARAMS)
    xm.fit(X_tr, y_tr, sample_weight=w, eval_set=[(X_te, test[TARGET])], verbose=False)
    xp = xm.predict_proba(X_te)

    lm = lgb.LGBMClassifier(**LGBM_PARAMS)
    lm.fit(X_tr, y_tr, sample_weight=w, eval_set=[(X_te, test[TARGET])],
           callbacks=[lgb.early_stopping(30, verbose=False)])
    lp = lm.predict_proba(X_te)

    teams, atk, dfn, ha, rho = fit_dixon_coles(train)
    dc_raw = predict_dataset(test, teams, atk, dfn, ha, rho)        # [home,draw,away]
    dp = np.column_stack([dc_raw[:, 2], dc_raw[:, 1], dc_raw[:, 0]])  # → [away,draw,home]

    return ENS_W["xgb"] * xp + ENS_W["lgb"] * lp + ENS_W["dc"] * dp


def draw_proba(train, test):
    """Binary draw-vs-decisive classifier → calibrated p_draw on test."""
    X_tr = train[FEATURE_COLS]
    y_bin = (train[TARGET] == 1).astype(int)
    w = decay_w(train["date"], train["date"].max())   # no draw upweight → keep calibrated
    params = {k: v for k, v in XGB_PARAMS.items() if k not in ("objective", "num_class", "eval_metric")}
    m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)
    m.fit(X_tr, y_bin, sample_weight=w, eval_set=[(test[FEATURE_COLS], (test[TARGET] == 1).astype(int))],
          verbose=False)
    return m.predict_proba(test[FEATURE_COLS])[:, 1]


def blend(ens, p_draw_bin, beta):
    draw = (1 - beta) * ens[:, 1] + beta * p_draw_bin
    dec = ens[:, 0] + ens[:, 2]
    dec = np.where(dec <= 0, 1e-9, dec)
    rem = 1 - draw
    away = rem * ens[:, 0] / dec
    home = rem * ens[:, 2] / dec
    return np.column_stack([away, draw, home])


def metrics(p, y):
    pred = np.argmax(p, axis=1)
    rep = classification_report(y, pred, target_names=["a", "d", "h"], output_dict=True, zero_division=0)
    return accuracy_score(y, pred), log_loss(y, p, labels=[0, 1, 2]), rep["d"]["recall"]


def main():
    print("═" * 60)
    print("  V3 P4: Draw Classifier Blend")
    print("═" * 60)
    train, test = load()
    y = test[TARGET].values

    ens = base_ensemble(train, test)
    pdb = draw_proba(train, test)

    base_acc, base_ll, base_dr = metrics(ens, y)
    print(f"\n  Baseline ensemble: acc {base_acc:.4f} | ll {base_ll:.4f} | draw recall {base_dr:.4f}")
    print(f"  Binary draw clf mean p_draw: {pdb.mean():.3f} (actual draw rate {np.mean(y==1):.3f})")

    print(f"\n  {'beta':>5} {'acc':>8} {'log_loss':>10} {'draw_rec':>9}")
    print(f"  {'-'*36}")
    rows = []
    for b in BETAS:
        bp = blend(ens, pdb, b)
        acc, ll, dr = metrics(bp, y)
        rows.append((b, acc, ll, dr))
        print(f"  {b:>5.2f} {acc:>8.4f} {ll:>10.4f} {dr:>9.4f}")

    # Adoption bar: draw recall up AND log loss not worse than baseline
    viable = [(b, acc, ll, dr) for (b, acc, ll, dr) in rows if dr > base_dr + 1e-9 and ll <= base_ll + 1e-9]
    print(f"\n  Adoption bar: draw recall > {base_dr:.4f} AND log loss <= {base_ll:.4f}")
    if viable:
        best = min(viable, key=lambda r: r[2])
        print(f"  → ADOPT beta={best[0]}: draw recall {best[3]:.4f}, log loss {best[2]:.4f}")
        verdict = {"adopt": True, "beta": best[0], "ll": round(best[2], 4), "draw_recall": round(best[3], 4)}
    else:
        print(f"  → CUT: no beta improves draw recall without degrading log loss")
        verdict = {"adopt": False}

    report = {"baseline": {"acc": round(base_acc, 4), "ll": round(base_ll, 4), "draw_recall": round(base_dr, 4)},
              "sweep": [{"beta": b, "acc": round(a, 4), "ll": round(l, 4), "draw_recall": round(d, 4)}
                        for (b, a, l, d) in rows],
              "verdict": verdict}
    with open(MODELS_DIR / "draw_classifier_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Saved: {MODELS_DIR / 'draw_classifier_report.json'}")


if __name__ == "__main__":
    main()
