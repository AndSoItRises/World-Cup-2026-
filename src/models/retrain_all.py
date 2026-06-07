"""
Production retrain for the WC2026 forecast.
Trains XGB + LGBM + Dixon-Coles on ALL competitive data (train + test, through
the latest match ~2026-03) instead of just the pre-2022 train split. The test
split existed for validation (done in V3); the final forecast should learn from
every available match. Decay is off (V3), so weights are uniform × 1.75 on draws.

Saves to *_prod paths (validated v3 models kept intact):
  models/xgb_prod.json, models/lgbm_prod.txt, models/dixon_coles_params_prod.json

Run with:
  python -m src.models.retrain_all
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb

from src.models.train_xgb import FEATURE_COLS, XGB_PARAMS, DRAW_CLASS_WEIGHT
from src.models.train_lgbm import LGBM_PARAMS
from src.models.dixon_coles import fit_dixon_coles

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"
MODELS = BASE / "models"
TARGET = "result"


def main():
    print("═" * 60)
    print("  Production Retrain (all competitive data)")
    print("═" * 60)

    train = pd.read_csv(DATA_PROC / "train_features.csv", parse_dates=["date"])
    test  = pd.read_csv(DATA_PROC / "test_features.csv",  parse_dates=["date"])
    alld = pd.concat([train, test], ignore_index=True).sort_values("date").reset_index(drop=True)
    print(f"  All data: {len(alld):,} matches | {alld['date'].min().date()} → {alld['date'].max().date()}")

    for c in FEATURE_COLS:
        if c in alld.columns and alld[c].isna().any():
            alld[c] = alld[c].fillna(alld[c].median())
    alld["neutral"] = alld["neutral"].astype(int)

    X, y = alld[FEATURE_COLS], alld[TARGET]
    w = np.ones(len(alld))
    w[y.values == 1] *= DRAW_CLASS_WEIGHT

    # XGB — drop early stopping (no holdout when training on everything)
    xgb_params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X, y, sample_weight=w, verbose=False)
    xgb_model.save_model(str(MODELS / "xgb_prod.json"))
    print(f"  ✅ xgb_prod.json")

    lgb_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    lgb_model.fit(X, y, sample_weight=w)
    lgb_model.booster_.save_model(str(MODELS / "lgbm_prod.txt"))
    print(f"  ✅ lgbm_prod.txt")

    # Dixon-Coles on all data (uniform weights — decay off)
    alld["sample_weight"] = 1.0
    teams, attack, defense, home_adv, rho = fit_dixon_coles(alld)
    dc_out = {
        "model": "dixon_coles_prod",
        "home_advantage": round(float(home_adv), 6),
        "rho": round(float(rho), 6),
        "teams": {t: {"attack": round(float(attack[i]), 6),
                      "defense": round(float(defense[i]), 6)}
                  for i, t in enumerate(teams)},
    }
    with open(MODELS / "dixon_coles_params_prod.json", "w") as f:
        json.dump(dc_out, f, indent=2)
    print(f"  ✅ dixon_coles_params_prod.json ({len(teams)} teams)")
    print(f"\n  Production models trained on all {len(alld):,} matches.")


if __name__ == "__main__":
    main()
