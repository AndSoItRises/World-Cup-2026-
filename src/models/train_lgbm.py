"""
Phase 5c: LightGBM Model Training
Direct comparison against XGBoost using identical features and evaluation.

Outputs:
  models/lgbm_v2.txt           (trained model)
  models/lgbm_report_v2.json      (metrics, feature importance)

Run with:
  python -m src.models.train_lgbm
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss, classification_report
import lightgbm as lgb

warnings.filterwarnings("ignore")

BASE       = Path(__file__).resolve().parents[2]
DATA_PROC  = BASE / "data" / "processed"
MODELS_DIR = BASE / "models"

TRAIN_PATH = DATA_PROC / "train_features.csv"
TEST_PATH  = DATA_PROC / "test_features.csv"

FEATURE_COLS = [
    # FIFA rankings
    "home_fifa_rank", "away_fifa_rank", "fifa_rank_diff",
    # ELO (V2)
    "home_elo", "away_elo", "elo_diff",
    # Simple rolling form
    "home_win_rate_5", "home_avg_goals_5", "home_avg_gd_5",
    "home_win_rate_10", "home_avg_goals_10", "home_avg_gd_10",
    "away_win_rate_5", "away_avg_goals_5", "away_avg_gd_5",
    "away_win_rate_10", "away_avg_goals_10", "away_avg_gd_10",
    # Quality-weighted rolling form (V2)
    "home_weighted_win_rate_5",  "home_weighted_avg_goals_5",  "home_weighted_avg_gd_5",
    "home_weighted_win_rate_10", "home_weighted_avg_goals_10", "home_weighted_avg_gd_10",
    "away_weighted_win_rate_5",  "away_weighted_avg_goals_5",  "away_weighted_avg_gd_5",
    "away_weighted_win_rate_10", "away_weighted_avg_goals_10", "away_weighted_avg_gd_10",
    # H2H
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "h2h_total", "h2h_home_win_rate",
    # Context
    "home_days_rest", "away_days_rest",
    "is_knockout", "altitude_m",
    "tournament_tier", "neutral",
]

TARGET     = "result"
WEIGHT_COL = "sample_weight"
LABELS     = {0: "away_win", 1: "draw", 2: "home_win"}

# Draw upweighting — must match train_xgb.py so the ensemble blends
# two consistently-calibrated models. See train_xgb.py for rationale.
DRAW_CLASS_WEIGHT = 1.75


def load_and_prep():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["date"])

    for col in FEATURE_COLS:
        if col in train.columns and train[col].isna().any():
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col]  = test[col].fillna(median_val)

    train["neutral"] = train["neutral"].astype(int)
    test["neutral"]  = test["neutral"].astype(int)

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET]
    w_train = train[WEIGHT_COL].copy()

    # Upweight draws so the model stops assigning them near-zero probability
    w_train[y_train == 1] *= DRAW_CLASS_WEIGHT

    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET]

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Draw class weight: {DRAW_CLASS_WEIGHT}x")
    return train, X_train, y_train, w_train, X_test, y_test


LGBM_PARAMS = {
    "objective":       "multiclass",
    "num_class":       3,
    "n_estimators":    500,
    "learning_rate":   0.05,
    "max_depth":       4,
    "num_leaves":      15,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "random_state":    42,
    "n_jobs":          -1,
    "verbose":         -1,
}


def train_and_evaluate(X_train, y_train, w_train, X_test, y_test):
    print("\n── Training LightGBM ──")

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    print(f"  Best iteration: {model.best_iteration_}")

    proba = model.predict_proba(X_test)
    pred  = np.argmax(proba, axis=1)

    acc = accuracy_score(y_test, pred)
    ll  = log_loss(y_test, proba)

    print(f"  Test Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Test Log Loss : {ll:.4f}")

    report = classification_report(
        y_test, pred,
        target_names=["away_win", "draw", "home_win"],
        output_dict=True
    )
    print(f"\n  Per-Class Metrics:")
    for cls in ["away_win", "draw", "home_win"]:
        r = report[cls]
        print(f"  {cls:<12} precision: {r['precision']:.3f} | "
              f"recall: {r['recall']:.3f} | f1: {r['f1-score']:.3f}")

    return model, acc, ll, report


def print_feature_importance(model, top_n=15):
    print(f"\n── Top {top_n} Feature Importances (gain) ──")
    importance = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: -x[1]
    )
    for feat, score in importance[:top_n]:
        bar = "█" * int(score / importance[0][1] * 30)
        print(f"  {feat:<30} {score:>8.1f}  {bar}")
    return dict(importance)


def save_artifacts(model, importance, acc, ll, report):
    model.booster_.save_model(str(MODELS_DIR / "lgbm_v2.txt"))
    print(f"\n✅ Model saved: models/lgbm_v2.txt")

    out = {
        "model": "lgbm_v1",
        "features": FEATURE_COLS,
        "test_accuracy": round(acc, 4),
        "test_log_loss": round(ll, 4),
        "lgbm_params": LGBM_PARAMS,
        "classification_report": report,
        "feature_importance": {k: round(float(v), 2) for k, v in list(importance.items())[:20]},
    }
    with open(MODELS_DIR / "lgbm_report.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"✅ Report saved: models/lgbm_report_v2.json")


def main():
    print("═" * 60)
    print("  Phase 5c: LightGBM Model Training")
    print("═" * 60)

    _, X_train, y_train, w_train, X_test, y_test = load_and_prep()
    model, acc, ll, report = train_and_evaluate(X_train, y_train, w_train, X_test, y_test)
    importance = print_feature_importance(model)
    save_artifacts(model, importance, acc, ll, report)

    # Side-by-side vs XGBoost
    xgb_report_path = MODELS_DIR / "training_report_v2.json"
    if xgb_report_path.exists():
        with open(xgb_report_path) as f:
            xgb = json.load(f)
        print(f"\n── XGBoost vs LightGBM ──")
        print(f"  {'':20} {'XGBoost':>10} {'LightGBM':>10}")
        print(f"  {'Test Accuracy':20} {xgb['test_accuracy']:>10.4f} {acc:>10.4f}")
        print(f"  {'Test Log Loss':20} {xgb['test_log_loss']:>10.4f} {ll:>10.4f}")

    print(f"\nPhase 5c complete ✅")


if __name__ == "__main__":
    main()
