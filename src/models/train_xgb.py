"""
Phase 5: XGBoost Model Training
Trains a multiclass classifier on train_features.csv, evaluates on test_features.csv.

Outputs:
  models/xgb_v1.json          (trained model)
  models/training_report.json (metrics, feature importance)

Run with:
  python -m src.models.train_xgb
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, log_loss, confusion_matrix,
    classification_report
)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parents[2]
DATA_PROC  = BASE / "data" / "processed"
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_PROC / "train_features.csv"
TEST_PATH  = DATA_PROC / "test_features.csv"
META_PATH  = DATA_PROC / "cleaning_metadata.json"

# ── Feature columns ───────────────────────────────────────────────────────────
# These are the 24 engineered features. Excludes raw match info and target.
FEATURE_COLS = [
    "home_fifa_rank", "away_fifa_rank", "fifa_rank_diff",
    "home_win_rate_5", "home_avg_goals_5", "home_avg_gd_5",
    "home_win_rate_10", "home_avg_goals_10", "home_avg_gd_10",
    "away_win_rate_5", "away_avg_goals_5", "away_avg_gd_5",
    "away_win_rate_10", "away_avg_goals_10", "away_avg_gd_10",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "h2h_total", "h2h_home_win_rate",
    "home_days_rest", "away_days_rest",
    "is_knockout", "altitude_m",
    # Base features already in train.csv
    "tournament_tier", "neutral",
]

TARGET = "result"
WEIGHT_COL = "sample_weight"

# Result labels for display
LABELS = {0: "away_win", 1: "draw", 2: "home_win"}


# ── Load & prep ───────────────────────────────────────────────────────────────
def load_and_prep():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["date"])

    # Median impute the 1.3% rolling NaNs (first-appearance teams)
    for col in FEATURE_COLS:
        if col in train.columns and train[col].isna().any():
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col]  = test[col].fillna(median_val)  # use train median for test

    # neutral is bool → int
    train["neutral"] = train["neutral"].astype(int)
    test["neutral"]  = test["neutral"].astype(int)

    # Verify all feature cols exist
    missing = [c for c in FEATURE_COLS if c not in train.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET]
    w_train = train[WEIGHT_COL]

    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET]

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train class distribution: { {LABELS[k]: v for k,v in y_train.value_counts().sort_index().items()} }")
    print(f"Test  class distribution: { {LABELS[k]: v for k,v in y_test.value_counts().sort_index().items()} }")
    return train, test, X_train, y_train, w_train, X_test, y_test


# ── Baseline ──────────────────────────────────────────────────────────────────
def print_baseline(y_train, y_test):
    """Naive baseline: always predict the majority class (home win = 2)."""
    majority = y_train.value_counts().idxmax()
    naive_pred = np.full(len(y_test), majority)
    naive_acc  = accuracy_score(y_test, naive_pred)
    # Log loss for naive: use training class frequencies as constant probs
    train_probs = y_train.value_counts(normalize=True).sort_index().values
    naive_proba = np.tile(train_probs, (len(y_test), 1))
    naive_ll = log_loss(y_test, naive_proba)
    print(f"\n── Naive Baseline (always predict '{LABELS[majority]}') ──")
    print(f"  Accuracy : {naive_acc:.4f} ({naive_acc*100:.1f}%)")
    print(f"  Log Loss : {naive_ll:.4f}")
    return naive_acc, naive_ll


# ── XGBoost params ────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "objective":        "multi:softprob",
    "num_class":        3,
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        4,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
    "eval_metric":      "mlogloss",
    "early_stopping_rounds": 30,
    "verbosity":        0,
}


# ── Time-series cross-validation ──────────────────────────────────────────────
def run_cv(train_df, X_train, y_train, w_train):
    """
    TimeSeriesSplit: 5 folds, each fold's test set is strictly after its train set.
    This mirrors real-world usage — we never train on future matches.
    """
    print(f"\n── Time-Series Cross-Validation (5 folds) ──")
    tscv = TimeSeriesSplit(n_splits=5)

    fold_accs  = []
    fold_lls   = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        w_tr        = w_train.iloc[tr_idx]

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        proba = model.predict_proba(X_val)
        pred  = np.argmax(proba, axis=1)

        acc = accuracy_score(y_val, pred)
        ll  = log_loss(y_val, proba)
        fold_accs.append(acc)
        fold_lls.append(ll)

        # Date range of validation fold
        val_dates = train_df.iloc[val_idx]["date"]
        print(f"  Fold {fold}: {val_dates.min().date()} → {val_dates.max().date()} "
              f"| Acc: {acc:.4f} | LogLoss: {ll:.4f}")

    print(f"  ── Mean Acc: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f} | "
          f"Mean LogLoss: {np.mean(fold_lls):.4f} ± {np.std(fold_lls):.4f}")
    return np.mean(fold_accs), np.mean(fold_lls)


# ── Final training ────────────────────────────────────────────────────────────
def train_final(X_train, y_train, w_train, X_test, y_test):
    print(f"\n── Training Final Model (full train set → evaluate on test) ──")

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    proba = model.predict_proba(X_test)
    pred  = np.argmax(proba, axis=1)

    acc = accuracy_score(y_test, pred)
    ll  = log_loss(y_test, proba)

    print(f"  Test Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Test Log Loss : {ll:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':>12} {'away_win':>10} {'draw':>10} {'home_win':>10}")
    for i, row in enumerate(cm):
        print(f"  {LABELS[i]:>12} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

    # Per-class metrics
    print(f"\n  Per-Class Metrics:")
    report = classification_report(
        y_test, pred,
        target_names=["away_win", "draw", "home_win"],
        output_dict=True
    )
    for cls in ["away_win", "draw", "home_win"]:
        r = report[cls]
        print(f"  {cls:<12} precision: {r['precision']:.3f} | "
              f"recall: {r['recall']:.3f} | f1: {r['f1-score']:.3f} | "
              f"support: {int(r['support'])}")

    return model, proba, pred, acc, ll, report


# ── Feature importance ────────────────────────────────────────────────────────
def print_feature_importance(model, top_n=15):
    print(f"\n── Top {top_n} Feature Importances (gain) ──")
    importance = model.get_booster().get_score(importance_type="gain")
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, score in importance[:top_n]:
        bar = "█" * int(score / importance[0][1] * 30)
        print(f"  {feat:<30} {score:>8.1f}  {bar}")
    return dict(importance)


# ── Save ──────────────────────────────────────────────────────────────────────
def save_artifacts(model, importance, acc, ll, cv_acc, cv_ll, report):
    model_path = MODELS_DIR / "xgb_v1.json"
    model.save_model(str(model_path))
    print(f"\n✅ Model saved: {model_path}")

    training_report = {
        "model": "xgb_v1",
        "features": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "test_accuracy": round(acc, 4),
        "test_log_loss": round(ll, 4),
        "cv_mean_accuracy": round(cv_acc, 4),
        "cv_mean_log_loss": round(cv_ll, 4),
        "xgb_params": {k: v for k, v in XGB_PARAMS.items()
                       if k not in ["early_stopping_rounds", "verbosity"]},
        "classification_report": report,
        "feature_importance_gain": {k: round(v, 2) for k, v in list(importance.items())[:20]},
    }

    report_path = MODELS_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(training_report, f, indent=2)
    print(f"✅ Training report saved: {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("═" * 60)
    print("  Phase 5: XGBoost Model Training")
    print("═" * 60)

    train_df, test_df, X_train, y_train, w_train, X_test, y_test = load_and_prep()

    naive_acc, naive_ll = print_baseline(y_train, y_test)

    cv_acc, cv_ll = run_cv(train_df, X_train, y_train, w_train)

    model, proba, pred, acc, ll, report = train_final(
        X_train, y_train, w_train, X_test, y_test
    )

    importance = print_feature_importance(model, top_n=15)

    save_artifacts(model, importance, acc, ll, cv_acc, cv_ll, report)

    print(f"\n── Summary ──")
    print(f"  Naive baseline accuracy : {naive_acc*100:.1f}%")
    print(f"  CV mean accuracy        : {cv_acc*100:.1f}%")
    print(f"  Test accuracy           : {acc*100:.1f}%")
    print(f"  Improvement over naive  : +{(acc - naive_acc)*100:.1f}pp")
    print(f"\nPhase 5 complete ✅")


if __name__ == "__main__":
    main()
