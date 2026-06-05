"""
Phase 6: Ensemble Model
Combines XGBoost, LightGBM, and Dixon-Coles via weighted average.

Weights: XGB 35% | LGBM 35% | DC 30%
(Adjust WEIGHTS dict below to experiment.)

Outputs:
  models/ensemble_test_proba.npy   (blended probabilities on test set)
  models/ensemble_report.json      (comparison metrics across all models)

Run with:
  python -m src.models.ensemble
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, log_loss, classification_report
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parents[2]
DATA_PROC  = BASE / "data" / "processed"
MODELS_DIR = BASE / "models"

TRAIN_PATH = DATA_PROC / "train_features.csv"
TEST_PATH  = DATA_PROC / "test_features.csv"

XGB_MODEL_PATH = MODELS_DIR / "xgb_v1.json"
LGB_MODEL_PATH = MODELS_DIR / "lgbm_v1.txt"
DC_PARAMS_PATH = MODELS_DIR / "dixon_coles_params.json"

# ── Ensemble weights (must sum to 1.0) ────────────────────────────────────────
WEIGHTS = {
    "xgb": 0.275,
    "lgb": 0.275,
    "dc":  0.45,
}

# ── Feature columns (same as training) ────────────────────────────────────────
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
    "tournament_tier", "neutral",
]

TARGET = "result"
LABELS = {0: "away_win", 1: "draw", 2: "home_win"}
MAX_GOALS = 10


# ── Load & prep data ──────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["date"])

    for col in FEATURE_COLS:
        if col in train.columns and train[col].isna().any():
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col]  = test[col].fillna(median_val)

    train["neutral"] = train["neutral"].astype(int)
    test["neutral"]  = test["neutral"].astype(int)

    print(f"Train: {train.shape} | Test: {test.shape}")
    return train, test


# ── XGBoost probabilities ─────────────────────────────────────────────────────
def get_xgb_proba(test: pd.DataFrame) -> np.ndarray:
    """Load saved XGB model, return proba array shape (n, 3)."""
    model = xgb.XGBClassifier()
    model.load_model(str(XGB_MODEL_PATH))
    X_test = test[FEATURE_COLS]
    proba = model.predict_proba(X_test)   # cols: [away_win, draw, home_win]
    print(f"  XGB loaded  | shape: {proba.shape}")
    return proba


# ── LightGBM probabilities ────────────────────────────────────────────────────
def get_lgb_proba(test: pd.DataFrame) -> np.ndarray:
    """Load saved LGBM model, return proba array shape (n, 3)."""
    booster = lgb.Booster(model_file=str(LGB_MODEL_PATH))
    X_test = test[FEATURE_COLS].values
    raw = booster.predict(X_test)         # shape (n, 3), cols: [away_win, draw, home_win]
    print(f"  LGBM loaded | shape: {raw.shape}")
    return raw


# ── Dixon-Coles probabilities ─────────────────────────────────────────────────
def tau(x, y, lam, mu, rho):
    if x == 0 and y == 0:   return 1 - lam * mu * rho
    elif x == 0 and y == 1: return 1 + lam * rho
    elif x == 1 and y == 0: return 1 + mu * rho
    elif x == 1 and y == 1: return 1 - rho
    else:                   return 1.0


def dc_predict_match(home_team, away_team, is_neutral, dc):
    """Return (p_away_win, p_draw, p_home_win) matching column order of XGB/LGBM."""
    teams = dc["teams"]
    if home_team not in teams or away_team not in teams:
        return (1/3, 1/3, 1/3)

    ht = teams[home_team]
    at = teams[away_team]
    home_adv = dc["home_advantage"]
    rho      = dc["rho"]

    lam = ht["attack"] * at["defense"] * (1.0 if is_neutral else home_adv)
    mu  = at["attack"] * ht["defense"]

    p_home = p_draw = p_away = 0.0
    for x in range(MAX_GOALS + 1):
        for y in range(MAX_GOALS + 1):
            t = tau(x, y, lam, mu, rho)
            p = t * poisson.pmf(x, lam) * poisson.pmf(y, mu)
            if x > y:   p_home += p
            elif x == y: p_draw += p
            else:        p_away += p

    total = p_home + p_draw + p_away
    # Return order: [away_win, draw, home_win] — matches XGB/LGBM
    return (p_away / total, p_draw / total, p_home / total)


def get_dc_proba(test: pd.DataFrame) -> np.ndarray:
    """Load saved DC params, return proba array shape (n, 3)."""
    with open(DC_PARAMS_PATH) as f:
        dc = json.load(f)

    proba = []
    for _, row in test.iterrows():
        p = dc_predict_match(
            row["home_team"], row["away_team"], bool(row["neutral"]), dc
        )
        proba.append(p)

    proba = np.array(proba)
    print(f"  DC loaded   | shape: {proba.shape}")
    return proba


# ── Blend probabilities ───────────────────────────────────────────────────────
def blend(xgb_p, lgb_p, dc_p) -> np.ndarray:
    """Weighted average of three probability arrays."""
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"
    blended = (
        WEIGHTS["xgb"] * xgb_p +
        WEIGHTS["lgb"] * lgb_p +
        WEIGHTS["dc"]  * dc_p
    )
    return blended


# ── Evaluate one set of probabilities ─────────────────────────────────────────
def evaluate(proba: np.ndarray, y_true: np.ndarray, label: str) -> dict:
    pred = np.argmax(proba, axis=1)
    acc  = accuracy_score(y_true, pred)
    ll   = log_loss(y_true, proba)

    report = classification_report(
        y_true, pred,
        target_names=["away_win", "draw", "home_win"],
        output_dict=True
    )
    draw_recall = report["draw"]["recall"]

    return {
        "label":       label,
        "accuracy":    acc,
        "log_loss":    ll,
        "draw_recall": draw_recall,
        "report":      report,
    }


# ── Print comparison table ────────────────────────────────────────────────────
def print_comparison(results: list[dict]):
    print(f"\n{'═'*60}")
    print(f"  Ensemble Comparison")
    print(f"{'═'*60}")
    header = f"  {'Model':<16} {'Accuracy':>10} {'Log Loss':>10} {'Draw Recall':>12}"
    print(header)
    print(f"  {'-'*50}")
    for r in results:
        marker = " ◄" if r["label"] == "Ensemble" else ""
        print(f"  {r['label']:<16} {r['accuracy']:>10.4f} {r['log_loss']:>10.4f} {r['draw_recall']:>12.4f}{marker}")
    print(f"{'═'*60}")


# ── Save outputs ──────────────────────────────────────────────────────────────
def save_outputs(ensemble_proba: np.ndarray, results: list[dict]):
    # Save raw probabilities for Phase 6 Step 2
    npy_path = MODELS_DIR / "ensemble_test_proba.npy"
    np.save(str(npy_path), ensemble_proba)
    print(f"\n✅ Ensemble probabilities saved: {npy_path}")

    # Save report
    report_out = {
        "weights": WEIGHTS,
        "models": [
            {
                "label":       r["label"],
                "accuracy":    round(r["accuracy"], 4),
                "log_loss":    round(r["log_loss"], 4),
                "draw_recall": round(r["draw_recall"], 4),
            }
            for r in results
        ]
    }
    report_path = MODELS_DIR / "ensemble_report.json"
    with open(report_path, "w") as f:
        json.dump(report_out, f, indent=2)
    print(f"✅ Ensemble report saved:        {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("═" * 60)
    print("  Phase 6: Ensemble Model")
    print(f"  Weights — XGB: {WEIGHTS['xgb']} | LGBM: {WEIGHTS['lgb']} | DC: {WEIGHTS['dc']}")
    print("═" * 60)

    train, test = load_data()
    y_true = test[TARGET].values

    print("\n── Loading models ──")
    xgb_proba = get_xgb_proba(test)
    lgb_proba = get_lgb_proba(test)
    dc_proba  = get_dc_proba(test)

    print("\n── Blending ──")
    ensemble_proba = blend(xgb_proba, lgb_proba, dc_proba)
    print(f"  Ensemble proba shape: {ensemble_proba.shape}")

    print("\n── Evaluating ──")
    results = [
        evaluate(xgb_proba,      y_true, "XGBoost"),
        evaluate(lgb_proba,      y_true, "LightGBM"),
        evaluate(dc_proba,       y_true, "Dixon-Coles"),
        evaluate(ensemble_proba, y_true, "Ensemble"),
    ]

    print_comparison(results)
    save_outputs(ensemble_proba, results)

    print(f"\nPhase 6 Step 1 complete ✅")
    print(f"Next: python -m src.models.predict_wc2026")


if __name__ == "__main__":
    main()
