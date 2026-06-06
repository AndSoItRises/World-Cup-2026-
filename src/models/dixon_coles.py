"""
Phase 5b: Dixon-Coles Poisson Model
Fits attack/defense parameters per team via maximum likelihood.
Includes rho correction for low-scoring games (0-0, 1-0, 0-1, 1-1).
Uses time-decay sample weights from Phase 3.

Outputs:
  models/dixon_coles_params.json   (fitted team parameters)
  models/dixon_coles_report.json   (evaluation metrics)

Run with:
  python -m src.models.dixon_coles
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parents[2]
DATA_PROC  = BASE / "data" / "processed"
MODELS_DIR = BASE / "models"

TRAIN_PATH = DATA_PROC / "train_features.csv"
TEST_PATH  = DATA_PROC / "test_features.csv"

MAX_GOALS = 10   # cap scoreline sum for probability calculation


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["date"])

    # Drop rows with missing scores (can't fit on them)
    train = train.dropna(subset=["home_score", "away_score"])
    test  = test.dropna(subset=["home_score", "away_score"])

    train["home_score"] = train["home_score"].astype(int)
    train["away_score"] = train["away_score"].astype(int)
    test["home_score"]  = test["home_score"].astype(int)
    test["away_score"]  = test["away_score"].astype(int)

    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")
    return train, test


# ── Dixon-Coles correction (tau) ──────────────────────────────────────────────
def tau(x, y, lam, mu, rho):
    """
    Correction factor for low-scoring games.
    Pure Poisson underestimates 0-0 and 1-1 draws, overestimates 1-0 and 0-1.
    rho < 0 inflates 0-0 and 1-1, deflates 1-0 and 0-1.
    """
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0


# ── Log likelihood (vectorized) ───────────────────────────────────────────────
def neg_log_likelihood(params, n, h_idx, a_idx,
                       home_goals, away_goals, weights, neutral_flags):
    """
    Fully vectorized negative log likelihood — no Python loops over matches.
    params layout: [attack_0..n, defense_0..n, home_advantage, rho]
    """
    attack   = np.exp(params[:n])
    defense  = np.exp(params[n:2*n])
    home_adv = np.exp(params[2*n])
    rho      = params[2*n + 1]

    lam = attack[h_idx] * defense[a_idx] * np.where(neutral_flags, 1.0, home_adv)
    mu  = attack[a_idx] * defense[h_idx]

    x = home_goals
    y = away_goals

    # Vectorized tau correction
    tau_vals = np.ones(len(x))
    m00 = (x == 0) & (y == 0);  tau_vals[m00] = 1 - lam[m00] * mu[m00] * rho
    m01 = (x == 0) & (y == 1);  tau_vals[m01] = 1 + lam[m01] * rho
    m10 = (x == 1) & (y == 0);  tau_vals[m10] = 1 + mu[m10]  * rho
    m11 = (x == 1) & (y == 1);  tau_vals[m11] = 1 - rho

    # Penalize invalid tau
    if np.any(tau_vals <= 0):
        return 1e10

    # Vectorized Poisson log pmf: log(e^-l * l^x / x!) = -l + x*log(l) - log(x!)
    log_fac_x = np.array([np.sum(np.log(np.arange(1, xi + 1))) if xi > 0 else 0.0 for xi in x])
    log_fac_y = np.array([np.sum(np.log(np.arange(1, yi + 1))) if yi > 0 else 0.0 for yi in y])

    ll = (np.log(tau_vals)
          + (-lam + x * np.log(lam) - log_fac_x)
          + (-mu  + y * np.log(mu)  - log_fac_y))

    return -np.dot(weights, ll)


# ── Fit model ─────────────────────────────────────────────────────────────────
def fit_dixon_coles(train: pd.DataFrame):
    print("\n── Fitting Dixon-Coles model ──")

    # All teams seen in training data
    teams = sorted(set(train["home_team"]) | set(train["away_team"]))
    n = len(teams)
    print(f"  Teams: {n}")

    team_idx  = {t: i for i, t in enumerate(teams)}
    h_idx     = np.array([team_idx[t] for t in train["home_team"]])
    a_idx     = np.array([team_idx[t] for t in train["away_team"]])
    home_goals   = train["home_score"].values.astype(int)
    away_goals   = train["away_score"].values.astype(int)
    weights      = train["sample_weight"].values
    neutral_flags = train["neutral"].astype(int).values

    # Precompute log factorials for all goal values seen
    # (avoids recomputing inside optimizer — further speedup)

    # Initial params: all log(1) = 0, rho = -0.1
    x0 = np.zeros(2 * n + 2)
    x0[-1] = -0.1  # rho starting value

    print(f"  Optimizing {len(x0):,} parameters via L-BFGS-B...")
    result = minimize(
        neg_log_likelihood,
        x0,
        args=(n, h_idx, a_idx, home_goals, away_goals, weights, neutral_flags),
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-6}
    )

    if not result.success:
        print(f"  Warning: optimizer did not fully converge — {result.message}")
    else:
        print(f"  Converged in {result.nit} iterations")

    params = result.x
    attack   = np.exp(params[:n])
    defense  = np.exp(params[n:2*n])
    home_adv = np.exp(params[2*n])
    rho      = params[2*n + 1]

    print(f"  Home advantage multiplier : {home_adv:.4f}")
    print(f"  Rho (low-score correction): {rho:.4f}")

    # Top 10 strongest attacks
    team_attack = sorted(zip(teams, attack), key=lambda x: -x[1])
    print(f"\n  Top 10 attack ratings:")
    for t, a in team_attack[:10]:
        print(f"    {t:<25} {a:.4f}")

    return teams, attack, defense, home_adv, rho


# ── Predict win/draw/loss probabilities ───────────────────────────────────────
def predict_match(home_team, away_team, is_neutral,
                  teams, attack, defense, home_adv, rho):
    """
    Returns (p_home_win, p_draw, p_away_win) by summing Poisson
    probabilities over all scorelines up to MAX_GOALS each.
    """
    team_idx = {t: i for i, t in enumerate(teams)}
    h = team_idx.get(home_team)
    a = team_idx.get(away_team)

    if h is None or a is None:
        # Unknown team — return uninformed prior
        return (1/3, 1/3, 1/3)

    lam = attack[h] * defense[a] * (1.0 if is_neutral else home_adv)
    mu  = attack[a] * defense[h]

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0

    for x in range(MAX_GOALS + 1):
        for y in range(MAX_GOALS + 1):
            t = tau(x, y, lam, mu, rho)
            p = t * poisson.pmf(x, lam) * poisson.pmf(y, mu)
            if x > y:
                p_home += p
            elif x == y:
                p_draw += p
            else:
                p_away += p

    # Normalize (should sum to ~1 already, but floating point)
    total = p_home + p_draw + p_away
    return (p_home / total, p_draw / total, p_away / total)


def predict_dataset(df: pd.DataFrame, teams, attack, defense, home_adv, rho):
    proba = []
    for _, row in df.iterrows():
        p = predict_match(
            row["home_team"], row["away_team"], row["neutral"],
            teams, attack, defense, home_adv, rho
        )
        proba.append(p)
    return np.array(proba)   # shape (n, 3): [p_away_win, p_draw, p_home_win]
    # Note: columns are ordered 0=away_win, 1=draw, 2=home_win to match XGB


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(df: pd.DataFrame, proba: np.ndarray, label: str):
    # proba columns: [p_home_win, p_draw, p_away_win] from predict_match
    # Reorder to match result encoding: 0=away_win, 1=draw, 2=home_win
    proba_reordered = np.column_stack([proba[:, 2], proba[:, 1], proba[:, 0]])
    # proba_reordered[:, 0] = p_away_win
    # proba_reordered[:, 1] = p_draw
    # proba_reordered[:, 2] = p_home_win

    pred  = np.argmax(proba_reordered, axis=1)
    y_true = df["result"].values

    acc = accuracy_score(y_true, pred)
    ll  = log_loss(y_true, proba_reordered)

    print(f"\n── {label} Evaluation ──")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Log Loss : {ll:.4f}")

    # Per-class breakdown
    from sklearn.metrics import classification_report
    report = classification_report(
        y_true, pred,
        target_names=["away_win", "draw", "home_win"],
        output_dict=True
    )
    for cls in ["away_win", "draw", "home_win"]:
        r = report[cls]
        print(f"  {cls:<12} precision: {r['precision']:.3f} | "
              f"recall: {r['recall']:.3f} | f1: {r['f1-score']:.3f}")

    return acc, ll, proba_reordered, report


# ── Save ──────────────────────────────────────────────────────────────────────
def save_params(teams, attack, defense, home_adv, rho,
                test_acc, test_ll, report):
    params_out = {
        "model": "dixon_coles_v1",
        "home_advantage": round(float(home_adv), 6),
        "rho": round(float(rho), 6),
        "teams": {
            t: {
                "attack":  round(float(attack[i]), 6),
                "defense": round(float(defense[i]), 6),
            }
            for i, t in enumerate(teams)
        },
        "test_accuracy": round(test_acc, 4),
        "test_log_loss": round(test_ll, 4),
        "classification_report": report,
    }

    out_path = MODELS_DIR / "dixon_coles_params_v3.json"
    with open(out_path, "w") as f:
        json.dump(params_out, f, indent=2)
    print(f"\n✅ Params saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("═" * 60)
    print("  Phase 5b: Dixon-Coles Poisson Model")
    print("═" * 60)

    train, test = load_data()

    teams, attack, defense, home_adv, rho = fit_dixon_coles(train)

    # Evaluate on train (sanity check) and test
    train_proba = predict_dataset(train, teams, attack, defense, home_adv, rho)
    test_proba  = predict_dataset(test,  teams, attack, defense, home_adv, rho)

    train_acc, train_ll, _, _        = evaluate(train, train_proba, "Train")
    test_acc,  test_ll,  _, report   = evaluate(test,  test_proba,  "Test")

    save_params(teams, attack, defense, home_adv, rho, test_acc, test_ll, report)

    print(f"\n── Summary ──")
    print(f"  Train accuracy : {train_acc*100:.1f}%")
    print(f"  Test accuracy  : {test_acc*100:.1f}%")
    print(f"\nPhase 5b complete ✅")
    print("\nNext: python -m src.models.train_lgbm  (LightGBM)")


if __name__ == "__main__":
    main()
