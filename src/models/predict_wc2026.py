"""
Phase 6 Step 2: WC2026 Match Predictions
Builds feature vectors for all 104 fixtures and runs them through the V2 ensemble.

V2: uses xgb_v2 / lgbm_v2, the full 41-feature set (ELO + quality-weighted
rolling form), and the 80-rank sentinel for WC-qualified teams. Lookup logic
mirrors monte_carlo.py so the two stay consistent.

Outputs:
  data/processed/wc2026_predictions.csv

Run with:
  python -m src.models.predict_wc2026
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from scipy.stats import poisson
import xgboost as xgb
import lightgbm as lgb

from src.features.data_cleaning import standardize_name
from src.features.build_squad_strength import SQUAD_MODEL_FEATURES, squad_fields

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parents[2]
DATA_RAW   = BASE / "data" / "raw"
DATA_PROC  = BASE / "data" / "processed"
MODELS_DIR = BASE / "models"

FIXTURES_PATH  = DATA_RAW  / "wc2026_fixtures.csv"
RANKINGS_PATH  = DATA_RAW  / "current_fifa_rankings.csv"
TRAIN_PATH     = DATA_PROC / "train_features.csv"
TEST_PATH      = DATA_PROC / "test_features.csv"
# Production models — trained on ALL competitive data (retrain_all.py) for the
# live WC2026 forecast. Validated v3 models stay at their own paths.
DC_PARAMS_PATH = MODELS_DIR / "dixon_coles_params_prod.json"
XGB_MODEL_PATH = MODELS_DIR / "xgb_prod.json"
LGB_MODEL_PATH = MODELS_DIR / "lgbm_prod.txt"

WEIGHTS = {"xgb": 0.275, "lgb": 0.275, "dc": 0.45}
MAX_GOALS = 10
RANK_SENTINEL = 80   # WC-qualified teams: conservative floor (see decision log)

FEATURE_COLS = [
    "home_fifa_rank", "away_fifa_rank", "fifa_rank_diff",
    "home_elo", "away_elo", "elo_diff",
    "home_win_rate_5", "home_avg_goals_5", "home_avg_gd_5",
    "home_win_rate_10", "home_avg_goals_10", "home_avg_gd_10",
    "away_win_rate_5", "away_avg_goals_5", "away_avg_gd_5",
    "away_win_rate_10", "away_avg_goals_10", "away_avg_gd_10",
    "home_weighted_win_rate_5",  "home_weighted_avg_goals_5",  "home_weighted_avg_gd_5",
    "home_weighted_win_rate_10", "home_weighted_avg_goals_10", "home_weighted_avg_gd_10",
    "away_weighted_win_rate_5",  "away_weighted_avg_goals_5",  "away_weighted_avg_gd_5",
    "away_weighted_win_rate_10", "away_weighted_avg_goals_10", "away_weighted_avg_gd_10",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "h2h_total", "h2h_home_win_rate",
    "home_days_rest", "away_days_rest",
    "is_knockout", "altitude_m",
    "tournament_tier", "neutral",
]
FEATURE_COLS = FEATURE_COLS + SQUAD_MODEL_FEATURES  # V4: squad strength + depth

# ── Name mapping: fixture names → training data names ─────────────────────────
# Fixtures file uses different spellings for some teams.
NAME_MAP = {
    "Ivory Coast":             "Côte d'Ivoire",
    "Bosnia and Herzegovina":  "Bosnia-Herzegovina",
    "South Korea":             "Korea Republic",
    "Cape Verde":              "Cape Verde Islands",
    "Curacao":                 "Curaçao",
    "United States":           "USA",
    "DR Congo":                "Congo DR",   # V3 P1: was falling back to default form
}


def normalize(name: str) -> str:
    return NAME_MAP.get(name, name)


# ── Load data ─────────────────────────────────────────────────────────────────
def load_all():
    fixtures  = pd.read_csv(FIXTURES_PATH, parse_dates=["date"])
    rankings  = pd.read_csv(RANKINGS_PATH)
    train     = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    test      = pd.read_csv(TEST_PATH,  parse_dates=["date"])
    all_data  = pd.concat([train, test], ignore_index=True).sort_values("date")

    print(f"Fixtures : {len(fixtures)} rows")
    print(f"Rankings : {len(rankings)} teams")
    print(f"History  : {len(all_data)} matches")
    return fixtures, rankings, all_data


# ── Build FIFA rank lookup ────────────────────────────────────────────────────
def build_rank_lookup(rankings: pd.DataFrame) -> dict:
    """Map standardized team_name → rank.
    - Keys standardized ('IR Iran'→'Iran') to match normalize() output.
    - Rank re-derived from points: the current rankings CSV has a corrupted
      rank column for some teams (Austria '231', Algeria '291', Cabo Verde '681');
      points are clean and FIFA rank is points-descending, so we re-rank."""
    r = rankings.copy()
    r["rank"] = r["points"].rank(ascending=False, method="min").astype(int)
    return {standardize_name(row["team_name"]): int(row["rank"])
            for _, row in r.iterrows()}


# ── Build rolling form lookup (V2: includes ELO + quality-weighted form) ──────
def build_form_lookup(all_data: pd.DataFrame) -> dict:
    """
    For each team, find their most recent match in history and extract rolling
    form, quality-weighted form, and ELO. Mirrors monte_carlo.build_form_lookup.
    """
    form = {}

    for team, grp in all_data.groupby("home_team"):
        latest = grp.sort_values("date").iloc[-1]
        form[team] = {
            "win_rate_5":   latest["home_win_rate_5"],
            "avg_goals_5":  latest["home_avg_goals_5"],
            "avg_gd_5":     latest["home_avg_gd_5"],
            "win_rate_10":  latest["home_win_rate_10"],
            "avg_goals_10": latest["home_avg_goals_10"],
            "avg_gd_10":    latest["home_avg_gd_10"],
            "weighted_win_rate_5":   latest["home_weighted_win_rate_5"],
            "weighted_avg_goals_5":  latest["home_weighted_avg_goals_5"],
            "weighted_avg_gd_5":     latest["home_weighted_avg_gd_5"],
            "weighted_win_rate_10":  latest["home_weighted_win_rate_10"],
            "weighted_avg_goals_10": latest["home_weighted_avg_goals_10"],
            "weighted_avg_gd_10":    latest["home_weighted_avg_gd_10"],
            "elo":          latest["home_elo"],
            "days_rest":    latest["home_days_rest"],
            "date":         latest["date"],
        }

    for team, grp in all_data.groupby("away_team"):
        latest = grp.sort_values("date").iloc[-1]
        if team not in form or latest["date"] > form[team]["date"]:
            form[team] = {
                "win_rate_5":   latest["away_win_rate_5"],
                "avg_goals_5":  latest["away_avg_goals_5"],
                "avg_gd_5":     latest["away_avg_gd_5"],
                "win_rate_10":  latest["away_win_rate_10"],
                "avg_goals_10": latest["away_avg_goals_10"],
                "avg_gd_10":    latest["away_avg_gd_10"],
                "weighted_win_rate_5":   latest["away_weighted_win_rate_5"],
                "weighted_avg_goals_5":  latest["away_weighted_avg_goals_5"],
                "weighted_avg_gd_5":     latest["away_weighted_avg_gd_5"],
                "weighted_win_rate_10":  latest["away_weighted_win_rate_10"],
                "weighted_avg_goals_10": latest["away_weighted_avg_goals_10"],
                "weighted_avg_gd_10":    latest["away_weighted_avg_gd_10"],
                "elo":          latest["away_elo"],
                "days_rest":    latest["away_days_rest"],
                "date":         latest["date"],
            }

    return form


# ── H2H lookup ────────────────────────────────────────────────────────────────
def build_h2h_lookup(all_data: pd.DataFrame) -> dict:
    """For each (home, away) pair, find most recent H2H stats from history."""
    h2h = {}
    for _, row in all_data.iterrows():
        key = (row["home_team"], row["away_team"])
        if key not in h2h or row["date"] > h2h[key]["date"]:
            h2h[key] = {
                "h2h_home_wins":     row["h2h_home_wins"],
                "h2h_draws":         row["h2h_draws"],
                "h2h_away_wins":     row["h2h_away_wins"],
                "h2h_total":         row["h2h_total"],
                "h2h_home_win_rate": row["h2h_home_win_rate"],
                "date":              row["date"],
            }
    return h2h


# ── Default form (median of historical data, V2 fields) ───────────────────────
def compute_defaults(all_data: pd.DataFrame) -> dict:
    return {
        "win_rate_5":   all_data["home_win_rate_5"].median(),
        "avg_goals_5":  all_data["home_avg_goals_5"].median(),
        "avg_gd_5":     all_data["home_avg_gd_5"].median(),
        "win_rate_10":  all_data["home_win_rate_10"].median(),
        "avg_goals_10": all_data["home_avg_goals_10"].median(),
        "avg_gd_10":    all_data["home_avg_gd_10"].median(),
        "weighted_win_rate_5":   all_data["home_weighted_win_rate_5"].median(),
        "weighted_avg_goals_5":  all_data["home_weighted_avg_goals_5"].median(),
        "weighted_avg_gd_5":     all_data["home_weighted_avg_gd_5"].median(),
        "weighted_win_rate_10":  all_data["home_weighted_win_rate_10"].median(),
        "weighted_avg_goals_10": all_data["home_weighted_avg_goals_10"].median(),
        "weighted_avg_gd_10":    all_data["home_weighted_avg_gd_10"].median(),
        "elo":          1500.0,
        "days_rest":    all_data["home_days_rest"].median(),
    }


# ── Build feature vector for one fixture ─────────────────────────────────────
def build_feature_row(row, rank_lookup, form_lookup, h2h_lookup, defaults):
    home = normalize(row["home_team"])
    away = normalize(row["away_team"])

    # FIFA ranks (80 sentinel for WC-qualified teams missing from lookup)
    home_rank = rank_lookup.get(home, RANK_SENTINEL)
    away_rank = rank_lookup.get(away, RANK_SENTINEL)

    # Form (V2: includes ELO + weighted)
    hf = form_lookup.get(home, defaults)
    af = form_lookup.get(away, defaults)

    home_elo = hf.get("elo", 1500.0)
    away_elo = af.get("elo", 1500.0)

    # H2H — try both orderings (fixtures may flip home/away vs history)
    key     = (home, away)
    key_rev = (away, home)
    if key in h2h_lookup:
        h2h = h2h_lookup[key]
        h2h_home_wins     = h2h["h2h_home_wins"]
        h2h_draws         = h2h["h2h_draws"]
        h2h_away_wins     = h2h["h2h_away_wins"]
        h2h_total         = h2h["h2h_total"]
        h2h_home_win_rate = h2h["h2h_home_win_rate"]
    elif key_rev in h2h_lookup:
        h2h = h2h_lookup[key_rev]
        # Flip perspective
        h2h_home_wins     = h2h["h2h_away_wins"]
        h2h_draws         = h2h["h2h_draws"]
        h2h_away_wins     = h2h["h2h_home_wins"]
        h2h_total         = h2h["h2h_total"]
        h2h_home_win_rate = 1 - h2h["h2h_home_win_rate"] if h2h["h2h_total"] > 0 else 0.5
    else:
        h2h_home_wins = h2h_draws = h2h_away_wins = h2h_total = 0
        h2h_home_win_rate = 0.5

    # Stage → is_knockout
    stage = str(row.get("stage", "Group Stage"))
    is_knockout = 0 if stage == "Group Stage" else 1

    return {
        "home_fifa_rank":    home_rank,
        "away_fifa_rank":    away_rank,
        "fifa_rank_diff":    home_rank - away_rank,
        "home_elo":          home_elo,
        "away_elo":          away_elo,
        "elo_diff":          home_elo - away_elo,
        "home_win_rate_5":   hf["win_rate_5"],
        "home_avg_goals_5":  hf["avg_goals_5"],
        "home_avg_gd_5":     hf["avg_gd_5"],
        "home_win_rate_10":  hf["win_rate_10"],
        "home_avg_goals_10": hf["avg_goals_10"],
        "home_avg_gd_10":    hf["avg_gd_10"],
        "away_win_rate_5":   af["win_rate_5"],
        "away_avg_goals_5":  af["avg_goals_5"],
        "away_avg_gd_5":     af["avg_gd_5"],
        "away_win_rate_10":  af["win_rate_10"],
        "away_avg_goals_10": af["avg_goals_10"],
        "away_avg_gd_10":    af["avg_gd_10"],
        "home_weighted_win_rate_5":   hf["weighted_win_rate_5"],
        "home_weighted_avg_goals_5":  hf["weighted_avg_goals_5"],
        "home_weighted_avg_gd_5":     hf["weighted_avg_gd_5"],
        "home_weighted_win_rate_10":  hf["weighted_win_rate_10"],
        "home_weighted_avg_goals_10": hf["weighted_avg_goals_10"],
        "home_weighted_avg_gd_10":    hf["weighted_avg_gd_10"],
        "away_weighted_win_rate_5":   af["weighted_win_rate_5"],
        "away_weighted_avg_goals_5":  af["weighted_avg_goals_5"],
        "away_weighted_avg_gd_5":     af["weighted_avg_gd_5"],
        "away_weighted_win_rate_10":  af["weighted_win_rate_10"],
        "away_weighted_avg_goals_10": af["weighted_avg_goals_10"],
        "away_weighted_avg_gd_10":    af["weighted_avg_gd_10"],
        "h2h_home_wins":     h2h_home_wins,
        "h2h_draws":         h2h_draws,
        "h2h_away_wins":     h2h_away_wins,
        "h2h_total":         h2h_total,
        "h2h_home_win_rate": h2h_home_win_rate,
        "home_days_rest":    hf["days_rest"],
        "away_days_rest":    af["days_rest"],
        "is_knockout":       is_knockout,
        "altitude_m":        row.get("altitude_m", 0) or 0,
        "tournament_tier":   1,   # WC = tier 1
        "neutral":           1,   # all WC2026 matches neutral
        **squad_fields(home, away),   # V4: squad strength + depth (latest edition)
    }


# ── Dixon-Coles predict ───────────────────────────────────────────────────────
def tau(x, y, lam, mu, rho):
    if x == 0 and y == 0:    return 1 - lam * mu * rho
    elif x == 0 and y == 1:  return 1 + lam * rho
    elif x == 1 and y == 0:  return 1 + mu * rho
    elif x == 1 and y == 1:  return 1 - rho
    else:                    return 1.0


def dc_predict(home, away, is_neutral, dc):
    teams = dc["teams"]
    if home not in teams or away not in teams:
        return np.array([1/3, 1/3, 1/3])  # [away_win, draw, home_win]

    ht = teams[home]
    at = teams[away]
    home_adv = dc["home_advantage"]
    rho      = dc["rho"]

    lam = ht["attack"] * at["defense"] * (1.0 if is_neutral else home_adv)
    mu  = at["attack"] * ht["defense"]

    p_home = p_draw = p_away = 0.0
    for x in range(MAX_GOALS + 1):
        for y in range(MAX_GOALS + 1):
            t = tau(x, y, lam, mu, rho)
            p = t * poisson.pmf(x, lam) * poisson.pmf(y, mu)
            if x > y:    p_home += p
            elif x == y: p_draw += p
            else:        p_away += p

    total = p_home + p_draw + p_away
    return np.array([p_away / total, p_draw / total, p_home / total])


# ── Load models ───────────────────────────────────────────────────────────────
def load_models():
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(XGB_MODEL_PATH))

    lgb_booster = lgb.Booster(model_file=str(LGB_MODEL_PATH))

    with open(DC_PARAMS_PATH) as f:
        dc_params = json.load(f)

    print("  XGB v2, LGBM v2, Dixon-Coles loaded")
    return xgb_model, lgb_booster, dc_params


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("═" * 60)
    print("  Phase 6 Step 2: WC2026 Predictions (V2)")
    print("═" * 60)

    fixtures, rankings, all_data = load_all()

    print("\n── Building lookups ──")
    rank_lookup = build_rank_lookup(rankings)
    form_lookup = build_form_lookup(all_data)
    h2h_lookup  = build_h2h_lookup(all_data)
    defaults    = compute_defaults(all_data)
    print(f"  Rank lookup : {len(rank_lookup)} teams")
    print(f"  Form lookup : {len(form_lookup)} teams")
    print(f"  H2H lookup  : {len(h2h_lookup)} pairs")

    print("\n── Loading models ──")
    xgb_model, lgb_booster, dc_params = load_models()

    # ── Filter to predictable fixtures (no TBD teams) ────────────────────────
    predictable = fixtures[
        ~fixtures["home_team"].str.startswith("TBD") &
        ~fixtures["away_team"].str.startswith("TBD")
    ].copy()
    skipped = len(fixtures) - len(predictable)
    print(f"\n── Fixtures: {len(predictable)} predictable | {skipped} TBD (skipped) ──")

    # ── Build feature matrix ──────────────────────────────────────────────────
    print("\n── Building feature vectors ──")
    rows = []
    unknown_teams = set()

    for _, fix_row in predictable.iterrows():
        home = normalize(fix_row["home_team"])
        away = normalize(fix_row["away_team"])

        if home not in form_lookup:
            unknown_teams.add(home)
        if away not in form_lookup:
            unknown_teams.add(away)

        feat = build_feature_row(fix_row, rank_lookup, form_lookup, h2h_lookup, defaults)
        rows.append(feat)

    if unknown_teams:
        print(f"  Teams using default form (no history): {sorted(unknown_teams)}")

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    print(f"  Feature matrix: {X.shape}")

    # ── Get probabilities from each model ─────────────────────────────────────
    print("\n── Running predictions ──")
    xgb_proba = xgb_model.predict_proba(X)           # (n, 3)
    lgb_proba = lgb_booster.predict(X.values)        # (n, 3)

    dc_proba = np.array([
        dc_predict(
            normalize(row["home_team"]),
            normalize(row["away_team"]),
            True,   # all WC matches are neutral
            dc_params
        )
        for _, row in predictable.iterrows()
    ])

    # ── Blend ─────────────────────────────────────────────────────────────────
    # V6: log-pool + draw-shrink calibrator if adopted (v4 linear blend fallback)
    from src.models.calibrator import pool_and_calibrate, banner
    print(f"  {banner()}")
    ensemble_proba = pool_and_calibrate(
        xgb_proba, lgb_proba, dc_proba,
        weights=(WEIGHTS["xgb"], WEIGHTS["lgb"], WEIGHTS["dc"]))
    # Columns: [p_away_win, p_draw, p_home_win]

    RESULT_LABELS = {0: "away_win", 1: "draw", 2: "home_win"}
    predicted_result = np.argmax(ensemble_proba, axis=1)

    # ── Assemble output ───────────────────────────────────────────────────────
    out = predictable[["match_id", "date", "stage", "group",
                        "home_team", "away_team", "altitude_m"]].copy()
    out["p_home_win"] = ensemble_proba[:, 2].round(4)
    out["p_draw"]     = ensemble_proba[:, 1].round(4)
    out["p_away_win"] = ensemble_proba[:, 0].round(4)
    out["predicted_result"] = [RESULT_LABELS[r] for r in predicted_result]

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = DATA_PROC / "wc2026_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\n✅ Predictions saved: {out_path}")
    print(f"   {len(out)} matches predicted")

    # ── Quick preview ─────────────────────────────────────────────────────────
    print(f"\n── Sample Predictions (Group Stage) ──")
    group_preds = out[out["stage"] == "Group Stage"].head(12)
    print(f"  {'Home':<25} {'Away':<25} {'P(H)':>6} {'P(D)':>6} {'P(A)':>6} {'Pred':<12}")
    print(f"  {'-'*85}")
    for _, r in group_preds.iterrows():
        print(f"  {r['home_team']:<25} {r['away_team']:<25} "
              f"{r['p_home_win']:>6.3f} {r['p_draw']:>6.3f} {r['p_away_win']:>6.3f} "
              f"{r['predicted_result']:<12}")

    # ── Result distribution ────────────────────────────────────────────────────
    dist = out["predicted_result"].value_counts()
    print(f"\n── Predicted Result Distribution ──")
    for k, v in dist.items():
        print(f"  {k:<12}: {v} ({v/len(out)*100:.1f}%)")

    print(f"\nPhase 6 Step 2 complete ✅")
    print(f"Next: python -m src.models.monte_carlo")


if __name__ == "__main__":
    main()
