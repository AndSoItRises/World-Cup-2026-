"""
Phase 7: Monte Carlo Bracket Simulator
Simulates the WC2026 tournament N times, tracking how often each team
reaches each stage. Outputs tournament win probabilities per team.

Method:
  - Group stage: simulate exact scores via Poisson (Dixon-Coles expected goals)
    so goal difference can break ties
  - Top 2 per group + best 8 third-place teams advance (WC2026 format)
  - Knockout: ensemble probabilities pre-computed for all team pairs
  - N = 10,000 simulations

Outputs:
  data/processed/tournament_probs.csv
  data/processed/group_stage_probs.csv

Run with:
  python -m src.models.monte_carlo
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from scipy.stats import poisson
import xgboost as xgb
import lightgbm as lgb
from itertools import combinations

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
# Production models (retrain_all.py) — all competitive data, for the live forecast.
DC_PARAMS_PATH = MODELS_DIR / "dixon_coles_params_prod.json"
XGB_MODEL_PATH = MODELS_DIR / "xgb_prod.json"
LGB_MODEL_PATH = MODELS_DIR / "lgbm_prod.txt"

N_SIMULATIONS = 10_000
WEIGHTS = {"xgb": 0.275, "lgb": 0.275, "dc": 0.45}
MAX_GOALS = 8  # cap for Poisson score simulation

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

NAME_MAP = {
    "Ivory Coast":            "Côte d'Ivoire",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "South Korea":            "Korea Republic",
    "Cape Verde":             "Cape Verde Islands",
    "Curacao":                "Curaçao",
    "United States":          "USA",
    "DR Congo":               "Congo DR",   # V3 P1: was hitting 80 sentinel + default form
}

# Round of 32 bracket: match_id → (home_slot, away_slot)
# Slots: "W<group>" = group winner, "R<group>" = runner-up, "3rd" = third-place slot
R32_BRACKET = {
    73: ("R_A", "R_B"),
    74: ("W_E", "3rd_ABCDF"),
    75: ("W_F", "R_C"),
    76: ("W_C", "R_F"),
    77: ("W_I", "3rd_CDFGH"),
    78: ("R_E", "R_I"),
    79: ("W_A", "3rd_CEFHI"),
    80: ("W_L", "3rd_EHIJK"),
    81: ("W_D", "3rd_BEFIJ"),
    82: ("W_G", "3rd_AEHIJ"),
    83: ("R_K", "R_L"),
    84: ("W_H", "R_J"),
    85: ("W_B", "3rd_EFGIJ"),
    86: ("W_J", "R_H"),
    87: ("W_K", "3rd_DEIJL"),
    88: ("R_D", "R_G"),
}

# Round of 16 bracket: match_id → (winner of match_id_A, winner of match_id_B)
R16_BRACKET = {
    89: (74, 77),
    90: (73, 75),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
}

QF_BRACKET = {
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
}

SF_BRACKET = {
    101: (97, 98),
    102: (99, 100),
}


def normalize(name):
    return NAME_MAP.get(name, name)


# ── Load data ─────────────────────────────────────────────────────────────────
def load_all():
    fixtures = pd.read_csv(FIXTURES_PATH, parse_dates=["date"])
    rankings = pd.read_csv(RANKINGS_PATH)
    train    = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    test     = pd.read_csv(TEST_PATH,  parse_dates=["date"])
    all_data = pd.concat([train, test], ignore_index=True).sort_values("date")
    return fixtures, rankings, all_data


def build_rank_lookup(rankings):
    # V3 P1: standardize keys ('IR Iran'→'Iran'); re-derive rank from points
    # (current rankings CSV has a corrupted rank column for some teams).
    r = rankings.copy()
    r["rank"] = r["points"].rank(ascending=False, method="min").astype(int)
    return {standardize_name(row["team_name"]): int(row["rank"])
            for _, row in r.iterrows()}


def build_form_lookup(all_data):
    form = {}
    for team, grp in all_data.groupby("home_team"):
        latest = grp.sort_values("date").iloc[-1]
        form[team] = {
            "win_rate_5": latest["home_win_rate_5"],
            "avg_goals_5": latest["home_avg_goals_5"],
            "avg_gd_5": latest["home_avg_gd_5"],
            "win_rate_10": latest["home_win_rate_10"],
            "avg_goals_10": latest["home_avg_goals_10"],
            "avg_gd_10": latest["home_avg_gd_10"],
            "weighted_win_rate_5": latest["home_weighted_win_rate_5"],
            "weighted_avg_goals_5": latest["home_weighted_avg_goals_5"],
            "weighted_avg_gd_5": latest["home_weighted_avg_gd_5"],
            "weighted_win_rate_10": latest["home_weighted_win_rate_10"],
            "weighted_avg_goals_10": latest["home_weighted_avg_goals_10"],
            "weighted_avg_gd_10": latest["home_weighted_avg_gd_10"],
            "elo": latest["home_elo"],
            "days_rest": latest["home_days_rest"],
            "date": latest["date"],
        }
    for team, grp in all_data.groupby("away_team"):
        latest = grp.sort_values("date").iloc[-1]
        if team not in form or latest["date"] > form[team]["date"]:
            form[team] = {
                "win_rate_5": latest["away_win_rate_5"],
                "avg_goals_5": latest["away_avg_goals_5"],
                "avg_gd_5": latest["away_avg_gd_5"],
                "win_rate_10": latest["away_win_rate_10"],
                "avg_goals_10": latest["away_avg_goals_10"],
                "avg_gd_10": latest["away_avg_gd_10"],
                "weighted_win_rate_5": latest["away_weighted_win_rate_5"],
                "weighted_avg_goals_5": latest["away_weighted_avg_goals_5"],
                "weighted_avg_gd_5": latest["away_weighted_avg_gd_5"],
                "weighted_win_rate_10": latest["away_weighted_win_rate_10"],
                "weighted_avg_goals_10": latest["away_weighted_avg_goals_10"],
                "weighted_avg_gd_10": latest["away_weighted_avg_gd_10"],
                "elo": latest["away_elo"],
                "days_rest": latest["away_days_rest"],
                "date": latest["date"],
            }
    return form


def build_h2h_lookup(all_data):
    h2h = {}
    for _, row in all_data.iterrows():
        key = (row["home_team"], row["away_team"])
        if key not in h2h or row["date"] > h2h[key]["date"]:
            h2h[key] = {
                "h2h_home_wins": row["h2h_home_wins"],
                "h2h_draws": row["h2h_draws"],
                "h2h_away_wins": row["h2h_away_wins"],
                "h2h_total": row["h2h_total"],
                "h2h_home_win_rate": row["h2h_home_win_rate"],
                "date": row["date"],
            }
    return h2h


def compute_defaults(all_data):
    return {
        "win_rate_5": all_data["home_win_rate_5"].median(),
        "avg_goals_5": all_data["home_avg_goals_5"].median(),
        "avg_gd_5": all_data["home_avg_gd_5"].median(),
        "win_rate_10": all_data["home_win_rate_10"].median(),
        "avg_goals_10": all_data["home_avg_goals_10"].median(),
        "avg_gd_10": all_data["home_avg_gd_10"].median(),
        "weighted_win_rate_5": all_data["home_weighted_win_rate_5"].median(),
        "weighted_avg_goals_5": all_data["home_weighted_avg_goals_5"].median(),
        "weighted_avg_gd_5": all_data["home_weighted_avg_gd_5"].median(),
        "weighted_win_rate_10": all_data["home_weighted_win_rate_10"].median(),
        "weighted_avg_goals_10": all_data["home_weighted_avg_goals_10"].median(),
        "weighted_avg_gd_10": all_data["home_weighted_avg_gd_10"].median(),
        "elo": 1500.0,
        "days_rest": all_data["home_days_rest"].median(),
    }


# ── Load models ───────────────────────────────────────────────────────────────
def load_models():
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(XGB_MODEL_PATH))
    lgb_booster = lgb.Booster(model_file=str(LGB_MODEL_PATH))
    with open(DC_PARAMS_PATH) as f:
        dc_params = json.load(f)
    return xgb_model, lgb_booster, dc_params


# ── Ensemble probability for one matchup ─────────────────────────────────────
def get_matchup_prob(home, away, is_knockout,
                     rank_lookup, form_lookup, h2h_lookup, defaults,
                     xgb_model, lgb_booster, dc_params):
    """
    Returns (p_home_win, p_draw, p_away_win) for a given matchup.
    All WC matches treated as neutral.
    """
    hf = form_lookup.get(home, defaults)
    af = form_lookup.get(away, defaults)

    home_rank = rank_lookup.get(home, 80)
    away_rank = rank_lookup.get(away, 80)

    key     = (home, away)
    key_rev = (away, home)
    if key in h2h_lookup:
        h = h2h_lookup[key]
        h2h_hw, h2h_d, h2h_aw = h["h2h_home_wins"], h["h2h_draws"], h["h2h_away_wins"]
        h2h_tot, h2h_hwr = h["h2h_total"], h["h2h_home_win_rate"]
    elif key_rev in h2h_lookup:
        h = h2h_lookup[key_rev]
        h2h_hw, h2h_d, h2h_aw = h["h2h_away_wins"], h["h2h_draws"], h["h2h_home_wins"]
        h2h_tot = h["h2h_total"]
        h2h_hwr = 1 - h["h2h_home_win_rate"] if h["h2h_total"] > 0 else 0.5
    else:
        h2h_hw = h2h_d = h2h_aw = h2h_tot = 0
        h2h_hwr = 0.5

    home_elo = hf.get("elo", 1500.0)
    away_elo = af.get("elo", 1500.0)
    feat = {
        "home_fifa_rank": home_rank, "away_fifa_rank": away_rank,
        "fifa_rank_diff": home_rank - away_rank,
        "home_elo": home_elo, "away_elo": away_elo, "elo_diff": home_elo - away_elo,
        "home_win_rate_5": hf["win_rate_5"], "home_avg_goals_5": hf["avg_goals_5"],
        "home_avg_gd_5": hf["avg_gd_5"], "home_win_rate_10": hf["win_rate_10"],
        "home_avg_goals_10": hf["avg_goals_10"], "home_avg_gd_10": hf["avg_gd_10"],
        "away_win_rate_5": af["win_rate_5"], "away_avg_goals_5": af["avg_goals_5"],
        "away_avg_gd_5": af["avg_gd_5"], "away_win_rate_10": af["win_rate_10"],
        "away_avg_goals_10": af["avg_goals_10"], "away_avg_gd_10": af["avg_gd_10"],
        "home_weighted_win_rate_5": hf["weighted_win_rate_5"],
        "home_weighted_avg_goals_5": hf["weighted_avg_goals_5"],
        "home_weighted_avg_gd_5": hf["weighted_avg_gd_5"],
        "home_weighted_win_rate_10": hf["weighted_win_rate_10"],
        "home_weighted_avg_goals_10": hf["weighted_avg_goals_10"],
        "home_weighted_avg_gd_10": hf["weighted_avg_gd_10"],
        "away_weighted_win_rate_5": af["weighted_win_rate_5"],
        "away_weighted_avg_goals_5": af["weighted_avg_goals_5"],
        "away_weighted_avg_gd_5": af["weighted_avg_gd_5"],
        "away_weighted_win_rate_10": af["weighted_win_rate_10"],
        "away_weighted_avg_goals_10": af["weighted_avg_goals_10"],
        "away_weighted_avg_gd_10": af["weighted_avg_gd_10"],
        "h2h_home_wins": h2h_hw, "h2h_draws": h2h_d, "h2h_away_wins": h2h_aw,
        "h2h_total": h2h_tot, "h2h_home_win_rate": h2h_hwr,
        "home_days_rest": hf["days_rest"], "away_days_rest": af["days_rest"],
        "is_knockout": int(is_knockout), "altitude_m": 0,
        "tournament_tier": 1, "neutral": 1,
        **squad_fields(home, away),   # V4: squad strength + depth (latest edition)
    }

    X = pd.DataFrame([feat], columns=FEATURE_COLS)
    xgb_p = xgb_model.predict_proba(X)[0]       # [away_win, draw, home_win]
    lgb_p = lgb_booster.predict(X.values)[0]    # [away_win, draw, home_win]

    # Dixon-Coles
    teams = dc_params["teams"]
    if home in teams and away in teams:
        ht, at = teams[home], teams[away]
        lam = ht["attack"] * at["defense"]   # neutral → no home_adv
        mu  = at["attack"] * ht["defense"]
        p_home = p_draw = p_away = 0.0
        for x in range(MAX_GOALS + 1):
            for y in range(MAX_GOALS + 1):
                t = _tau(x, y, lam, mu, dc_params["rho"])
                p = t * poisson.pmf(x, lam) * poisson.pmf(y, mu)
                if x > y:    p_home += p
                elif x == y: p_draw += p
                else:        p_away += p
        total = p_home + p_draw + p_away
        dc_p = np.array([p_away / total, p_draw / total, p_home / total])
    else:
        dc_p = np.array([1/3, 1/3, 1/3])

    # V6: log-pool + draw-shrink calibrator if adopted (falls back to the v4
    # linear blend when models/calibrator_v6.json is absent)
    from src.models.calibrator import pool_and_calibrate
    blended = pool_and_calibrate(
        xgb_p, lgb_p, dc_p,
        weights=(WEIGHTS["xgb"], WEIGHTS["lgb"], WEIGHTS["dc"]))
    # blended: [p_away_win, p_draw, p_home_win]
    return blended[2], blended[1], blended[0]   # p_home, p_draw, p_away


def _tau(x, y, lam, mu, rho):
    if x == 0 and y == 0:    return 1 - lam * mu * rho
    elif x == 0 and y == 1:  return 1 + lam * rho
    elif x == 1 and y == 0:  return 1 + mu * rho
    elif x == 1 and y == 1:  return 1 - rho
    return 1.0


# ── DC expected goals (for score simulation) ──────────────────────────────────
def get_expected_goals(home, away, dc_params):
    """Returns (lambda, mu) for Poisson score simulation."""
    teams = dc_params["teams"]
    if home in teams and away in teams:
        ht, at = teams[home], teams[away]
        lam = ht["attack"] * at["defense"]  # neutral, no home_adv
        mu  = at["attack"] * ht["defense"]
    else:
        lam = mu = 1.15  # league average
    return lam, mu


# ── Pre-compute pairwise knockout probabilities ───────────────────────────────
def precompute_pairwise(wc_teams, rank_lookup, form_lookup, h2h_lookup,
                        defaults, xgb_model, lgb_booster, dc_params):
    """
    Build a lookup: (home, away) → (p_home_win, p_draw, p_away_win)
    for all team pairs. Used during knockout simulation.
    """
    print(f"  Pre-computing {len(wc_teams)*(len(wc_teams)-1)} pairwise matchups...")
    prob_cache = {}
    for home in wc_teams:
        for away in wc_teams:
            if home == away:
                continue
            prob_cache[(home, away)] = get_matchup_prob(
                home, away, True,
                rank_lookup, form_lookup, h2h_lookup, defaults,
                xgb_model, lgb_booster, dc_params
            )
    return prob_cache


# ── Group stage simulation ────────────────────────────────────────────────────
def simulate_group(matches, dc_params, rng):
    """
    Simulate one group's matches. Returns standings dict per team:
      {team: {"pts": int, "gd": int, "gf": int, "rank": int}}
    matches: list of (home, away) tuples
    """
    records = {}
    for home, away in matches:
        lam, mu = get_expected_goals(home, away, dc_params)
        hg = rng.poisson(lam)
        ag = rng.poisson(mu)

        for t in [home, away]:
            if t not in records:
                records[t] = {"pts": 0, "gd": 0, "gf": 0}

        records[home]["gf"] += hg
        records[away]["gf"] += ag
        records[home]["gd"] += hg - ag
        records[away]["gd"] += ag - hg

        if hg > ag:
            records[home]["pts"] += 3
        elif hg == ag:
            records[home]["pts"] += 1
            records[away]["pts"] += 1
        else:
            records[away]["pts"] += 3

    return records


def rank_group(records, rank_lookup):
    """
    Sort teams by: points DESC, GD DESC, GF DESC, FIFA rank ASC (lower = better).
    Returns ordered list of team names.
    """
    teams = list(records.keys())
    teams.sort(key=lambda t: (
        -records[t]["pts"],
        -records[t]["gd"],
        -records[t]["gf"],
        rank_lookup.get(t, 80),
    ))
    return teams


# ── Select best 8 third-place teams ──────────────────────────────────────────
# Third-place slot → eligible groups
THIRD_PLACE_SLOTS = [
    ("3rd_ABCDF",  set("ABCDF")),
    ("3rd_CDFGH",  set("CDFGH")),
    ("3rd_CEFHI",  set("CEFHI")),
    ("3rd_EHIJK",  set("EHIJK")),
    ("3rd_BEFIJ",  set("BEFIJ")),
    ("3rd_AEHIJ",  set("AEHIJ")),
    ("3rd_EFGIJ",  set("EFGIJ")),
    ("3rd_DEIJL",  set("DEIJL")),
]

THIRD_PLACE_COMBOS = {
    "ABCDEFGH": ["H", "G", "B", "C", "A", "F", "D", "E"],
    "ABCDEFGI": ["C", "G", "B", "D", "A", "F", "E", "I"],
    "ABCDEFGJ": ["C", "G", "B", "D", "A", "F", "E", "J"],
    "ABCDEFGK": ["C", "G", "B", "D", "A", "F", "E", "K"],
    "ABCDEFGL": ["C", "G", "B", "D", "A", "F", "L", "E"],
    "ABCDEFHI": ["H", "E", "B", "C", "A", "F", "D", "I"],
    "ABCDEFHJ": ["H", "J", "B", "C", "A", "F", "D", "E"],
    "ABCDEFHK": ["H", "E", "B", "C", "A", "F", "D", "K"],
    "ABCDEFHL": ["H", "F", "B", "C", "A", "D", "L", "E"],
    "ABCDEFIJ": ["C", "J", "B", "D", "A", "F", "E", "I"],
    "ABCDEFIK": ["C", "E", "B", "D", "A", "F", "I", "K"],
    "ABCDEFIL": ["C", "E", "B", "D", "A", "F", "L", "I"],
    "ABCDEFJK": ["C", "J", "B", "D", "A", "F", "E", "K"],
    "ABCDEFJL": ["C", "J", "B", "D", "A", "F", "L", "E"],
    "ABCDEFKL": ["C", "E", "B", "D", "A", "F", "L", "K"],
    "ABCDEGHI": ["H", "G", "B", "C", "A", "D", "E", "I"],
    "ABCDEGHJ": ["H", "G", "B", "C", "A", "D", "E", "J"],
    "ABCDEGHK": ["H", "G", "B", "C", "A", "D", "E", "K"],
    "ABCDEGHL": ["H", "G", "B", "C", "A", "D", "L", "E"],
    "ABCDEGIJ": ["E", "G", "B", "C", "A", "D", "I", "J"],
    "ABCDEGIK": ["E", "G", "B", "C", "A", "D", "I", "K"],
    "ABCDEGIL": ["E", "G", "B", "C", "A", "D", "L", "I"],
    "ABCDEGJK": ["E", "G", "B", "C", "A", "D", "J", "K"],
    "ABCDEGJL": ["E", "G", "B", "C", "A", "D", "L", "J"],
    "ABCDEGKL": ["E", "G", "B", "C", "A", "D", "L", "K"],
    "ABCDEHIJ": ["H", "J", "B", "C", "A", "D", "E", "I"],
    "ABCDEHIK": ["H", "E", "B", "C", "A", "D", "I", "K"],
    "ABCDEHIL": ["H", "E", "B", "C", "A", "D", "L", "I"],
    "ABCDEHJK": ["H", "J", "B", "C", "A", "D", "E", "K"],
    "ABCDEHJL": ["H", "J", "B", "C", "A", "D", "L", "E"],
    "ABCDEHKL": ["H", "E", "B", "C", "A", "D", "L", "K"],
    "ABCDEIJK": ["E", "J", "B", "C", "A", "D", "I", "K"],
    "ABCDEIJL": ["E", "J", "B", "C", "A", "D", "L", "I"],
    "ABCDEIKL": ["E", "I", "B", "C", "A", "D", "L", "K"],
    "ABCDEJKL": ["E", "J", "B", "C", "A", "D", "L", "K"],
    "ABCDFGHI": ["H", "G", "B", "C", "A", "F", "D", "I"],
    "ABCDFGHJ": ["H", "G", "B", "C", "A", "F", "D", "J"],
    "ABCDFGHK": ["H", "G", "B", "C", "A", "F", "D", "K"],
    "ABCDFGHL": ["C", "G", "B", "D", "A", "F", "L", "H"],
    "ABCDFGIJ": ["C", "G", "B", "D", "A", "F", "I", "J"],
    "ABCDFGIK": ["C", "G", "B", "D", "A", "F", "I", "K"],
    "ABCDFGIL": ["C", "G", "B", "D", "A", "F", "L", "I"],
    "ABCDFGJK": ["C", "G", "B", "D", "A", "F", "J", "K"],
    "ABCDFGJL": ["C", "G", "B", "D", "A", "F", "L", "J"],
    "ABCDFGKL": ["C", "G", "B", "D", "A", "F", "L", "K"],
    "ABCDFHIJ": ["H", "J", "B", "C", "A", "F", "D", "I"],
    "ABCDFHIK": ["H", "F", "B", "C", "A", "D", "I", "K"],
    "ABCDFHIL": ["H", "F", "B", "C", "A", "D", "L", "I"],
    "ABCDFHJK": ["H", "J", "B", "C", "A", "F", "D", "K"],
    "ABCDFHJL": ["C", "J", "B", "D", "A", "F", "L", "H"],
    "ABCDFHKL": ["H", "F", "B", "C", "A", "D", "L", "K"],
    "ABCDFIJK": ["C", "J", "B", "D", "A", "F", "I", "K"],
    "ABCDFIJL": ["C", "J", "B", "D", "A", "F", "L", "I"],
    "ABCDFIKL": ["C", "I", "B", "D", "A", "F", "L", "K"],
    "ABCDFJKL": ["C", "J", "B", "D", "A", "F", "L", "K"],
    "ABCDGHIJ": ["H", "G", "B", "C", "A", "D", "I", "J"],
    "ABCDGHIK": ["H", "G", "B", "C", "A", "D", "I", "K"],
    "ABCDGHIL": ["H", "G", "B", "C", "A", "D", "L", "I"],
    "ABCDGHJK": ["H", "G", "B", "C", "A", "D", "J", "K"],
    "ABCDGHJL": ["H", "G", "B", "C", "A", "D", "L", "J"],
    "ABCDGHKL": ["H", "G", "B", "C", "A", "D", "L", "K"],
    "ABCDGIJK": ["C", "J", "B", "D", "A", "G", "I", "K"],
    "ABCDGIJL": ["C", "J", "B", "D", "A", "G", "L", "I"],
    "ABCDGIKL": ["I", "G", "B", "C", "A", "D", "L", "K"],
    "ABCDGJKL": ["C", "J", "B", "D", "A", "G", "L", "K"],
    "ABCDHIJK": ["H", "J", "B", "C", "A", "D", "I", "K"],
    "ABCDHIJL": ["H", "J", "B", "C", "A", "D", "L", "I"],
    "ABCDHIKL": ["H", "I", "B", "C", "A", "D", "L", "K"],
    "ABCDHJKL": ["H", "J", "B", "C", "A", "D", "L", "K"],
    "ABCDIJKL": ["I", "J", "B", "C", "A", "D", "L", "K"],
    "ABCEFGHI": ["H", "G", "B", "C", "A", "F", "E", "I"],
    "ABCEFGHJ": ["H", "G", "B", "C", "A", "F", "E", "J"],
    "ABCEFGHK": ["H", "G", "B", "C", "A", "F", "E", "K"],
    "ABCEFGHL": ["H", "G", "B", "C", "A", "F", "L", "E"],
    "ABCEFGIJ": ["E", "G", "B", "C", "A", "F", "I", "J"],
    "ABCEFGIK": ["E", "G", "B", "C", "A", "F", "I", "K"],
    "ABCEFGIL": ["E", "G", "B", "C", "A", "F", "L", "I"],
    "ABCEFGJK": ["E", "G", "B", "C", "A", "F", "J", "K"],
    "ABCEFGJL": ["E", "G", "B", "C", "A", "F", "L", "J"],
    "ABCEFGKL": ["E", "G", "B", "C", "A", "F", "L", "K"],
    "ABCEFHIJ": ["H", "J", "B", "C", "A", "F", "E", "I"],
    "ABCEFHIK": ["H", "E", "B", "C", "A", "F", "I", "K"],
    "ABCEFHIL": ["H", "E", "B", "C", "A", "F", "L", "I"],
    "ABCEFHJK": ["H", "J", "B", "C", "A", "F", "E", "K"],
    "ABCEFHJL": ["H", "J", "B", "C", "A", "F", "L", "E"],
    "ABCEFHKL": ["H", "E", "B", "C", "A", "F", "L", "K"],
    "ABCEFIJK": ["E", "J", "B", "C", "A", "F", "I", "K"],
    "ABCEFIJL": ["E", "J", "B", "C", "A", "F", "L", "I"],
    "ABCEFIKL": ["E", "I", "B", "C", "A", "F", "L", "K"],
    "ABCEFJKL": ["E", "J", "B", "C", "A", "F", "L", "K"],
    "ABCEGHIJ": ["H", "J", "B", "C", "A", "G", "E", "I"],
    "ABCEGHIK": ["E", "G", "B", "C", "A", "H", "I", "K"],
    "ABCEGHIL": ["E", "G", "B", "C", "A", "H", "L", "I"],
    "ABCEGHJK": ["H", "J", "B", "C", "A", "G", "E", "K"],
    "ABCEGHJL": ["H", "J", "B", "C", "A", "G", "L", "E"],
    "ABCEGHKL": ["E", "G", "B", "C", "A", "H", "L", "K"],
    "ABCEGIJK": ["E", "J", "B", "C", "A", "G", "I", "K"],
    "ABCEGIJL": ["E", "J", "B", "C", "A", "G", "L", "I"],
    "ABCEGIKL": ["E", "G", "B", "A", "I", "C", "L", "K"],
    "ABCEGJKL": ["E", "J", "B", "C", "A", "G", "L", "K"],
    "ABCEHIJK": ["E", "J", "B", "C", "A", "H", "I", "K"],
    "ABCEHIJL": ["E", "J", "B", "C", "A", "H", "L", "I"],
    "ABCEHIKL": ["E", "I", "B", "C", "A", "H", "L", "K"],
    "ABCEHJKL": ["E", "J", "B", "C", "A", "H", "L", "K"],
    "ABCEIJKL": ["E", "J", "B", "A", "I", "C", "L", "K"],
    "ABCFGHIJ": ["H", "G", "B", "C", "A", "F", "I", "J"],
    "ABCFGHIK": ["H", "G", "B", "C", "A", "F", "I", "K"],
    "ABCFGHIL": ["H", "G", "B", "C", "A", "F", "L", "I"],
    "ABCFGHJK": ["H", "G", "B", "C", "A", "F", "J", "K"],
    "ABCFGHJL": ["H", "G", "B", "C", "A", "F", "L", "J"],
    "ABCFGHKL": ["H", "G", "B", "C", "A", "F", "L", "K"],
    "ABCFGIJK": ["C", "J", "B", "F", "A", "G", "I", "K"],
    "ABCFGIJL": ["C", "J", "B", "F", "A", "G", "L", "I"],
    "ABCFGIKL": ["I", "G", "B", "C", "A", "F", "L", "K"],
    "ABCFGJKL": ["C", "J", "B", "F", "A", "G", "L", "K"],
    "ABCFHIJK": ["H", "J", "B", "C", "A", "F", "I", "K"],
    "ABCFHIJL": ["H", "J", "B", "C", "A", "F", "L", "I"],
    "ABCFHIKL": ["H", "I", "B", "C", "A", "F", "L", "K"],
    "ABCFHJKL": ["H", "J", "B", "C", "A", "F", "L", "K"],
    "ABCFIJKL": ["I", "J", "B", "C", "A", "F", "L", "K"],
    "ABCGHIJK": ["H", "J", "B", "C", "A", "G", "I", "K"],
    "ABCGHIJL": ["H", "J", "B", "C", "A", "G", "L", "I"],
    "ABCGHIKL": ["I", "G", "B", "C", "A", "H", "L", "K"],
    "ABCGHJKL": ["H", "J", "B", "C", "A", "G", "L", "K"],
    "ABCGIJKL": ["I", "J", "B", "C", "A", "G", "L", "K"],
    "ABCHIJKL": ["I", "J", "B", "C", "A", "H", "L", "K"],
    "ABDEFGHI": ["H", "G", "B", "D", "A", "F", "E", "I"],
    "ABDEFGHJ": ["H", "G", "B", "D", "A", "F", "E", "J"],
    "ABDEFGHK": ["H", "G", "B", "D", "A", "F", "E", "K"],
    "ABDEFGHL": ["H", "G", "B", "D", "A", "F", "L", "E"],
    "ABDEFGIJ": ["E", "G", "B", "D", "A", "F", "I", "J"],
    "ABDEFGIK": ["E", "G", "B", "D", "A", "F", "I", "K"],
    "ABDEFGIL": ["E", "G", "B", "D", "A", "F", "L", "I"],
    "ABDEFGJK": ["E", "G", "B", "D", "A", "F", "J", "K"],
    "ABDEFGJL": ["E", "G", "B", "D", "A", "F", "L", "J"],
    "ABDEFGKL": ["E", "G", "B", "D", "A", "F", "L", "K"],
    "ABDEFHIJ": ["H", "J", "B", "D", "A", "F", "E", "I"],
    "ABDEFHIK": ["H", "E", "B", "D", "A", "F", "I", "K"],
    "ABDEFHIL": ["H", "E", "B", "D", "A", "F", "L", "I"],
    "ABDEFHJK": ["H", "J", "B", "D", "A", "F", "E", "K"],
    "ABDEFHJL": ["H", "J", "B", "D", "A", "F", "L", "E"],
    "ABDEFHKL": ["H", "E", "B", "D", "A", "F", "L", "K"],
    "ABDEFIJK": ["E", "J", "B", "D", "A", "F", "I", "K"],
    "ABDEFIJL": ["E", "J", "B", "D", "A", "F", "L", "I"],
    "ABDEFIKL": ["E", "I", "B", "D", "A", "F", "L", "K"],
    "ABDEFJKL": ["E", "J", "B", "D", "A", "F", "L", "K"],
    "ABDEGHIJ": ["H", "J", "B", "D", "A", "G", "E", "I"],
    "ABDEGHIK": ["E", "G", "B", "D", "A", "H", "I", "K"],
    "ABDEGHIL": ["E", "G", "B", "D", "A", "H", "L", "I"],
    "ABDEGHJK": ["H", "J", "B", "D", "A", "G", "E", "K"],
    "ABDEGHJL": ["H", "J", "B", "D", "A", "G", "L", "E"],
    "ABDEGHKL": ["E", "G", "B", "D", "A", "H", "L", "K"],
    "ABDEGIJK": ["E", "J", "B", "D", "A", "G", "I", "K"],
    "ABDEGIJL": ["E", "J", "B", "D", "A", "G", "L", "I"],
    "ABDEGIKL": ["E", "G", "B", "A", "I", "D", "L", "K"],
    "ABDEGJKL": ["E", "J", "B", "D", "A", "G", "L", "K"],
    "ABDEHIJK": ["E", "J", "B", "D", "A", "H", "I", "K"],
    "ABDEHIJL": ["E", "J", "B", "D", "A", "H", "L", "I"],
    "ABDEHIKL": ["E", "I", "B", "D", "A", "H", "L", "K"],
    "ABDEHJKL": ["E", "J", "B", "D", "A", "H", "L", "K"],
    "ABDEIJKL": ["E", "J", "B", "A", "I", "D", "L", "K"],
    "ABDFGHIJ": ["H", "G", "B", "D", "A", "F", "I", "J"],
    "ABDFGHIK": ["H", "G", "B", "D", "A", "F", "I", "K"],
    "ABDFGHIL": ["H", "G", "B", "D", "A", "F", "L", "I"],
    "ABDFGHJK": ["H", "G", "B", "D", "A", "F", "J", "K"],
    "ABDFGHJL": ["H", "G", "B", "D", "A", "F", "L", "J"],
    "ABDFGHKL": ["H", "G", "B", "D", "A", "F", "L", "K"],
    "ABDFGIJK": ["F", "J", "B", "D", "A", "G", "I", "K"],
    "ABDFGIJL": ["F", "J", "B", "D", "A", "G", "L", "I"],
    "ABDFGIKL": ["I", "G", "B", "D", "A", "F", "L", "K"],
    "ABDFGJKL": ["F", "J", "B", "D", "A", "G", "L", "K"],
    "ABDFHIJK": ["H", "J", "B", "D", "A", "F", "I", "K"],
    "ABDFHIJL": ["H", "J", "B", "D", "A", "F", "L", "I"],
    "ABDFHIKL": ["H", "I", "B", "D", "A", "F", "L", "K"],
    "ABDFHJKL": ["H", "J", "B", "D", "A", "F", "L", "K"],
    "ABDFIJKL": ["I", "J", "B", "D", "A", "F", "L", "K"],
    "ABDGHIJK": ["H", "J", "B", "D", "A", "G", "I", "K"],
    "ABDGHIJL": ["H", "J", "B", "D", "A", "G", "L", "I"],
    "ABDGHIKL": ["I", "G", "B", "D", "A", "H", "L", "K"],
    "ABDGHJKL": ["H", "J", "B", "D", "A", "G", "L", "K"],
    "ABDGIJKL": ["I", "J", "B", "D", "A", "G", "L", "K"],
    "ABDHIJKL": ["I", "J", "B", "D", "A", "H", "L", "K"],
    "ABEFGHIJ": ["H", "J", "B", "F", "A", "G", "E", "I"],
    "ABEFGHIK": ["E", "G", "B", "F", "A", "H", "I", "K"],
    "ABEFGHIL": ["E", "G", "B", "F", "A", "H", "L", "I"],
    "ABEFGHJK": ["H", "J", "B", "F", "A", "G", "E", "K"],
    "ABEFGHJL": ["H", "J", "B", "F", "A", "G", "L", "E"],
    "ABEFGHKL": ["E", "G", "B", "F", "A", "H", "L", "K"],
    "ABEFGIJK": ["E", "J", "B", "F", "A", "G", "I", "K"],
    "ABEFGIJL": ["E", "J", "B", "F", "A", "G", "L", "I"],
    "ABEFGIKL": ["E", "G", "B", "A", "I", "F", "L", "K"],
    "ABEFGJKL": ["E", "J", "B", "F", "A", "G", "L", "K"],
    "ABEFHIJK": ["E", "J", "B", "F", "A", "H", "I", "K"],
    "ABEFHIJL": ["E", "J", "B", "F", "A", "H", "L", "I"],
    "ABEFHIKL": ["E", "I", "B", "F", "A", "H", "L", "K"],
    "ABEFHJKL": ["E", "J", "B", "F", "A", "H", "L", "K"],
    "ABEFIJKL": ["E", "J", "B", "A", "I", "F", "L", "K"],
    "ABEGHIJK": ["E", "J", "B", "A", "H", "G", "I", "K"],
    "ABEGHIJL": ["E", "J", "B", "A", "H", "G", "L", "I"],
    "ABEGHIKL": ["E", "G", "B", "A", "I", "H", "L", "K"],
    "ABEGHJKL": ["E", "J", "B", "A", "H", "G", "L", "K"],
    "ABEGIJKL": ["E", "J", "B", "A", "I", "G", "L", "K"],
    "ABEHIJKL": ["E", "J", "B", "A", "I", "H", "L", "K"],
    "ABFGHIJK": ["H", "J", "B", "F", "A", "G", "I", "K"],
    "ABFGHIJL": ["H", "J", "B", "F", "A", "G", "L", "I"],
    "ABFGHIKL": ["H", "G", "B", "A", "I", "F", "L", "K"],
    "ABFGHJKL": ["H", "J", "B", "F", "A", "G", "L", "K"],
    "ABFGIJKL": ["I", "J", "B", "F", "A", "G", "L", "K"],
    "ABFHIJKL": ["H", "J", "B", "A", "I", "F", "L", "K"],
    "ABGHIJKL": ["H", "J", "B", "A", "I", "G", "L", "K"],
    "ACDEFGHI": ["H", "G", "E", "C", "A", "F", "D", "I"],
    "ACDEFGHJ": ["H", "G", "J", "C", "A", "F", "D", "E"],
    "ACDEFGHK": ["H", "G", "E", "C", "A", "F", "D", "K"],
    "ACDEFGHL": ["H", "G", "F", "C", "A", "D", "L", "E"],
    "ACDEFGIJ": ["C", "G", "J", "D", "A", "F", "E", "I"],
    "ACDEFGIK": ["C", "G", "E", "D", "A", "F", "I", "K"],
    "ACDEFGIL": ["C", "G", "E", "D", "A", "F", "L", "I"],
    "ACDEFGJK": ["C", "G", "J", "D", "A", "F", "E", "K"],
    "ACDEFGJL": ["C", "G", "J", "D", "A", "F", "L", "E"],
    "ACDEFGKL": ["C", "G", "E", "D", "A", "F", "L", "K"],
    "ACDEFHIJ": ["H", "J", "E", "C", "A", "F", "D", "I"],
    "ACDEFHIK": ["H", "E", "F", "C", "A", "D", "I", "K"],
    "ACDEFHIL": ["H", "E", "F", "C", "A", "D", "L", "I"],
    "ACDEFHJK": ["H", "J", "E", "C", "A", "F", "D", "K"],
    "ACDEFHJL": ["H", "J", "F", "C", "A", "D", "L", "E"],
    "ACDEFHKL": ["H", "E", "F", "C", "A", "D", "L", "K"],
    "ACDEFIJK": ["C", "J", "E", "D", "A", "F", "I", "K"],
    "ACDEFIJL": ["C", "J", "E", "D", "A", "F", "L", "I"],
    "ACDEFIKL": ["C", "E", "I", "D", "A", "F", "L", "K"],
    "ACDEFJKL": ["C", "J", "E", "D", "A", "F", "L", "K"],
    "ACDEGHIJ": ["H", "G", "J", "C", "A", "D", "E", "I"],
    "ACDEGHIK": ["H", "G", "E", "C", "A", "D", "I", "K"],
    "ACDEGHIL": ["H", "G", "E", "C", "A", "D", "L", "I"],
    "ACDEGHJK": ["H", "G", "J", "C", "A", "D", "E", "K"],
    "ACDEGHJL": ["H", "G", "J", "C", "A", "D", "L", "E"],
    "ACDEGHKL": ["H", "G", "E", "C", "A", "D", "L", "K"],
    "ACDEGIJK": ["E", "G", "J", "C", "A", "D", "I", "K"],
    "ACDEGIJL": ["E", "G", "J", "C", "A", "D", "L", "I"],
    "ACDEGIKL": ["E", "G", "I", "C", "A", "D", "L", "K"],
    "ACDEGJKL": ["E", "G", "J", "C", "A", "D", "L", "K"],
    "ACDEHIJK": ["H", "J", "E", "C", "A", "D", "I", "K"],
    "ACDEHIJL": ["H", "J", "E", "C", "A", "D", "L", "I"],
    "ACDEHIKL": ["H", "E", "I", "C", "A", "D", "L", "K"],
    "ACDEHJKL": ["H", "J", "E", "C", "A", "D", "L", "K"],
    "ACDEIJKL": ["E", "J", "I", "C", "A", "D", "L", "K"],
    "ACDFGHIJ": ["H", "G", "J", "C", "A", "F", "D", "I"],
    "ACDFGHIK": ["H", "G", "F", "C", "A", "D", "I", "K"],
    "ACDFGHIL": ["H", "G", "F", "C", "A", "D", "L", "I"],
    "ACDFGHJK": ["H", "G", "J", "C", "A", "F", "D", "K"],
    "ACDFGHJL": ["C", "G", "J", "D", "A", "F", "L", "H"],
    "ACDFGHKL": ["H", "G", "F", "C", "A", "D", "L", "K"],
    "ACDFGIJK": ["C", "G", "J", "D", "A", "F", "I", "K"],
    "ACDFGIJL": ["C", "G", "J", "D", "A", "F", "L", "I"],
    "ACDFGIKL": ["C", "G", "I", "D", "A", "F", "L", "K"],
    "ACDFGJKL": ["C", "G", "J", "D", "A", "F", "L", "K"],
    "ACDFHIJK": ["H", "J", "F", "C", "A", "D", "I", "K"],
    "ACDFHIJL": ["H", "J", "F", "C", "A", "D", "L", "I"],
    "ACDFHIKL": ["H", "F", "I", "C", "A", "D", "L", "K"],
    "ACDFHJKL": ["H", "J", "F", "C", "A", "D", "L", "K"],
    "ACDFIJKL": ["C", "J", "I", "D", "A", "F", "L", "K"],
    "ACDGHIJK": ["H", "G", "J", "C", "A", "D", "I", "K"],
    "ACDGHIJL": ["H", "G", "J", "C", "A", "D", "L", "I"],
    "ACDGHIKL": ["H", "G", "I", "C", "A", "D", "L", "K"],
    "ACDGHJKL": ["H", "G", "J", "C", "A", "D", "L", "K"],
    "ACDGIJKL": ["I", "G", "J", "C", "A", "D", "L", "K"],
    "ACDHIJKL": ["H", "J", "I", "C", "A", "D", "L", "K"],
    "ACEFGHIJ": ["H", "G", "J", "C", "A", "F", "E", "I"],
    "ACEFGHIK": ["H", "G", "E", "C", "A", "F", "I", "K"],
    "ACEFGHIL": ["H", "G", "E", "C", "A", "F", "L", "I"],
    "ACEFGHJK": ["H", "G", "J", "C", "A", "F", "E", "K"],
    "ACEFGHJL": ["H", "G", "J", "C", "A", "F", "L", "E"],
    "ACEFGHKL": ["H", "G", "E", "C", "A", "F", "L", "K"],
    "ACEFGIJK": ["E", "G", "J", "C", "A", "F", "I", "K"],
    "ACEFGIJL": ["E", "G", "J", "C", "A", "F", "L", "I"],
    "ACEFGIKL": ["E", "G", "I", "C", "A", "F", "L", "K"],
    "ACEFGJKL": ["E", "G", "J", "C", "A", "F", "L", "K"],
    "ACEFHIJK": ["H", "J", "E", "C", "A", "F", "I", "K"],
    "ACEFHIJL": ["H", "J", "E", "C", "A", "F", "L", "I"],
    "ACEFHIKL": ["H", "E", "I", "C", "A", "F", "L", "K"],
    "ACEFHJKL": ["H", "J", "E", "C", "A", "F", "L", "K"],
    "ACEFIJKL": ["E", "J", "I", "C", "A", "F", "L", "K"],
    "ACEGHIJK": ["E", "G", "J", "C", "A", "H", "I", "K"],
    "ACEGHIJL": ["E", "G", "J", "C", "A", "H", "L", "I"],
    "ACEGHIKL": ["E", "G", "I", "C", "A", "H", "L", "K"],
    "ACEGHJKL": ["E", "G", "J", "C", "A", "H", "L", "K"],
    "ACEGIJKL": ["E", "J", "I", "C", "A", "G", "L", "K"],
    "ACEHIJKL": ["E", "J", "I", "C", "A", "H", "L", "K"],
    "ACFGHIJK": ["H", "G", "J", "C", "A", "F", "I", "K"],
    "ACFGHIJL": ["H", "G", "J", "C", "A", "F", "L", "I"],
    "ACFGHIKL": ["H", "G", "I", "C", "A", "F", "L", "K"],
    "ACFGHJKL": ["H", "G", "J", "C", "A", "F", "L", "K"],
    "ACFGIJKL": ["I", "G", "J", "C", "A", "F", "L", "K"],
    "ACFHIJKL": ["H", "J", "I", "C", "A", "F", "L", "K"],
    "ACGHIJKL": ["H", "J", "I", "C", "A", "G", "L", "K"],
    "ADEFGHIJ": ["H", "G", "J", "D", "A", "F", "E", "I"],
    "ADEFGHIK": ["H", "G", "E", "D", "A", "F", "I", "K"],
    "ADEFGHIL": ["H", "G", "E", "D", "A", "F", "L", "I"],
    "ADEFGHJK": ["H", "G", "J", "D", "A", "F", "E", "K"],
    "ADEFGHJL": ["H", "G", "J", "D", "A", "F", "L", "E"],
    "ADEFGHKL": ["H", "G", "E", "D", "A", "F", "L", "K"],
    "ADEFGIJK": ["E", "G", "J", "D", "A", "F", "I", "K"],
    "ADEFGIJL": ["E", "G", "J", "D", "A", "F", "L", "I"],
    "ADEFGIKL": ["E", "G", "I", "D", "A", "F", "L", "K"],
    "ADEFGJKL": ["E", "G", "J", "D", "A", "F", "L", "K"],
    "ADEFHIJK": ["H", "J", "E", "D", "A", "F", "I", "K"],
    "ADEFHIJL": ["H", "J", "E", "D", "A", "F", "L", "I"],
    "ADEFHIKL": ["H", "E", "I", "D", "A", "F", "L", "K"],
    "ADEFHJKL": ["H", "J", "E", "D", "A", "F", "L", "K"],
    "ADEFIJKL": ["E", "J", "I", "D", "A", "F", "L", "K"],
    "ADEGHIJK": ["E", "G", "J", "D", "A", "H", "I", "K"],
    "ADEGHIJL": ["E", "G", "J", "D", "A", "H", "L", "I"],
    "ADEGHIKL": ["E", "G", "I", "D", "A", "H", "L", "K"],
    "ADEGHJKL": ["E", "G", "J", "D", "A", "H", "L", "K"],
    "ADEGIJKL": ["E", "J", "I", "D", "A", "G", "L", "K"],
    "ADEHIJKL": ["E", "J", "I", "D", "A", "H", "L", "K"],
    "ADFGHIJK": ["H", "G", "J", "D", "A", "F", "I", "K"],
    "ADFGHIJL": ["H", "G", "J", "D", "A", "F", "L", "I"],
    "ADFGHIKL": ["H", "G", "I", "D", "A", "F", "L", "K"],
    "ADFGHJKL": ["H", "G", "J", "D", "A", "F", "L", "K"],
    "ADFGIJKL": ["I", "G", "J", "D", "A", "F", "L", "K"],
    "ADFHIJKL": ["H", "J", "I", "D", "A", "F", "L", "K"],
    "ADGHIJKL": ["H", "J", "I", "D", "A", "G", "L", "K"],
    "AEFGHIJK": ["E", "G", "J", "F", "A", "H", "I", "K"],
    "AEFGHIJL": ["E", "G", "J", "F", "A", "H", "L", "I"],
    "AEFGHIKL": ["E", "G", "I", "F", "A", "H", "L", "K"],
    "AEFGHJKL": ["E", "G", "J", "F", "A", "H", "L", "K"],
    "AEFGIJKL": ["E", "J", "I", "F", "A", "G", "L", "K"],
    "AEFHIJKL": ["E", "J", "I", "F", "A", "H", "L", "K"],
    "AEGHIJKL": ["E", "J", "I", "A", "H", "G", "L", "K"],
    "AFGHIJKL": ["H", "J", "I", "F", "A", "G", "L", "K"],
    "BCDEFGHI": ["C", "G", "B", "D", "H", "F", "E", "I"],
    "BCDEFGHJ": ["H", "G", "B", "C", "J", "F", "D", "E"],
    "BCDEFGHK": ["C", "G", "B", "D", "H", "F", "E", "K"],
    "BCDEFGHL": ["C", "G", "B", "D", "H", "F", "L", "E"],
    "BCDEFGIJ": ["C", "G", "B", "D", "J", "F", "E", "I"],
    "BCDEFGIK": ["C", "G", "B", "D", "E", "F", "I", "K"],
    "BCDEFGIL": ["C", "G", "B", "D", "E", "F", "L", "I"],
    "BCDEFGJK": ["C", "G", "B", "D", "J", "F", "E", "K"],
    "BCDEFGJL": ["C", "G", "B", "D", "J", "F", "L", "E"],
    "BCDEFGKL": ["C", "G", "B", "D", "E", "F", "L", "K"],
    "BCDEFHIJ": ["C", "J", "B", "D", "H", "F", "E", "I"],
    "BCDEFHIK": ["C", "E", "B", "D", "H", "F", "I", "K"],
    "BCDEFHIL": ["C", "E", "B", "D", "H", "F", "L", "I"],
    "BCDEFHJK": ["C", "J", "B", "D", "H", "F", "E", "K"],
    "BCDEFHJL": ["C", "J", "B", "D", "H", "F", "L", "E"],
    "BCDEFHKL": ["C", "E", "B", "D", "H", "F", "L", "K"],
    "BCDEFIJK": ["C", "J", "B", "D", "E", "F", "I", "K"],
    "BCDEFIJL": ["C", "J", "B", "D", "E", "F", "L", "I"],
    "BCDEFIKL": ["C", "E", "B", "D", "I", "F", "L", "K"],
    "BCDEFJKL": ["C", "J", "B", "D", "E", "F", "L", "K"],
    "BCDEGHIJ": ["H", "G", "B", "C", "J", "D", "E", "I"],
    "BCDEGHIK": ["E", "G", "B", "C", "H", "D", "I", "K"],
    "BCDEGHIL": ["E", "G", "B", "C", "H", "D", "L", "I"],
    "BCDEGHJK": ["H", "G", "B", "C", "J", "D", "E", "K"],
    "BCDEGHJL": ["H", "G", "B", "C", "J", "D", "L", "E"],
    "BCDEGHKL": ["E", "G", "B", "C", "H", "D", "L", "K"],
    "BCDEGIJK": ["E", "G", "B", "C", "J", "D", "I", "K"],
    "BCDEGIJL": ["E", "G", "B", "C", "J", "D", "L", "I"],
    "BCDEGIKL": ["E", "G", "B", "C", "I", "D", "L", "K"],
    "BCDEGJKL": ["E", "G", "B", "C", "J", "D", "L", "K"],
    "BCDEHIJK": ["E", "J", "B", "C", "H", "D", "I", "K"],
    "BCDEHIJL": ["E", "J", "B", "C", "H", "D", "L", "I"],
    "BCDEHIKL": ["E", "I", "B", "C", "H", "D", "L", "K"],
    "BCDEHJKL": ["E", "J", "B", "C", "H", "D", "L", "K"],
    "BCDEIJKL": ["E", "J", "B", "C", "I", "D", "L", "K"],
    "BCDFGHIJ": ["H", "G", "B", "C", "J", "F", "D", "I"],
    "BCDFGHIK": ["C", "G", "B", "D", "H", "F", "I", "K"],
    "BCDFGHIL": ["C", "G", "B", "D", "H", "F", "L", "I"],
    "BCDFGHJK": ["H", "G", "B", "C", "J", "F", "D", "K"],
    "BCDFGHJL": ["C", "G", "B", "D", "H", "F", "L", "J"],
    "BCDFGHKL": ["C", "G", "B", "D", "H", "F", "L", "K"],
    "BCDFGIJK": ["C", "G", "B", "D", "J", "F", "I", "K"],
    "BCDFGIJL": ["C", "G", "B", "D", "J", "F", "L", "I"],
    "BCDFGIKL": ["C", "G", "B", "D", "I", "F", "L", "K"],
    "BCDFGJKL": ["C", "G", "B", "D", "J", "F", "L", "K"],
    "BCDFHIJK": ["C", "J", "B", "D", "H", "F", "I", "K"],
    "BCDFHIJL": ["C", "J", "B", "D", "H", "F", "L", "I"],
    "BCDFHIKL": ["C", "I", "B", "D", "H", "F", "L", "K"],
    "BCDFHJKL": ["C", "J", "B", "D", "H", "F", "L", "K"],
    "BCDFIJKL": ["C", "J", "B", "D", "I", "F", "L", "K"],
    "BCDGHIJK": ["H", "G", "B", "C", "J", "D", "I", "K"],
    "BCDGHIJL": ["H", "G", "B", "C", "J", "D", "L", "I"],
    "BCDGHIKL": ["H", "G", "B", "C", "I", "D", "L", "K"],
    "BCDGHJKL": ["H", "G", "B", "C", "J", "D", "L", "K"],
    "BCDGIJKL": ["I", "G", "B", "C", "J", "D", "L", "K"],
    "BCDHIJKL": ["H", "J", "B", "C", "I", "D", "L", "K"],
    "BCEFGHIJ": ["H", "G", "B", "C", "J", "F", "E", "I"],
    "BCEFGHIK": ["E", "G", "B", "C", "H", "F", "I", "K"],
    "BCEFGHIL": ["E", "G", "B", "C", "H", "F", "L", "I"],
    "BCEFGHJK": ["H", "G", "B", "C", "J", "F", "E", "K"],
    "BCEFGHJL": ["H", "G", "B", "C", "J", "F", "L", "E"],
    "BCEFGHKL": ["E", "G", "B", "C", "H", "F", "L", "K"],
    "BCEFGIJK": ["E", "G", "B", "C", "J", "F", "I", "K"],
    "BCEFGIJL": ["E", "G", "B", "C", "J", "F", "L", "I"],
    "BCEFGIKL": ["E", "G", "B", "C", "I", "F", "L", "K"],
    "BCEFGJKL": ["E", "G", "B", "C", "J", "F", "L", "K"],
    "BCEFHIJK": ["E", "J", "B", "C", "H", "F", "I", "K"],
    "BCEFHIJL": ["E", "J", "B", "C", "H", "F", "L", "I"],
    "BCEFHIKL": ["E", "I", "B", "C", "H", "F", "L", "K"],
    "BCEFHJKL": ["E", "J", "B", "C", "H", "F", "L", "K"],
    "BCEFIJKL": ["E", "J", "B", "C", "I", "F", "L", "K"],
    "BCEGHIJK": ["E", "J", "B", "C", "H", "G", "I", "K"],
    "BCEGHIJL": ["E", "J", "B", "C", "H", "G", "L", "I"],
    "BCEGHIKL": ["E", "G", "B", "C", "I", "H", "L", "K"],
    "BCEGHJKL": ["E", "J", "B", "C", "H", "G", "L", "K"],
    "BCEGIJKL": ["E", "J", "B", "C", "I", "G", "L", "K"],
    "BCEHIJKL": ["E", "J", "B", "C", "I", "H", "L", "K"],
    "BCFGHIJK": ["H", "G", "B", "C", "J", "F", "I", "K"],
    "BCFGHIJL": ["H", "G", "B", "C", "J", "F", "L", "I"],
    "BCFGHIKL": ["H", "G", "B", "C", "I", "F", "L", "K"],
    "BCFGHJKL": ["H", "G", "B", "C", "J", "F", "L", "K"],
    "BCFGIJKL": ["I", "G", "B", "C", "J", "F", "L", "K"],
    "BCFHIJKL": ["H", "J", "B", "C", "I", "F", "L", "K"],
    "BCGHIJKL": ["H", "J", "B", "C", "I", "G", "L", "K"],
    "BDEFGHIJ": ["H", "G", "B", "D", "J", "F", "E", "I"],
    "BDEFGHIK": ["E", "G", "B", "D", "H", "F", "I", "K"],
    "BDEFGHIL": ["E", "G", "B", "D", "H", "F", "L", "I"],
    "BDEFGHJK": ["H", "G", "B", "D", "J", "F", "E", "K"],
    "BDEFGHJL": ["H", "G", "B", "D", "J", "F", "L", "E"],
    "BDEFGHKL": ["E", "G", "B", "D", "H", "F", "L", "K"],
    "BDEFGIJK": ["E", "G", "B", "D", "J", "F", "I", "K"],
    "BDEFGIJL": ["E", "G", "B", "D", "J", "F", "L", "I"],
    "BDEFGIKL": ["E", "G", "B", "D", "I", "F", "L", "K"],
    "BDEFGJKL": ["E", "G", "B", "D", "J", "F", "L", "K"],
    "BDEFHIJK": ["E", "J", "B", "D", "H", "F", "I", "K"],
    "BDEFHIJL": ["E", "J", "B", "D", "H", "F", "L", "I"],
    "BDEFHIKL": ["E", "I", "B", "D", "H", "F", "L", "K"],
    "BDEFHJKL": ["E", "J", "B", "D", "H", "F", "L", "K"],
    "BDEFIJKL": ["E", "J", "B", "D", "I", "F", "L", "K"],
    "BDEGHIJK": ["E", "J", "B", "D", "H", "G", "I", "K"],
    "BDEGHIJL": ["E", "J", "B", "D", "H", "G", "L", "I"],
    "BDEGHIKL": ["E", "G", "B", "D", "I", "H", "L", "K"],
    "BDEGHJKL": ["E", "J", "B", "D", "H", "G", "L", "K"],
    "BDEGIJKL": ["E", "J", "B", "D", "I", "G", "L", "K"],
    "BDEHIJKL": ["E", "J", "B", "D", "I", "H", "L", "K"],
    "BDFGHIJK": ["H", "G", "B", "D", "J", "F", "I", "K"],
    "BDFGHIJL": ["H", "G", "B", "D", "J", "F", "L", "I"],
    "BDFGHIKL": ["H", "G", "B", "D", "I", "F", "L", "K"],
    "BDFGHJKL": ["H", "G", "B", "D", "J", "F", "L", "K"],
    "BDFGIJKL": ["I", "G", "B", "D", "J", "F", "L", "K"],
    "BDFHIJKL": ["H", "J", "B", "D", "I", "F", "L", "K"],
    "BDGHIJKL": ["H", "J", "B", "D", "I", "G", "L", "K"],
    "BEFGHIJK": ["E", "J", "B", "F", "H", "G", "I", "K"],
    "BEFGHIJL": ["E", "J", "B", "F", "H", "G", "L", "I"],
    "BEFGHIKL": ["E", "G", "B", "F", "I", "H", "L", "K"],
    "BEFGHJKL": ["E", "J", "B", "F", "H", "G", "L", "K"],
    "BEFGIJKL": ["E", "J", "B", "F", "I", "G", "L", "K"],
    "BEFHIJKL": ["E", "J", "B", "F", "I", "H", "L", "K"],
    "BEGHIJKL": ["E", "J", "I", "B", "H", "G", "L", "K"],
    "BFGHIJKL": ["H", "J", "B", "F", "I", "G", "L", "K"],
    "CDEFGHIJ": ["C", "G", "J", "D", "H", "F", "E", "I"],
    "CDEFGHIK": ["C", "G", "E", "D", "H", "F", "I", "K"],
    "CDEFGHIL": ["C", "G", "E", "D", "H", "F", "L", "I"],
    "CDEFGHJK": ["C", "G", "J", "D", "H", "F", "E", "K"],
    "CDEFGHJL": ["C", "G", "J", "D", "H", "F", "L", "E"],
    "CDEFGHKL": ["C", "G", "E", "D", "H", "F", "L", "K"],
    "CDEFGIJK": ["C", "G", "E", "D", "J", "F", "I", "K"],
    "CDEFGIJL": ["C", "G", "E", "D", "J", "F", "L", "I"],
    "CDEFGIKL": ["C", "G", "E", "D", "I", "F", "L", "K"],
    "CDEFGJKL": ["C", "G", "E", "D", "J", "F", "L", "K"],
    "CDEFHIJK": ["C", "J", "E", "D", "H", "F", "I", "K"],
    "CDEFHIJL": ["C", "J", "E", "D", "H", "F", "L", "I"],
    "CDEFHIKL": ["C", "E", "I", "D", "H", "F", "L", "K"],
    "CDEFHJKL": ["C", "J", "E", "D", "H", "F", "L", "K"],
    "CDEFIJKL": ["C", "J", "E", "D", "I", "F", "L", "K"],
    "CDEGHIJK": ["E", "G", "J", "C", "H", "D", "I", "K"],
    "CDEGHIJL": ["E", "G", "J", "C", "H", "D", "L", "I"],
    "CDEGHIKL": ["E", "G", "I", "C", "H", "D", "L", "K"],
    "CDEGHJKL": ["E", "G", "J", "C", "H", "D", "L", "K"],
    "CDEGIJKL": ["E", "G", "I", "C", "J", "D", "L", "K"],
    "CDEHIJKL": ["E", "J", "I", "C", "H", "D", "L", "K"],
    "CDFGHIJK": ["C", "G", "J", "D", "H", "F", "I", "K"],
    "CDFGHIJL": ["C", "G", "J", "D", "H", "F", "L", "I"],
    "CDFGHIKL": ["C", "G", "I", "D", "H", "F", "L", "K"],
    "CDFGHJKL": ["C", "G", "J", "D", "H", "F", "L", "K"],
    "CDFGIJKL": ["C", "G", "I", "D", "J", "F", "L", "K"],
    "CDFHIJKL": ["C", "J", "I", "D", "H", "F", "L", "K"],
    "CDGHIJKL": ["H", "G", "I", "C", "J", "D", "L", "K"],
    "CEFGHIJK": ["E", "G", "J", "C", "H", "F", "I", "K"],
    "CEFGHIJL": ["E", "G", "J", "C", "H", "F", "L", "I"],
    "CEFGHIKL": ["E", "G", "I", "C", "H", "F", "L", "K"],
    "CEFGHJKL": ["E", "G", "J", "C", "H", "F", "L", "K"],
    "CEFGIJKL": ["E", "G", "I", "C", "J", "F", "L", "K"],
    "CEFHIJKL": ["E", "J", "I", "C", "H", "F", "L", "K"],
    "CEGHIJKL": ["E", "J", "I", "C", "H", "G", "L", "K"],
    "CFGHIJKL": ["H", "G", "I", "C", "J", "F", "L", "K"],
    "DEFGHIJK": ["E", "G", "J", "D", "H", "F", "I", "K"],
    "DEFGHIJL": ["E", "G", "J", "D", "H", "F", "L", "I"],
    "DEFGHIKL": ["E", "G", "I", "D", "H", "F", "L", "K"],
    "DEFGHJKL": ["E", "G", "J", "D", "H", "F", "L", "K"],
    "DEFGIJKL": ["E", "G", "I", "D", "J", "F", "L", "K"],
    "DEFHIJKL": ["E", "J", "I", "D", "H", "F", "L", "K"],
    "DEGHIJKL": ["E", "J", "I", "D", "H", "G", "L", "K"],
    "DFGHIJKL": ["H", "G", "I", "D", "J", "F", "L", "K"],
    "EFGHIJKL": ["E", "J", "I", "F", "H", "G", "L", "K"],
}

# Ordered list of Python slot names corresponding to combo table indices 0-7
COMBO_SLOT_ORDER = [
    "3rd_CEFHI",   # i=0: 3rd-place opponent for W_A (M79)
    "3rd_EFGIJ",   # i=1: 3rd-place opponent for W_B (M85)
    "3rd_BEFIJ",   # i=2: 3rd-place opponent for W_D (M81)
    "3rd_ABCDF",   # i=3: 3rd-place opponent for W_E (M74)
    "3rd_AEHIJ",   # i=4: 3rd-place opponent for W_G (M82)
    "3rd_CDFGH",   # i=5: 3rd-place opponent for W_I (M77)
    "3rd_DEIJL",   # i=6: 3rd-place opponent for W_K (M87)
    "3rd_EHIJK",   # i=7: 3rd-place opponent for W_L (M80)
]


def assign_third_place_teams(third_place_teams, rank_lookup):
    """
    Assign the best 8 third-place teams to the 8 bracket slots using the
    official FIFA 495-combination table. Key = sorted 8-char group string.
    Falls back to greedy only if the key is not found (should never happen
    with valid 12-group input).
    """
    best8 = third_place_teams[:8]
    group_to_team = {group: team for team, group, pts, gd, gf in best8}
    groups_key = "".join(sorted(group_to_team.keys()))

    assignment = THIRD_PLACE_COMBOS.get(groups_key)
    if assignment is None:
        # Greedy fallback - should not trigger with valid 12-group data
        slot_assignments = {}
        used = set()
        for slot_name, eligible in THIRD_PLACE_SLOTS:
            for team, group, pts, gd, gf in best8:
                if team not in used and group in eligible:
                    slot_assignments[slot_name] = team
                    used.add(team)
                    break
        return slot_assignments

    return {
        COMBO_SLOT_ORDER[i]: group_to_team[group]
        for i, group in enumerate(assignment)
    }


# ── Knockout round simulation ─────────────────────────────────────────────────
def simulate_knockout_match(home, away, prob_cache, rng):
    """
    Draw a winner. No draws in knockout — use home_win / (home_win + away_win).
    """
    if (home, away) in prob_cache:
        p_home, _, p_away = prob_cache[(home, away)]
    else:
        p_home, p_away = 0.5, 0.5

    p_home_adj = p_home / (p_home + p_away)
    return home if rng.random() < p_home_adj else away


# ── One full tournament simulation ───────────────────────────────────────────
def simulate_tournament(group_fixtures, dc_params, prob_cache,
                        rank_lookup, rng):
    """
    Returns (results, group_position):
      results[team]        = highest round reached (1=group..7=Winner)
      group_position[team] = finishing place in its group (1..4)
    """
    results = {}   # team → round reached
    group_position = {}  # team → group finish place (1=winner, 2=runner-up, ...)

    # ── Group stage ───────────────────────────────────────────────────────────
    group_winners  = {}   # group letter → winner
    group_runners  = {}   # group letter → runner-up
    third_place_list = [] # (team, group, pts, gd, gf)

    for group, matches in group_fixtures.items():
        records = simulate_group(matches, dc_params, rng)
        ranked  = rank_group(records, rank_lookup)

        for place, t in enumerate(ranked, 1):
            group_position[t] = place

        group_winners[group] = ranked[0]
        group_runners[group] = ranked[1]

        t3 = ranked[2]
        r3 = records[t3]
        third_place_list.append((t3, group, r3["pts"], r3["gd"], r3["gf"]))

        for t in ranked:
            results[t] = 1   # reached group stage (all teams)

        # Mark qualifiers
        results[ranked[0]] = 2  # at minimum, into R32
        results[ranked[1]] = 2

    # Sort third-place teams by pts DESC, gd DESC, gf DESC, rank ASC
    third_place_list.sort(key=lambda x: (
        -x[2], -x[3], -x[4], rank_lookup.get(x[0], 80)
    ))

    third_assignments = assign_third_place_teams(third_place_list, rank_lookup)

    # Mark 8 best third-place as R32 qualifiers
    for slot_name, team in third_assignments.items():
        results[team] = 2

    # ── Build slot → team lookup ───────────────────────────────────────────────
    slot_to_team = {}
    for g, t in group_winners.items():
        slot_to_team[f"W_{g}"] = t
    for g, t in group_runners.items():
        slot_to_team[f"R_{g}"] = t
    for slot_name, team in third_assignments.items():
        slot_to_team[slot_name] = team

    # ── Round of 32 ───────────────────────────────────────────────────────────
    match_winners = {}   # match_id → winner

    for match_id, (home_slot, away_slot) in R32_BRACKET.items():
        home = slot_to_team.get(home_slot)
        away = slot_to_team.get(away_slot)
        if home is None or away is None:
            continue
        winner = simulate_knockout_match(home, away, prob_cache, rng)
        match_winners[match_id] = winner
        results[winner] = max(results.get(winner, 0), 3)

    # ── Round of 16 ───────────────────────────────────────────────────────────
    for match_id, (r32_a, r32_b) in R16_BRACKET.items():
        home = match_winners.get(r32_a)
        away = match_winners.get(r32_b)
        if home is None or away is None:
            continue
        winner = simulate_knockout_match(home, away, prob_cache, rng)
        match_winners[match_id] = winner
        results[winner] = max(results.get(winner, 0), 4)

    # ── Quarterfinals ─────────────────────────────────────────────────────────
    for match_id, (r16_a, r16_b) in QF_BRACKET.items():
        home = match_winners.get(r16_a)
        away = match_winners.get(r16_b)
        if home is None or away is None:
            continue
        winner = simulate_knockout_match(home, away, prob_cache, rng)
        match_winners[match_id] = winner
        results[winner] = max(results.get(winner, 0), 5)

    # ── Semifinals ────────────────────────────────────────────────────────────
    sf_losers = []
    for match_id, (qf_a, qf_b) in SF_BRACKET.items():
        home = match_winners.get(qf_a)
        away = match_winners.get(qf_b)
        if home is None or away is None:
            continue
        winner = simulate_knockout_match(home, away, prob_cache, rng)
        loser  = away if winner == home else home
        match_winners[match_id] = winner
        results[winner] = max(results.get(winner, 0), 6)
        sf_losers.append(loser)

    # ── Final ─────────────────────────────────────────────────────────────────
    finalist_a = match_winners.get(101)
    finalist_b = match_winners.get(102)
    if finalist_a and finalist_b:
        champion = simulate_knockout_match(finalist_a, finalist_b, prob_cache, rng)
        results[champion] = 7

    return results, group_position


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("═" * 60)
    print(f"  Phase 7: Monte Carlo Bracket Simulator ({N_SIMULATIONS:,} runs)")
    print("═" * 60)

    # Load
    fixtures, rankings, all_data = load_all()
    rank_lookup = build_rank_lookup(rankings)
    form_lookup = build_form_lookup(all_data)
    h2h_lookup  = build_h2h_lookup(all_data)
    defaults    = compute_defaults(all_data)

    print("\n── Loading models ──")
    xgb_model, lgb_booster, dc_params = load_models()

    # Group stage fixtures
    group_df = fixtures[fixtures["stage"] == "Group Stage"].copy()
    group_df["home_team"] = group_df["home_team"].map(normalize)
    group_df["away_team"] = group_df["away_team"].map(normalize)

    wc_teams = sorted(
        set(group_df["home_team"]) | set(group_df["away_team"])
    )
    print(f"\n── WC2026 Teams: {len(wc_teams)} ──")

    # Build group fixture dict: group → [(home, away), ...]
    group_fixtures = {}
    for _, row in group_df.iterrows():
        g = row["group"]
        group_fixtures.setdefault(g, []).append((row["home_team"], row["away_team"]))

    # Pre-compute pairwise knockout probs
    print("\n── Pre-computing pairwise probabilities ──")
    prob_cache = precompute_pairwise(
        wc_teams, rank_lookup, form_lookup, h2h_lookup, defaults,
        xgb_model, lgb_booster, dc_params
    )
    print(f"  Cached {len(prob_cache)} matchups")

    # Run simulations
    print(f"\n── Running {N_SIMULATIONS:,} simulations ──")
    rng = np.random.default_rng(42)

    # Track: team → count of times reaching each round, and group finish place
    ROUNDS = {1: "Group", 2: "R32", 3: "R16", 4: "QF", 5: "SF", 6: "Final", 7: "Winner"}
    counts = {team: {r: 0 for r in ROUNDS} for team in wc_teams}
    pos_counts = {team: {1: 0, 2: 0, 3: 0, 4: 0} for team in wc_teams}
    team_group = {}  # team → group letter (for the standings table)
    for g, ms in group_fixtures.items():
        for h, a in ms:
            team_group[h] = g
            team_group[a] = g

    for sim in range(N_SIMULATIONS):
        if sim % 1000 == 0:
            print(f"  Simulation {sim:,}/{N_SIMULATIONS:,}...")

        sim_results, sim_pos = simulate_tournament(
            group_fixtures, dc_params, prob_cache, rank_lookup, rng
        )

        for team in wc_teams:
            reached = sim_results.get(team, 1)
            for r in ROUNDS:
                if reached >= r:
                    counts[team][r] += 1
            place = sim_pos.get(team)
            if place in pos_counts[team]:
                pos_counts[team][place] += 1

    # Build advancement DataFrame
    rows = []
    for team in wc_teams:
        c = counts[team]
        rows.append({
            "team":           team,
            "fifa_rank":      rank_lookup.get(team, 80),
            "p_group_adv":    round(c[2] / N_SIMULATIONS, 4),
            "p_r16":          round(c[3] / N_SIMULATIONS, 4),
            "p_quarterfinal": round(c[4] / N_SIMULATIONS, 4),
            "p_semifinal":    round(c[5] / N_SIMULATIONS, 4),
            "p_final":        round(c[6] / N_SIMULATIONS, 4),
            "p_winner":       round(c[7] / N_SIMULATIONS, 4),
        })

    out_df = pd.DataFrame(rows).sort_values("p_winner", ascending=False)

    # Save
    out_path = DATA_PROC / "tournament_probs.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n✅ Saved: {out_path}")

    # Build group-standings DataFrame (P of finishing 1st / 2nd / 3rd, + advance)
    g_rows = []
    for team in wc_teams:
        pc = pos_counts[team]
        g_rows.append({
            "group":        team_group.get(team, "?"),
            "team":         team,
            "fifa_rank":    rank_lookup.get(team, 80),
            "p_win_group":  round(pc[1] / N_SIMULATIONS, 4),
            "p_2nd":        round(pc[2] / N_SIMULATIONS, 4),
            "p_3rd":        round(pc[3] / N_SIMULATIONS, 4),
            "p_advance":    round(counts[team][2] / N_SIMULATIONS, 4),
        })
    g_df = pd.DataFrame(g_rows).sort_values(["group", "p_win_group"], ascending=[True, False])
    g_path = DATA_PROC / "group_standings.csv"
    g_df.to_csv(g_path, index=False)
    print(f"✅ Saved: {g_path}")

    # Print top 20
    print(f"\n── Top 20 Tournament Win Probabilities ──")
    print(f"  {'Team':<25} {'Rank':>5} {'Advance':>8} {'R16':>6} {'QF':>6} {'SF':>6} {'Final':>7} {'Win':>7}")
    print(f"  {'-'*75}")
    for _, r in out_df.head(20).iterrows():
        print(f"  {r['team']:<25} {int(r['fifa_rank']):>5} "
              f"{r['p_group_adv']:>8.1%} {r['p_r16']:>6.1%} "
              f"{r['p_quarterfinal']:>6.1%} {r['p_semifinal']:>6.1%} "
              f"{r['p_final']:>7.1%} {r['p_winner']:>7.1%}")

    # Sanity check: win probs should sum to ~100%
    total_win = out_df["p_winner"].sum()
    print(f"\n  Win probability sum: {total_win:.4f} (expect ~1.0)")

    print(f"\nPhase 7 complete ✅")
    print(f"Next: Phase 8 — Visualization")


if __name__ == "__main__":
    main()
