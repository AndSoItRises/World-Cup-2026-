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
DC_PARAMS_PATH = MODELS_DIR / "dixon_coles_params.json"
XGB_MODEL_PATH = MODELS_DIR / "xgb_v2.json"
LGB_MODEL_PATH = MODELS_DIR / "lgbm_v2.txt"

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

    blended = WEIGHTS["xgb"] * xgb_p + WEIGHTS["lgb"] * lgb_p + WEIGHTS["dc"] * dc_p
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

def assign_third_place_teams(third_place_teams, rank_lookup):
    """
    third_place_teams: list of (team, group, pts, gd, gf) sorted best-first.
    Select best 8 and assign to the 8 bracket slots.
    Returns dict: slot_name → team
    """
    best8 = third_place_teams[:8]

    slot_assignments = {}
    used_teams = set()

    # Try to assign each team to a slot where their group is eligible
    for slot_name, eligible_groups in THIRD_PLACE_SLOTS:
        # Find the best unassigned team from an eligible group
        for team, group, pts, gd, gf in best8:
            if team not in used_teams and group in eligible_groups:
                slot_assignments[slot_name] = team
                used_teams.add(team)
                break
        else:
            # Fallback: best unassigned team
            for team, group, pts, gd, gf in best8:
                if team not in used_teams:
                    slot_assignments[slot_name] = team
                    used_teams.add(team)
                    break

    return slot_assignments


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
    Returns dict: team → highest round reached (int)
    Rounds: 1=group, 2=R32, 3=R16, 4=QF, 5=SF, 6=Final, 7=Winner
    """
    results = {}   # team → round reached

    # ── Group stage ───────────────────────────────────────────────────────────
    group_winners  = {}   # group letter → winner
    group_runners  = {}   # group letter → runner-up
    third_place_list = [] # (team, group, pts, gd, gf)

    for group, matches in group_fixtures.items():
        records = simulate_group(matches, dc_params, rng)
        ranked  = rank_group(records, rank_lookup)

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

    return results


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

    # Track: team → count of times reaching each round
    ROUNDS = {1: "Group", 2: "R32", 3: "R16", 4: "QF", 5: "SF", 6: "Final", 7: "Winner"}
    counts = {team: {r: 0 for r in ROUNDS} for team in wc_teams}

    for sim in range(N_SIMULATIONS):
        if sim % 1000 == 0:
            print(f"  Simulation {sim:,}/{N_SIMULATIONS:,}...")

        sim_results = simulate_tournament(
            group_fixtures, dc_params, prob_cache, rank_lookup, rng
        )

        for team in wc_teams:
            reached = sim_results.get(team, 1)
            for r in ROUNDS:
                if reached >= r:
                    counts[team][r] += 1

    # Build output DataFrame
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
