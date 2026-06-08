"""
Phase 4: Feature Engineering
Builds enriched feature sets from train.csv and test.csv.

Outputs:
  data/processed/train_features.csv
  data/processed/test_features.csv

Run with:
  python -m src.features.feature_engineering
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

from src.features.data_cleaning import standardize_name
from src.features.build_squad_strength import SQUAD_SENTINEL

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
DATA_RAW = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"

TRAIN_PATH = DATA_PROC / "train.csv"
TEST_PATH  = DATA_PROC / "test.csv"
RANKINGS_DIR = DATA_RAW / "FIFA_rankings_training"
CURRENT_RANKINGS = DATA_RAW / "current_fifa_rankings.csv"
FIXTURES_PATH = DATA_RAW / "wc2026_fixtures.csv"
ELO_PATH      = DATA_PROC / "elo_ratings.csv"


# ── 1. Load data ─────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["date"])
    print(f"Loaded train: {len(train):,} rows | test: {len(test):,} rows")
    return train, test


# ── 2. Build unified FIFA rankings table ─────────────────────────────────────
def load_rankings():
    """
    Combines all historical ranking CSVs into one table sorted by date.
    Columns needed: rank_date (datetime), team (str), rank (int)
    """
    frames = []
    for csv_file in sorted(RANKINGS_DIR.glob("*.csv")):
        df = pd.read_csv(csv_file, parse_dates=["rank_date"])
        df = df[["rank_date", "country_full", "rank"]].rename(
            columns={"country_full": "team"}
        )
        frames.append(df)

    # Current rankings — treat as a single snapshot at today's date
    curr = pd.read_csv(CURRENT_RANKINGS)
    curr["rank_date"] = pd.Timestamp("2026-06-03")
    curr = curr.rename(columns={"team_name": "team"})[["rank_date", "team", "rank"]]
    frames.append(curr)

    rankings = pd.concat(frames, ignore_index=True)
    # V3 P1 (DL-01): standardize ranking team names to model names BEFORE merge.
    # Rankings data uses raw names ("IR Iran", "Czech Republic"); matches use
    # standardized names ("Iran", "Czechia"). Without this, the asof-merge misses
    # and ranks fall back to the 150 sentinel (Iran's bug — stored 150, actual ~22).
    rankings["team"] = rankings["team"].apply(standardize_name)
    rankings = rankings.sort_values("rank_date").reset_index(drop=True)
    print(f"Rankings table: {len(rankings):,} rows covering "
          f"{rankings['rank_date'].min().date()} → {rankings['rank_date'].max().date()}")
    return rankings

# Note (V3 P2): a team→confederation map is trivially rebuildable from the
# training ranking CSVs' `confederation` column if a future confederation-STRENGTH
# feature is tried. The conf_match_pct feature itself was cut (no signal — DL-02).


# ── 3. Merge FIFA ranking onto matches (no leakage via merge_asof) ────────────
def add_fifa_ranking_diff(matches: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, find the most recent ranking available BEFORE match date
    for both home and away team. Diff = home_rank - away_rank.
    Positive = home team is ranked worse (higher number).
    """
    matches = matches.copy()

    def get_rank_for_team(team_col: str, out_col: str) -> pd.Series:
        # Build a per-team lookup: for each unique team, asof-merge on date
        result = pd.Series(index=matches.index, dtype=float, name=out_col)
        for team, group in matches.groupby(team_col):
            team_rankings = rankings[rankings["team"] == team].copy()
            if team_rankings.empty:
                result.loc[group.index] = np.nan
                continue
            merged = pd.merge_asof(
                group[["date"]].sort_values("date").reset_index(),
                team_rankings[["rank_date", "rank"]].rename(columns={"rank_date": "date"}),
                on="date",
                direction="backward"
            ).set_index("index")
            result.loc[merged.index] = merged["rank"].values
        return result

    matches["home_fifa_rank"] = get_rank_for_team("home_team", "home_fifa_rank")
    matches["away_fifa_rank"] = get_rank_for_team("away_team", "away_fifa_rank")
    matches["fifa_rank_diff"] = matches["home_fifa_rank"] - matches["away_fifa_rank"]

    na_count = matches["fifa_rank_diff"].isna().sum()
    print(f"FIFA rank diff: {na_count:,} matches missing ({na_count/len(matches)*100:.1f}%)")

    # Impute unranked teams with rank 150 (conservative "unranked nation" value)
    UNRANKED = 150
    matches["home_fifa_rank"] = matches["home_fifa_rank"].fillna(UNRANKED)
    matches["away_fifa_rank"] = matches["away_fifa_rank"].fillna(UNRANKED)
    matches["fifa_rank_diff"] = matches["home_fifa_rank"] - matches["away_fifa_rank"]
    print(f"FIFA rank NaNs imputed with {UNRANKED} (unranked sentinel)")
    return matches


# ── Name map: training data names → results.csv / ELO names ─────────────────
ELO_NAME_MAP = {
    "USA":                        "United States",
    "Congo DR":                   "DR Congo",
    "Côte d'Ivoire":              "Ivory Coast",
    "Korea Republic":             "South Korea",
    "Korea DPR":                  "North Korea",
    "Bosnia-Herzegovina":         "Bosnia and Herzegovina",
    "Czechia":                    "Czech Republic",
    "Ireland":                    "Republic of Ireland",
    "Cape Verde Islands":         "Cape Verde",
    "Kyrgyz Republic":            "Kyrgyzstan",
    "Brunei Darussalam":          "Brunei",
    "St. Kitts and Nevis":        "Saint Kitts and Nevis",
    "St. Lucia":                  "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
}


# ── 4. ELO ratings ───────────────────────────────────────────────────────────
def add_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Merges pre-match ELO for home and away teams onto each match.
    Uses the ELO recorded at match time (already pre-match in elo_ratings.csv).
    Falls back to 1500 for any team not found.
    """
    matches = matches.copy()
    elo = pd.read_csv(ELO_PATH, parse_dates=["date"])

    # Translate training data team names to match results.csv naming
    lookup = matches.copy()
    lookup["home_team"] = lookup["home_team"].replace(ELO_NAME_MAP)
    lookup["away_team"] = lookup["away_team"].replace(ELO_NAME_MAP)

    # Merge home ELO
    home_elo = lookup.merge(
        elo[["date", "home_team", "away_team", "home_elo", "away_elo"]],
        on=["date", "home_team", "away_team"],
        how="left"
    )
    matches["home_elo"] = home_elo["home_elo"].fillna(1500).values
    matches["away_elo"] = home_elo["away_elo"].fillna(1500).values
    matches["elo_diff"] = matches["home_elo"] - matches["away_elo"]

    missing = home_elo["home_elo"].isna().sum()
    print(f"ELO features added | {missing:,} matches fell back to default 1500")
    return matches


# ── 4b. Squad strength + depth (V4) ──────────────────────────────────────────
SQUAD_PATH = DATA_PROC / "squad_strength_by_year.csv"
# "Not in FIFA" ≈ weaker than the weakest rated nation (1st pct strength ≈ 62).
# Same idea as the FIFA-rank-150 sentinel: a defensible low value, not a NaN.
SQUAD_COLS = ["squad_strength", "squad_top11", "squad_depth", "squad_n_quality"]
# SQUAD_SENTINEL imported from build_squad_strength (single source of truth).


def add_squad_strength(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pre-match national-team squad strength + depth (FIFA video-game ratings,
    built by build_squad_strength.py) onto each match.

    Leakage-safe: merge_asof BACKWARD on each edition's availability_date, so a
    match only ever sees the most recent FIFA edition released BEFORE it. Teams
    with no rated squad (minnows / thin FIFA coverage) get a low sentinel — the
    same convention as the FIFA-rank-150 fill.
    """
    matches = matches.copy()
    if not SQUAD_PATH.exists():
        print(f"  ⚠ {SQUAD_PATH.name} missing — run build_squad_strength; filling sentinel")
        for side in ["home", "away"]:
            for k in SQUAD_COLS:
                matches[f"{side}_{k}"] = SQUAD_SENTINEL[k]
        matches["squad_both_covered"] = 0
    else:
        sq = pd.read_csv(SQUAD_PATH, parse_dates=["availability_date"]).sort_values("availability_date")

        def lookup(team_col: str, side: str):
            out = pd.DataFrame(index=matches.index, columns=SQUAD_COLS, dtype=float)
            for team, group in matches.groupby(team_col):
                ts = sq[sq["team"] == team]
                if ts.empty:
                    continue
                merged = pd.merge_asof(
                    group[["date"]].sort_values("date").reset_index(),
                    ts[["availability_date"] + SQUAD_COLS].rename(columns={"availability_date": "date"}),
                    on="date", direction="backward",
                ).set_index("index")
                out.loc[merged.index, SQUAD_COLS] = merged[SQUAD_COLS].values
            for k in SQUAD_COLS:
                matches[f"{side}_{k}"] = out[k]

        lookup("home_team", "home")
        lookup("away_team", "away")

        # Coverage flag BEFORE imputation: a squad diff is only meaningful when
        # BOTH teams have real data. signal_test showed the orthogonal signal
        # lives entirely in the covered subset, so this flag lets the trees gate
        # on it instead of being polluted by sentinel rows.
        both_covered = (matches["home_squad_strength"].notna() &
                        matches["away_squad_strength"].notna())
        matches["squad_both_covered"] = both_covered.astype(int)
        for side in ["home", "away"]:
            for k in SQUAD_COLS:
                matches[f"{side}_{k}"] = matches[f"{side}_{k}"].fillna(SQUAD_SENTINEL[k])
        print(f"Squad strength/depth added (leakage-safe asof) | both-covered: {both_covered.mean()*100:.1f}%")

    # Diffs (home − away): positive ⇒ home has the stronger / deeper squad
    for k in SQUAD_COLS:
        matches[f"{k}_diff"] = matches[f"home_{k}"] - matches[f"away_{k}"]
    return matches


# ── 4c. Confederation / schedule strength (V5 candidate) ─────────────────────
def _build_conf_map() -> dict:
    """team → confederation, from the training ranking CSVs' `confederation` column."""
    frames = []
    for f in sorted(RANKINGS_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        if {"country_full", "confederation"}.issubset(df.columns):
            frames.append(df[["country_full", "confederation"]])
    allc = pd.concat(frames, ignore_index=True).dropna()
    allc["country_full"] = allc["country_full"].apply(standardize_name)
    return allc.groupby("country_full")["confederation"].agg(lambda s: s.mode().iloc[0]).to_dict()


def add_confederation_strength(matches: pd.DataFrame) -> pd.DataFrame:
    """
    V5 candidate: each team's CONFEDERATION average ELO (prior year) — a
    schedule-quality prior ("how strong is the region you farm results in?").
    Tests whether regional strength adds signal beyond a team's own ELO + squad
    value. Leakage-safe: uses the PRIOR-year confederation aggregate (an average
    over other teams, computed before the match year). Must run after add_elo.
    """
    matches = matches.copy()
    conf_map = _build_conf_map()
    yr = matches["date"].dt.year

    hv = pd.DataFrame({"team": matches["home_team"], "year": yr, "elo": matches["home_elo"]})
    av = pd.DataFrame({"team": matches["away_team"], "year": yr, "elo": matches["away_elo"]})
    long = pd.concat([hv, av], ignore_index=True)
    long["conf"] = long["team"].map(conf_map)
    long = long.dropna(subset=["conf"])
    conf_year = long.groupby(["conf", "year"])["elo"].mean().to_dict()
    global_mean = float(long["elo"].mean())

    def lookup(conf, year):
        return conf_year.get((conf, year - 1), conf_year.get((conf, year), global_mean))

    home_conf = matches["home_team"].map(conf_map)
    away_conf = matches["away_team"].map(conf_map)
    matches["home_conf_strength"] = [lookup(c, y) for c, y in zip(home_conf, yr)]
    matches["away_conf_strength"] = [lookup(c, y) for c, y in zip(away_conf, yr)]
    matches["conf_strength_diff"] = matches["home_conf_strength"] - matches["away_conf_strength"]
    print("Confederation-strength features added (prior-year conf avg ELO, leakage-safe)")
    return matches


# ── 6. Rolling form features ──────────────────────────────────────────────────
def add_rolling_form(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Per team, compute over their last 5 and 10 matches (before current):
      Simple:           win rate, avg goals, avg goal diff
      Quality-weighted: same three stats, weighted by opponent ELO / 1500
    Uses shift(1) to avoid leaking current match result.
    """
    matches = matches.copy()
    matches = matches.sort_values("date").reset_index(drop=True)

    # Build a flat per-team match history (each match appears twice: once as home, once as away)
    # Include opponent ELO for quality weighting
    home_view = matches[["date", "home_team", "home_score", "away_score", "result", "away_elo"]].copy()
    home_view.columns = ["date", "team", "goals_for", "goals_against", "result_raw", "opp_elo"]
    home_view["win"] = (home_view["result_raw"] == 2).astype(int)
    home_view["goal_diff"] = home_view["goals_for"] - home_view["goals_against"]

    away_view = matches[["date", "away_team", "away_score", "home_score", "result", "home_elo"]].copy()
    away_view.columns = ["date", "team", "goals_for", "goals_against", "result_raw", "opp_elo"]
    away_view["win"] = (away_view["result_raw"] == 0).astype(int)
    away_view["goal_diff"] = away_view["goals_for"] - away_view["goals_against"]

    team_history = pd.concat([home_view, away_view], ignore_index=True)
    team_history = team_history.sort_values(["team", "date"]).reset_index(drop=True)

    # Opponent quality weight: normalized around default ELO of 1500
    team_history["opp_weight"] = team_history["opp_elo"] / 1500.0

    def rolling_stats(df, window):
        grp = df.groupby("team")
        win_rate  = grp["win"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        avg_goals = grp["goals_for"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        avg_gd    = grp["goal_diff"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        return win_rate, avg_goals, avg_gd

    def weighted_rolling_stats(df, window):
        """Weighted mean using opponent ELO as weight."""
        results = {}
        for team, grp in df.groupby("team"):
            grp = grp.copy()
            for col in ["win", "goals_for", "goal_diff"]:
                vals    = grp[col].shift(1)
                weights = grp["opp_weight"].shift(1)
                def wmean(v, w, win=window):
                    out = []
                    for i in range(len(v)):
                        start = max(0, i - win)
                        v_win = v.iloc[start:i]
                        w_win = w.iloc[start:i]
                        w_win = w_win.fillna(1.0)
                        if w_win.sum() == 0 or len(v_win.dropna()) == 0:
                            out.append(np.nan)
                        else:
                            mask = v_win.notna()
                            out.append(
                                np.average(v_win[mask], weights=w_win[mask])
                            )
                    return pd.Series(out, index=v.index)
                results.setdefault(col, {})[team] = wmean(vals, weights)

        def combine(col_dict, index):
            s = pd.Series(index=index, dtype=float)
            for team, series in col_dict.items():
                s.update(series)
            return s

        idx = df.index
        return (
            combine(results["win"],        idx),
            combine(results["goals_for"],  idx),
            combine(results["goal_diff"],  idx),
        )

    for w in [5, 10]:
        wr, ag, agd = rolling_stats(team_history, w)
        team_history[f"win_rate_{w}"]  = wr
        team_history[f"avg_goals_{w}"] = ag
        team_history[f"avg_gd_{w}"]    = agd

        wwr, wag, wagd = weighted_rolling_stats(team_history, w)
        team_history[f"weighted_win_rate_{w}"]  = wwr
        team_history[f"weighted_avg_goals_{w}"] = wag
        team_history[f"weighted_avg_gd_{w}"]    = wagd

    # Deduplicate to one row per (team, date) — take last if same team played twice same day
    stat_cols = [
        "win_rate_5", "avg_goals_5", "avg_gd_5",
        "win_rate_10", "avg_goals_10", "avg_gd_10",
        "weighted_win_rate_5", "weighted_avg_goals_5", "weighted_avg_gd_5",
        "weighted_win_rate_10", "weighted_avg_goals_10", "weighted_avg_gd_10",
    ]
    team_stats = (
        team_history.groupby(["team", "date"])[stat_cols]
        .last()
        .reset_index()
    )

    # Merge back onto matches for home and away
    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        merged = matches[["date", team_col]].merge(
            team_stats,
            left_on=["date", team_col],
            right_on=["date", "team"],
            how="left"
        )
        for col in stat_cols:
            matches[f"{side}_{col}"] = merged[col].values

    print(f"Rolling form features added (simple + quality-weighted, windows: 5, 10)")
    return matches


# ── 5. Head-to-head record ────────────────────────────────────────────────────
def add_h2h(matches: pd.DataFrame) -> pd.DataFrame:
    """
    For each match: count of prior wins/draws/losses between home_team and away_team.
    Only uses matches that occurred BEFORE the current match date.
    """
    matches = matches.copy()
    matches = matches.sort_values("date").reset_index(drop=True)

    h2h_home_wins  = np.zeros(len(matches), dtype=float)
    h2h_draws      = np.zeros(len(matches), dtype=float)
    h2h_away_wins  = np.zeros(len(matches), dtype=float)
    h2h_total      = np.zeros(len(matches), dtype=float)

    # For efficiency: iterate once, maintain a dict of cumulative H2H counts
    # key = frozenset(team_a, team_b), value = {(a_wins_as_home_vs_b, draws, b_wins)}
    # Simpler: track per ordered pair (home, away) counts
    from collections import defaultdict
    record = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0})

    for i, row in matches.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        key = (h, a)
        rev = (a, h)

        h_wins  = record[key]["wins"]   + record[rev]["losses"]
        draws   = record[key]["draws"]  + record[rev]["draws"]
        h_loss  = record[key]["losses"] + record[rev]["wins"]
        total   = h_wins + draws + h_loss

        h2h_home_wins[i] = h_wins
        h2h_draws[i]     = draws
        h2h_away_wins[i] = h_loss
        h2h_total[i]     = total

        # Update record with current match result
        if row["result"] == 2:
            record[key]["wins"]   += 1
        elif row["result"] == 1:
            record[key]["draws"]  += 1
        else:
            record[key]["losses"] += 1

    matches["h2h_home_wins"]  = h2h_home_wins
    matches["h2h_draws"]      = h2h_draws
    matches["h2h_away_wins"]  = h2h_away_wins
    matches["h2h_total"]      = h2h_total
    matches["h2h_home_win_rate"] = np.where(
        matches["h2h_total"] > 0,
        matches["h2h_home_wins"] / matches["h2h_total"],
        0.5  # no history → neutral prior
    )

    print(f"H2H features added")
    return matches


# ── 6. Days rest ──────────────────────────────────────────────────────────────
def add_days_rest(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Days since each team's last competitive match before the current one.
    """
    matches = matches.copy()
    matches = matches.sort_values("date").reset_index(drop=True)

    all_dates = pd.concat([
        matches[["date", "home_team"]].rename(columns={"home_team": "team"}),
        matches[["date", "away_team"]].rename(columns={"away_team": "team"}),
    ]).sort_values(["team", "date"]).reset_index(drop=True)

    all_dates["prev_date"] = all_dates.groupby("team")["date"].shift(1)
    all_dates["days_rest"] = (all_dates["date"] - all_dates["prev_date"]).dt.days

    # Deduplicate (same team same day → take first occurrence)
    last_rest = all_dates.groupby(["team", "date"])["days_rest"].first().reset_index()

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        merged = matches[["date", team_col]].merge(
            last_rest,
            left_on=["date", team_col],
            right_on=["date", "team"],
            how="left"
        )
        matches[f"{side}_days_rest"] = merged["days_rest"].values

    # Fill NaN (first-ever appearance) with median
    for col in ["home_days_rest", "away_days_rest"]:
        median_val = matches[col].median()
        matches[col] = matches[col].fillna(median_val)

    print(f"Days rest features added")
    return matches


# ── 7. Tournament stage ───────────────────────────────────────────────────────
KNOCKOUT_KEYWORDS = ["final", "semifinal", "semi-final", "quarter", "round of",
                     "knockout", "elimination", "third place"]

def add_tournament_stage(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    lower = matches["tournament"].str.lower()
    matches["is_knockout"] = lower.apply(
        lambda t: int(any(kw in t for kw in KNOCKOUT_KEYWORDS))
    )
    print(f"Tournament stage: {matches['is_knockout'].sum():,} knockout matches detected")
    return matches


# ── 8. Altitude (WC2026 fixtures only) ───────────────────────────────────────
def add_altitude(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Altitude is only meaningful for WC2026 venue predictions.
    For training data, we don't have reliable venue altitude, so we default to 0.
    This column will be populated from fixtures in Phase 6 (simulator).
    """
    matches = matches.copy()
    matches["altitude_m"] = 0
    print("Altitude: defaulted to 0 for training data (populated from fixtures in Phase 6)")
    return matches


# ── Pipeline ──────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, rankings: pd.DataFrame, label: str) -> pd.DataFrame:
    print(f"\n── Building features for {label} ({len(df):,} rows) ──")
    df = add_fifa_ranking_diff(df, rankings)
    df = add_elo(df)
    df = add_squad_strength(df)
    df = add_rolling_form(df)
    df = add_h2h(df)
    df = add_days_rest(df)
    df = add_tournament_stage(df)
    df = add_altitude(df)
    return df


def main():
    train, test = load_data()
    rankings    = load_rankings()

    # Combine train+test so rolling/H2H features for test matches can use
    # all training history (test matches are later in time, so no leakage)
    combined = pd.concat([train, test], ignore_index=True).sort_values("date").reset_index(drop=True)
    print(f"\nCombined for feature building: {len(combined):,} rows")

    combined = add_fifa_ranking_diff(combined, rankings)
    combined = add_elo(combined)
    combined = add_squad_strength(combined)   # V4: squad strength + depth
    # V5 (DL-03): confederation-strength tested via signal_test and CUT — redundant
    # (incr IC ≈ 0/neg, ΔLL −0.0005). Squad value already captures "true quality vs
    # schedule-inflated form", subsuming the regional proxy. add_confederation_strength
    # retained for reference but not in the pipeline.
    combined = add_rolling_form(combined)
    # V3 P2: conf_match_pct tested here, CUT — no CV improvement (DL-02 result).
    # ~93% of all qualifying matches are intra-confederation, so the feature is
    # near-constant (~0.93) for every team and carries no discriminative signal.
    combined = add_h2h(combined)
    combined = add_days_rest(combined)
    combined = add_tournament_stage(combined)
    combined = add_altitude(combined)

    # Split back
    cutoff = pd.Timestamp("2022-11-20")
    train_feat = combined[combined["date"] <  cutoff].copy()
    test_feat  = combined[combined["date"] >= cutoff].copy()

    # Save
    train_out = DATA_PROC / "train_features.csv"
    test_out  = DATA_PROC / "test_features.csv"
    train_feat.to_csv(train_out, index=False)
    test_feat.to_csv(test_out,  index=False)

    print(f"\n✅ Saved train_features.csv: {len(train_feat):,} rows, {len(train_feat.columns)} cols")
    print(f"✅ Saved test_features.csv:  {len(test_feat):,} rows,  {len(test_feat.columns)} cols")

    # Feature summary
    feat_cols = [c for c in train_feat.columns if c not in
                 ["date","home_team","away_team","home_score","away_score",
                  "tournament","city","country","neutral","tournament_tier",
                  "result","sample_weight"]]
    print(f"\nNew feature columns ({len(feat_cols)}):")
    for c in feat_cols:
        na = train_feat[c].isna().sum()
        print(f"  {c:<30} NaN: {na:,} ({na/len(train_feat)*100:.1f}%)")

    # Save metadata
    meta_path = DATA_PROC / "cleaning_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["feature_columns"] = feat_cols
    meta["train_features_shape"] = list(train_feat.shape)
    meta["test_features_shape"]  = list(test_feat.shape)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n✅ Updated cleaning_metadata.json with feature column list")


if __name__ == "__main__":
    main()
