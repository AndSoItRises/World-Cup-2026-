"""
Live Tournament Update Pipeline — WC2026.

After each matchday, ingest actual results, update ELO forward from the
pre-tournament baseline, lock known results, and re-run the 10k Monte Carlo on
the remaining bracket. Outputs fresh win probabilities reflecting what's happened.

Live-updated feature: ELO only (highest-signal, only one that meaningfully moves
within a 3-game group stage). Rolling form + H2H stay at pre-tournament values.

Input  (Jake maintains): data/raw/wc2026_live_results.csv
Outputs: data/processed/tournament_probs_live.csv  (+ timestamped archive)

Baseline note: the spec named tournament_probs_v2.csv, but the live forecast uses
the production models (retrain_all.py). The correct "no results yet" baseline is
therefore the current pre-tournament prod forecast (tournament_probs.csv); v2 is
never touched. With an empty live file this script reproduces that baseline.

Run with:
  python -m src.models.live_update
"""

import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime
from pathlib import Path

from src.models.monte_carlo import (
    normalize, load_all, build_rank_lookup, build_form_lookup, build_h2h_lookup,
    compute_defaults, load_models, get_matchup_prob, get_expected_goals,
    rank_group, assign_third_place_teams, simulate_knockout_match,
    THIRD_PLACE_SLOTS, R32_BRACKET, R16_BRACKET, QF_BRACKET, SF_BRACKET,
    N_SIMULATIONS,
)
from src.features.elo import update_elo

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_RAW = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"

LIVE_PATH     = DATA_RAW  / "wc2026_live_results.csv"
BASELINE_PATH = DATA_PROC / "tournament_probs.csv"          # pre-tournament prod forecast
LIVE_OUT      = DATA_PROC / "tournament_probs_live.csv"

WC_K = 40.0   # World Cup = high K-factor


# ── Step 1: load live results ─────────────────────────────────────────────────
def load_live():
    if not LIVE_PATH.exists():
        print(f"  No live results file — creating header-only {LIVE_PATH.name}")
        cols = "match_id,stage,group,date,home_team,away_team,home_goals,away_goals,decided_by,winner"
        LIVE_PATH.write_text(cols + "\n")
        return pd.DataFrame(columns=cols.split(","))
    live = pd.read_csv(LIVE_PATH)
    if len(live) == 0:
        return live
    live["date"] = pd.to_datetime(live["date"], errors="coerce")
    live["home_team"] = live["home_team"].map(normalize)
    live["away_team"] = live["away_team"].map(normalize)
    if "decided_by" not in live.columns:
        live["decided_by"] = "90min"
    live["decided_by"] = live["decided_by"].fillna("90min")
    return live


def build_known(live):
    group_results = live[live["stage"] == "Group Stage"]
    knockout_results = live[live["stage"] != "Group Stage"]

    known_group_scores = {
        (r["home_team"], r["away_team"]): (int(r["home_goals"]), int(r["away_goals"]))
        for _, r in group_results.iterrows()
    }

    known_knockout_winners = {}
    for _, r in knockout_results.iterrows():
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        if hg > ag:
            winner = r["home_team"]
        elif ag > hg:
            winner = r["away_team"]
        else:
            winner = normalize(r["winner"]) if isinstance(r.get("winner"), str) else None
        if winner:
            known_knockout_winners[int(r["match_id"])] = winner

    return known_group_scores, known_knockout_winners


# ── Step 2: update ELO with actual results ────────────────────────────────────
def update_live_elo(live, form_lookup):
    """Seed current ELO from the pre-tournament form_lookup (same source as the
    sim) and walk confirmed matches chronologically applying update_elo()."""
    current_elo = {t: form_lookup[t].get("elo", 1500.0) for t in form_lookup}
    if len(live) == 0:
        return current_elo

    for _, m in live.sort_values("date").iterrows():
        home, away = m["home_team"], m["away_team"]
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        # ELO uses the 90-min score; AET/PKs level → it was a 90-min draw
        nh, na = update_elo(
            current_elo.get(home, 1500.0), current_elo.get(away, 1500.0),
            hg, ag, k=WC_K, home_advantage=0.0,
        )
        current_elo[home] = nh
        current_elo[away] = na
    return current_elo


# ── Step 4: group simulation with known scores ────────────────────────────────
def simulate_group_live(matches, dc_params, rng, known_scores):
    records = {team: {"pts": 0, "gd": 0, "gf": 0} for pair in matches for team in pair}
    for home, away in matches:
        if (home, away) in known_scores:
            hg, ag = known_scores[(home, away)]
        elif (away, home) in known_scores:
            ag, hg = known_scores[(away, home)]
        else:
            lam, mu = get_expected_goals(home, away, dc_params)
            hg, ag = rng.poisson(lam), rng.poisson(mu)
        records[home]["gf"] += hg; records[away]["gf"] += ag
        records[home]["gd"] += hg - ag; records[away]["gd"] += ag - hg
        if hg > ag:
            records[home]["pts"] += 3
        elif hg == ag:
            records[home]["pts"] += 1; records[away]["pts"] += 1
        else:
            records[away]["pts"] += 3
    return records


def simulate_knockout_match_live(home, away, match_id, known_winners, prob_cache, rng):
    if match_id in known_winners:
        return known_winners[match_id]
    return simulate_knockout_match(home, away, prob_cache, rng)


# ── Step 5: tournament simulation locking known results ───────────────────────
def simulate_tournament_live(group_fixtures, dc_params, prob_cache, rank_lookup, rng,
                             known_group_scores, known_knockout_winners):
    results = {}
    group_winners, group_runners, third_place_list = {}, {}, []

    for group, matches in group_fixtures.items():
        records = simulate_group_live(matches, dc_params, rng, known_group_scores)
        ranked = rank_group(records, rank_lookup)
        group_winners[group] = ranked[0]
        group_runners[group] = ranked[1]
        t3 = ranked[2]; r3 = records[t3]
        third_place_list.append((t3, group, r3["pts"], r3["gd"], r3["gf"]))
        for t in ranked:
            results[t] = 1
        results[ranked[0]] = 2
        results[ranked[1]] = 2

    third_place_list.sort(key=lambda x: (-x[2], -x[3], -x[4], rank_lookup.get(x[0], 80)))
    third_assignments = assign_third_place_teams(third_place_list, rank_lookup)
    for team in third_assignments.values():
        results[team] = 2

    slot_to_team = {}
    for g, t in group_winners.items():
        slot_to_team[f"W_{g}"] = t
    for g, t in group_runners.items():
        slot_to_team[f"R_{g}"] = t
    for slot, team in third_assignments.items():
        slot_to_team[slot] = team

    match_winners = {}
    for mid, (hs, as_) in R32_BRACKET.items():
        h, a = slot_to_team.get(hs), slot_to_team.get(as_)
        if h is None or a is None:
            continue
        w = simulate_knockout_match_live(h, a, mid, known_knockout_winners, prob_cache, rng)
        match_winners[mid] = w
        results[w] = max(results.get(w, 0), 3)

    for bracket, rnd in [(R16_BRACKET, 4), (QF_BRACKET, 5), (SF_BRACKET, 6)]:
        for mid, (a_id, b_id) in bracket.items():
            h, a = match_winners.get(a_id), match_winners.get(b_id)
            if h is None or a is None:
                continue
            w = simulate_knockout_match_live(h, a, mid, known_knockout_winners, prob_cache, rng)
            match_winners[mid] = w
            results[w] = max(results.get(w, 0), rnd)

    fa, fb = match_winners.get(101), match_winners.get(102)
    if fa and fb:
        champ = simulate_knockout_match_live(fa, fb, 103, known_knockout_winners, prob_cache, rng)
        results[champ] = 7

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("═" * 60)
    print("  WC2026 Live Tournament Update")
    print("═" * 60)

    fixtures, rankings, all_data = load_all()
    rank_lookup = build_rank_lookup(rankings)
    form_lookup = build_form_lookup(all_data)
    h2h_lookup  = build_h2h_lookup(all_data)
    defaults    = compute_defaults(all_data)
    xgb_model, lgb_booster, dc_params = load_models()

    live = load_live()
    print(f"\n  Live results loaded: {len(live)} matches")
    known_group_scores, known_knockout_winners = build_known(live)
    print(f"  Group results: {len(known_group_scores)} | knockout results: {len(known_knockout_winners)}")

    # Warn on unresolved teams (no silent NaN)
    for _, m in live.iterrows():
        for side in ("home_team", "away_team"):
            t = m[side]
            if t not in form_lookup:
                print(f"  ⚠️  '{t}' not in form_lookup — using defaults")
            if t not in rank_lookup:
                print(f"  ⚠️  '{t}' not in rank_lookup — using rank sentinel")

    # Update ELO, patch into form_lookup
    current_elo = update_live_elo(live, form_lookup)
    moved = 0
    for team in form_lookup:
        if team in current_elo and abs(current_elo[team] - form_lookup[team].get("elo", 1500.0)) > 0.01:
            moved += 1
        if team in current_elo:
            form_lookup[team]["elo"] = current_elo[team]
    print(f"  ELO updated for {moved} teams from live results")

    # Group fixtures + WC teams
    group_df = fixtures[fixtures["stage"] == "Group Stage"].copy()
    group_df["home_team"] = group_df["home_team"].map(normalize)
    group_df["away_team"] = group_df["away_team"].map(normalize)
    wc_teams = sorted(set(group_df["home_team"]) | set(group_df["away_team"]))
    group_fixtures = {}
    for _, row in group_df.iterrows():
        group_fixtures.setdefault(row["group"], []).append((row["home_team"], row["away_team"]))

    # Pairwise knockout probs with LIVE ELO
    print("\n  Pre-computing pairwise probabilities (live ELO)...")
    prob_cache = {}
    for home in wc_teams:
        for away in wc_teams:
            if home != away:
                prob_cache[(home, away)] = get_matchup_prob(
                    home, away, True, rank_lookup, form_lookup, h2h_lookup, defaults,
                    xgb_model, lgb_booster, dc_params)

    # Run sims
    print(f"  Running {N_SIMULATIONS:,} simulations...")
    rng = np.random.default_rng(42)
    ROUNDS = {1: "Group", 2: "R32", 3: "R16", 4: "QF", 5: "SF", 6: "Final", 7: "Winner"}
    counts = {t: {r: 0 for r in ROUNDS} for t in wc_teams}
    for sim in range(N_SIMULATIONS):
        sr = simulate_tournament_live(group_fixtures, dc_params, prob_cache, rank_lookup, rng,
                                      known_group_scores, known_knockout_winners)
        for t in wc_teams:
            reached = sr.get(t, 1)
            for r in ROUNDS:
                if reached >= r:
                    counts[t][r] += 1

    rows = []
    for t in wc_teams:
        c = counts[t]
        rows.append({
            "team": t, "fifa_rank": rank_lookup.get(t, 80),
            "p_group_adv": round(c[2] / N_SIMULATIONS, 4),
            "p_r16": round(c[3] / N_SIMULATIONS, 4),
            "p_quarterfinal": round(c[4] / N_SIMULATIONS, 4),
            "p_semifinal": round(c[5] / N_SIMULATIONS, 4),
            "p_final": round(c[6] / N_SIMULATIONS, 4),
            "p_winner": round(c[7] / N_SIMULATIONS, 4),
            "eliminated": bool(c[2] == 0),
        })
    out_df = pd.DataFrame(rows).sort_values("p_winner", ascending=False)

    out_df.to_csv(LIVE_OUT, index=False)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    archive = DATA_PROC / f"tournament_probs_live_{stamp}.csv"
    out_df.to_csv(archive, index=False)
    print(f"\n✅ Saved: {LIVE_OUT}")
    print(f"✅ Saved: {archive}")

    # Comparison vs pre-tournament baseline
    if BASELINE_PATH.exists():
        base = pd.read_csv(BASELINE_PATH).set_index("team")["p_winner"]
        cmp = out_df.set_index("team")
        cmp["pre"] = cmp.index.map(base).fillna(0.0)
        cmp["delta"] = cmp["p_winner"] - cmp["pre"]
        cmp = cmp.reindex(cmp["delta"].abs().sort_values(ascending=False).index)
        print(f"\n── Biggest movers vs pre-tournament ──")
        print(f"  {'Team':<22}{'Pre':>8}{'Now':>8}{'Delta':>9}")
        print(f"  {'-'*46}")
        for team, r in cmp.head(15).iterrows():
            print(f"  {team:<22}{r['pre']*100:>7.1f}%{r['p_winner']*100:>7.1f}%{r['delta']*100:>+8.1f}%")
        if len(live) == 0:
            maxd = cmp["delta"].abs().max() * 100
            print(f"\n  [validation] empty live file → max drift vs baseline {maxd:.2f}%"
                  f" ({'PASS <1%' if maxd < 1.0 else 'check'} — stochastic variance only)")


if __name__ == "__main__":
    main()
