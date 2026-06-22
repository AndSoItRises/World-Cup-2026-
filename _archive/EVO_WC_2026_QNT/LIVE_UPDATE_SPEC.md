# Spec: `src/models/live_update.py`
### Live Tournament Update Pipeline — WC2026

**Purpose:** After each WC2026 matchday, ingest actual results, update ELO ratings, and re-run the
Monte Carlo bracket simulator locking in known results. Outputs fresh tournament win probabilities
that reflect what has already happened.

---

## Context (read before touching any code)

The existing `monte_carlo.py` simulates the entire bracket stochastically from scratch — it has no
concept of results that have already happened. This script wraps and extends that logic to:
1. Accept known match results as ground truth
2. Update ELO ratings forward from the pre-tournament baseline using those results
3. Lock known results during simulation (don't re-simulate what already happened)
4. Re-run 10k Monte Carlo sims on the remaining bracket

**What NOT to update live:** Rolling form (win_rate_5/10, avg_goals, etc.) and H2H lookups are
left as pre-tournament values. ELO is the only live-updated feature — it's the highest-signal
feature and the only one that meaningfully changes within a 3-game group stage.

---

## New Files This Script Creates

### Input (Jake manually maintains this):
```
data/raw/wc2026_live_results.csv
```
Columns: `match_id, stage, group, date, home_team, away_team, home_goals, away_goals`

- `stage`: one of `"Group Stage"`, `"Round of 32"`, `"Round of 16"`, `"Quarterfinal"`, `"Semifinal"`, `"Final"`
- `group`: group letter (A–L) for group stage matches; empty string for knockout
- `home_team` / `away_team`: use the model's standardized names (same as `wc2026_fixtures.csv`)
- `home_goals` / `away_goals`: integer goals scored in 90 min (no extra time — for ELO purposes, treat
  AET/PKs as a draw at 90 min if the score was level; add a `decided_by` column: `"90min"`, `"AET"`, `"PKs"`)
- Jake adds rows to this file after each real match is played

### Outputs:
```
data/processed/tournament_probs_live.csv      ← always overwritten (latest run)
data/processed/tournament_probs_live_{YYYYMMDD_HHMM}.csv  ← timestamped archive
```
Same columns as `tournament_probs_v2.csv`: `team, fifa_rank, p_group_adv, p_r16, p_quarterfinal,
p_semifinal, p_final, p_winner`. Add one extra column: `eliminated` (bool — True if team is
mathematically eliminated from advancing).

Also print a comparison table: pre-tournament prob vs current prob for each team (sorted by biggest
mover), so Jake can see what's changed.

---

## Architecture

### Step 1 — Load live results
```python
live = pd.read_csv("data/raw/wc2026_live_results.csv", parse_dates=["date"])
```
Apply the same `normalize()` name map from `monte_carlo.py` to `home_team` and `away_team`.

Split into:
- `group_results`: rows where `stage == "Group Stage"`
- `knockout_results`: all other rows

Build two lookups:
```python
# For group simulation: {(home, away): (home_goals, away_goals)}
known_group_scores = {
    (row.home_team, row.away_team): (row.home_goals, row.away_goals)
    for _, row in group_results.iterrows()
}

# For ELO update (all confirmed matches): list of dicts with home, away, hg, ag, decided_by
confirmed_matches = live.to_dict("records")

# For knockout bracket: {match_id: winner_team}
known_knockout_winners = {}
for _, row in knockout_results.iterrows():
    hg, ag = row.home_goals, row.away_goals
    # Use 90-min score; if decided_by == "PKs" or "AET" and scores level, coin-flip is irrelevant
    # because we know the actual winner — derive it:
    if hg > ag:
        winner = row.home_team
    elif ag > hg:
        winner = row.away_team
    else:
        # AET/PKs — must have a decided_by winner; require a `winner` column for these rows
        winner = row.get("winner", None)  # Jake fills this manually for AET/PK matches
    if winner:
        known_knockout_winners[int(row.match_id)] = winner
```

### Step 2 — Update ELO with actual results

Load the pre-tournament ELO baseline. The end-state ELO for each team is stored in
`data/processed/elo_ratings.csv` — take each team's LAST row as its pre-tournament rating.

```python
elo_df = pd.read_csv("data/processed/elo_ratings.csv", parse_dates=["date"])
# Get latest pre-tournament ELO per team
current_elo = {}
for team, grp in elo_df.groupby("home_team"):
    current_elo[team] = grp.sort_values("date").iloc[-1]["home_elo"]
for team, grp in elo_df.groupby("away_team"):
    latest = grp.sort_values("date").iloc[-1]
    if team not in current_elo or latest["date"] > ...:
        current_elo[team] = latest["away_elo"]
```

Then walk confirmed matches chronologically and apply `update_elo()` from `src.features.elo`:
```python
from src.features.elo import update_elo, get_k_factor

for match in sorted(confirmed_matches, key=lambda x: x["date"]):
    home, away = match["home_team"], match["away_team"]
    hg, ag = match["home_goals"], match["away_goals"]

    # For ELO, use 90-min score only. If decided_by is AET/PKs and score was level,
    # treat as draw (0.5 result) — that's what happened in 90 min.
    decided_by = match.get("decided_by", "90min")
    if decided_by in ("AET", "PKs") and hg == ag:
        elo_hg, elo_ag = hg, ag  # treat as draw
    else:
        elo_hg, elo_ag = hg, ag

    k = 40.0  # World Cup = high K
    home_advantage = 0.0  # all WC matches neutral

    new_home, new_away = update_elo(
        current_elo.get(home, 1500.0),
        current_elo.get(away, 1500.0),
        elo_hg, elo_ag, k=k, home_advantage=home_advantage
    )
    current_elo[home] = new_home
    current_elo[away] = new_away
```

After this loop, `current_elo` reflects real WC2026 match outcomes.

### Step 3 — Rebuild form_lookup with updated ELO

Copy `build_form_lookup()` from `monte_carlo.py` as-is, then patch ELO values:
```python
form_lookup = build_form_lookup(all_data)  # pre-tournament rolling form (unchanged)
for team in form_lookup:
    if team in current_elo:
        form_lookup[team]["elo"] = current_elo[team]  # overwrite with live ELO
# Also patch teams not in form_lookup but in current_elo (rare edge case)
```

### Step 4 — Modified group simulation

Replace `simulate_group()` with a version that accepts `known_scores`:

```python
def simulate_group_live(matches, dc_params, rng, known_scores):
    """
    matches: list of (home, away) tuples for the full group schedule
    known_scores: dict {(home, away): (home_goals, away_goals)} for played matches
    Unplayed matches are Poisson-simulated as before.
    """
    records = {team: {"pts": 0, "gd": 0, "gf": 0}
               for pair in matches for team in pair}

    for home, away in matches:
        if (home, away) in known_scores:
            hg, ag = known_scores[(home, away)]
        elif (away, home) in known_scores:
            ag, hg = known_scores[(away, home)]  # reversed fixture
        else:
            lam, mu = get_expected_goals(home, away, dc_params)
            hg = rng.poisson(lam)
            ag = rng.poisson(mu)

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
```

### Step 5 — Modified tournament simulation

Replace `simulate_tournament()` with `simulate_tournament_live()`:

Same as the original but:
1. Pass `known_group_scores` into `simulate_group_live()` instead of `simulate_group()`
2. For knockout rounds: before simulating a match, check `known_knockout_winners`:
```python
def simulate_knockout_match_live(home, away, match_id, known_knockout_winners, prob_cache, rng):
    if match_id in known_knockout_winners:
        return known_knockout_winners[match_id]  # lock actual result
    return simulate_knockout_match(home, away, prob_cache, rng)  # simulate as before
```
3. For the bracket slot resolution in R32 / R16 / QF / SF:
   - Groups that are fully played have deterministic winners/runners-up (no simulation needed for
     those slots — resolve them directly from `known_group_scores` standings before the sim loop)
   - Groups that are partially played: simulate the remaining matches, which will vary per sim

### Step 6 — Run simulations and output

Same N_SIMULATIONS = 10,000, same RNG seeding logic, same output format.

Add `eliminated` column: a team is eliminated if it cannot mathematically advance from its group
(0 pts after 3 games, or knocked out). For simplicity, mark as `True` if `p_group_adv == 0.0`
across all 10k sims.

Print comparison vs `tournament_probs_v2.csv` (the pre-tournament baseline):
```
Team                   Pre-tournament    Current    Delta
France                      4.55%         8.2%     +3.65%
...
```
Sort by abs(Delta) descending — biggest movers at top.

---

## Run command
```
python -m src.models.live_update
```

---

## Non-negotiables (from CONTEXT_V4.md)

- Apply the `normalize()` name map to all live result team names before any lookup
- If a team from live results doesn't resolve in `form_lookup` or `rank_lookup`, print a warning
  and fall back to defaults (do not silently NaN)
- Do not modify any existing model files (`xgb_v3.json`, `lgbm_v3.txt`, `dixon_coles_params_v3.json`)
- Do not modify `tournament_probs_v2.csv` — it's the pre-tournament baseline; never overwrite it
- Save to `tournament_probs_live.csv` (always) + timestamped archive copy

---

## Starter data file

Create `data/raw/wc2026_live_results.csv` with headers only (no data yet):
```
match_id,stage,group,date,home_team,away_team,home_goals,away_goals,decided_by,winner
```
Jake populates this manually after each match. The `winner` column is only needed for AET/PK matches
where `home_goals == away_goals` at full time. `decided_by` defaults to `"90min"` if left blank.

---

## Validation check to run after implementation

```python
# Sanity: if no live results exist yet (empty CSV), output should match tournament_probs_v2.csv
# within ~1% on all teams (stochastic variance only). Run with empty live_results and diff.
```
