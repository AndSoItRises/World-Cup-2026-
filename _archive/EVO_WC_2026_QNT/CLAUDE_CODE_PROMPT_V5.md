# Claude Code Prompt — WC2026 V5 Final Iteration

Paste this entire prompt into Claude Code from the repo root (`World-Cup-2026-/`).

---

```
You are executing the final iteration (V5) of the WC2026 ML prediction model.
Read CONTEXT_V5_FINAL.md in full before doing anything. Then execute each step
below in order. Do not skip steps. Verify each step before proceeding.

---

## STEP 1 — Fix assign_third_place_teams() in src/models/monte_carlo.py

The current greedy implementation is wrong. Replace it with the official FIFA
495-combination table lookup.

### 1a. Extract the combo table

Run this Python snippet to extract the THIRD_PLACE_COMBOS dict from the
existing HTML (the source of truth, already verified correct):

    import re, json
    html = open("outputs/bracket_simulator.html").read()
    m = re.search(r"const COMBO_TABLE\s*=\s*(\{[\s\S]+?\n\s*\});", html)
    combo_js = m.group(1)
    # JS uses unquoted 8-char keys — add double quotes
    combo_js_quoted = re.sub(r'([A-L]{8}):', r'"\1":', combo_js)
    combos = json.loads(combo_js_quoted)
    print(f"Loaded {len(combos)} combos")
    # Emit as Python dict literal
    lines = ["THIRD_PLACE_COMBOS = {"]
    for k, v in sorted(combos.items()):
        lines.append(f'    "{k}": {v},')
    lines.append("}")
    print("\n".join(lines[:5]))  # preview

Verify the output has exactly 495 entries before proceeding.

### 1b. Edit src/models/monte_carlo.py

After the existing `THIRD_PLACE_SLOTS` list (around line 433), insert:

1. The full `THIRD_PLACE_COMBOS` dict (all 495 entries from step 1a)
2. The `COMBO_SLOT_ORDER` list:

    COMBO_SLOT_ORDER = [
        "3rd_CEFHI",   # i=0: opponent for W_A (M79)
        "3rd_EFGIJ",   # i=1: opponent for W_B (M85)
        "3rd_BEFIJ",   # i=2: opponent for W_D (M81)
        "3rd_ABCDF",   # i=3: opponent for W_E (M74)
        "3rd_AEHIJ",   # i=4: opponent for W_G (M82)
        "3rd_CDFGH",   # i=5: opponent for W_I (M77)
        "3rd_DEIJL",   # i=6: opponent for W_K (M87)
        "3rd_EHIJK",   # i=7: opponent for W_L (M80)
    ]

3. Replace the existing `assign_third_place_teams()` function with:

    def assign_third_place_teams(third_place_teams, rank_lookup):
        """
        Assign best 8 third-place teams to bracket slots using the official
        FIFA 495-combination table. Key = sorted 8-char group string.
        Falls back to greedy only if key not found (should never happen).
        """
        best8 = third_place_teams[:8]
        group_to_team = {group: team for team, group, pts, gd, gf in best8}
        groups_key = "".join(sorted(group_to_team.keys()))

        assignment = THIRD_PLACE_COMBOS.get(groups_key)
        if assignment is None:
            # Greedy fallback — should never trigger with valid 12-group data
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

### 1c. Verify the fix

Run a quick sanity check:

    python3 -c "
    from src.models.monte_carlo import assign_third_place_teams, THIRD_PLACE_COMBOS
    print(f'Combo table loaded: {len(THIRD_PLACE_COMBOS)} entries')
    # Simulate EFGHIJKL scenario
    fake = [('TeamE','E',7,5,6), ('TeamJ','J',6,3,4), ('TeamI','I',6,2,3),
            ('TeamF','F',5,1,2), ('TeamH','H',5,0,1), ('TeamG','G',4,-1,0),
            ('TeamL','L',4,-2,0), ('TeamK','K',3,-3,0)]
    result = assign_third_place_teams(fake, {})
    print('Slot assignments:', result)
    # Expected for EFGHIJKL: 3rd_CEFHI=TeamE, 3rd_EFGIJ=TeamJ, 3rd_BEFIJ=TeamI,
    #   3rd_ABCDF=TeamF, 3rd_AEHIJ=TeamH, 3rd_CDFGH=TeamG, 3rd_DEIJL=TeamL, 3rd_EHIJK=TeamK
    "

Confirm the output matches the expected comment above before proceeding.

---

## STEP 2 — Re-run Monte Carlo

    python -m src.models.monte_carlo

This overwrites data/processed/tournament_probs.csv and group_standings.csv.
Confirm the run completes without errors and prints final log loss / accuracy.
Note the new top-5 win probabilities for use in the README update.

---

## STEP 3 — Update bracket_simulator.html with corrected probabilities

Read the new tournament_probs.csv. Generate an updated TEAM_PROBS JavaScript
object. Find the existing TEAM_PROBS block in outputs/bracket_simulator.html
and replace it.

The TEAM_PROBS format is:
    const TEAM_PROBS = {
        "Spain": { win: 0.1854, sf: 0.2766, qf: 0.5372, r16: 0.7049, advance: 0.9262 },
        "England": { win: 0.0894, sf: 0.1448, qf: 0.3596, r16: 0.5818, advance: 0.796 },
        ...
    };

Columns to use from tournament_probs.csv:
    win     ← p_winner
    sf      ← p_semifinal
    qf      ← p_quarterfinal
    r16     ← p_r16
    advance ← p_group_adv

Include all teams in the CSV. Preserve the rest of the HTML file exactly.

---

## STEP 4 — Add 2 new charts and regenerate all probability-dependent charts

Edit src/visualization/charts.py to add two new functions:

### chart_group_winner_prob(df_groups)
- Data: group_standings.csv (columns: group, team, p_win_group, p_2nd, p_advance)
- 12-panel subplot (3 rows × 4 cols), one subplot per group (A–L)
- Each subplot: horizontal bar chart of p_win_group for teams in that group
- Bars sorted descending. Color: top team = GOLD, rest = ACCENT (#58a6ff)
- Value labels on bars (e.g., "64.1%")
- Group letter as subplot title
- Overall figure title: "Group Winner Probability — WC2026"
- figsize=(16, 12), dark theme (same rcParams as existing charts)
- Save to: outputs/viz/05_group_winner_prob.png

### chart_advancement_ladder(df)
- Data: tournament_probs.csv, top 24 teams by p_winner
- Stacked horizontal bar, sorted by p_winner descending
- For each team, show 6 incremental probability segments:
    seg1 = p_group_adv
    seg2 = p_r16 - p_group_adv         (negative? clamp to 0 — shouldn't happen)
    seg3 = p_quarterfinal - p_r16
    seg4 = p_semifinal - p_quarterfinal
    seg5 = p_final - p_semifinal
    seg6 = p_winner - p_final
  Wait — this isn't right. The probabilities are P(reached that round or further).
  Incremental segments should be:
    seg1 = p_group_adv - p_r16          (advanced group but eliminated R32)
    seg2 = p_r16 - p_quarterfinal       (reached R16 but eliminated)
    seg3 = p_quarterfinal - p_semifinal
    seg4 = p_semifinal - p_final
    seg5 = p_final - p_winner
    seg6 = p_winner
  The full bar width = p_group_adv for all teams.
- Use a 6-color sequential palette: very dark blue → light gold
  colors = ["#1c3a5e", "#1f6feb", "#388bfd", "#58a6ff", "#f0c040", "#ffd700"]
  labels = ["R32 exit", "R16 exit", "QF exit", "SF exit", "Final exit", "Winner"]
- Add legend
- Title: "Tournament Advancement Ladder — WC2026"
- figsize=(12, 14), dark theme
- Save to: outputs/viz/06_advancement_ladder.png

### Update main() in charts.py

In the main() function, add calls to the two new functions:
    df_groups = pd.read_csv(DATA / "group_standings.csv")
    chart_group_winner_prob(df_groups)
    chart_advancement_ladder(df)

Then run:
    python -m src.visualization.charts

Confirm 6 PNGs now exist in outputs/viz/.

### Copy updated outputs to key_insights/

    cp outputs/viz/01_win_probability.png outputs/key_insights/02_tournament_win_probs.png
    cp outputs/viz/02_stage_heatmap.png outputs/key_insights/03_stage_heatmap.png
    cp outputs/viz/03_advance_vs_win_scatter.png outputs/key_insights/08_advance_scatter.png
    cp outputs/viz/04_market_divergence.png outputs/key_insights/04_market_divergence_v4.png
    cp outputs/viz/05_group_winner_prob.png outputs/key_insights/09_group_winner_prob.png
    cp outputs/viz/06_advancement_ladder.png outputs/key_insights/10_advancement_ladder.png

---

## STEP 5 — Update README.md

### 5a. Update WC2026 Live Forecast table

Replace the existing top-10 table with the new values from tournament_probs.csv.
Format: Team | FIFA Rank | Group Advance % | Win %
Sort by Win % descending, top 10 teams.

### 5b. Update model version table

Change the V5 row (currently "V5 — Confirmed | — | — | — | All levers tested") to:
    | V5 — 495-Combo Fix | 62.0% | 0.8461 | 9.1% | 3rd-place slot assignment corrected (official FIFA table) |

### 5c. Add new visualization entries

After the existing 7 visualization entries, add:

    ### Group Winner Probability — All 12 Groups
    ![Group Winner Prob](outputs/key_insights/09_group_winner_prob.png)

    ### Tournament Advancement Ladder — Top 24 Teams
    ![Advancement Ladder](outputs/key_insights/10_advancement_ladder.png)

### 5d. Add Live Update Instructions section

Add this section before the final section of the README:

    ## Live Update Instructions

    After each matchday, add actual results to `data/raw/wc2026_live_results.csv`:

    ```
    match_id,stage,group,date,home_team,away_team,home_goals,away_goals,decided_by,winner
    1,Group Stage,A,2026-06-11,Mexico,TeamX,2,0,FT,Mexico
    ```

    Then run:
    ```bash
    python -m src.models.live_update
    ```

    Output: `data/processed/tournament_probs_live.csv` — fresh probabilities with actual
    results locked. Copy updated values into `outputs/bracket_simulator.html` TEAM_PROBS
    and push to GitHub Pages.

---

## STEP 6 — Verify live_update.py runs cleanly

    python -m src.models.live_update

With an empty live results file, this should reproduce near-identical probabilities to
the pre-tournament baseline (max drift < 1% stochastic variance). Confirm the output
matches or note any discrepancy.

---

## STEP 7 — Update GitHub About section

If the GitHub CLI is available:
    gh repo edit andsoitrises/World-Cup-2026- \
      --description "WC2026 ML prediction model — XGB/LGBM/Dixon-Coles ensemble, 62% accuracy. Live bracket simulator with official FIFA 495-combo 3rd-place rules. Updated after each matchday." \
      --homepage "https://andsoitrises.github.io/World-Cup-2026-/outputs/bracket_simulator.html"

If gh is not available, note this step as manual and move on.

---

## STEP 8 — Commit everything

    git add src/models/monte_carlo.py \
            src/visualization/charts.py \
            data/processed/tournament_probs.csv \
            data/processed/group_standings.csv \
            outputs/bracket_simulator.html \
            outputs/viz/ \
            outputs/key_insights/ \
            README.md

    git commit -m "fix: replace greedy 3rd-place slot assignment with official FIFA 495-combo table

- monte_carlo.py: add THIRD_PLACE_COMBOS (495 entries) and rewrite
  assign_third_place_teams() to use official lookup; greedy kept as fallback
- Re-run Monte Carlo (10k sims) — corrected tournament_probs.csv + group_standings.csv
- Update bracket_simulator.html TEAM_PROBS with corrected probabilities
- charts.py: add chart_group_winner_prob() and chart_advancement_ladder()
- Regenerate all probability-dependent visualizations (charts 01-04 + new 05-06)
- README: update forecast table, model history, add live update instructions
- Stage live_update.py pipeline confirmed clean (0 live results, <1% drift)"

Then provide the user with the exact push command for their local machine.

---

## SUCCESS CRITERIA

Before reporting done, verify all of the following:

1. `len(THIRD_PLACE_COMBOS) == 495` — combo table complete
2. Sanity check for EFGHIJKL produces the correct slot assignments (see Step 1c)
3. `data/processed/tournament_probs.csv` has new timestamps and the top team's p_winner
   differs from the old value by at least 0.001 (confirming the fix changed something)
4. `outputs/bracket_simulator.html` TEAM_PROBS block reflects the new CSV values
5. 6 PNG files exist in `outputs/viz/`
6. 10 PNG files exist in `outputs/key_insights/` (01–07 old + 08–10 new)
7. README Live Forecast table matches the new tournament_probs.csv top-10
8. `python -m src.models.live_update` runs without error
9. All files staged and committed

If any step fails, report the exact error and what was tried before stopping.
```
