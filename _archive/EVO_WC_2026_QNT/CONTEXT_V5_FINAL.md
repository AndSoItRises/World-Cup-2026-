# WC2026 Model — V5 Final Iteration Handoff
### Context Document for Claude Code Execution

*Purpose: Onboard Claude Code into the exact state of the project, the one confirmed bug, and the full
scope of the final iteration. Read this before touching any file.*

---

## Part 1 — Repo State (June 2026)

**Repo:** `andsoitrises/World-Cup-2026-` (local clone likely at `~/World-Cup-2026-` or wherever user has it)
**GitHub Pages:** `https://andsoitrises.github.io/World-Cup-2026-/outputs/bracket_simulator.html`

**Working model:** V4 Final — XGB + LGBM + Dixon-Coles ensemble (weights 0.275 / 0.275 / 0.45)
- Accuracy: 62.0% | Log loss: 0.8461 | Draw recall: 9.1%
- WC2022 backtest log loss: 1.0447 (24.3% better than naive baseline)

**Key files:**
```
src/models/monte_carlo.py        ← CONTAINS THE BUG (assign_third_place_teams)
src/models/live_update.py        ← imports assign_third_place_teams from monte_carlo; fixed automatically
src/visualization/charts.py      ← 4 chart functions (add 2 more here)
data/processed/tournament_probs.csv      ← pre-tournament MC output (STALE — needs re-run after fix)
data/processed/group_standings.csv       ← group-stage MC output (also stale)
outputs/bracket_simulator.html           ← HTML simulator (data needs update after re-run)
outputs/key_insights/                    ← 7 existing PNG visualizations
README.md                                ← needs tournament table update
```

---

## Part 2 — The One Confirmed Bug

### `assign_third_place_teams()` in `src/models/monte_carlo.py`

**Lines ~434–477.** This function receives the 12 third-place teams (ranked), selects the best 8, and
assigns them to the 8 bracket slots. The current implementation uses a **greedy sequential algorithm**:
it iterates through slots in order and picks the best unassigned team from an eligible group for each
slot. This does not match the official FIFA 495-combination table.

**Why it matters:** FIFA's official rules specify exactly which third-place team fills which bracket slot
based on which 8 groups the third-place qualifiers came from. The 495 combinations cover every possible
combination of 8 groups from 12. The greedy algorithm produces wrong slot assignments in a meaningful
fraction of simulations — meaning teams are assigned to play against wrong opponents in the R32, which
propagates into all subsequent knockout probabilities. Tournament win percentages are affected for every
team in the simulation.

**What correct behavior looks like (from bracket_simulator.html, already working):**
```javascript
const thirdGroups = best8.map(t => t.group).sort().join('');
const lookup = COMBO_TABLE[thirdGroups];
const slots = ['A','B','D','E','G','I','K','L'];
if (lookup) {
  slots.forEach((grp, i) => { thirdAssign[grp] = thirdsByGroup[lookup[i]]; });
}
```

The lookup result `[v0, v1, ..., v7]` maps to these 8 Python slot names (by index):
```
index 0 → "3rd_CEFHI"   (opponent for 1A, Match 79)
index 1 → "3rd_EFGIJ"   (opponent for 1B, Match 85)
index 2 → "3rd_BEFIJ"   (opponent for 1D, Match 81)
index 3 → "3rd_ABCDF"   (opponent for 1E, Match 74)
index 4 → "3rd_AEHIJ"   (opponent for 1G, Match 82)
index 5 → "3rd_CDFGH"   (opponent for 1I, Match 77)
index 6 → "3rd_DEIJL"   (opponent for 1K, Match 87)
index 7 → "3rd_EHIJK"   (opponent for 1L, Match 80)
```

### The Fix — Replace `assign_third_place_teams()`

Add the 495-combination table as a Python dict (`THIRD_PLACE_COMBOS`) and rewrite the function:

```python
# Ordered list of Python slot names corresponding to combo table indices 0–7
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
    Assign best 8 third-place teams to the 8 bracket slots using the
    official FIFA 495-combination table. Key = sorted 8-char group string.
    """
    best8 = third_place_teams[:8]
    group_to_team = {group: team for team, group, pts, gd, gf in best8}
    groups_key = "".join(sorted(group_to_team.keys()))

    assignment = THIRD_PLACE_COMBOS.get(groups_key)
    if assignment is None:
        # Fallback: greedy (should not happen with valid 12-group input)
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
```

**The full `THIRD_PLACE_COMBOS` dict (495 entries) must be embedded in the Python file.**
The source of truth is the `COMBO_TABLE` object already in `outputs/bracket_simulator.html` (lines ~10–1100
of that file, ~27k chars). You can also load it from the JSON at the path below:

```
/tmp/combos.json  ← Python dict, 495 keys, format: "EFGHIJKL" → ["E","J","I","F","H","G","L","K"]
```

If `/tmp/combos.json` is gone (sandbox cleared), extract from `outputs/bracket_simulator.html`:
```python
import re, json
html = open("outputs/bracket_simulator.html").read()
m = re.search(r"const COMBO_TABLE\s*=\s*(\{[\s\S]+?\});", html)
combo_js = m.group(1)
# JS object keys are unquoted — add quotes:
combo_js2 = re.sub(r'(\w{8}):', r'"\1":', combo_js)
combos = json.loads(combo_js2)
```

---

## Part 3 — What Is Already Correct (Do Not Change)

These structures in `monte_carlo.py` have been verified against the official FIFA WC2026 bracket:

```python
R32_BRACKET = {
    73: ("R_A",  "R_B"),         74: ("W_E",  "3rd_ABCDF"),
    75: ("W_F",  "R_C"),         76: ("W_C",  "R_F"),
    77: ("W_I",  "3rd_CDFGH"),   78: ("R_E",  "R_I"),
    79: ("W_A",  "3rd_CEFHI"),   80: ("W_L",  "3rd_EHIJK"),
    81: ("W_D",  "3rd_BEFIJ"),   82: ("W_G",  "3rd_AEHIJ"),
    83: ("R_K",  "R_L"),         84: ("W_H",  "R_J"),
    85: ("W_B",  "3rd_EFGIJ"),   86: ("W_J",  "R_H"),
    87: ("W_K",  "3rd_DEIJL"),   88: ("R_D",  "R_G"),
}
R16_BRACKET = {89:(74,77), 90:(73,75), 91:(76,78), 92:(79,80),
               93:(83,84), 94:(81,82), 95:(86,88), 96:(85,87)}
QF_BRACKET  = {97:(89,90), 98:(93,94), 99:(91,92), 100:(95,96)}
SF_BRACKET  = {101:(97,98), 102:(99,100)}
```

The group simulation (Poisson + correct tiebreakers) is correct.
The knockout probability draw (`p_home / (p_home + p_away)`) is correct.
The `live_update.py` pipeline is correct in structure — it will inherit the 3rd-place fix automatically
because it imports `assign_third_place_teams` from `monte_carlo`.

---

## Part 4 — Full Scope of the Final Iteration

### Step 1: Fix the bug
Replace `assign_third_place_teams()` and add `THIRD_PLACE_COMBOS` dict in `src/models/monte_carlo.py`.
Keep `THIRD_PLACE_SLOTS` in place — it is still used as the greedy fallback.

### Step 2: Re-run Monte Carlo
```bash
python -m src.models.monte_carlo
```
This overwrites `data/processed/tournament_probs.csv` and `data/processed/group_standings.csv` with
corrected probabilities.

### Step 3: Regenerate visualizations
Run the existing chart pipeline and add 2 new charts to `src/visualization/charts.py`:

**Existing charts (regenerate from corrected data):**
- `01_win_probability.png` — top 20 tournament win %
- `02_stage_heatmap.png` — stage progression heatmap, top 16
- `03_advance_vs_win_scatter.png` — group advance % vs win %
- `04_market_divergence.png` — model vs market edge

**New charts to add:**

**Chart 05 — Group Winner Probability (all 12 groups)**
- 12-panel subplot (3 rows × 4 columns), one per group
- For each group: horizontal bar chart showing each team's P(win group)
- Data from `group_standings.csv`, column `p_win_group`
- Dark theme, match existing style
- Title: "Group Winner Probability — All 12 Groups"
- Save to `outputs/viz/05_group_winner_prob.png`

**Chart 06 — Path to Glory: Likelihood of Advancement by Stage**
- Stacked horizontal bar chart, top 24 teams (all that matter)
- 6 bars per team: Group Adv, R16, QF, SF, Final, Win
- Each bar segment is the INCREMENTAL probability (delta between stages)
- Dark theme, gradient colormap from dim to bright
- Title: "Tournament Advancement Ladder — WC2026"
- Save to `outputs/viz/06_advancement_ladder.png`

**Existing key_insights images (copy updated viz output there):**
```
outputs/viz/01_win_probability.png  → outputs/key_insights/02_tournament_win_probs.png
outputs/viz/02_stage_heatmap.png    → outputs/key_insights/03_stage_heatmap.png
outputs/viz/03_advance_vs_win_scatter.png  → outputs/key_insights/  (new: 08_advance_scatter.png)
outputs/viz/04_market_divergence.png → outputs/key_insights/04_market_divergence_v4.png
outputs/viz/05_group_winner_prob.png → outputs/key_insights/09_group_winner_prob.png
outputs/viz/06_advancement_ladder.png → outputs/key_insights/10_advancement_ladder.png
```
(Key insights 01, 05, 06, 07 — version progression, bracket shift, feature importance, bias resolution —
do not depend on MC output and don't need regenerating unless explicitly requested.)

### Step 4: Update bracket_simulator.html

The HTML file at `outputs/bracket_simulator.html` contains hardcoded tournament probability data.
It needs to be updated to reflect the re-run MC output. Specifically, update the `TEAM_PROBS` JavaScript
object (near the top of the script section) with values from the new `tournament_probs.csv`.

The TEAM_PROBS format in the HTML is:
```javascript
const TEAM_PROBS = {
    "Spain":   { win: 0.1854, sf: 0.2766, qf: 0.5372, r16: 0.7049, advance: 0.9262 },
    ...
};
```
Read the new `tournament_probs.csv` and generate updated JavaScript object literals, then sed/replace
the TEAM_PROBS block in the HTML file.

### Step 5: Update README.md

Update the **WC2026 Live Forecast** table with the new top-10 from corrected `tournament_probs.csv`.
The table columns are: Team | FIFA Rank | Group Advance % | Win %.
Sort by Win % descending, show top 10.

Also update the model performance table — V5 row:
```
| V5 — 495-Combo Fix | 62.0% | 0.8461 | 9.1% | 3rd-place slot assignment corrected |
```

### Step 6: Update GitHub About section

The repo's About description (shown on the GitHub repo homepage) should read:
> WC2026 ML prediction model — XGB/LGBM/Dixon-Coles ensemble, 62% accuracy. Live bracket simulator with official 495-combo 3rd-place rules. Updated after each matchday.

Use the GitHub CLI if available:
```bash
gh repo edit andsoitrises/World-Cup-2026- --description "WC2026 ML prediction model — XGB/LGBM/Dixon-Coles ensemble, 62% accuracy. Live bracket simulator with official 495-combo 3rd-place rules. Updated after each matchday."
```

### Step 7: Stage for live updates

Verify `data/raw/wc2026_live_results.csv` exists (created by `live_update.py` if absent) and the
`live_update.py` pipeline runs cleanly end-to-end:
```bash
python -m src.models.live_update
```
It should print: "0 matches loaded" and confirm max delta < 1% vs baseline.

Create a usage instructions block in the README under a new `## Live Update Instructions` section:
```markdown
## Live Update Instructions

After each matchday, open `data/raw/wc2026_live_results.csv` and add rows with actual results:

| match_id | stage | group | date | home_team | away_team | home_goals | away_goals | decided_by | winner |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Group Stage | A | 2026-06-11 | Mexico | ... | 2 | 0 | FT | Mexico |

Then run:
```bash
python -m src.models.live_update
```
Output: `data/processed/tournament_probs_live.csv` — fresh MC probabilities with actual results locked.
Copy updated data into `outputs/bracket_simulator.html` TEAM_PROBS block and push to GitHub Pages.
```

---

## Part 5 — Known Limitations (Do Not Fix in This Iteration)

**Host nation home advantage:** USA, Canada, Mexico are treated as neutral (`"neutral": 1`) in all
WC matches. In reality they have partial home advantage. This is a known modeling choice — fixing it
requires changing the `predict_wc2026.py` feature builder and re-running the full pipeline. Out of scope
for V5.

**Squad value in training:** Historical Transfermarkt values (2002–2022) are still not sourced. Squad
value remains a diagnostic result only, not a validated model feature. Out of scope for V5.

**Draw recall ceiling (~9%):** All tested approaches to improve draw recall have been exhausted at this
feature set. Not a regression — model is at its ceiling.

---

## Part 6 — Commit Message for V5

```
fix: replace greedy 3rd-place slot assignment with official FIFA 495-combo table

- monte_carlo.py: add THIRD_PLACE_COMBOS (495 entries) and rewrite
  assign_third_place_teams() to use official lookup instead of greedy algorithm
- Re-run Monte Carlo (10k sims) to produce corrected tournament_probs.csv
- Update bracket_simulator.html TEAM_PROBS with corrected data
- Add charts 05 (group winner prob) and 06 (advancement ladder)
- Regenerate all probability-dependent visualizations
- README: update live forecast table and model version history
- Stage live_update.py pipeline (confirmed clean run, 0 live results)
```

---

*Document version: CONTEXT_V5_FINAL.md — written June 2026*
*Read alongside CONTEXT_QUANT_TRANSITION.md for full intellectual history.*
