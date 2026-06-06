# WC2026 Model — V4 Context Document

---

## ══ HOOKS (Permanent — Copy Into Every New Context Doc) ══

### How We Work
- Explain decisions before writing code — options considered, why we chose this path
- One phase at a time — don't jump ahead
- Claude Code executes scripts autonomously — no need to prompt Jake to run them
- Venv must be activated before any python execution: `venv\Scripts\activate`
- Scripts run as: `python -m src.<module>.<script>`
- Full file replacements when edits get complex — rewrite the whole file cleanly
- On Windows, set `PYTHONUTF8=1` before running scripts (console is cp1252; box-drawing prints crash otherwise)

### Session Start Ritual (Claude Code — Every Session)
1. Read the current context doc fully
2. Run `git status` — confirm clean state or note what's uncommitted
3. State which phase we're in and what the next action is
4. Confirm understanding before touching any file

### Non-Negotiables
- No leakage — all features must use only information available before the match
- Every new feature must be validated: retrain → compare CV log loss to prior version baseline. If no improvement, cut it.
- Keep prior version models intact — new versions save to new file paths (xgb_v4, lgbm_v4, etc.)
- Any new data source requires a team name audit before merging — mismatches have caused silent NaN bugs (V3 found Iran rank=150 and a corrupt rank column this way)
- Git checkpoint commits during active work; version tag only when version is fully closed
- Context doc decision log gets written immediately after a decision is made — not at end of session

### End-of-Phase Checklist (Before Moving to Next Phase)
- [ ] All outputs saved to correct paths
- [ ] No unexpected NaNs in new features (check the feature summary printout)
- [ ] Prior version model files untouched
- [ ] Decision log updated in context doc
- [ ] Checkpoint commit pushed

### Git Protocol
- Active work: `git add . && git commit -m "descriptive message"`
- Close a version: final commit → `git tag vX.0` → `git push --tags` → create new `CONTEXT_VX+1.md`
- Claude Code reads the current context doc at the start of every session
- GitHub: https://github.com/AndSoItRises/World-Cup-2026-

### Project Location
`C:\Users\jakeh\OneDrive\Documents\Claude\Projects\World Cup 2026 Model`

### Tech Stack
Python 3.10+, pandas, numpy, scikit-learn, xgboost, lightgbm, scipy, statsmodels, matplotlib, plotly

### Tooling
- Editor: Cursor
- Terminal: PowerShell (venv activated)
- Agent: Claude Code (autonomous execution)
- Version tracking: Git + context doc per version

---

## ══ VERSION BODY ══

## Project Summary
World Cup 2026 match prediction + bracket simulation. V1 baseline → V2 made it intelligent (ELO,
opponent-weighted form, draw fix, tuning) → V3 fixed data bugs and proved the model is at its ceiling
for the current feature set. **V4's mandate is the one lever V3 left standing: add genuinely NEW signal.
Reweighting existing features is exhausted — the residual biases (Mexico/CONCACAF inflation, Euro-power
underrating) are baked into what the model can currently see. V4 must give it new eyes.**

---

## Inherited State (V3 — tagged v3.0, closed)

### Metrics (unchanged since V2 — the model is at its ceiling)
| Metric | V1 | V2 | V3 |
|---|---|---|---|
| Test Accuracy | 60.6% | 62.1% | 61.7% |
| Test Log Loss | 0.8605 | 0.8458 | 0.8462 |
| Draw Recall | 0.4% | 9.7% | 8.7% |

### What exists (all working, in `main`)
- Pipeline: `data_cleaning → elo → feature_engineering → dixon_coles → train_xgb → train_lgbm →
  ensemble → predict_wc2026 → monte_carlo` + `market_divergence`, `charts`, `validate_v3`
- 41 features: FIFA rank, ELO, simple + opponent-ELO-weighted rolling form, H2H, rest, tier, neutral
- Models: XGB + LGBM (depth-3, draw-upweighted 1.75×) + Dixon-Coles; fixed ensemble 0.275/0.275/0.45
  (validated near-optimal vs a stacked meta-learner). Saved at `*_v3` paths; v1/v2 preserved.
- `verify_teams.py` — asserts all 48 WC2026 teams resolve to a real rank + ELO (run it after any data change)

### What V3 PROVED doesn't work (do not retry without new data)
- conf_match_pct (no variance — ~93% intra-conf for everyone)
- partial time-decay (amplifies CONCACAF inflation; no LL gain — DL-03/08 of V3)
- dedicated draw classifier (can't beat ~9% recall ceiling without overforecasting)

### Persistent biases V4 must attack (with NEW signal)
1. **Mexico / CONCACAF inflation** — model ~11% win prob vs market 1.1%. Strongest single bias.
2. **Euro powers underrated** — France ~5% vs market 14.8%; Portugal, England similar.
3. These are NOT fixable by reweighting existing features (V3 exhausted that). They reflect missing
   information: the model can't see that CONCACAF qualifying inflates form vs genuinely weak squads.

---

## V4 Goals (DRAFT — refine in brainstorm)
Thesis: **add new signal that distinguishes a team's true quality from its schedule-inflated form.**
Candidate sources (each new source ⇒ TEAM-NAME AUDIT first, then validate-or-cut):
- **Squad market value** (e.g. Transfermarkt) — total/median player value; a strong talent prior that
  doesn't care who you beat in qualifying. Likely the highest-leverage single feature.
- **Player availability / injuries** near tournament time; **club form** of the player pool.
- **Confederation-STRENGTH** feature (opponent confederation avg ELO) — distinct from the cut conf_match_pct.
- Manager tenure / continuity; travel/altitude/rest already partially covered.

Validation bar (carried, and reinforced by V3): adopt only if CV log loss improves AND the BRACKET
moves the right way (V3's DL-08 lesson — LL alone missed that decay worsened the bracket).

DEFERRED until a clean out-of-sample edge is demonstrated: Kelly bet sizing, betting product, sim engine.

---

## Open Questions for Brainstorm (seed the cowork session with these)
1. Squad market value: best accessible source and snapshot date handling (values change over time —
   need pre-tournament values, no leakage). How to join 48 national teams without another name-mismatch bug?
2. Will squad value actually fix CONCACAF inflation, or just correlate with FIFA rank/ELO we already have?
   How do we test incremental value beyond existing features (e.g. residual analysis)?
3. Player availability/injuries: is reliable historical injury data even obtainable for a backtest, or
   is it only usable forward (WC2026) and thus unvalidatable?
4. Honest edge validation: build the PAST-tournament backtest (model vs market on WC2018/2022) BEFORE
   trusting any WC2026 divergence — so we don't launder the market's opinion into the model.
5. Is there a cleaner target than 1X2? (e.g. predict expected goals / supremacy, derive 1X2) — would a
   different target better absorb a squad-value feature?
6. What's the minimum viable new feature that could be added + validated in one phase?

---

## V4 Decision Log
*(empty — fill as decisions are made)*

---

## V4 Phase Status
*(to be defined after brainstorm — replace with the agreed phase plan)*

| Phase | Description | Status |
|---|---|---|
| _TBD from brainstorm_ | | ⬜ |

---

## Notes on the Data (carried from V3)
- results.csv: "Korea Republic", "Côte d'Ivoire", "Bosnia-Herzegovina", "United States"
- wc2026_fixtures.csv: "South Korea", "Ivory Coast", "Bosnia and Herzegovina", "DR Congo", "Iran"
- Name standardization is centralized in `data_cleaning.standardize_name` / `TEAM_NAME_MAP`; rankings
  load + prediction rank-lookups now apply it (V3 P1). Current-rankings rank column is CORRUPT for ~8
  teams → `build_rank_lookup` re-derives rank from points. `verify_teams.py` guards all 48 resolve.
- `wc2026_market_odds.csv` uses ESPN naming; `MARKET_TO_MODEL` in market_divergence.py maps it.
- **Any new V4 data source → audit names against the model's standardized names BEFORE merging, then
  run `verify_teams.py`-style coverage check.**

---

## ══ END-OF-VERSION REVIEW ══
*(Filled in when V4 is fully closed)*
