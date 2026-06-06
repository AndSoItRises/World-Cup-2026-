# WC2026 Model — V3 Context Document

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
- Keep prior version models intact — new versions save to new file paths (xgb_v3, lgbm_v3, etc.)
- Any new data source requires a team name audit before merging — mismatches have caused silent NaN bugs
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
World Cup 2026 match prediction + bracket simulation. V1 built the baseline pipeline. V2 made the
model intelligent (ELO, opponent-quality-weighted form, draw fix, CV tuning) and tested it against the
betting market. **V3's job is decided by what the V2 market analysis found: the model has no trustworthy
edge yet because its biggest market disagreements are its own known biases. So V3 is about fixing those
biases and earning a real, out-of-sample edge — not about betting infrastructure (deferred until edge exists).**

---

## Inherited State (V2 — tagged v2.0, closed)

### Final V2 metrics (test set, temporal split @ 2022-11-20)
| Metric | V1 | V2 |
|---|---|---|
| Test Accuracy | 60.6% | 62.1% |
| Log Loss | 0.8605 | 0.8458 |
| Draw Recall | 0.4% | 9.7% |

### What V2 built (all working, in `main`)
- **ELO ratings** (`src/features/elo.py`) — tiered K 40/30/20, used as features + opponent-quality signal
- **41-feature set** — FIFA rank, ELO, simple + opponent-ELO-weighted rolling form (5/10), H2H, rest, tier, neutral
- **Models** — XGBoost + LightGBM (depth-3, draw-upweighted 1.75×) + Dixon-Coles; fixed-weight ensemble (0.275/0.275/0.45)
- **Monte Carlo** — 10k-sim 48-team bracket → `tournament_probs.csv`
- **Tuning** (`tune_hyperparams.py`) — time decay DISABLED, depth-3 chosen via TimeSeriesSplit CV
- **Market analysis** (`market_divergence.py`) — odds de-vigged, compared to model
- **Stacking** (`stacking.py`) — evaluated, NOT adopted (fixed weights near-optimal)
- **Charts** (`charts.py`) — 4 PNGs incl. market divergence

### V2 left these OPEN (the V3 to-do seeds)
1. **Mexico / CONCACAF inflation** — model 10.4% vs market 1.1%. Biggest single bias. ELO weighting didn't fully fix strength-of-schedule.
2. **European powers underrated** — France 4.5% (market 14.8%), Portugal 3.1% (market 9.0%), England 4.9% (market 10.6%). No-decay over-rewards recent CONMEBOL dominance.
3. **Draw recall ceiling** — 9.7%; DC fits rho≈0 under no-decay and rarely calls draws, capping the blend.
4. **ELO K-factors never CV-tuned** — blocked on the O(n²) weighted-rolling-stats feature rebuild being slow.
5. **80 rank-sentinel** may overrate unranked WC teams (Iran 4.2%).
6. **DR Congo** name-audit gap in `predict_wc2026` (fixture name unmatched).

---

## V3 Goals (DRAFT — to be refined in brainstorm)

Primary thesis: **fix the biases, then re-test for edge.** Candidate directions (not yet committed):
- Strength-of-schedule signal beyond ELO weighting; confederation-strength priors/adjustments
- Squad quality features (market value, key-player availability) — new data source → name audit required
- Partial time-decay (e.g. 1460d) to balance the bias-vs-Euro-underrating tradeoff revealed in V2
- Vectorize weighted rolling stats → makes K-factor tuning and decay sweeps cheap
- Re-run market divergence after bias fixes; only then consider edge/Kelly work
- Engineering: faster sim engine, live result → ELO update → re-sim, deployment

DEFERRED until a clean out-of-sample edge is demonstrated: Kelly bet sizing, betting-facing product.

---

## Open Questions for Brainstorm (seed the cowork session with these)
1. Is the Mexico/CONCACAF inflation a *feature* problem (need better SoS signal) or a *training-data* problem (too many easy CONCACAF qualifiers in the rolling window)? How would we tell them apart?
2. No-decay fixed Argentina/Brazil but seems to underrate France/Portugal/England. Is a single global decay the wrong lever — should decay be per-confederation, or replaced by an explicit recency feature the model can weigh itself?
3. What's the cleanest way to validate "edge" honestly? (e.g. backtest predicted vs market on a held-out historical tournament before trusting WC2026 divergences.)
4. Squad/market-value data: best source, and how to join it without another silent name-mismatch bug?
5. Should the draw model be a separate dedicated classifier rather than relying on the blend?
6. What's the smallest change that would most move the bias needle — and how do we avoid overfitting to "make the bracket look right"?

---

## V3 Decision Log

### DL-01 — Iran rank sentinel is a data bug, not a model problem
Iran's FIFA rank is stored as 150 in the model but their actual rank is ~22. This is the sentinel value
assigned to teams not in the FIFA rankings CSV at training time. Fix: audit `data_cleaning.py` and
`feature_engineering.py` sentinel assignment; use current FIFA rank (22) for Iran at prediction time.
Do NOT model around this — just fix the data join. Eliminates the 34x false positive entirely.

### DL-02 — CONCACAF inflation is a training-data volume problem, not a SoS problem
Quantified via SoS audit: Mexico's avg opponent ELO over last 20 training matches = 1709 vs own ELO
1861 (gap +152). This is actually the *smallest* gap of any team audited — France +259, England +319,
Brazil +222. Mexico is not anomalously easy-schedule relative to others. The inflation comes instead
from the sheer volume of CONCACAF qualifier matches in the rolling window: high win-rate, high goals,
against weak sides that the ELO-weighting partially but not fully corrects. Fix: add a
`confederation_match_pct` feature (% of rolling window that was same-confederation) so the model can
discount intra-confederation form on its own. Do NOT hard-code a CONCACAF penalty — let the model learn it.

### DL-03 — European underrating is a decay problem
No-decay was correct to fix Argentina/Brazil (over-rewarded CONMEBOL dominance under old decay). But
it now punishes consistently elite UEFA teams who have steady but unspectacular qualifier records.
France/England/Portugal all play DOWN (+259/+319 ELO gap vs opponents) — their form numbers look
mediocre because they rarely face tough opposition in qualifiers. Fix: implement partial decay
(half-life ~3 years / 1095 days) and sweep via TimeSeriesSplit CV. Do NOT tune decay by hand to
"make France look right" — only adopt if CV log loss improves.

### DL-04 — Draw recall is a structural ceiling under current Dixon-Coles blend
DC fits rho≈0 under no-decay → rarely predicts draws → caps ensemble draw recall at ~9.7%.
Confederation breakdown confirms: draw_pred averages 0.25-0.31 across all confederations while
draw_actual is 0.20-0.32 — the model is not systematically wrong on draw *rates* overall, but it
concentrates draws poorly (wrong matches). Fix for V3: evaluate a dedicated draw classifier (binary:
draw vs. decisive) as a third signal. Only adopt if it improves draw recall without hurting overall LL.

### DL-05 — WC2022 backtest confirms real predictive power
Model log loss on WC2022 matches (n=348): 0.8313 vs naive baseline 1.0986. Skill score = +24.3% over
uniform, +22.7% over historical base rates. Accuracy 62.1%. This is the honest out-of-sample
validation. The model works — the biases above are fixable signal problems, not fundamental failures.
Confederation LL breakdown: UEFA 0.8121, AFC 0.7768, CONCACAF 0.9245, CONMEBOL 0.9255, CAF 0.9558.
CAF is the worst-performing confederation — flagged for V3 investigation (likely same training-volume issue as CONCACAF).

### DL-07 — P1 EXECUTED: name standardization + a second data bug (corrupt rank column)
Root cause of Iran=150 confirmed: `feature_engineering.load_rankings()` never standardized
ranking-table names, so "IR Iran" (rankings) never matched "Iran" (matches) → 150 sentinel.
Fix: apply `standardize_name` to the rankings table + to prediction rank-lookup keys; added
`DR Congo→Congo DR`, `Türkiye→Turkey`, `Cabo Verde→Cape Verde Islands` to the maps.
**Second bug found:** `current_fifa_rankings.csv` has a CORRUPTED rank column for ~8 teams
(Austria "231"/1597pts, Algeria "291", Cabo Verde "681", Ghana "731", Haiti "821"). Points are
clean and FIFA rank = points-descending, so `build_rank_lookup` now re-derives rank from points.
Result: all 48 WC2026 teams resolve to real rank + ELO (Iran 21, Algeria 29, Turkey 22, …).
Training FIFA-rank coverage improved (missing 8.5% → 4.4%). New: `src/features/verify_teams.py`.
Vectorization of weighted rolling (planned P3 prereq) DEFERRED — the decay sweep recomputes
weights from dates (no feature rebuild), so it isn't needed; revisit only if K-factor tuning happens.

### DL-06 — Calibration is acceptable, draw is slightly overforecast
away_win: mean_pred=0.300 vs actual=0.329 (UNDER by 3pp, cal_err=0.049)
draw:     mean_pred=0.259 vs actual=0.222 (OVER  by 4pp, cal_err=0.032)
home_win: mean_pred=0.441 vs actual=0.449 (UNDER by 1pp, cal_err=0.033)
No Platt scaling or isotonic regression needed at this stage — errors are small and may close with
bias fixes. Revisit after V3 retraining.

---

## V3 Phase Status

| Phase | Description | Status |
|---|---|---|
| P1 | **Data fixes** — Iran rank sentinel + corrupt rank column fixed via name standardization + points-rerank; all 48 teams resolve (see DL-07) | ✅ |
| P2 | **Confederation feature** — Add `home_conf_match_pct` + `away_conf_match_pct` to rolling feature set (% of rolling-10 window that was same-confederation). Retrain → CV LL must improve vs V2 baseline to keep. | ⬜ |
| P3 | **Decay sweep** — Vectorize weighted rolling stats (unblock O(n²) bottleneck first). Then sweep half-life [365, 730, 1095, 1460, inf] via TimeSeriesSplit CV. Adopt whichever minimizes CV LL — do not pick by bracket feel. | ⬜ |
| P4 | **Draw classifier** — Train a binary draw-vs-decisive classifier. Blend as fourth signal if draw recall improves without degrading overall LL. | ⬜ |
| P5 | **Retrain + validate** — Full retrain with P1-P4 changes. Rerun market divergence. Compare V3 vs V2 on: test LL, confederation breakdown LL, WC2022 backtest LL, draw recall. Only accept V3 if it beats V2 on >2 of 4 metrics. | ⬜ |
| P6 | **Re-run bracket + charts** — Monte Carlo 10k sims with V3 models. Update all 4 charts. Update market divergence. Flag any remaining high-ratio divergences and classify: likely-edge vs likely-bias. | ⬜ |

---

## Notes on the Data (carried from V2)
- results.csv uses: "Korea Republic", "Côte d'Ivoire", "Bosnia-Herzegovina", "United States"
- wc2026_fixtures.csv uses: "South Korea", "Ivory Coast", "Bosnia and Herzegovina", "DR Congo"
- `ELO_NAME_MAP` in feature_engineering.py, `TEAM_NAME_MAP` in data_cleaning.py, `NAME_MAP` in ensemble/predict_wc2026/monte_carlo handle the mismatches
- `wc2026_market_odds.csv` (raw) uses ESPN naming; `MARKET_TO_MODEL` in market_divergence.py maps it
- Any new V3 data source → audit names against training names BEFORE merging

---

## ══ END-OF-VERSION REVIEW ══
*(Filled in when V3 is fully closed)*
