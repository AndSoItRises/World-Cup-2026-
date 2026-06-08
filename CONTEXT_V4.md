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

### DL-01 — Squad value is PROVEN missing signal, but rigorous integration is data-blocked
Diagnostic (`squad_value_diagnostic.py`, 47/48 WC2026 teams, Transfermarkt values vs market_divergence):
- corr(log squad value, **market** win%) = **+0.66** vs corr(log value, **model** win%) = **+0.42** — the
  market tracks talent far more than our model does; the model is partly blind to it.
- **Incremental R² = +0.178** predicting market prob (model alone 0.443 → +value 0.621), coef +0.014.
  Squad value carries real information BEYOND ELO/form — NOT redundant.
- corr(value, edge=model−market) = −0.40 → high-value squads are systematically UNDER-rated (Euro bias
  = talent-blindness). Over-rated by value gap: Mexico (val rk 27 / model rk 3), Iran, Japan, Korea, Australia.
**Conclusion:** the V4 premise holds — there is identifiable residual signal (~18% incremental R²) the
current feature set cannot see. BUT adopting squad value as a validated MODEL feature needs historical
squad values (2002–2022) aligned to the 11.5k training matches for CV — we don't have that, and the
non-negotiable forbids unvalidated features. A forward-only hand-weighted bracket adjustment was
considered and REJECTED: the blend weight would be arbitrary (validating against the market just
reproduces market odds — circular). So V4's real work is a DATA-ACQUISITION project, not modeling.
**Answer to "are we at max utility?":** for what we can *rigorously* build with data in hand — yes.
In absolute terms — no, ~18% more explanatory signal is provably available, but it's gated by historical
squad-value data, not by model cleverness. Next gain requires acquiring that data, then validate-or-cut.

### DL-02 — Historical squad data ACQUIRED (FIFA ratings) and squad value+depth ADOPTED into V4
The DL-01 blocker ("we don't have historical squad values") is resolved. FIFA/EA video-game player
ratings are a leakage-safe, by-edition, broad-coverage proxy for squad quality AND depth:
- Source: lbenz730/fifa_model (FIFA 2005–2020, fifaindex) + abineshta FIFA22 + EAFC26-DataHub FC26.
  FIFA `overall` is one 0–99 scale across sources. `build_squad_strength.py` aggregates per nation-year:
  `squad_strength` (top-23 mean), `squad_top11`, `squad_depth` (12th–23rd mean = bench quality),
  `squad_n_quality` (#players ≥75). Cached raw under data/raw/fifa_ratings (gitignored, re-downloadable);
  derived table at data/processed/squad_strength_by_year.csv.
- Leakage safety: each edition stamped availability = (Y-1)-10-01 (games ship ~Sept); merged onto matches
  with `merge_asof` BACKWARD. Prediction uses the latest edition (FC26, ships 2025 → pre-WC2026, no leak).
- Name audit (non-negotiable): 46/48 WC2026 teams covered; Jordan & Uzbekistan have <11 rated players
  (thin FIFA coverage) → low sentinel (≈60, the "not in FIFA"≈weakest-nation convention, like rank-150).
- **signal_test.py (new; = Learning Step 6):** squad strength/depth is REDUNDANT on the full match set
  (ΔLL≈0 — 65% of competitive matches involve non-FIFA minnows → sentinel-diluted) but carries strong
  ORTHOGONAL signal on the COVERED subset (both teams real): incremental IC +0.125 (strength) / +0.100
  (depth), ΔLL +0.0062. Fix: a `squad_both_covered` flag lets the trees gate on real-vs-sentinel.
  `squad_top11` (collinear w/ strength) and `squad_n_quality` (noisy, IC sign-flips) were CUT.
- **Validation (validate-or-cut, DL-08 rule):** v4 = v3 + 7 squad features (strength/depth per-side + diffs
  + coverage flag). XGB CV log loss 0.8701→0.8668 (**+0.0033**). Ensemble vs v3 on the 4 metrics: wins 3/4
  — test_ll 0.8462→0.8461, **WC2022 finals LL 1.0608→1.0447 (−0.016)**, draw recall 0.087→0.091; loses
  macro_conf_ll 0.8628→0.8653 (the one regression). Squad features land at XGB importance rank 3
  (squad_strength_diff, gain 16.3) and rank 5 (squad_depth_diff) — the model leans on them.
- **Bracket (the binding DL-08 test) moves the RIGHT way** (`bracket_v4.py`): mean single-match P(win)
  CONCACAF −1.28pp (Panama −4.6, Canada −2.3, Mexico −0.5) and Euro +2.36pp (**France +4.3**, Belgium +5.7,
  Portugal +3.9, Netherlands +3.2, Spain +2.8). Defensible exceptions: USA ↑ (genuinely high squad value),
  Croatia ↓ (aging, low market value). This directly attacks both persistent biases.
- **Tradeoff (honest):** the gain is concentrated in matches between real footballing nations (incl. ALL
  WC2026 ties); on the full minnow-heavy test set the overall LL barely moves and macro_conf_ll dips
  slightly. Net: adopted — CV + WC2022 + bracket + importance all positive, one minor metric regresses.
- Production: `retrain_all`, `predict_wc2026`, `monte_carlo` now use the 48-feature v4 set (squad_fields
  helper, leakage-safe). Prior models preserved: v1/v2/v3 untouched; old prod backed up to *_prod_v3.*.
- **Production forecast impact (the proof):** model↔market correlation **0.665 → 0.838**. The two flagship
  biases shrank — Mexico edge +10.0pp→**+6.1pp** (model win% 11.0→7.1), France −10.0pp→**−7.7pp**
  (4.8→7.1), England now near-aligned (−1.7pp). Not fully closed (ELO still dominant; squad coverage is
  ~34% of all matches) but the largest single bias-correction since V2. WC2026 top: Spain 18.5%, England
  8.9%, Mexico 7.1%, France 7.1%, Brazil 6.3%.

---

## V4 Phase Status

| Phase | Description | Status |
|---|---|---|
| P1 | **Squad-value diagnostic** — confirmed squad value is real missing signal (+0.178 incremental R²); explains both biases. See DL-01. | ✅ |
| P2 | **Acquire historical squad values** — RESOLVED via FIFA/EA video-game ratings (leakage-safe, by-edition, broad coverage) instead of Transfermarkt history. `build_squad_strength.py`; 46/48 WC teams. See DL-02. | ✅ |
| P3 | **Add squad value+depth features + validate** — `signal_test.py` (orthogonality), coverage-flag fix, retrain v4, CV +0.0033, 3/4 metrics, bracket corrects CONCACAF↓/Euro↑. ADOPTED. See DL-02. | ✅ |
| P4 | **Past-tournament market backtest** — `market_backtest.py` computes model IC (v4 0.595 > v3 0.593) + market-as-benchmark on WC2026. FULL past-tournament odds backtest still needs historical bookmaker lines. | ◑ partial |

**Status note:** P1–P3 done — V4 delivered a genuine, validated model improvement (squad value + depth),
the first gain since V2. P4 is partially built (IC machinery exists); a true edge test still needs
historical odds. The hands-on quant **Learning Path** scripts are now built and runnable (see below):
`signal_test.py` (Step 6), `calibration.py` (Step 1), `backtest.py` (Step 3), `market_backtest.py` (Step 2).

---

## ══ LEARNING PATH: Transitioning into Quantitative Modeling ══
*Hands-on next steps. Theory lives in `CONTEXT_QUANT_TRANSITION.md`; this is the do-it sequence —
the fastest way to learn quant is to extend a model you already understand. Each step = a concept,
why it matters, and a concrete thing to build in THIS repo. Do them in order; each is ~a session.*

**The meta-point:** you've already done real quant work here — proper scoring (log loss), temporal
cross-validation, de-vigging odds, edge detection, and the validate-or-cut discipline. That discipline
(*"adopt only if it beats the baseline out-of-sample"*) IS the quant mindset. The steps below deepen it.

### Step 1 — Calibration & proper scoring rules  ✅ BUILT: `src/models/calibration.py`
**Concept:** A probability of 30% should be right 30% of the time. "Proper" scoring rules (log loss, Brier)
are minimized only by *honest* probabilities — that's why we never optimized accuracy.
**Build:** A `calibration.py` that bins your test predictions, plots reliability diagrams per outcome
(you have one calibration plot already — generalize it), computes the **Brier score**, then tries
**Platt scaling** and **isotonic regression** to recalibrate. Measure if recalibration improves test log loss.
**Learn:** sharpness vs calibration, why a well-calibrated 55% beats a miscalibrated 90%.

### Step 2 — The market as benchmark (you compete with a price, not reality)  ✅ BUILT: `src/models/market_backtest.py`
**Concept:** V3's big lesson — being "right" is worthless; being *righter than the price* is everything.
The market already embeds squad value, injuries, everything. Your edge = your probability − the fair price.
**Build:** Extend `market_divergence.py` into a backtest: for past matches with odds, compute the model's
**information coefficient** (rank correlation of predicted vs realized outcomes) and compare it to the
market's IC. If the market's IC ≥ yours, you have no edge — quantify that honestly.
**Learn:** information coefficient, hit-rate vs edge, market efficiency.

### Step 3 — Backtesting rigor & the overfitting trap  ✅ BUILT: `src/models/backtest.py`
**Concept:** The #1 way quant models fail is look-ahead bias and tuning-to-the-test. You used a single
temporal split; real quant uses **walk-forward** (expanding-window) validation.
**Build:** A `backtest.py` that retrains at each historical season boundary and predicts the next season
only (walk-forward). Then deliberately *p-hack* — tune hyperparameters on the test set — and watch the
walk-forward performance get worse. Feel the failure mode in your own data.
**Learn:** look-ahead bias, multiple-testing / p-hacking, why the non-negotiables exist.

### Step 4 — From probability to decision: Expected Value & Kelly
**Concept:** A 5% edge means nothing without **sizing**. The Kelly criterion maximizes long-run growth;
bet too big and variance ruins you, too small and you leave growth on the table.
**Build:** A `bet_sim.py`: walk historical matches, and whenever model_prob > market_implied (an edge),
"bet" a **Kelly fraction** of a simulated bankroll; track bankroll, drawdown, and variance over a season.
Compare full-Kelly vs half-Kelly vs flat-staking. (This is the V3 "deferred until edge exists" work —
build the machinery now to *learn* it, even though we found no edge yet.)
**Learn:** EV, Kelly, bankroll growth vs ruin, why variance management is half of quant.

### Step 5 — Bayesian thinking (ELO is already a baby version of this)
**Concept:** ELO updates a belief about strength after each result — that's Bayesian updating with a fixed
learning rate. A real Bayesian model gives you a *distribution* over strength (uncertainty), not a point.
**Build:** A small **hierarchical Poisson** model (PyMC or numpyro) for team attack/defense with priors
and shrinkage toward the mean — compare its team ratings + *uncertainty bands* to your Dixon-Coles MLE.
**Learn:** priors, posteriors, shrinkage, uncertainty quantification, why a point estimate hides risk.

### Step 6 — Signal research: orthogonality & incremental information  ✅ BUILT: `src/models/signal_test.py`
**Concept:** V4 P1's `+0.178 incremental R²` is the single most important quant idea you've touched.
New signal is only worth adding if it's **orthogonal** to what you already have — raw correlation lies.
**Build:** A reusable `signal_test.py`: given a candidate feature, regress it on the existing feature set,
take the **residual**, and measure the residual's incremental IC / R² on the target. Run it on squad value
(once you have history) and on any future feature. This is literally how quant researchers vet alphas.
**Learn:** feature orthogonality, incremental IC, why "it correlates with winning" isn't enough.

### Step 7 — Where the noise floor is (aleatoric vs epistemic uncertainty)
**Concept:** Your log loss plateaued at ~0.846. Some of that is *model* error (epistemic, reducible with
better signal); some is *irreducible match randomness* (aleatoric — a ball off the post). Knowing the split
tells you whether more modeling can even help.
**Build:** Bootstrap the training set (resample with replacement, retrain, repeat) and measure the variance
of predictions across bootstraps = epistemic uncertainty. The gap to perfect log loss that *doesn't* shrink
with more data ≈ the aleatoric floor.
**Learn:** the two kinds of uncertainty, when to stop modeling, why football caps near ~62% accuracy.

### Two destinations (same toolkit)
- **Sports/betting quant:** Steps 2→4→6 are the core loop (find edge, size it, vet new signals).
- **Quant finance / research:** identical machinery — log loss→Sharpe, edge→alpha, Kelly→portfolio sizing,
  IC and orthogonality are used verbatim. This project is a legitimate on-ramp to either.

**Suggested order:** 1 → 3 → 2 → 4 → 6 → 5 → 7. (Calibration and backtesting first — they're the
foundation everything else trusts; Bayesian and noise-floor last — they're the deepest.)

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

**V4 shipped the first model gain since V2 — squad value + DEPTH, rigorously validated.**

What landed:
- **New signal acquired & validated.** Squad strength + depth from FIFA/EA ratings (2005→FC26), leakage-safe
  asof merge, name-audited (46/48 WC teams). The DL-01 "data-blocked" verdict is overturned — FIFA ratings
  were the accessible historical proxy Transfermarkt history wasn't.
- **Disciplined validate-or-cut.** `signal_test.py` exposed that the signal is real only on covered matches
  (incr IC +0.125/+0.10); a `squad_both_covered` flag recovered it; collinear/noisy squad cols were cut.
  XGB CV +0.0033, ensemble 3/4 metrics, WC2022 finals −0.016, squad features at importance rank 3 & 5.
- **Both target biases reduced** in the live forecast: model↔market corr 0.665→0.838; Mexico/CONCACAF
  deflated, France/Euro lifted (with self-consistent exceptions USA↑, Croatia↓).
- **Quant learning stack built & runnable** (for cowork): calibration (Step 1), market-IC (Step 2),
  walk-forward backtest (Step 3), signal_test/orthogonality (Step 6).

Honest limitations / carried to V5:
- Squad coverage is ~34% of all competitive matches (FIFA omits minnows) → gain is concentrated in
  real-nation matches (incl. all WC ties); overall test LL barely moves and macro_conf_ll dipped slightly.
  Filling 2021–2025 editions (FIFA23/FC24/FC25) and minnow nations would lift coverage.
- Biases reduced, not closed — ELO still dominates; squad value is a complementary prior.
- P4 (true past-tournament odds backtest) still needs historical bookmaker lines.
- **Next obvious feature:** confederation/schedule-strength — now a one-step `signal_test.py` vet.

Models: v1/v2/v3 intact; v4 at xgb_v4/lgbm_v4/ensemble_test_proba_v4; prod = v4 (old prod → *_prod_v3).
