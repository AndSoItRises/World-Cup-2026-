# Quant Learning Stack — Context Doc

*A hands-on curriculum for transitioning into quantitative modeling, built around the
World Cup 2026 model (your own code is the lab). Each step = a concept, why it matters,
the script that demonstrates it in the repo, what to run, what to look for, and an exercise.*

**How to use this in cowork:** open this doc + the repo. Go in the suggested order
(1 → 3 → 2 → 6 → 4 → 5 → 7). For each step: read the concept, run the script, read its
output against "what to look for," then do the exercise (modify the script / try it on new
data). The point isn't to read about quant — it's to *extend a model you already understand*.

Repo: `World Cup 2026 Model` (GitHub: AndSoItRises/World-Cup-2026-). Run scripts from the repo
root with the venv active: `python -m src.models.<script>` (set `PYTHONUTF8=1` on Windows first).

**The meta-point:** you've already done real quant work — proper scoring (log loss), temporal
cross-validation, de-vigging odds, edge detection, and the *validate-or-cut* discipline ("adopt
only if it beats the baseline out-of-sample"). That discipline IS the quant mindset. These steps
deepen it. The single most important habit: **you are never trying to be right about reality —
you are trying to be less wrong than a price, out of sample.**

---

## Status at a glance

| Step | Concept | Script | Built? |
|---|---|---|---|
| 1 | Calibration & proper scoring | `src/models/calibration.py` | ✅ |
| 2 | Market as benchmark / Information Coefficient | `src/models/market_backtest.py` | ✅ |
| 3 | Backtesting rigor / walk-forward / overfitting | `src/models/backtest.py` | ✅ |
| 4 | Expected value & Kelly sizing | `src/models/bet_sim.py` | ⬜ to build |
| 5 | Bayesian thinking (hierarchical Poisson) | `src/models/bayes_ratings.py` | ⬜ to build |
| 6 | Signal research: orthogonality & incremental IC | `src/models/signal_test.py` | ✅ |
| 7 | Noise floor: aleatoric vs epistemic uncertainty | `src/models/noise_floor.py` | ⬜ to build |

---

## Step 1 — Calibration & proper scoring rules  ·  `calibration.py` ✅
**Concept.** A probability of 30% should be right 30% of the time. "Proper" scoring rules (log
loss, Brier) are minimised only by *honest* probabilities — that's why this project never optimised
accuracy. Accuracy rewards confident guessing; log loss punishes confident wrongness.
**Why it matters.** Every downstream decision (edge, bet size) trusts your probabilities. A
miscalibrated model leaks money even when it "looks accurate."
**Run.** `python -m src.models.calibration` (defaults to the v4 ensemble; pass `v3` to compare).
**Look for.** The reliability table — predicted vs observed per bin should track the diagonal
(small `gap`). Then the recalibration block: if Platt/isotonic barely beat "raw," the model is
already well-calibrated (good). Big gains ⇒ systematic over/under-confidence.
**Exercise.** Add a reliability *plot* (matplotlib) and overlay v3 vs v4. Which is better calibrated
in the 0.7–0.8 bin? Why might the ensemble be under-confident there?
**Vocabulary.** sharpness vs calibration; proper scoring rule; Brier score; Platt scaling; isotonic.

## Step 3 — Backtesting rigor & the overfitting trap  ·  `backtest.py` ✅
*(Do this 2nd — backtesting is the foundation everything else trusts.)*
**Concept.** The #1 way quant models die is look-ahead bias and tuning-to-the-test. A single
temporal split is the minimum; real quant uses **walk-forward** (expanding window): retrain at each
season boundary, predict only the *next* season, never peek forward.
**Why it matters.** Every impressive backtest you'll ever see is guilty until proven innocent.
Walk-forward is how you prove it.
**Run.** `python -m src.models.backtest`.
**Look for.** Two columns: walk-forward log loss (the number to trust) vs "peek" log loss (a model
that tuned its early-stopping on the very season it's scored on). The **optimism gap** is the cost
of peeking — felt in your own data, not a textbook.
**Exercise.** Make the peek worse: also tune `max_depth` by peeking at the test season. Watch the
gap grow. That's p-hacking, quantified.
**Vocabulary.** look-ahead bias; walk-forward / expanding window; multiple testing / p-hacking;
in-sample vs out-of-sample.

## Step 2 — The market as benchmark (you compete with a price)  ·  `market_backtest.py` ✅
**Concept.** Being "right" is worthless; being *righter than the price* is everything. The market
already embeds squad value, injuries, everything. Your edge = your probability − the fair price.
The **Information Coefficient (IC)** = rank correlation between your predictive score and the
realised outcome; higher IC ⇒ your *ordering* of matches tracks reality better.
**Why it matters.** This is the line between "interesting model" and "tradeable model." Most models
have an IC; few have an IC the market doesn't already have.
**Run.** `python -m src.models.market_backtest`.
**Look for.** Overall IC (v4 ≈ 0.595) and IC by confederation (where does the model order best?
worst?). Then the WC2026 model-vs-market edges. Note the honest caveat: an edge is only real money
if your IC beats the *market's* IC — which needs realised results for matches that had odds
(Step → the V5 historical-odds backtest).
**Exercise.** Compute IC on only the matches where the model and market most *disagree*. Is the
model's ordering still good there, or is disagreement just noise?
**Vocabulary.** information coefficient; edge vs hit-rate; market efficiency; de-vigging (overround).

## Step 6 — Signal research: orthogonality & incremental information  ·  `signal_test.py` ✅
**Concept.** A new feature is only worth adding if it's **orthogonal** to what you already have —
raw correlation with winning lies (a feature can be strongly predictive yet fully redundant with
ELO). The test: regress the candidate on the existing features, take the **residual**, and measure
the residual's incremental IC / log-loss improvement.
**Why it matters.** This is literally how quant researchers vet alphas. V4's squad-value feature
lived or died on exactly this test (it passed only on the covered subset → hence the coverage flag).
**Run.** `python -m src.models.signal_test` (defaults to the squad diffs; pass any feature columns).
**Look for.** "raw IC" high but "incr IC" ≈ 0 ⇒ redundant, cut it. Positive ΔLL ⇒ genuinely helps.
The covered-vs-all split shows how coverage dilution can hide real signal.
**Exercise.** Run it on a feature you *expect* to be redundant (e.g. `home_fifa_rank` vs the set
that already includes ELO). Confirm the incremental IC collapses. That's orthogonality in action.
**Vocabulary.** orthogonality; residualisation; incremental IC; alpha vetting; collinearity.

## Step 4 — From probability to decision: Expected Value & Kelly  ·  `bet_sim.py` ⬜
**Concept.** A 5% edge means nothing without **sizing**. The Kelly criterion maximises long-run
log-growth; bet too big and variance ruins you, too small and you leave growth on the table.
**Build (in cowork).** Walk historical matches; whenever `model_prob > market_implied` (an edge),
"bet" a Kelly fraction of a simulated bankroll. Track bankroll, drawdown, variance. Compare
full-Kelly vs half-Kelly vs flat-staking.
**Look for.** Full Kelly has the highest growth *and* terrifying drawdowns; half-Kelly is the
practical sweet spot. This is why variance management is half of quant.
**Prereq.** Needs historical odds (the V5 Lever-4 dataset) — build it after that lands.
**Vocabulary.** expected value; Kelly fraction; bankroll growth vs ruin; drawdown; variance drag.

## Step 5 — Bayesian thinking (ELO is already a baby version)  ·  `bayes_ratings.py` ⬜
**Concept.** ELO updates a belief about strength after each result — Bayesian updating with a fixed
learning rate. A real Bayesian model gives a *distribution* over strength (uncertainty), not a point.
**Build (in cowork).** A small **hierarchical Poisson** model (PyMC or numpyro) for team
attack/defense with priors + shrinkage toward the mean. Compare its ratings + *uncertainty bands*
to the Dixon-Coles MLE already in the repo (`models/dixon_coles_params_*.json`).
**Look for.** Shrinkage pulls small-sample teams toward the mean (fixes the "minnow with one fluke
win" problem); the posterior width tells you which ratings to trust.
**Vocabulary.** prior / posterior; shrinkage / partial pooling; hierarchical model; credible interval.

## Step 7 — Where the noise floor is (aleatoric vs epistemic)  ·  `noise_floor.py` ⬜
**Concept.** Log loss plateaued near ~0.846. Some is *model* error (epistemic — reducible with
better signal); some is *irreducible match randomness* (aleatoric — a ball off the post). Knowing
the split tells you whether more modelling can even help.
**Build (in cowork).** Bootstrap the training set (resample, retrain, repeat); the variance of
predictions across bootstraps = epistemic uncertainty. The gap to perfect log loss that *doesn't*
shrink with more data ≈ the aleatoric floor.
**Look for.** If you're near the floor, stop adding features and start managing variance/sizing
instead. This is the quant version of knowing when to quit.
**Vocabulary.** aleatoric vs epistemic uncertainty; bootstrap; irreducible error; the Bayes error rate.

---

## Two destinations (same toolkit)
- **Sports / betting quant:** Steps 2 → 4 → 6 are the core loop — find edge, size it, vet new signal.
- **Quant finance / research:** identical machinery — log loss → Sharpe, edge → alpha, Kelly →
  position sizing; IC and orthogonality are used verbatim. This project is a legitimate on-ramp to either.

## Suggested order
**1 → 3 → 2 → 6 → 4 → 5 → 7.** Calibration + backtesting first (the foundation everything trusts),
then market/IC and signal research (the daily loop), then Kelly (decisions), then Bayesian + noise
floor (the deepest). Steps 4/5/7 are "to build" — do them in cowork; the data/skeleton is ready.

## One-line glossary
- **Proper scoring rule** — a loss minimised only by honest probabilities (log loss, Brier).
- **Calibration** — predicted probabilities match observed frequencies.
- **Information Coefficient (IC)** — rank corr of predicted score vs realised outcome.
- **Edge** — your probability minus the fair (de-vigged) market probability.
- **Orthogonality** — the part of a new feature not explained by existing features.
- **Walk-forward** — retrain on the past, test only on the unseen next period.
- **Kelly fraction** — the bet size that maximises long-run log-growth.
- **Shrinkage** — pulling noisy small-sample estimates toward a prior/mean.
- **Aleatoric vs epistemic** — irreducible randomness vs reducible model error.
