# WC2026 Model — Quantitative Transition Document
### From Heuristic Baseline to Algorithmic Generation

*Purpose: Onboard any future model/agent/collaborator into the full intellectual history of this project.
Written to be educational — each method is explained, not just named.*

---

## Part 1 — What We Built and Why (V1 → V4)

This project started as a structured learning exercise and became a genuine prediction system. Each version
had a specific intellectual mandate.

### V1 — Baseline (Prove the pipeline works)

**What it does:** Trains a multiclass classifier (XGBoost + LightGBM) on ~45k historical international
football results (1872–present, Kaggle `international_results.csv`). Predicts Win/Draw/Loss probabilities
for each WC2026 match. Runs 10,000 Monte Carlo simulations of the bracket.

**Features at V1:** FIFA ranking difference, simple rolling form (win% last 5/10 games), head-to-head
record, neutral ground flag, tournament stage, rest days.

**Key quant concepts introduced:**
- **Multiclass log loss** as the evaluation metric. Unlike accuracy, log loss penalizes confident wrong
  predictions — a model that says "90% home win" and is wrong is punished more than one that says "55%."
  This is the right metric for probability prediction, not accuracy.
- **Temporal train/test split.** Standard cross-validation randomly shuffles data. That would let a model
  trained on 2022 data predict a 2015 match — information leakage. We split on time: train on everything
  before a cutoff, test on everything after.
- **Class imbalance.** International football results are roughly 45/25/30 Win/Draw/Loss. If we don't
  correct for this, the model learns to almost never predict draws (which minimizes raw error but is wrong).
  We used class weights, not SMOTE — SMOTE (Synthetic Minority Oversampling) creates synthetic draw
  examples by interpolating features, but football draws don't interpolate cleanly.

**V1 result:** 60.6% accuracy, log loss 0.8605, draw recall ~0.4% (basically never predicted draws).

---

### V2 — Make It Intelligent (Add real signal)

**Mandate:** V1's features were naive. FIFA ranking is slow-moving and politically influenced. Simple form
doesn't account for who you beat. V2 added economically meaningful features.

**New features:**
- **Elo rating** (computed from scratch on the full results history). Elo is a dynamic rating system
  originally designed for chess. Each team has a rating; when a stronger team beats a weaker one, few
  points transfer. An upset transfers many. The formula: `new_rating = old_rating + K * (result - expected)`,
  where `expected = 1 / (1 + 10^((opponent_elo - own_elo)/400))`. Elo compounds history in a principled
  way — it's more informative than FIFA rank.
- **Opponent-ELO-weighted form.** Instead of "won 4 of last 5," we weight each result by the opponent's
  Elo at the time. Beating Brazil counts more than beating Andorra.
- **Dixon-Coles Poisson model.** Instead of classifying outcomes directly, we model goals scored and
  conceded as Poisson-distributed random variables. Each team has an attack parameter (λ) and a defense
  parameter (μ). Expected goals for team A vs team B = λ_A * μ_B. We then integrate over the Poisson
  probability mass to get P(Win), P(Draw), P(Loss). Dixon-Coles adds a correction for the empirically
  observed overfrequency of 0-0 and 1-0 draws — the Poisson naive model underestimates these scorelines.
- **Draw upweighting (1.75×).** We upweighted the draw class in the loss function. This is a pragmatic
  lever when you know a class is systematically underpredicted.

**V2 result:** 62.1% accuracy, log loss 0.8458, draw recall 9.7%. The Elo feature was the single
largest driver. Dixon-Coles improved calibration on close matches.

---

### V3 — Find the Ceiling (Prove the model is at its limit)

**Mandate:** We had suspicions the model was biased toward CONCACAF teams (especially Mexico) and
underrating European powers (France, Portugal). V3's job was to confirm or deny this, fix data bugs,
and rigorously test whether more feature engineering could close the gap.

**What V3 proved doesn't work:**
- `conf_match_pct` (share of matches against same confederation): ~93% for nearly everyone — no variance,
  no signal.
- Partial time decay (exponential downweighting of old matches): amplified CONCACAF inflation rather
  than fixing it. This is a critical lesson — decay rewards teams for playing lots of games, and CONCACAF
  has a lot of qualifying matches against weak opponents.
- Dedicated draw classifier (binary classifier on top of the main model): couldn't exceed ~9% draw recall
  ceiling without simultaneously overforecasting draws in matches that weren't close.
- Hyperparameter tuning: marginal (depth-3 trees already near-optimal; deeper trees overfitted).

**Key quant concept introduced:**
- **Residual analysis.** After comparing model predictions vs betting market odds, we computed
  `edge = model_prob - market_prob` for each team's tournament win probability. Plotting this revealed
  the systematic biases clearly: Mexico edge = +10%, France edge = −9%. The market is treating Transfermarkt
  squad value as a strong prior; our model can't see it.
- **Ensemble validation.** We tested a stacked meta-learner (a second-level model trained on the outputs
  of XGB, LGBM, and Dixon-Coles) against the fixed-weight ensemble (0.275/0.275/0.45). The meta-learner
  didn't improve CV log loss. This told us the bottleneck is features, not blending.

**V3 result:** 61.7% accuracy, log loss 0.8462, draw recall 8.7%. Nearly identical to V2 — the model
hit its ceiling for the current feature set.

---

### V4 — Find the New Signal (Squad value diagnostic)

**Mandate:** If the model is at its ceiling, the only path forward is new information, not smarter
algorithms. V4's first act was to prove (not assume) that squad market value carries information beyond
what we already have.

**What the diagnostic showed:**
- `corr(log squad value, market win%) = +0.66` vs `corr(log squad value, model win%) = +0.42`
  The market tracks talent far better than our model.
- **Incremental R² = +0.178.** Adding squad value to a regression predicting market probability (with
  model probability already included) explains an additional 17.8% of variance. This is not redundant
  with ELO/form — it's genuinely new signal.
- `corr(squad value, edge) = −0.40` — high-value squads (France, England, Portugal) are systematically
  underrated by our model. Over-rated by value gap: Mexico, Iran, Japan, Korea, Australia.

**Key quant concepts:**
- **Incremental R² / partial correlation.** The right way to ask "does feature X add anything beyond
  features I already have?" is not to compute corr(X, target) — that measures raw correlation. Instead,
  fit a regression with the existing features, then add X, and observe how much R² improves. If it
  doesn't improve, X is redundant (already explained by what you have).
- **Why the model is blocked:** To use squad value as a validated training feature, we need historical
  squad values for each team at each match date going back to 2002. We don't have that. Using only
  current values in training would introduce leakage (2022 market values shouldn't predict 2010 match
  outcomes). This is a data problem, not a modeling problem.

**V4 status:** P1 (diagnostic) complete. P2 (acquire historical data) is the critical path blocker.

---

## Part 2 — Quantitative Methods Map

This section catalogs every method in the model, organized from foundational to advanced.
Think of it as a curriculum — you can locate yourself on the map and know what's next.

### Tier 1 — Foundations (you've built all of these)

| Method | What it is | Where used |
|---|---|---|
| **Multiclass classification** | Predict one of N categories (Win/Draw/Loss) with probabilities | XGBoost, LightGBM |
| **Log loss (cross-entropy)** | Measures calibration quality of probability predictions | Primary evaluation metric |
| **Temporal train/test split** | Prevents future information from leaking into training | `data_cleaning.py` |
| **Class weights** | Upweight underrepresented classes in the loss function | draw_weight = 1.75× in XGB/LGBM |
| **Gradient boosting** | Sequentially fit weak learners (decision trees) to residuals | XGBoost, LightGBM |
| **Cross-validation** | Evaluate model on multiple train/test splits to get stable estimates | `StratifiedKFold` on features |
| **Elo rating system** | Dynamic paired-comparison rating with mean reversion | `elo.py` |
| **Rolling window statistics** | Compute metrics over recent N matches (form) | `feature_engineering.py` |
| **Ensemble / model averaging** | Combine multiple models' predictions to reduce variance | `ensemble.py` (0.275/0.275/0.45) |
| **Monte Carlo simulation** | Run the bracket 10k times, sampling from match probabilities | `monte_carlo.py` |

### Tier 2 — Intermediate (partially built; some gaps)

| Method | What it is | Status |
|---|---|---|
| **Poisson regression (Dixon-Coles)** | Model goal counts as Poisson RVs; derive match probs | ✅ Built |
| **Exponential decay weighting** | Downweight old matches; recent form matters more | ⚠️ Tested, caused inflation — not in final model |
| **Stacked meta-learning** | Train a second model on first-layer outputs | ✅ Tested, outperformed by fixed weights — excluded |
| **Residual analysis** | Diagnose systematic model error vs market | ✅ Used in V3/V4 |
| **Incremental R² / partial correlation** | Measure whether a new feature adds signal beyond existing ones | ✅ Used in squad-value diagnostic |
| **Hyperparameter optimization** | Grid/random search over model hyperparameters | ✅ Used in `tune_hyperparams.py` |

### Tier 3 — Advanced (the next frontier)

These are the methods that separate a manual model from a genuinely algorithmic one.

| Method | What it is | Why it's the next step |
|---|---|---|
| **Bayesian inference (PyMC / Stan)** | Model parameters as probability distributions, not point estimates. Produces full posterior distributions over match outcomes — not just "France wins 68%" but "with what confidence?" | Replaces point-estimate XGB predictions with uncertainty quantification. Better for close matches and bracket propagation. |
| **Dixon-Coles with time-varying attack/defense** | Extend the Poisson model so team strength parameters drift over time (using a state-space model or rolling MLE). | Elo is already time-varying but coarse. DC with time-varying params captures recent form at the goals level. |
| **Bradley-Terry model** | A paired-comparison model that directly estimates team strength as a latent variable from the full match graph. More principled than Elo for sparse head-to-head data. | Better theoretical grounding than Elo; accounts for the network structure of who-beat-whom. |
| **Latent factor models / matrix factorization** | Decompose the match result matrix into latent team-strength vectors. Similar to how collaborative filtering works in recommendation systems. | Can extract signal from sparse matchups that Elo misses. |
| **Conformal prediction** | Produce valid prediction intervals (not just point probabilities) with statistical coverage guarantees, regardless of model architecture. | Allows honest communication of model uncertainty: "we're 90% confident this team's win prob is between 28% and 51%." |
| **Calibration (Platt scaling / isotonic regression)** | Post-hoc adjustment of raw model probabilities to make them match empirical frequencies. | If the model says "70% win," it should win 70% of those matches. Calibration measures and fixes this. |
| **SHAP values** | Game-theoretic feature attribution — how much did each feature push this specific prediction? | Goes beyond global feature importance. Lets you say "for Mexico vs Argentina, the draw prediction is driven by H2H, not form." |
| **Hierarchical models** | Pool information across groups (confederations, tournaments) while allowing group-level variation. Shrinks estimates toward confederation averages when data is sparse. | Directly addresses the CONCACAF inflation problem — teams with weak qualifying opponents get their strength estimates shrunk toward a confederation baseline. |
| **Neural network (MLP or LSTM)** | Learn nonlinear feature interactions automatically; LSTM can process match sequences as time series. | After sufficient data engineering, a deep model can capture interactions the gradient boosting trees miss. High complexity, requires careful regularization. |

---

## Part 3 — The Algorithmic Generation Transition

### What "algorithmic generation" means

Right now the model is **manual and supervised.** Every decision — which features to try, how to weight
the ensemble, which matches to include — was made by a human and hardcoded. The model can retrain on
new data, but it doesn't learn how to improve itself.

**Algorithmic generation** means the model architecture, feature set, and weights are determined by
optimization, not by hand. The human defines the objective (minimize log loss on held-out data) and the
search space (what features are allowed, what model families are in scope). The algorithm finds the
configuration.

### Phase Map — Manual → Algorithmic

**Stage 0 (current):** Hand-engineered features → XGBoost/LGBM → fixed ensemble weights.
Every feature is coded by a human. Ensemble weights chosen by intuition + validation.

**Stage 1 — Automated feature selection.** Use SHAP values to identify which of the 41 current
features are actually carrying weight. Drop the zero-signal ones. Use recursive feature elimination (RFE)
to find the minimum feature set that preserves CV log loss. This is already within reach.

**Stage 2 — Automated hyperparameter optimization.** Replace the manual grid search with Bayesian
optimization (Optuna). Bayesian HPO builds a probabilistic model of the loss landscape and samples from
promising regions — far more efficient than grid search. This is the canonical next step after building
a baseline.

**Stage 3 — Automated ensemble weighting.** Instead of fixed 0.275/0.275/0.45, train a meta-learner
on out-of-fold predictions. The stacked meta-learner (tested in V3 and excluded because it didn't help)
failed because the base models weren't sufficiently diverse. If we add a Bayesian model as a third leg,
the diversity increases and stacking becomes worthwhile.

**Stage 4 — Feature store + pipeline automation.** Historical squad values, live match results, ranking
updates, and injury data become inputs to a scheduled pipeline. After each WC2026 matchday, the pipeline:
(1) ingests results, (2) updates ELO, (3) retrains, (4) re-runs Monte Carlo, (5) outputs new tournament
probabilities. This is the live-prediction product.

**Stage 5 — Probabilistic model (Bayesian / Bradley-Terry).** Replace point-estimate gradient boosting
with a full probabilistic model. Predictions come with credible intervals. Bracket simulation samples
from the posterior distribution of team strength, not just from the point-estimate win probabilities.
This is where the model becomes genuinely research-grade.

---

## Part 4 — Quantitative Learning Path

If you want to go deep on the methods underpinning this model, here's a structured path.
Each entry is ordered: understand the concept → see it in the code → know what to read.

### Elo and Rating Systems
- **Core concept:** Bayesian updating of a latent strength variable from paired comparisons.
- **In the code:** `src/features/elo.py` — the `update_elo()` function is the whole model.
- **Read:** Arpad Elo's original paper, or the Wikipedia derivation. Then read Glicko-2 (adds rating
  deviation, which Elo lacks).

### Poisson Models for Count Data
- **Core concept:** When outcomes are counts (goals), Poisson regression models the rate (expected goals)
  as a function of covariates.
- **In the code:** `src/models/dixon_coles.py` — MLE over the attack/defense parameters.
- **Read:** Dixon & Coles (1997), "Modelling Association Football Scores and Inefficiencies in the Football
  Betting Market." This is the canonical paper.

### Gradient Boosting
- **Core concept:** Build an ensemble of weak learners (shallow trees) by sequentially fitting each to the
  residuals of the ensemble so far. XGBoost adds L1/L2 regularization and second-order gradient approximation.
- **In the code:** `src/models/train_xgb.py` — pay attention to `scale_pos_weight` and `eval_metric`.
- **Read:** Chen & Guestrin (2016) XGBoost paper. Then Friedman (2001) for the mathematical foundation.

### Log Loss and Calibration
- **Core concept:** Log loss = `−(y * log(p) + (1−y) * log(1−p))`. It punishes confident wrong
  predictions exponentially. A model with 90% accuracy but poor calibration can have worse log loss
  than a model with 85% accuracy but well-calibrated probabilities.
- **In the code:** `src/models/validate_v3.py` — the CV loop computes log loss per fold.
- **Read:** Niculescu-Mizil & Caruana (2005), "Predicting Good Probabilities with Supervised Learning."

### Monte Carlo Simulation
- **Core concept:** When you can't compute a probability analytically (e.g., "what's the probability
  France wins the tournament?"), simulate thousands of runs sampling from match-level probabilities.
  The empirical frequency over runs approximates the true probability.
- **In the code:** `src/models/monte_carlo.py` — the outer loop runs 10,000 bracket simulations.
- **Read:** Chapter 1 of "Monte Carlo Statistical Methods" by Robert & Casella is the rigorous treatment.
  For football specifically, read Groll et al. (2018), "Prediction of the FIFA World Cup 2018."

### Bayesian Inference (next step)
- **Core concept:** Instead of finding the single best parameters (MLE), compute the full posterior
  distribution P(parameters | data) using Bayes' theorem. Uncertainty in parameters propagates to
  uncertainty in predictions.
- **In the code:** Not yet built — this is the Tier 3 frontier.
- **Read:** McElreath, *Statistical Rethinking* (best practical introduction). Then Davidson-Pilon,
  *Probabilistic Programming & Bayesian Methods for Hackers* (free online, code-first).

### SHAP Values
- **Core concept:** Shapley values from cooperative game theory, applied to ML. For a specific prediction,
  each feature's contribution is the average marginal contribution across all possible orderings of feature
  coalitions. Unlike simple feature importance, SHAP is local (per-prediction) and has a firm theoretical
  basis.
- **In the code:** Not yet built — straightforward to add (`pip install shap`, then wrap the XGB model).
- **Read:** Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions."

---

## Part 5 — Current State and What's Next

### Where V4 stands (June 2026)
- **Working model:** 62.1% accuracy, log loss 0.8458, draw recall ~9%. Ensemble of XGB + LGBM + Dixon-Coles.
- **Proven missing signal:** Squad market value carries +0.178 incremental R² beyond existing features.
- **Blocker:** Historical squad values (2002–2022) needed to backtest feature against training matches.
  Without this, squad value can't be a validated model feature (only a forward-looking manual adjustment,
  which we rejected as arbitrary).
- **Bias signature:** Mexico +10% over market, France −9%, Portugal −7%. CONCACAF overrated, Euro powers
  underrated. This is talent-blindness — the model can't see squad quality.

### Decision point
The project faces a fork:

**Option A — Acquire historical squad-value data.** Scrape or source Transfermarkt historical snapshots
(archived pages, Wayback Machine, third-party datasets). Join to training matches. Validate squad value
feature in cross-validation. This is the only path to an improved validated model. Medium-high effort.

**Option B — Forward-only product.** Accept v3.0 as the validated model. For WC2026 specifically,
incorporate squad-value adjustments as a manually-documented post-processing step (not part of the
trained model, clearly labeled as analyst overlay). Build the live pipeline (daily retrain, Monte Carlo
refresh after each matchday, visualization).

**Option C — Architecture shift.** Pivot to a Bayesian or Bradley-Terry model. This doesn't solve the
squad-value data problem, but it produces better-calibrated uncertainty intervals and is a significant
learning milestone. Can be done with existing data.

### Suggested next sequence (purely algorithmic learning path)
1. Add SHAP values to the existing model — one afternoon of work, immediate interpretability gain.
2. Implement Optuna-based Bayesian hyperparameter optimization — replaces the manual grid search.
3. Build the live retrain pipeline — after each WC2026 matchday, auto-retrain and re-simulate.
4. (If historical squad data is sourced) — add squad value feature, validate, close V4.
5. Implement a Bradley-Terry model as a parallel model for V5 — compare to XGB ensemble on WC2022 backtest.

---

*Document version: CONTEXT_QUANT_TRANSITION.md — written June 2026*
*For session handoff, read CONTEXT_V4.md first (operational state), then this document (intellectual history).*