# WC 2026 Prediction Model — Key Insights

> **Model status:** V4 is the final validated model. V5 confirmed the ceiling — all accessible
> levers tested and exhausted. Further gains require new data (historical international odds;
> full-universe historical squad values), not new model cleverness.

---

## Quick Reference

| Version | Accuracy | Log Loss | Draw Recall | Key Addition |
|---------|----------|----------|-------------|--------------|
| V1 | 60.6% | 0.8605 | 0.4% | Baseline pipeline |
| V2 | 62.1% | 0.8458 | 9.7% | ELO, opponent-weighted form, Dixon-Coles |
| V3 | 61.7% | 0.8462 | 8.7% | Data bug fixes, ceiling confirmed |
| **V4** | **62.0%** | **0.8461** | **9.1%** | **Squad strength + depth (FIFA/EA ratings)** |

**WC2022 backtest log loss: 1.0447** (V4) vs 1.0608 (V3) — 24.3% better than a naive baseline.
**Model↔market correlation: 0.838** (V4) vs 0.665 (V3).

---

## Charts

| # | Chart | What it shows |
|---|-------|---------------|
| [01](outputs/key_insights/01_version_progression.png) | Version Progression | Accuracy, log loss, and draw recall across V1→V4 |
| [02](outputs/key_insights/02_tournament_win_probs.png) | Tournament Win Probabilities | V4 model top-20 win probabilities |
| [03](outputs/key_insights/03_stage_heatmap.png) | Stage Heatmap | Probability of reaching each round, top 16 teams |
| [04](outputs/key_insights/04_market_divergence_v4.png) | Model vs Market (V4) | Where the model disagrees with betting markets |
| [05](outputs/key_insights/05_v3_v4_bracket_shift.png) | V3→V4 Bracket Shift | Who gained and who lost when squad features were added |
| [06](outputs/key_insights/06_feature_importance_v4.png) | Feature Importance (V4) | XGBoost gain scores, highlighting new squad features |
| [07](outputs/key_insights/07_bias_resolution_v3_v4.png) | Bias Resolution | V3 vs V4 model-vs-market edge — how biases changed |

---

## Current Forecast (V4 Model, Group Stage Live)

| Team | Win % | Group Advance % | Final % |
|------|-------|-----------------|---------|
| Spain | 15.9% | 92.6% | 24.4% |
| Mexico | 9.8% | 97.0% | 17.3% |
| England | 7.5% | 79.6% | 12.9% |
| Argentina | 7.3% | 83.6% | 12.8% |
| Brazil | 6.2% | 91.6% | 11.6% |
| Japan | 6.1% | 94.1% | 11.4% |
| France | 4.5% | 81.0% | 8.9% |
| Iran | 3.9% | 86.3% | 8.0% |
| USA | 3.7% | 94.5% | 8.0% |
| Netherlands | 3.5% | 74.2% | 7.2% |

*Source: `data/processed/tournament_probs_live.csv` — updated by `src/models/live_update.py` after each matchday.*

---

## Key Findings

### 1. ELO is the dominant signal — by a wide margin
`elo_diff` (feature importance rank #1) and `fifa_rank_diff` (#2) together account for the majority of
predictive gain in XGBoost. Everything else is marginal by comparison. This reflects a fundamental truth
about international football: strength differential predicts outcomes, but not precisely.

### 2. Squad quality was the proven missing signal — and we found it
V4's diagnostic confirmed that squad market value carried +0.178 incremental R² beyond existing features
(Transfermarkt route was data-blocked). The solution was FIFA/EA video-game ratings (2005→FC26), which
are freely available, historically deep, and leakage-safe via backward asof merge.

The squad features (`squad_strength_diff` at rank #3, `squad_depth_diff` at #5) landed in the top 5
most important features. The signal is concentrated on matches where both teams have coverage (~34% of
the full training set, but all WC2026 ties).

### 3. The CONCACAF inflation problem was real and partially fixed
V3 showed Mexico overrated by +10.0pp vs market. Root cause: the model couldn't see squad quality, so
teams that accumulate wins against weak CONCACAF opponents inflate their ELO and form features without
reflecting their true talent level.

V4 reduced the Mexico bias to +6.1pp and improved the model↔market correlation from 0.665 to 0.838.
The bias is reduced, not closed — ELO still dominates and squad value is a complementary prior.

**Teams most corrected downward by V4:** Panama (−4.6pp), Australia (−4.2pp), Iran (−4.0pp),
Ecuador (−3.6pp), Uzbekistan (−3.3pp), Korea Republic (−3.2pp).

**Teams most corrected upward:** Belgium (+5.7pp), Turkey (+4.7pp), Czechia (+4.6pp), France (+4.3pp),
Portugal (+3.9pp).

### 4. Draws are systematically overpredicted
The model predicts 25.9% draws vs 22.2% actual — a persistent 3.7pp overestimate. This is worst in
CONCACAF (27.9% predicted vs 17.2% actual). Dixon-Coles corrects for the empirical low-score
overfrequency, but draw recall is capped at ~9% across all versions — this appears to be a structural
ceiling given the available features.

### 5. The model is now largely aligned with the market
V4's model↔market correlation of 0.838 means the bulk of pricing signal has been captured. The
remaining divergences (Mexico still +6pp, France still −7.7pp) are concentrated bets the model is
making against the market. They should be treated as known biases, not actionable edges — no
out-of-sample historical odds backtest has been run to validate them.

### 6. The practical ceiling has been reached
V5 tested every remaining accessible lever under validate-or-cut:
- **Confederation/schedule-strength feature:** CUT — squad value already subsumes the regional proxy.
- **Extend squad coverage (FIFA23/FC24/FC25):** DATA-BLOCKED — those editions aren't freely hosted.
  Also wouldn't fix the structural ~66% coverage gap from minnow nations.
- **Ensemble reweight:** KEEP fixed weights — tuned weights overfit (one fold zeroed LGBM).
- **Historical odds edge backtest:** DATA-GATED — free international historical odds don't exist.

The model did not change in V5. The convergence of all levers to negative/blocked is itself the evidence
the ceiling has been reached with accessible data.

---

## Model Architecture

```
Data pipeline:
  international_results.csv (49k matches, 1872–present)
  → data_cleaning.py (standardize names, fix rank corruption)
  → elo.py (tiered K-factor ELO: 40/30/20 by tournament type)
  → feature_engineering.py (41 base features + 7 squad features)
  → build_squad_strength.py (FIFA/EA ratings 2005→FC26, asof merge)

Models (ensemble 0.275 XGBoost / 0.275 LightGBM / 0.45 Dixon-Coles):
  - XGBoost / LightGBM: depth-3, draw-upweighted 1.75×, temporal CV split
  - Dixon-Coles: Poisson goal model with low-score correction (rho parameter)
  - Fixed ensemble weights validated near-optimal vs stacked meta-learner

Simulation:
  monte_carlo.py → 10,000 bracket simulations → tournament_probs_live.csv
  live_update.py → ingest actual results → update ELO → re-simulate
```

**48 features total:** ELO diff, FIFA rank diff, rolling form (5/10 games, raw + opponent-ELO-weighted),
H2H record, days rest, altitude, tournament tier, neutral ground, is_knockout,
squad strength + depth (per team, diff, coverage flag).

---

## Non-Negotiables (Methodology Guardrails)

These rules were established in V1 and held through V5. They are the reason the model is trustworthy:

1. **No leakage.** All features use only information available before the match date.
2. **Temporal train/test split.** CV splits on time boundaries, never random shuffle.
3. **Validate-or-cut.** Every feature retrained and compared to prior CV log loss. If no improvement → cut.
4. **Log loss as primary metric,** not accuracy. Log loss penalizes confident wrong predictions.
5. **Team name audits** for every new data source — name mismatches cause silent NaN bugs.

---

## What's Left (Data-Gated, Not Cleverness-Gated)

| Item | What it unlocks | Gate |
|------|-----------------|------|
| Historical international bookmaker odds | True out-of-sample edge test + Kelly bet sizing | Paid/scrape |
| Full-universe historical squad values | Lift squad coverage past ~34%, fix minnow gap | Transfermarkt history / full FIFA editions |

Both are data problems. The model machinery to use them is already built:
`market_backtest.py` (IC harness), `signal_test.py` (orthogonality gate), `bet_sim.py` (Kelly — Step 4, not yet built).

---

## Quant Learning Path Status

Steps built as part of V4 — runnable scripts, not just theory:

| Step | Concept | Script | Status |
|------|---------|--------|--------|
| 1 | Calibration & proper scoring | `src/models/calibration.py` | ✅ Built |
| 2 | Market as benchmark / IC | `src/models/market_backtest.py` | ✅ Built |
| 3 | Walk-forward backtesting | `src/models/backtest.py` | ✅ Built |
| 4 | EV & Kelly criterion | `src/models/bet_sim.py` | ⬜ Not yet |
| 5 | Bayesian / hierarchical model | — | ⬜ Not yet |
| 6 | Signal orthogonality / IC | `src/models/signal_test.py` | ✅ Built |
| 7 | Aleatoric vs epistemic uncertainty | — | ⬜ Not yet |

---

## Repository Structure

```
World-Cup-2026-/
├── CONTEXT_V4.md              ← Operational state + learning path
├── CONTEXT_V5.md              ← V5 decisions + final verdict
├── CONTEXT_QUANT_TRANSITION.md ← Full intellectual history V1→V4
├── KEY_INSIGHTS.md            ← This document
├── data/
│   ├── raw/                   ← Source data (results, rankings, fixtures, odds)
│   └── processed/             ← Engineered features, ELO, tournament probs
├── models/                    ← Saved model artifacts (xgb/lgbm/dc, v1–v4 + prod)
├── outputs/
│   ├── key_insights/          ← Charts for this document
│   └── viz/                   ← Original V3 charts
└── src/
    ├── features/              ← data_cleaning, elo, feature_engineering, build_squad_strength
    ├── models/                ← All model scripts + quant learning stack
    └── visualization/         ← charts.py
```

---

*Document version: June 2026 — reflects V4 final model + V5 ceiling confirmation.*
*For full session history: read CONTEXT_V4.md (operational) then CONTEXT_QUANT_TRANSITION.md (intellectual).*
