# World Cup 2026 Prediction Model

A machine learning model that predicts match outcomes and simulates the full WC2026 bracket using an ensemble of XGBoost, LightGBM, and Dixon-Coles — with ELO ratings, opponent-quality-weighted form, and a betting-market comparison.

---

## Results (V2)

| Model | Test Accuracy | Log Loss | Draw Recall |
|---|---|---|---|
| Naive baseline | 45.9% | 1.059 | — |
| Dixon-Coles | 60.7% | 0.8626 | 0.4% |
| XGBoost | 58.2% | 0.8635 | 26.1% |
| LightGBM | 58.6% | 0.8651 | 26.9% |
| **Ensemble** | **62.1%** | **0.8458** | 9.7% |

**V1 → V2 improvement:** accuracy 60.6% → 62.1%, log loss 0.8605 → 0.8458, draw recall 0.4% → 9.7%.

**Top tournament win probabilities (10,000 Monte Carlo simulations):**

| Team | FIFA Rank | Win % | vs Market (de-vig) |
|---|---|---|---|
| Spain | 2 | 13.9% | 15.5% |
| Brazil | 6 | 11.9% | 8.1% |
| Mexico | 15 | 10.4% | 1.1% |
| Argentina | 3 | 7.0% | 8.5% |
| England | 4 | 4.9% | 10.6% |
| France | 1 | 4.5% | 14.8% |

---

## What's New in V2

- **ELO ratings** — running ELO for every team across 49k matches (tiered K-factors 40/30/20), used as features and as the opponent-quality signal.
- **Opponent-quality-weighted form** — rolling stats weighted by opponent ELO, correcting the V1 confederation bias (CONCACAF teams inflated by weak schedules, CONMEBOL teams suppressed by hard ones). 41 features total (up from 26).
- **Draw fix** — gradient-boosted models were upweighted on draws (1.75×), lifting draw recall from a broken 0.6% to functional.
- **Hyperparameter tuning** — CV-tuned via TimeSeriesSplit: time decay disabled (longer memory improved log loss monotonically), shallower depth-3 trees. This also fixed the Argentina/Brazil underrating and reduced Canada's inflation.
- **Market divergence analysis** — live WC2026 odds de-vigged and compared to the model. Verdict: the model's biggest disagreements track its *known biases*, so no trustworthy betting edge yet.
- **Stacking meta-learner** — evaluated but not adopted; it couldn't beat the hand-set ensemble weights without zeroing draw recall.

---

## How It Works

### Data
- **Match results:** 49,368 international matches (1872–present), filtered to 11,538 competitive matches post-2002
- **FIFA rankings:** ~200k historical ranking records used to assign team strength at match time
- **WC2026 fixtures:** 104 matches across group stage and knockout rounds
- **Market odds:** 48-team WC2026 outright board (de-vigged for the divergence analysis)

> Raw data not included in this repo. Sources: [Kaggle international football results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017), FIFA ranking CSVs, manually assembled WC2026 fixtures, and a public odds board snapshot.

### Feature Engineering (41 features)
- FIFA rank differential + **ELO ratings** (home/away/diff) at match date
- Rolling win rate, avg goals, avg goal difference (last 5 and 10) — **simple and opponent-ELO-weighted**
- Head-to-head history (cumulative, no leakage)
- Days rest, altitude, tournament tier, neutral venue flag
- All rolling features use `shift(1)` — no data leakage

### Models
- **XGBoost** / **LightGBM** — multiclass classifiers (away / draw / home), depth-3, draw-upweighted
- **Dixon-Coles** — Poisson model with rho correction for low-scoring games
- **Ensemble** — weighted average (XGB 27.5% / LGBM 27.5% / DC 45%); validated near-optimal against a stacked meta-learner

### Monte Carlo Simulator
Simulates the full 48-team bracket 10,000 times: Poisson score simulation for group-stage tiebreakers, top 2 per group + best 8 third-place teams advance, ensemble probabilities for knockout rounds.

---

## Project Structure

```
data/
  raw/              # results, rankings, fixtures, market odds
  processed/        # cleaned CSVs, features, predictions, tournament + divergence probs
models/             # trained models, reports (tuning, market, stacking)
outputs/viz/        # 4 charts
src/
  features/         # data_cleaning, feature_engineering, elo
  models/           # train_xgb, train_lgbm, dixon_coles, ensemble, predict_wc2026,
                    # monte_carlo, tune_draw_weight, tune_hyperparams,
                    # market_divergence, stacking
  visualization/    # charts
```

---

## How to Run

```bash
# 1. Set up environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Data + features (ELO must run before feature engineering)
python -m src.features.data_cleaning
python -m src.features.elo
python -m src.features.feature_engineering

# 3. Train models
python -m src.models.dixon_coles
python -m src.models.train_xgb
python -m src.models.train_lgbm

# 4. Ensemble + WC2026 predictions
python -m src.models.ensemble
python -m src.models.predict_wc2026

# 5. Monte Carlo bracket simulation
python -m src.models.monte_carlo

# 6. Analysis + visualization
python -m src.models.market_divergence
python -m src.visualization.charts

# Optional: tuning / stacking experiments
python -m src.models.tune_hyperparams
python -m src.models.stacking
```

> On Windows, set `PYTHONUTF8=1` so the console renders the scripts' output.

---

## Known Limitations (V3 Roadmap)

- **Mexico / CONCACAF inflation** — model gives Mexico 10.4% vs a 1.1% market price; residual strength-of-schedule bias not fully corrected by ELO weighting.
- **European powers underrated** — France (4.5% vs 14.8% market) and Portugal underrated; no-decay over-rewards recent CONMEBOL dominance.
- **No demonstrated betting edge** — divergences from market track known model biases, so Kelly bet sizing is deferred until biases are fixed.
- **ELO K-factors** — still hardcoded (40/30/20), never CV-tuned (blocked on a faster feature rebuild).

---

## Tech Stack

Python 3.10+ · pandas · numpy · scikit-learn · XGBoost · LightGBM · scipy · matplotlib
