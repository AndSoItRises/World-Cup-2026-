# World Cup 2026 Prediction Model

A machine learning model that predicts match outcomes and simulates the full WC2026 bracket using an ensemble of XGBoost, LightGBM, and Dixon-Coles.

---

## Results (V1)

| Model | Test Accuracy | Log Loss | Draw Recall |
|---|---|---|---|
| Naive baseline | 45.9% | 1.059 | — |
| Dixon-Coles | 58.3% | 0.8815 | 10.1% |
| XGBoost | 59.8% | 0.8814 | 0.6% |
| LightGBM | 60.0% | 0.8835 | 0.6% |
| **Ensemble** | **60.6%** | **0.8605** | 0.4% |

**Top tournament win probabilities (10,000 Monte Carlo simulations):**

| Team | Win % | Final % |
|---|---|---|
| Spain | 12.2% | 20.1% |
| USA | 11.5% | 19.6% |
| Mexico | 10.0% | 17.5% |
| England | 7.2% | 13.3% |
| France | 6.2% | 11.5% |

---

## How It Works

### Data
- **Match results:** 49,368 international matches (1872–present), filtered to 11,538 competitive matches post-2002
- **FIFA rankings:** ~200k historical ranking records used to assign team strength at match time
- **WC2026 fixtures:** 104 matches across group stage and knockout rounds
- **Current FIFA rankings:** Used for WC2026 feature vectors

> Raw data not included in this repo. Sources: [Kaggle international football results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017), FIFA ranking CSVs, and manually assembled WC2026 fixture list.

### Feature Engineering (26 features)
- FIFA rank differential at match date
- Rolling win rate, avg goals, avg goal difference (last 5 and 10 matches)
- Head-to-head history (cumulative, no leakage)
- Days rest, altitude, tournament tier, neutral venue flag
- All rolling features use `shift(1)` — no data leakage

### Models
- **XGBoost** — multiclass classifier (away win / draw / home win)
- **LightGBM** — same target, direct XGB comparison
- **Dixon-Coles** — Poisson-based statistical model with rho correction for low-scoring games
- **Ensemble** — weighted average (XGB 27.5% / LGBM 27.5% / DC 45%)

### Monte Carlo Simulator
Simulates the full WC2026 bracket 10,000 times:
- Group stage: Poisson score simulation for accurate tiebreakers
- Top 2 per group + best 8 third-place teams advance (WC2026 48-team format)
- Knockout rounds: ensemble probabilities pre-computed for all team pairs

---

## Project Structure

```
data/
  processed/        # Cleaned CSVs, features, predictions, tournament probs
models/             # Trained model files (XGB, LGBM, Dixon-Coles)
outputs/viz/        # Charts
src/
  features/
    data_cleaning.py
    feature_engineering.py
  models/
    train_xgb.py
    dixon_coles.py
    train_lgbm.py
    ensemble.py
    monte_carlo.py
    predict_wc2026.py
  visualization/
    charts.py
```

---

## How to Run

```bash
# 1. Set up environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Data cleaning
python -m src.features.data_cleaning

# 3. Feature engineering
python -m src.features.feature_engineering

# 4. Train models
python -m src.models.train_xgb
python -m src.models.dixon_coles
python -m src.models.train_lgbm

# 5. Ensemble + WC2026 predictions
python -m src.models.ensemble
python -m src.models.predict_wc2026

# 6. Monte Carlo bracket simulation
python -m src.models.monte_carlo

# 7. Visualization
python -m src.visualization.charts
```

---

## Known Limitations (V2 Roadmap)

- **Strength of schedule** — rolling form doesn't adjust for opponent quality, inflating CONCACAF teams (USA, Mexico)
- **Draw prediction** — all ML models predict draws at <1% recall; dedicated draw classifier planned
- **Decay half-life** — hardcoded at 730 days; should be tuned via cross-validation
- **Argentina underrated** — likely a CONMEBOL qualifying artifact; related to strength-of-schedule issue

---

## Tech Stack

Python 3.10+ · pandas · numpy · scikit-learn · XGBoost · LightGBM · scipy · matplotlib
