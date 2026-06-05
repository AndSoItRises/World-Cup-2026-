World Cup 2026 Model — Phase 6 Context Doc
Paste this at the start of the next conversation

---

## Project Summary
Building a World Cup 2026 match prediction model in Python. Jake is learning as we go — every step is explained in plain English before code is written. Jake runs everything himself in PowerShell (venv activated) and Cursor (code editor). We go one step at a time: explain → write code → Jake pastes into Cursor → Jake runs in PowerShell → paste output back → confirm → next step.

## How We Work Together
- Explain first, code second — always describe what a step does and why before showing code
- One step at a time — never jump ahead
- Jake runs everything himself — never auto-execute; always give PowerShell commands explicitly
- Questions are welcomed mid-step — answer them directly before continuing
- Full file replacements — when edits get complex, do Ctrl+A replace of the whole file
- PowerShell only ever runs one command: `python -m src.<module>.<script_name>`
- Never paste Python into PowerShell

## Project Location
`C:\Users\jakeh\OneDrive\Documents\Claude\Projects\World Cup 2026 Model`

## Tech Stack
Python 3.10+, pandas, numpy, scikit-learn, xgboost, lightgbm, scipy, statsmodels, matplotlib, plotly

## Folder Structure
```
data/
  raw/
    international_results/results.csv     (49,368 match results, 1872–present)
    FIFA_rankings_training/               (3 CSVs, ~200k ranking records, cols: rank, country_full, country_abrv, total_points, rank_date)
    current_fifa_rankings.csv             (211 rows, cols: rank, team_name, team_code, points)
    wc2026_fixtures.csv                   (104 matches, cols: match_id, date, stage, group, home_team, away_team, stadium, city, country, altitude_m, neutral)
  processed/
    train.csv                             (9,281 rows) ✅
    test.csv                              (2,257 rows) ✅
    train_features.csv                    (9,281 rows, 36 cols) ✅
    test_features.csv                     (2,257 rows, 36 cols) ✅
    cleaning_metadata.json                ✅
models/
  xgb_v1.json                            ✅
  training_report.json                   ✅
  dixon_coles_params.json                ✅
  lgbm_v1.txt                            ✅
  lgbm_report.json                       ✅
src/
  features/
    data_cleaning.py                      ✅
    feature_engineering.py               ✅
  models/
    train_xgb.py                         ✅
    dixon_coles.py                       ✅
    train_lgbm.py                        ✅
  utils/
    fetch_rankings.py
```

## Phase Status
- ✅ Phase 1: Environment setup (venv, folders, Git)
- ✅ Phase 2: Data acquisition (results CSV, FIFA rankings, fixtures, live rankings scraper)
- ✅ Phase 3: Data cleaning (`src/features/data_cleaning.py`)
- ✅ Phase 4: Feature engineering (`src/features/feature_engineering.py`)
- ✅ Phase 5: Model training (XGBoost, Dixon-Coles, LightGBM)
- 🔲 Phase 6: Ensemble + WC2026 predictions ← START HERE
- 🔲 Phase 7: Bracket simulator (Monte Carlo)
- 🔲 Phase 8: Visualization

---

## Phase 3 — Key Decisions
- Competitive match filter: kept 11,538 of 49,368 matches (dropped friendlies, etc.). Cutoff at 2002.
- Team name standardization across results/rankings CSVs
- Temporal train/test split: cutoff 2022-11-20 (Qatar WC start). Train: 9,281. Test: 2,257.
- Exponential decay weights: half-life 730 days. Saved as `sample_weight` in train.csv.
- Categorical encoding: tournament_tier (1–4), result (0=away win, 1=draw, 2=home win)
- Class weights: away_win 1.132, draw 1.566, home_win 0.677
- No SMOTE — class weights only

## Phase 4 — Features Built (24 engineered + 2 base = 26 total)
All rolling features use shift(1) — no leakage. Combined train+test for feature building, split back after.

| Feature | Notes |
|---|---|
| home_fifa_rank, away_fifa_rank | merge_asof by date, unranked → 150 |
| fifa_rank_diff | home_rank - away_rank |
| home/away_win_rate_5/10 | rolling win rate, last 5/10 matches |
| home/away_avg_goals_5/10 | rolling avg goals scored |
| home/away_avg_gd_5/10 | rolling avg goal difference |
| h2h_home_wins/draws/away_wins/total | cumulative H2H prior to match |
| h2h_home_win_rate | H2H rate, neutral prior 0.5 if no history |
| home/away_days_rest | days since last match, median-filled |
| is_knockout | 0 for all training data (no stage info in historical CSV) |
| altitude_m | 0 for training data, populated from fixtures in Phase 6 |
| tournament_tier | 1–4, already in train.csv |
| neutral | bool→int, already in train.csv |

## Phase 5 — Model Results

### Naive Baseline
- Always predict home win: **45.9% accuracy**

### XGBoost (xgb_v1.json)
- Test accuracy: **59.8%** | Log loss: 0.8814
- CV mean accuracy: 58.3% ± 3.8% (5 time-ordered folds, improving fold-over-fold ✅)
- Draw recall: **0.6%** — essentially never predicts draws
- Top features: fifa_rank_diff (42.6), h2h_home_win_rate (20.0), home_avg_gd_10 (10.6)

### Dixon-Coles (dixon_coles_params.json)
- Test accuracy: **58.3%** | Log loss: 0.8815
- Draw recall: **10.1%** — rho correction working as designed
- Home advantage multiplier: 1.3141
- Rho: -0.1731 (inflates 0-0 and 1-1, deflates 1-0 and 0-1)
- Convergence warning present but results are stable (identical across two runs)
- Attack ratings slightly off (Canada #1) — likely recency bias from decay weights, acceptable for V1

### LightGBM (lgbm_v1.txt)
- Test accuracy: **60.0%** | Log loss: 0.8835
- Draw recall: **0.6%** — same failure as XGBoost
- Best iteration: 54 (early stopped)
- Marginally higher accuracy than XGBoost, slightly worse log loss

### Summary Table
| Model | Test Acc | Log Loss | Draw Recall |
|---|---|---|---|
| Naive baseline | 45.9% | 1.059 | — |
| Dixon-Coles | 58.3% | 0.8815 | 10.1% |
| XGBoost | 59.8% | 0.8814 | 0.6% |
| LightGBM | 60.0% | 0.8835 | 0.6% |

---

## Phase 6 — Plan: Ensemble + WC2026 Predictions

### Step 1: Build ensemble
Combine all three models. Two approaches to try:
- **Simple weighted average**: e.g. XGB 40% + LGBM 40% + DC 20%. Weight DC higher if draw accuracy matters.
- **Meta-model (stacking)**: train a logistic regression on the three models' probability outputs using CV predictions. More principled but more complex.

Recommended starting point: weighted average first, then stacking if it's meaningfully better.

### Step 2: Predict WC2026 fixtures
Load `wc2026_fixtures.csv` (104 matches). For each match:
- Build the feature vector (same 26 features)
- Use current FIFA rankings (`current_fifa_rankings.csv`) for rank features
- Populate altitude_m from fixtures file
- Set is_knockout from stage column
- Run through ensemble → output win/draw/loss probabilities per match

### Step 3: Output
Save `data/processed/wc2026_predictions.csv` with columns:
match_id, date, stage, home_team, away_team, p_home_win, p_draw, p_away_win, predicted_result

### Key things to handle in Phase 6
- Some WC2026 teams may not be in training data (new qualifiers) → use FIFA rank fallback
- Rolling form for WC2026 teams needs to be computed from test set + recent matches
- is_knockout needs to be set correctly from fixtures stage column (Group Stage vs knockout rounds)
- altitude_m is real in fixtures — high-altitude venues: Mexico City (2240m), La Paz equivalent etc.

### V2 Notes (carry forward)
- Combine decay weights WITH tournament tier weights (multiply together)
- Decay half-life 730 days is arbitrary — tune via CV in V2
- Dixon-Coles attack ratings have recency bias — consider separate decay parameter for DC fit
- Draw prediction is the weakest point across all models — worth exploring dedicated draw classifier
