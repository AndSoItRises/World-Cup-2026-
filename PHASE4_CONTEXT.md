# World Cup 2026 Model — Phase 4 Context Doc
_Paste this at the start of the next conversation_

---

## Project Summary
Building a World Cup 2026 match prediction model in Python. Jake is learning as we go — every step is explained in plain English before code is written. Jake runs everything himself in PowerShell (venv activated) and Cursor (code editor). We go one step at a time: explain → write code → Jake pastes into Cursor → Jake runs in PowerShell → paste output back → confirm → next step.

---

## How We Work Together
- **Explain first, code second** — always describe what a step does and why before showing code
- **One step at a time** — never jump ahead
- **Jake runs everything himself** — never auto-execute; always give PowerShell commands explicitly
- **Questions are welcomed mid-step** — Jake asks conceptual questions frequently; answer them directly before continuing
- **Full file replacements** — when edits get complex, do Ctrl+A replace of the whole file to avoid Cursor autocomplete issues
- **PowerShell only ever runs one command**: `python -m src.features.<script_name>`
- **Never paste Python into PowerShell** — Jake did this once accidentally; remind him if it seems like it's happening again

---

## Project Location
`C:\Users\jakeh\OneDrive\Documents\Claude\Projects\World Cup 2026 Model`

## Tech Stack
Python 3.10+, pandas, numpy, scikit-learn, xgboost, lightgbm, statsmodels, matplotlib, plotly

## Folder Structure
```
data/
  raw/
    international_results/results.csv     (49,368 match results, 1872–present)
    FIFA_rankings_training/               (3 CSVs, ~200k ranking records)
    current_fifa_rankings.csv             (211 rows, live scraped)
    wc2026_fixtures.csv                   (104 matches with venue/altitude)
  processed/
    train.csv                             (9,281 rows) ✅
    test.csv                              (2,257 rows) ✅
    cleaning_metadata.json                ✅
src/
  features/
    data_cleaning.py                      ✅ Phase 3 complete
  models/
  utils/
    fetch_rankings.py                     (rankings scraper)
```

---

## Phases Status
- ✅ Phase 1: Environment setup (venv, folders, Git)
- ✅ Phase 2: Data acquisition (results CSV, FIFA rankings, fixtures, live rankings scraper)
- ✅ Phase 3: Data cleaning (`src/features/data_cleaning.py`)
- 🔲 Phase 4: Feature engineering ← **START HERE**
- 🔲 Phase 5: Model training
- 🔲 Phase 6: Bracket simulator + predictions
- 🔲 Phase 7: Visualization

---

## Phase 3 — What Was Built
File: `src/features/data_cleaning.py`

**Steps completed:**
1. **Competitive match filter** — kept 11,538 of 49,368 matches (dropped friendlies, Island Games, Gulf Cup, CECAFA, etc.). Cut off at 2002 (modern era + reliable FIFA rankings).
2. **Team name standardization** — mapped mismatched names between results CSV and FIFA rankings CSV (e.g. "South Korea" → "Korea Republic", "United States" → "USA")
3. **Temporal train/test split** — cutoff at 2022-11-20 (Qatar WC start). Train: 9,281 matches. Test: 2,257 matches. Split by date only — no random splits (prevents data leakage).
4. **Exponential decay weights** — half-life 730 days. Oldest matches weight ~0.004, most recent ~5.63. Saved as `sample_weight` column in train.csv.
5. **Categorical encoding** — tournament tier (1–4), result (0=away win, 1=draw, 2=home win)
6. **Class weights** — away_win: 1.132, draw: 1.566, home_win: 0.677. Corrects for home win bias (49% of matches).
7. **Save to disk** — train.csv, test.csv, cleaning_metadata.json

**Key decisions to carry forward:**
- No SMOTE — class weights only
- Decay half-life is 730 days (tunable in V2)
- Test set = Qatar 2022 onward (real evaluation set)
- `cleaning_metadata.json` stores class weights and params for use in later phases

**V2 notes (do not forget):**
- Consider combining decay weights WITH tournament tier weights (multiply them together) for a more principled weighting scheme
- Decay half-life is arbitrary at 730 days — worth tuning via cross-validation in V2

---

## Phase 4 — Feature Engineering Plan
Create: `src/features/feature_engineering.py`

Features to build (in order):
1. **FIFA ranking differential** — home_fifa_rank minus away_fifa_rank (need to merge rankings CSV onto matches by date using merge_asof to avoid leakage)
2. **Rolling form** — each team's win rate and avg goals over last 5 and 10 matches (rolling window, date-ordered, no leakage)
3. **Head-to-head record** — historical win/draw/loss between the two specific teams
4. **Goal difference rolling avg** — rolling avg goal diff per team
5. **Altitude effect** — from wc2026_fixtures.csv, altitude_m column; encode as feature
6. **Neutral ground flag** — already in data as `neutral` column
7. **Days rest** — days since each team's last match
8. **Tournament stage** — group stage vs knockout (from fixtures)

Output: enriched `data/processed/train_features.csv` and `test_features.csv`
