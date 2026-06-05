# WC2026 Model — V2 Context Document

---

## ══ HOOKS (Permanent — Copy Into Every New Context Doc) ══

### How We Work
- Explain decisions before writing code — options considered, why we chose this path
- One phase at a time — don't jump ahead
- Claude Code executes scripts autonomously — no need to prompt Jake to run them
- Venv must be activated before any python execution: `venv\Scripts\activate`
- Scripts run as: `python -m src.<module>.<script>`
- Full file replacements when edits get complex — rewrite the whole file cleanly

### Session Start Ritual (Claude Code — Every Session)
1. Read the current context doc fully
2. Run `git status` — confirm clean state or note what's uncommitted
3. State which phase we're in and what the next action is
4. Confirm understanding before touching any file

### Non-Negotiables
- No leakage — all features must use only information available before the match
- Every new feature must be validated: retrain → compare CV log loss to prior version baseline. If no improvement, cut it.
- Keep prior version models intact — new versions save to new file paths (xgb_v2, lgbm_v2, etc.)
- Any new data source requires a team name audit before merging — mismatches have caused silent NaN bugs twice
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
Building a World Cup 2026 match prediction model. V1 established the baseline pipeline. V2 focuses on making the model intelligent — fixing known biases and finding market edge.

## Folder Structure
```
data/
  raw/
    international_results/results.csv     (49,368 match results, 1872–present)
    FIFA_rankings_training/               (3 CSVs, ~200k ranking records)
    current_fifa_rankings.csv             (211 rows: rank, team_name, team_code, points)
    wc2026_fixtures.csv                   (104 matches)
  processed/
    train.csv                             (9,281 rows) ✅
    test.csv                              (2,257 rows) ✅
    train_features.csv                    (9,281 rows, 51 cols) ✅ V2
    test_features.csv                     (2,257 rows, 51 cols) ✅ V2
    elo_ratings.csv                       (49,296 rows) ✅ V2 NEW
    wc2026_predictions.csv                ✅ V1 (needs V2 rerun)
    tournament_probs.csv                  ✅ V1 (needs V2 rerun)
    cleaning_metadata.json                ✅
models/
  xgb_v1.json                            ✅ V1 (preserved)
  lgbm_v1.txt                            ✅ V1 (preserved)
  xgb_v2.json                            ✅ V2
  lgbm_v2.txt                            ✅ V2
  dixon_coles_params.json                 ✅ V1 (needs V2 rerun)
  ensemble_report_v2.json                ✅ V2
  ensemble_test_proba_v2.npy             ✅ V2
  training_report_v2.json                ✅ V2
  lgbm_report_v2.json                    ✅ V2
src/
  features/
    data_cleaning.py                      ✅
    feature_engineering.py               ✅ V2 updated
    elo.py                               ✅ V2 NEW
  models/
    train_xgb.py                         ✅ V2 updated
    dixon_coles.py                       ✅
    train_lgbm.py                        ✅ V2 updated
    ensemble.py                          ✅ V2 updated
    predict_wc2026.py                    ✅ (needs V2 update)
    monte_carlo.py                       ✅ V2 updated
  visualization/
    charts.py                            ✅
  utils/
    fetch_rankings.py
```

---

## V1 Summary (Closed)
- 49,368 match results, FIFA rankings, WC2026 fixtures
- Temporal train/test split at 2022-11-20
- Dixon-Coles + XGBoost + LightGBM ensemble
- Ensemble: 60.6% accuracy | 0.8605 log loss | 0.4% draw recall
- Tagged: v1.0

### V1 Known Issues (Root Causes)
1. **USA/Mexico inflation** — rolling stats didn't adjust for opponent quality. CONCACAF qualifying inflated form stats vs weak opponents.
2. **Argentina underrated** — same root cause. CONMEBOL harder schedule suppressed rolling stats.
3. **Draw recall 0.4%** — XGB/LGBM output near-zero draw probabilities. DC blended in but diluted below decision threshold.
4. **Decay half-life untested** — 730-day half-life set arbitrarily, never CV-tuned.

---

## V2 Decision Log

### Decision: Build ELO before opponent-weighted rolling stats
**Options considered:**
- A) Build opponent-weighted rolling stats first using FIFA rank as quality proxy
- B) Build ELO first, use it as the quality signal for weighted rolling stats

**Decision:** B — ELO first.
**Why:** If we used FIFA rank as the proxy in Phase 1, we'd have to rebuild Phase 1 again once ELO existed. ELO is a superior quality signal (dynamic, accounts for full match history) and it's self-contained to build. Doing it once cleanly is better than doing Phase 1 twice.

---

### Decision: K-factors for ELO (40 / 30 / 20)
**Options considered:**
- Fixed K=32 (standard chess ELO)
- Tiered K by match importance: 40 / 30 / 20
- Optimize K via grid search now

**Decision:** Tiered K, defer optimization to Phase 4.
**Why:** Tiered K is standard practice in football ELO (World Football Elo Ratings uses similar values). Optimizing K now would be premature — K tuning is only meaningful once the full feature set is locked. Tuning against an intermediate model risks optimizing toward the wrong target.

---

### Decision: Opponent quality weight = opponent_elo / 1500
**Options considered:**
- A) Opponent FIFA rank percentile
- B) Opponent ELO normalized by default rating (opp_elo / 1500)
- C) Raw ELO difference

**Decision:** B — ELO normalized to default.
**Why:** Normalizing around 1500 (the starting/default rating) keeps weights interpretable and centered. A match against a 2000 ELO team gets weight ~1.33, a match against a 1000 ELO team gets ~0.67. FIFA rank percentile would work but requires an additional percentile lookup per match. Raw ELO difference is unscaled and harder for the model to use.

---

### Decision: Keep simple rolling stats alongside weighted versions
**Options considered:**
- A) Replace simple rolling stats with weighted versions
- B) Keep both — 12 simple + 12 weighted features

**Decision:** B — keep both.
**Why:** Simple form still carries signal (momentum, fitness, team rhythm). Quality-weighted form corrects confederation bias. The model can learn the right blend from data. Replacing rather than adding would discard potentially useful signal.

---

### Decision: 150 → 80 sentinel for unranked teams in Monte Carlo
**Options considered:**
- Keep 150 (existing sentinel)
- Cap at 80 for WC-qualified teams

**Decision:** Cap at 80 in monte_carlo.py only (leave feature_engineering.py at 150).
**Why:** Every team at WC2026 qualified — none are genuinely rank-150 level. Using 150 caused Iran (unranked in our lookup) to appear unrealistically weak. 80 is a conservative but realistic floor for a WC-qualified nation. The 150 sentinel in feature_engineering.py is appropriate for training data where genuinely obscure nations appear in the full 49k match history.

---

## V2 Phase Status

| Phase | Description | Status |
|---|---|---|
| ELO ratings | src/features/elo.py, integrated as features | ✅ Done |
| Opponent-weighted rolling stats | 12 new weighted features in feature_engineering.py | ✅ Done |
| Retrain models | XGB V2, LGBM V2, Ensemble V2, Monte Carlo V2 | ✅ Done |
| Draw fix (Dixon-Coles improvement) | Fix structural draw recall problem | ⬜ Next |
| Hyperparameter tuning | Decay half-life, K-factors, model params via CV | ⬜ |
| Market divergence analysis | Scrape odds, find edge vs market | ⬜ Key goal |
| Stacking meta-learner | Replace fixed weights with trained meta-learner | ⬜ |
| Updated visualization + README | V1 vs V2 comparison charts | ⬜ |

---

## V2 Model Results (Phases 1–3 complete)

| Metric | V1 Ensemble | V2 Ensemble | Δ |
|---|---|---|---|
| Test Accuracy | 60.6% | 61.2% | +0.6pp |
| Log Loss | 0.8605 | 0.8477 | -0.013 |
| Draw Recall | 0.4% | 0.6% | flat |

### V2 Tournament Win Probabilities (10k simulations)
| Team | FIFA Rank | Win% | Final% |
|---|---|---|---|
| Spain | 2 | 16.1% | 24.9% |
| Mexico | 15 | 9.1% | 16.2% |
| Portugal | 5 | 6.9% | 12.2% |
| England | 4 | 6.6% | 12.2% |
| France | 1 | 6.5% | 12.1% |
| Brazil | 6 | 6.2% | 11.5% |
| Canada | 30 | 5.6% | 11.1% |
| USA | 16 | 5.1% | 10.7% |
| Argentina | 3 | 3.9% | 7.5% |

### Remaining Known Issues
- Mexico 9.1% — still inflated, residual CONCACAF form bias. Phase 4 (tuning).
- Argentina 3.9% — still low for defending champion. Phase 4.
- Draw recall 0.6% — structural. Phase 3 (Dixon-Coles fix).

---

## Notes on the Data
- results.csv uses: "Korea Republic", "Côte d'Ivoire", "Bosnia-Herzegovina", "United States"
- current_fifa_rankings.csv uses: "Korea Republic", "France" — mostly consistent
- wc2026_fixtures.csv uses: "South Korea", "Ivory Coast", "Bosnia and Herzegovina"
- ELO_NAME_MAP in feature_engineering.py handles 14 team name mismatches between training data and results.csv
- NAME_MAP in ensemble.py and predict_wc2026.py handles fixture name mismatches

---

## ══ END-OF-VERSION REVIEW ══
*(Filled in when V2 is fully closed)*

### Workflow Review
*What worked, what slowed us down, what to change for V3*

### Decision Review
*Which decisions held up, which should be revisited*

### Brainstorm: V3 Priorities
*From V2 context doc V3 Vision section:*
- C++/Rust simulation engine (1M sims, sub-second)
- API layer (accepts match result, updates ELO, reruns sim)
- Kelly criterion bet sizing
- Vercel deployment

*Only build V3 if market divergence analysis shows a real edge.*
