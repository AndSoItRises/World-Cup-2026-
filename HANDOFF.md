# WC2026 QUANT MODEL — AGENT HANDOFF (V6)
Updated: 2026-06-11 · repo: https://github.com/AndSoItRises/World-Cup-2026- · tournament starts 2026-06-11

This document is sufficient to start work cold. Read it top to bottom, then run the
SESSION START block. Detailed history lives in CONTEXT_V6.md (decision log DL-01..11)
and CONTEXT_V2–V5 (model evolution); you should not need them to begin.
Human-facing operator guide (dashboard tabs + results-entry workflow): DASHBOARD_GUIDE.md.

---

## 1. WHAT THIS IS
World Cup 2026 prediction model + Vegas research layer + betting dashboard.
- **Prediction model (v4, frozen)**: XGBoost(.275) + LightGBM(.275) + Dixon-Coles(.45),
  48 features, trained on internationals 2005→2022, tested on 2022-11→2026-03 (n=2257).
  Feature/architecture levers are EXHAUSTED (V5 verdict — do not re-litigate).
- **Calibrator (v6, ACTIVE)**: log-pool (geometric blend) + p(draw)×0.871, applied at
  inference on top of untouched v4 components. Test LL 0.8461→0.8405, held-out +0.0049.
  Gated on `models/calibrator_v6.json` — delete that file = full revert to v4 blend.
- **Vegas layer (v6)**: live DraftKings odds → de-vig (Shin) → edge/EV/Kelly →
  rule-based desk calls → single-file HTML dashboard.
- Owner: Jake (learning quant research through this project — explain decisions,
  show the math, never hide caveats).

## 2. SESSION START (run every session, in order)
```powershell
cd "C:\Users\jakeh\OneDrive\Documents\Claude\Projects\World Cup 2026 Model"
$env:PYTHONUTF8 = "1"             # console is cp1252; scripts print box-drawing
git status                        # expect clean; note anything uncommitted
```
State which queue item (section 7) you're on before touching files.
Python: `.\venv\Scripts\python.exe -m src.models.<script>` (module form, from repo root).

## 3. REFRESH PIPELINE (after results land / any time for odds)
Odds snapshots are AUTOMATED: Windows scheduled task "WC2026 odds snapshot" runs
`scripts/fetch_odds_task.ps1` hourly through the final (logs to logs/odds_fetch.log;
catches up after sleep/reboot via StartWhenAvailable). This keeps closing lines tight
for clv_tracker without manual fetches. Delete the task after 2026-07-19.

Run in this order — each feeds the next:
```powershell
# 1. Jake enters finished matches into data/raw/wc2026_live_results.csv
#    (columns: match_id,stage,group,date,home_team,away_team,home_goals,away_goals,decided_by,winner)
python -m src.models.live_update        # ELO update + 10k sims -> tournament_probs_live.csv (~3 min)
python -m src.models.predict_wc2026     # match-level 3-way probs (calibrated)
python -m src.models.fetch_live_odds    # scrape ESPN/DraftKings lines (appends snapshot history)
python -m src.models.market_monitor     # line movement vs model + arb scan
python -m src.models.bet_sim            # edge / EV / Kelly sheets
python -m src.models.desk_call          # BET/LEAN/PASS verdicts (+ CLV feedback)
python -m src.models.clv_tracker        # log new calls, score vs closing lines
python -m src.models.build_dashboard    # regenerates outputs/quant_dashboard.html
git add . ; git commit -m "..." ; git push
```

## 4. SYSTEM MAP (V6 modules, all in src/models/)
| module | reads | writes | purpose |
|---|---|---|---|
| calibrator.py | calibrator_v6.json | — | log-pool + draw-shrink at inference (used by monte_carlo + predict_wc2026) |
| calibrate_v6.py | test_features, v4 models | calibrator_v6.json | stage-1 experiments (ADOPTED — don't rerun unless test set changes) |
| calibrate_tilt.py | same | — | stage-2 dog-tilt experiments (CUT — see §6) |
| fetch_live_odds.py | ESPN scoreboard API, wc2026_fixtures.csv | data/raw/wc2026_match_odds.csv | DraftKings 3-way lines, open+current, appends per run |
| market_ingestion.py | odds CSVs | market_implied_probs/futures.csv | odds parsing + Shin de-vig + name audits |
| bet_sim.py | predictions, implied probs | value_bets.csv, value_bets_futures.csv | edge / EV / Kelly (¼, 5% cap), tail_risk flag |
| market_monitor.py | match odds history | line_movement.csv, arb_scan.csv | open→now shifts vs model; cross-book arb (needs ≥2 books) |
| desk_call.py | value_bets, movement, clv_report | desk_calls.csv | BET/LEAN/PASS + evidence; 25% portfolio cap; CLV confidence input (final closes, n≥8/category) |
| clv_tracker.py | desk_calls, match odds history, live results | bet_ledger.csv (append-only), clv_report.csv | CLV per logged bet vs closing line; settles DL-10 |
| build_dashboard.py | all of the above + HANDOFF.md | outputs/quant_dashboard.html | self-contained dashboard; embeds this doc in NOTES tab |
| live_update.py / monte_carlo.py / predict_wc2026.py | v4 models + calibrator | tournament_probs_live.csv, wc2026_predictions.csv | prod inference (pre-existing, now calibrated) |

Key data: `data/raw/wc2026_fixtures.csv` (match_id is THE key everywhere),
`data/raw/wc2026_market_odds.csv` (winner futures, American odds),
`data/processed/test_features.csv` + `models/ensemble_test_proba_v4.npy` (validation set).

## 5. NON-NEGOTIABLES (inherited, enforced)
- **Validate-or-cut**: any model/probability change must beat the held-out two-fold
  protocol (fit half of test by date, score the other, swap) by mean ΔLL ≥ +0.003 with
  both folds positive. See calibrate_v6.py for the reference implementation.
- **No leakage**: features use only pre-match information.
- **v1–v5 artifacts untouched**: new behavior ships as NEW files (e.g. calibrator_v6.json).
- **Name audit on every new data source**: print matched/unmatched loudly. Names differ
  across sources (ESPN "United States" = fixtures "USA" = futures "Korea Republic" ≠
  fixtures "South Korea"). Match odds are keyed by match_id to avoid this; team-keyed
  joins use the alias maps in market_ingestion.py / fetch_live_odds.py.
- **Decision log immediately** after any decision → CONTEXT_V6.md, then checkpoint commit.
- Communicate in probabilities + edge + EV + stake together; never a bet without uncertainty.

## 6. MODEL TRUTHS (memorize; they shape every output)
1. 62% accuracy, test LL 0.8405 (calibrated). Edge vs market UNPROVEN out-of-sample
   (needs historical international odds — paid; or this tournament's realized results).
2. Draw inflation is FIXED (DL-09): emitted draw rate now matches reality. Draw recall
   ~1% is the accepted cost — the betting layer prices off probabilities, not argmax.
3. Underdog tilt vs market is NOT a calibration bug (DL-10): favorite-bucket reliability
   is flat against realized outcomes; fitted dog-shrink ≈ 1.0. The model genuinely
   disagrees with the market on dogs/draws. Whether that's edge or the market knowing
   lineups gets decided by realized WC results — see queue item 1.
4. CONCACAF inflation (Mexico ≈ +6pp vs market) is documented model error — desk_call
   haircuts it; don't present those edges as value.
5. Futures with model_prob < 2% (<200 of 10k sims) are MC tail noise → tail_risk flag.
6. Market moved AGAINST the model on most significant pre-tournament line moves —
   books hardened favorites the model is lukewarm on. Track, don't panic.

## 7. WORK QUEUE (in order; each item is self-contained)
1. ~~**clv_tracker.py**~~ ✅ DONE 2026-06-11 (DL-11): append-only bet_ledger.csv +
   clv_report.csv; CLV tab on dashboard; desk_call confidence input gated on FINAL
   closes (settled matches) with n ≥ 8 per category. The dog-CLV KPI settles DL-10
   as the group stage plays out — check it after each matchday.
2. **uncertainty.py** (epistemic): save per-component probs (XGB/LGBM/DC) at predict
   time, ensemble disagreement = mean pairwise JS-divergence per match; high
   disagreement → desk_call stake haircut. Aleatoric (entropy) already on dashboard.
3. **Signal tests** via signal_test.py gate: tournament pressure score, manager tenure,
   travel/timezone, suspension exposure, confed-ELO deflation. Expect cuts (V5 pattern);
   only confed-deflation has real promise given §6.4.
4. **More books** into wc2026_match_odds.csv (any source; keep schema) → real arb
   scanning + best-line EV automatically activate.
5. **Stage futures** (group winner / R16 / QF) vs market when books price them —
   bet_sim pattern extends directly; model probs already in tournament_probs_live.csv.
6. **Bracket P10/P50/P90 bands + re-sim button**: dump per-sim outcomes from
   live_update.py, embed distribution in dashboard panel C.

## 8. DASHBOARD (outputs/quant_dashboard.html)
Self-contained, offline, dark terminal theme; mobile/iOS-friendly (open the GitHub
Pages URL in Safari). Tabs: DESK CALLS (landing — verdicts + evidence + how-to),
SCANNER, MATCHES, GROUPS (advance % + fair odds), BRACKET, DIVERGENCE, BANKROLL
(1k-path Kelly sim, truth=model/blend/market), MOVEMENT+ARB (incl. "where is the
arbitrage?" panel — gap-to-arb per match, activates with a 2nd book), CLV, UNCERTAINTY,
NOTES (this doc, embedded at build). "⟳ LIVE ODDS" (+ auto-4m toggle) re-fetches ESPN
in-browser: live scores in the ticker, EV/Kelly recomputed, AND desk verdicts/stakes
recomputed via a JS mirror of desk_call.py (keep the two in sync when rules change).
Model probs stay the local 10k-sim output — re-simulation is the Python pipeline.
Rebuild with `python -m src.models.build_dashboard` after ANY data change.

## 9. STYLE
Code: match existing module pattern (docstring header with Run:, BASE/DATA paths,
═-boxed prints, ✅/⚠️ markers, loud audits). Console needs PYTHONUTF8=1.
Commits: descriptive, Co-Authored-By Claude line, push after each closed phase.
With Jake: explain the why before the code; quantify uncertainty; flag every bias.
