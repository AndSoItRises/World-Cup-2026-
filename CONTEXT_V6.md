# WC2026 Model — V6 Context Document

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
V5 closed with the verdict that v4 is the final validated PREDICTION model — every accessible
modelling lever came back cut/flat/data-blocked. **V6 is the Vegas layer**: convert V5's
probabilities into betting intelligence (de-vigged market comparison, EV, Kelly, CLV,
uncertainty) and wrap it in an interactive research dashboard. The model itself does not change.

V6 spec lives in: `fable-wc2026-prompt.md` (Jake's prompt doc, outside the repo:
`OneDrive\Documents\Claude\Projects\mythos -lv-prompt\fable-wc2026-prompt.md`).

## Inherited State (V5 — tagged v5.0, closed)
- Prod model = v4 ensemble (XGB 27.5% / LGBM 27.5% / DC 45%), accuracy 62.0%, LL 0.8461,
  model↔market corr 0.838. V5 changed nothing — ceiling confirmed.
- Live pipeline: `live_update.py` ingests `data/raw/wc2026_live_results.csv` → ELO update →
  10k sims → `data/processed/tournament_probs_live.csv`.
- Match-level model probs: `data/processed/wc2026_predictions.csv` (72 group matches, 3-way).
- Futures market odds on hand: `data/raw/wc2026_market_odds.csv` (49 teams, American,
  tournament winner only). NO per-match market odds yet.
- Carried caveat (V5 DL-05): a real out-of-sample edge vs the market is UNPROVEN (needs
  historical international odds). V6 builds the betting machinery anyway — outputs are
  research signals, not validated +EV claims, and every output must carry that caveat.

## V6 Build Order (from the prompt doc)
| # | Deliverable | Status |
|---|---|---|
| 1 | `src/models/bet_sim.py` — EV engine + Kelly | ✅ done |
| 2 | `src/models/market_ingestion.py` — odds ingestion + juice stripping | ✅ done (merged into phase 1, see DL-01) |
| 3 | `outputs/quant_dashboard.html` — interactive research dashboard | ✅ done |
| + | `src/models/fetch_live_odds.py` — live ESPN/DraftKings line scraper | ✅ done (added scope, DL-05) |
| + | `src/models/market_monitor.py` — line movement + arb scanner | ✅ done (added scope, DL-06) |
| 4 | `src/models/uncertainty.py` — aleatoric/epistemic quantification | ⬜ (aleatoric entropy proxy already in dashboard) |
| 5 | `src/models/clv_tracker.py` — closing line value tracking | ✅ done (DL-11) |
| 6 | Signal tests for 5 new candidate features (orthogonality gate) | ⬜ |

The full next-steps list (incl. multi-book arb, stage futures, bracket re-sim bands) is embedded
in the dashboard's NOTES tab, so the research state travels with the HTML file.

---

## V6 Decision Log

### DL-01 — Phase 1 scope: build market_ingestion.py together with bet_sim.py
The prompt orders bet_sim first, ingestion second — but EV/Kelly math is meaningless without
de-vigged implied probabilities, so the dependency runs the other way. Phase 1 ships both:
`market_ingestion.py` owns odds parsing + juice stripping (Shin method default, proportional
fallback/comparison) and the team-name audit; `bet_sim.py` consumes it. Additive only — the
old proportional de-vig in `market_divergence.py` is untouched.

### DL-02 — Match-level odds: model-estimated proxy until real lines arrive
No per-match 3-way market odds are on hand. Per the prompt, matches without market odds use
model-implied fair odds flagged `market_source="model_estimated"` — these produce EV ≡ 0 by
construction (honest: the pipeline runs end-to-end but flags no fake value). A template
`data/raw/wc2026_match_odds.csv` (keyed by match_id — no team-name ambiguity) is auto-created;
the moment Jake fills it with real lines, the same run produces live EV/Kelly. Futures
(tournament winner) DO have real market odds, so that sheet is real from day one.

### DL-03 — Name audit caught a live bug: "Congo DR" was being silently dropped
The inherited MARKET_TO_MODEL map (from market_divergence.py, V2) renamed market "Congo DR"
→ "DR Congo", but every model file uses "Congo DR" verbatim — the team silently fell out of
every market comparison since V2 (47/48 matched, nobody noticed). V6's loud audit caught it.
Fixed in both market_ingestion.py and market_divergence.py; market_divergence.csv regenerated
(48/48 matched, model↔market corr 0.838 → 0.840). The name-audit non-negotiable earns its keep.

### DL-04 — Tail-risk flag: MC tail noise is not edge
Naive EV ranking put 1500:1 longshots (New Zealand, Uzbekistan) on top: p_winner from 10k
sims has huge relative error below ~200 sims (2%), and at 100:1+ odds that noise reads as
massive EV — the favorite-longshot trap. Added `tail_risk = model_prob < 0.02` to the futures
sheet; tail rows stay in the CSV but are excluded from the headline table. Also printed with
every run: the edge-vs-market is unproven (V5 DL-05), and the Mexico edge (+6.2pp) overlaps
the DOCUMENTED CONCACAF inflation bias — model error, not market error.

### DL-05 — Live match odds unlocked via ESPN's public scoreboard JSON
`site.api.espn.com/.../soccer/fifa.world/scoreboard` serves DraftKings 3-way moneylines for all
72 group matches with BOTH opening and current odds — free, no key, one GET. `fetch_live_odds.py`
appends snapshots to `wc2026_match_odds.csv` (opening written once per match/book; current
accumulates a line history per run). Ten ESPN events had home/away flipped vs our fixtures
(neutral-venue convention) — matcher tries both orientations and re-orients odds to the fixture's
home/away (what the model probs are keyed to). 72/72 matched. This closed DL-02's gap same-day:
match EV is now real, not model_estimated.

### DL-06 — Line movement + arb scanner (added scope)
`market_monitor.py`: per-outcome open→current implied shift (significant = ≥0.10 decimal or
≥5pp), classified toward/against the model (did the market move closer to our number?). First
read: 137/216 lines moved significantly since open, 81 AGAINST the model vs 56 toward — the
market has hardened favorites the model is lukewarm on. Arb scan takes best-line per outcome
across books (Σ 1/odds < 1 = riskless); with one book it correctly reports none (tightest
Σ=1.024) and activates automatically as more books are added to the odds CSV.

### DL-07 — Match-level +EV is draw/dog-heavy = documented model bias, surfaced everywhere
With real lines, 125/216 outcomes show +EV — concentrated in draws (the deliberate 1.75×
upweight) and lopsided-match underdogs (ELO compression). That's model bias, not market error.
Handled honestly: bet_sim prints the outcome breakdown + warning; the dashboard scanner defaults
to "hide draw bets"; the bankroll simulator offers truth = model / 50-50 blend / market-fair so
the optimistic-vs-pessimistic bounds are explicit.

### DL-08 — Desk Call layer: recommendations, not just information (Jake feedback)
Jake's read on the first dashboard: lots of data, no thesis. Added `desk_call.py` — a
rule-based verdict engine (BET ≥ 6 / LEAN ≥ 3 / PASS) where every input is a documented
finding: edge size, favorite-side trust, line movement toward/against the model, draw-bias
auto-pass, longshot + CONCACAF haircuts, coin-flip (entropy > 1.5 bits) half-stake, and a 25%
portfolio exposure cap on the whole book (raw Kelly wanted 51%). The DESK tab is now the
dashboard landing page: each pick shows stake + evidence chain (+) and risk haircuts (!), and
the PASS pile explains itself. First run: 8 BET / 16 LEAN / 240 PASS, $250 total book.
Ticker slowed 90s → 240s per loop (Jake request).

### DL-09 — ADOPTED: log-pool + draw-shrink calibrator (first model improvement since V4)
Jake challenged the V5 ceiling. The ceiling verdict covered FEATURE/architecture levers;
probability-level post-processing was never tested. `calibrate_v6.py` ran six candidates under
the honest two-fold held-out protocol (fit one chronological half of test, score the other,
swap; adopt bar mean ΔLL ≥ +0.003 with both folds positive):

  temperature −0.0032 ✂ · draw_shrink +0.0027 ✂ (just misses) · class_weight +0.0031 ✅ ·
  vector −0.0008 ✂ (overfits) · log_pool alone +0.0025 ✂ · **log_pool+draw_shrink +0.0049 ✅**

Winner: geometric (log) pooling of XGB/LGBM/DC at the same fixed weights, then p(draw)×0.871,
renormalized. Test LL 0.8461→0.8405, acc 0.6163→0.6176; emitted draw rate 25.9%→22.5% vs
realized 22.4% — the 1.75× draw-upweight inflation is removed at inference. Held-out gain
(+0.0049) exceeds the gain that justified V4 (+0.0033). Trade-off accepted with eyes open:
draw RECALL collapses 9%→1% (argmax almost never picks draw) — classification was already
structurally capped; the betting layer prices off probabilities, which are strictly better.

Strong independent validation: rerunning the live forecast, the calibrator moved the model
TOWARD the market on precisely the documented bias teams — Spain 18.4→22.1% (model fade
shrinks), Mexico 7.1→6.0%, Japan 5.9→4.7%, USA 4.2→3.4%, Iran/Korea/Canada all down. Nobody
told it about those biases; it found them via the draw/longshot miscalibration.

Integration (additive + reversible): `src/models/calibrator.py`, gated on
`models/calibrator_v6.json` — delete the file and monte_carlo/predict_wc2026/live_update all
revert to the v4 linear blend. v1–v4 artifacts untouched. Full chain regenerated (predictions,
10k sims, EV, desk calls, dashboard). Match +EV rows 125→119; draw +EV 70→58 — tilt reduced,
not gone (κ=0.871 is a calibration fix, not the dog-compression fix; that one is structural).

### DL-10 — Underdog tilt: CUT — it's a market disagreement, not a miscalibration
Stage-2 experiments (`calibrate_tilt.py`, stacked on DL-09, same held-out protocol) tested
dog_shrink / favorite-logit remap / lopsidedness-conditional temperature. First, the
diagnostic: bucketing test matches by favorite prob, predicted vs REALIZED favorite win rates
are flat (gaps −3.8%…+2.0%, no monotone pattern) — the model's underdog probabilities are
honest against reality on 2,257 held-out internationals. Fitted dog_shrink κ ≈ 1.02 ("do
nothing", ΔLL +0.0000); the 2-param transforms actively hurt held-out (−0.004…−0.006).

Implication: the dog/draw tilt vs the MARKET is the model genuinely disagreeing with the
price, not a calibration bug. Either the market knows things the model can't see (lineups,
injuries, motivation) or this is where the edge lives — indistinguishable until WC2026
results arrive. That makes the CLV tracker (phase 5) the decisive experiment: score the
dog-side desk calls against closing lines and realized outcomes as the group stage plays
out. Until then the desk-call haircuts (longshot/draw/CONCACAF) remain the risk treatment.

### DL-11 — CLV tracker shipped (queue #1, the DL-10 decider) + live desk on the dashboard
`clv_tracker.py`: every match-kind BET/LEAN is logged ONCE into an append-only
`bet_ledger.csv` (key match_id+outcome, line-at-log-time) — the desk changing its mind on
a re-run never rewrites history. Closing line = last snapshot per match in
`wc2026_match_odds.csv`; flagged `close_is_final` only after the match settles, because a
pre-kickoff "close" is just the latest fetch (provisional CLV would feed the desk its own
current line). First run: 29 bets logged opening day (2026-06-11T12:25Z). Futures excluded
(no close until July).

Feedback loop into `desk_call.py` (queue #1's confidence input): per-category (fav/dog/draw)
avg CLV over FINAL closes only, gated n ≥ 8 and |avg| ≥ 2% → ±1.5 score with an evidence
line. Dormant until enough matches settle; the dog category is the DL-10 verdict.

Dashboard upgrades in the same pass (Jake asks: live feed, value bets, advancement odds,
arb clarity, iOS): CLV tab with dog-CLV KPI; desk-call rules ported to JS so ⟳ LIVE ODDS
(+ new auto-4m toggle) recomputes BET/LEAN verdicts and capped stakes on live lines (model
probs remain the 10k-sim output — honest note shown); live scores in the ticker; "Where is
the arbitrage?" panel (per-match Σ implied + gap-to-arb; explains one book ⇒ no riskless
arb until a 2nd book's lines are added); fair-odds equivalents on GROUPS/BRACKET advancement
probs; how-to-use intro on the landing tab; mobile/iOS CSS (text-size-adjust, touch
scrolling, single-column cards). Maintenance cost accepted: the JS desk mirror must be kept
in sync with desk_call.py — flagged in HANDOFF §8. Meta line fixed to show calibrated LL
0.8405.

### DL-12 — Full unattended refresh loop: results auto-ingest + smart push (2026-06-12)
The last manual steps (entering results + running the ritual + pushing) are now automated.
`fetch_live_results.py` (NEW) reads ESPN's scoreboard — the same endpoint fetch_live_odds
already hits — detects `status.state == "post"` matches, maps them to fixtures by match_id
(reusing fetch_live_odds's normaliser + ESPN_TO_FIXTURE alias map), re-orients the score to
the fixture's home/away (ESPN flips neutral-venue ties), and writes wc2026_live_results.csv.
Idempotent: adds if missing, overwrites only if the SCORE changed, else no-op — so a human
hand-edit (e.g. a penalty-shootout knockout) survives re-runs. Exit code 10 = new/changed
result, 0 = none, signalling the orchestrator whether to run the expensive sims.

`scripts/refresh_all.ps1` (NEW) is the orchestrator: every cycle it fetches odds + results,
runs live_update + predict_wc2026 ONLY on a new result (gated on exit 10), always reprices
(market_monitor/bet_sim/desk_call/clv_tracker), then commits+pushes ONLY when a tracked
output materially changed — keyed on desk_calls/value_bets/predictions/tournament_probs_live,
deliberately NOT on line_movement/arb_scan (which jitter every odds tick). Rationale: the
dashboard already re-prices live odds client-side via ⟳ LIVE ODDS, so the Python push only
needs to carry new results + changed verdicts — this avoids ~48 timestamp-only commits/day
while keeping GitHub Pages live. `data/raw/` being gitignored means odds-snapshot noise never
reaches git regardless. `register_refresh_task.ps1` registers the "WC2026 full refresh"
scheduled task (every 30 min, StartWhenAvailable); it supersedes the hourly odds-only task.

Also fixed a live bug in fetch_live_odds.extract_moneyline — a null entry in ESPN's odds list
crashed the hourly task intermittently (`NoneType.get`); now guarded. First full run ingested
the two opening-day finals (Mexico 2-0 South Africa, South Korea 2-1 Czechia), re-simmed, and
pushed in ~70s. Carried caveat: knockout decided_by (AET vs 90-min vs pens) is best-effort from
this endpoint — group stage is exact; verify knockout rows before R32. The JS desk mirror /
desk_call.py sync obligation (DL-11, HANDOFF §8) is unchanged.

### DL-13 — Prediction scoreboard + "Next 5" desk board (Jake ask, 2026-06-12)
Jake asked for (a) correct/incorrect predictions stored and fed back to improve the model,
and (b) a "next 5 games" feature on the Quant Desk showing upcoming picks + conviction.

`prediction_tracker.py` (NEW): append-only `prediction_ledger.csv` logs every match's
model probs + Shin-fair market probs ONCE while the match is still unplayed — necessary
because wc2026_predictions.csv is regenerated after every result ingest, so a settled
match's current row postdates its own result (scoring it would be leakage; same fix as
DL-11's bet ledger). `prediction_scoreboard.csv` scores settled matches: argmax
correct/incorrect, p(realized), log-loss, and the market's log-loss on the same match.
The two matches that settled before the tracker existed were seeded from the
pre-tournament git snapshot (13b9979^ — rows verified byte-identical to the live file,
so predict_wc2026's features are confirmed as-of-date). Runs every refresh cycle;
prediction_scoreboard.csv added to the push triggers.

On "feed it back into the model": results ALREADY flow back — live_update re-rates ELO
after every final and the next sims price on it. Anything deeper is gated, deliberately:
the tracker prints a tournament reliability check only at n ≥ 40 settled (below that it's
noise), and any recalibration it suggests must still clear the validate-or-cut bar
(DL-09). Auto-refitting on a handful of results would chase noise AND contaminate the
DL-11 CLV experiment mid-sample (see the edge-vs-error memo,
outputs/research/wcup2026_edge_vs_error_memo.md — it recommends FREEZING desk-rule
feedback during the measurement window, not adding more).

Dashboard (DESK tab): "Next 5 games" board — per upcoming match: model prob bars vs
market fair ticks, model pick, and the desk's best call with verdict + desk score
(conviction after haircuts) + capped stake; re-renders on ⟳ LIVE ODDS via drawDesk.
"Model record" card — KPI strip (record, accuracy, model LL vs market LL, pending) +
full per-match table with ✓/✗. Both honest: post-result rows excluded from headline
numbers, n<40 flagged as too-small, LL comparison labeled diary-grade (the powered
statistic is CLV — memo §3a).

### DL-14 — Phase 7 Step 1: cloud results ingest (football-data.org) + GitHub Actions (2026-06-22)
Phase 7 wants the refresh loop to run with NO local machine. Added a parallel cloud ingest
path that mirrors fetch_live_results' contract but sources football-data.org instead of ESPN:
- `src/features/team_name_map.py` (NEW): football-data.org full names → fixtures convention
  (`United States`→`USA`, `Korea Republic`→`South Korea`, `Côte d'Ivoire`→`Ivory Coast`, …),
  accent-insensitive exact lookup + difflib fuzzy fallback. Targets the FIXTURES convention
  on purpose (not internal names) so output stays compatible with live_update.load_live(),
  which applies normalize() afterwards (two-layer naming — see §5 name-audit non-negotiable).
- `src/models/ingest_results.py` (NEW): GET /v4/competitions/{code}/matches?status=FINISHED
  (X-Auth-Token), parses 90-min scores (regularTime, fallback fullTime), joins to internal
  match_id via wc2026_fixtures.csv on (home,away), MERGE-writes wc2026_live_results.csv keyed
  on match_id (preserves existing/hand-edited rows). `--self-test` runs offline (no network).
  Verified: 72/72 group fixtures join with API-style names, 0 synthetic ids; empty pull
  preserves all 40 existing rows; load_live() consumes output unchanged.
- `.github/workflows/daily_update.yml` (NEW): cron 04:00 + 10:00 UTC + workflow_dispatch,
  `contents: write`, runs ingest_results → live_update, commits only if `git diff --staged`
  is non-empty. Reads secret FOOTBALL_DATA_API_KEY (already set in repo).

CARRIED CAVEATS / OPEN DECISIONS (not yet resolved with Jake):
- TWO ingestors now target wc2026_live_results.csv: fetch_live_results.py (ESPN, local 30-min
  task, DL-12) and ingest_results.py (football-data.org, GitHub Actions). They must not fight.
  Decision pending: does the cloud path REPLACE the local task, or is one canonical?
- Competition code defaulted to `WC2026` per the Phase 7 prompt; football-data.org has
  historically used `WC`. Overridable via env FOOTBALL_DATA_COMPETITION. UNVERIFIED against the
  live API (key not present in this dev shell) — confirm before relying on the cron.
- Workflow pins Python 3.13 (NOT the prompt's 3.10): requirements.txt pins numpy 2.4 / pandas
  3.0, which need ≥3.11; 3.10 would fail pip install. Matches the local venv.

### DL-15 — Phase 7 Step 2: settle_bets.py — the settlement fix (2026-06-22)
Phase 7's CRITICAL bug: picks sat at status=pending/pnl=0 after their match was played.
`src/models/settle_bets.py` (NEW) reconciles pending picks against wc2026_live_results.csv:
join by match_id (date+teams fuzzy fallback), set WON/LOST/VOID, pnl = (taken_decimal-1)*stake
on WON, -stake on LOST. 1X2 settled on the 90-min score (a draw loses a home/away ML),
matching how live_update records results. Outputs: clv_report.csv updated IN PLACE (only
status/pnl_usd of newly-settled rows; CLV data preserved); bet_ledger_settled.csv
(= bet_ledger.csv + status/pnl_usd, original untouched); settlement_log.csv (append-only
audit, one row per settlement, timestamped); group_standings.csv EXTENDED with actual
played/pts/gf/ga/gd (probability columns kept; fixtures names normalized to internal to join).
Idempotent — bet_ledger.csv has no status column, so prior settlements are re-seeded from
bet_ledger_settled.csv each run (caught the audit log growing 29→45 before the fix; now stable).
First run settled 13 pending picks (4 WON / 9 LOST, net +$11.02); formula cross-checked against
the 3 pre-existing settled rows (match_id 2/4/3 → 9.96/10.19/-19.35 exact). Future picks
(match_id>40) correctly stay pending.

CARRIED CAVEAT: this OVERLAPS clv_tracker.py, which already writes/settles bet_ledger.csv +
clv_report.csv (DL-10/DL-11). settle_bets.py is currently a standalone fix; it is NOT yet wired
into refresh_all.ps1 or live_update orchestration. Decision pending with Jake: does settle_bets
REPLACE clv_tracker's settlement, or run alongside it? Until resolved, run settle_bets manually
and do not double-settle.

### DL-16 — QUEUED (Jake ask, 2026-06-22): underdog "+0.5 insurance" companion tracker
Jake: when the model picks an underdog over a superior team (e.g. Senegal over Argentina),
also track a smaller-stake Senegal ML PLUS a Senegal +0.5 (wins if Senegal wins OR draws —
i.e. it cashes whenever the favorite does NOT win). Captures the draw outcome the straight ML
loses. CONFIRMED FEASIBLE from existing data with no new odds source: wc2026_predictions.csv /
prediction_ledger.csv carry full p_home/p_draw/p_away AND market mkt_home/mkt_draw/mkt_away, so
+0.5 model-cover = p_team + p_draw, market +0.5 implied = mkt_team + mkt_draw, edge = the diff,
fair decimal = 1/cover. Settlement reuses settle_bets' 90-min result_side: WIN if result_side ∈
{team, draw}, else LOSE.

DESIGN CONFIRMED with Jake (2026-06-22):
- TRIGGER (config-driven, edge-gated — only recommend a leg with positive edge): big-dog tier
  `market_implied_win ≤ 0.30` → ML + +0.5 insurance; toss-up tier `0.30 < implied < 0.50` →
  track BOTH ML and +0.5 as separate recommendations; favorites → no insurance leg.
- SIZING = JOINT multi-outcome Kelly, NOT independent per-leg (ML and +0.5 share the `win`
  state; independent Kelly double-counts it and over-stakes). Solve f₁,f₂ maximizing
  p_w·ln(1+f₁(d₁-1)+f₂(d₂-1)) + p_d·ln(1-f₁+f₂(d₂-1)) + p_l·ln(1-f₁-f₂) (concave, scipy),
  then apply the project's existing ½-Kelly fraction + 5% per-event cap. Rationale: harvests
  the model's full W/D/L disagreement AND de-variances it (the "consistently profitable" ask).
- TRACKING: 3 separate ledgers/streams (ML-only, +0.5-only, combined joint-Kelly strategy) for
  honest risk-adjusted attribution (Sharpe / max-drawdown A/B), unified into ONE dashboard panel
  (three overlaid equity curves). Separate technically, single digestible visualization.
- Config defaults (config.json): big_dog_threshold=0.30, tossup_band=[0.30,0.50],
  kelly_fraction=0.5, cap=0.05.
- HONEST CAVEAT: components (double chance, Kelly) are textbook; the defensible IP is the
  system — a calibrated W/D/L distribution that measurably disagrees with the market (DL-10)
  + correlated multi-outcome Kelly to express it, proven by realized CLV. It only monetizes an
  edge if the dog edge is REAL, which DL-10/§6.3 say is still unproven (decided by this
  tournament). The +0.5 layer de-variances an edge; it does not create one.

STILL OPEN before build: decision on the ingest/settlement canonical path (replace vs coexist —
the local ESPN/clv_tracker loop vs the cloud football-data/settle_bets path) gates the "auto-run"
wiring. NOT YET BUILT. See HANDOFF §7 queue item 9.

### DL-17 — Split decision (ingest vs settlement) + health-check monitor (2026-06-22)
Resolved the DL-14/DL-15 overlap between the existing local stack and the new cloud stack,
split by job (Jake's call):
- SETTLEMENT — single settler = `settle_bets`. Wired to run AFTER `clv_tracker` in
  `_active_scripts/refresh_all.ps1` (clv_tracker keeps CLV + ledger; settle_bets has the last
  word on status/pnl) and added as a step in daily_update.yml. Never run two settlers.
- INGEST — ESPN (`fetch_live_results`, local 30-min loop) stays PRIMARY. The cloud
  football-data.org path (daily_update.yml) is set to workflow_dispatch ONLY (schedule commented
  out) until the API is verified (WC2026 vs WC competition code, free-tier coverage) and Jake
  chooses to cut over — prevents two pushers fighting wc2026_live_results.csv and a misconfigured
  API silently blacking out. Re-enable = uncomment the cron block.

MONITOR — `src/models/health_check.py` (NEW): the "flag me if anything is broken" signal.
Checks, each mapping to a version-update risk:
  • live-results schema + non-empty (ingest wrote garbage / API blackout)
  • coverage/staleness — fixtures past their date with no result ingested
  • pending-after-played — a pick whose match has a result but status=pending (the CRITICAL bug
    regressing) → ERROR
  • settlement math — pnl_usd == (decimal-1)*stake on WON, -stake on LOST → ERROR on mismatch
  • status/pnl consistency — VOID/pending must have pnl 0
  • ledger consistency — clv_report vs bet_ledger_settled settled-count divergence
Writes data/processed/model_health.json (status HEALTHY/DEGRADED/BROKEN + issues + metrics; the
dashboard Model-Health panel reads it). Exit 1 on BROKEN. Wiring of the flag:
  • daily_update.yml runs it before commit — a non-zero exit FAILS the workflow and GitHub emails
    the repo owner (no extra infra; works with the PC off). Broken state is never pushed.
  • refresh_all.ps1 runs it each cycle and logs `WARN !!! PIPELINE HEALTH BROKEN ...` to refresh.log.
Verified: HEALTHY/exit 0 on current state (40 results, 0 pending-after-played, net +$11.82);
injected faults (match_id 30→pending, 36→bad pnl) correctly flipped it to BROKEN/exit 1 with the
exact match_ids, then restore returned HEALTHY/exit 0.

NOTE: the refresh_all.ps1 edit was made to the live file in _active_scripts/ (where a pending
repo reorg moved it). That reorg (the _active_scripts move, _archive project merge, CONTEXT_V2–5
archival) is left for Jake to commit deliberately — not swept into this commit, to avoid pushing
a large archived sub-project into the public repo.

### DL-18 — Underdog +0.5 insurance tracker SHIPPED (Jake ask, 2026-06-22)
Built the DL-16 design. For each model underdog pick, track ML (team win) + a +0.5
(team win-or-draw — cashes whenever the favorite does not win), sized JOINTLY (correlated
Kelly), settled against results, with three bankroll streams (ML-only / +0.5-only / combined)
for honest attribution. Files:
- `data/processed/config.json` (NEW): unit_size_usd + insurance block (big_dog_threshold 0.30,
  tossup_band [0.30,0.50], min_leg_edge 0.02, kelly_fraction 0.5, per_leg_cap 0.05, bank0 100).
- `src/models/insurance_sizing.py` (NEW): joint multi-outcome Kelly via scipy (maximizes
  expected log-growth over win/draw/loss for the two correlated legs), plus solo_kelly for the
  benchmark streams and an explain() rationale. Worked example confirms joint sizing CUTS the ML
  stake (2.0% vs 4.1% solo) because +0.5 already covers part of the win — the over-bet fix.
- `src/models/insurance_tracker.py` (NEW): reads prediction_ledger.csv (model + Shin-fair market
  1X2), derives +0.5 from mkt_team+mkt_draw, tiers/edge-gates, sizes, settles via settle_bets'
  result_side, compounds 3 bankrolls → insurance_ledger.csv + insurance_summary.json.
- `src/models/build_insurance_html.py` (NEW) → `outputs/insurance_tracker.html`: self-contained,
  mobile-friendly, plain-English explainer + 3 KPI cards + 3 overlaid equity curves (Chart.js CDN)
  + tier-badged recommendation table with rationale. "Cognitive and accessible" per the ask.
- Wired into refresh_all.ps1 (auto-run after settle_bets + prediction_tracker); insurance_summary.json
  added to the push triggers.
First run: 17 recs (6 settled). All 6 settlements verified against actual scores (incl. match_id 32
correctly dropping the ML leg — edge 1.9% < 2% gate — and keeping +0.5). CAVEAT surfaced everywhere:
fair (de-vigged) odds → research-grade P&L; tiny sample; the dogs settled so far mostly WON outright
so +0.5 insurance hasn't paid yet (its value is in draws). Edge realness still per DL-10.
Also wired as an INSURANCE tab in the main quant_dashboard.html (build_dashboard.py: payload
loads insurance_summary.json; inline-SVG 3-line equity chart so the dashboard stays self-contained
/ offline — NO Chart.js CDN; standalone outputs/insurance_tracker.html kept too).
OPEN: possible vig model for realistic pricing once live book odds for dogs are stored.

## Phase 1 Results (2026-06-10, pre-tournament — real futures odds, 17.5% vig, Shin de-vig)
- Credible (non-tail) positive-EV futures: Mexico +6.2pp edge (≤ known bias!), Japan +4.7pp,
  USA +2.9pp, Spain +1.9pp (EV +0.014 at 5.50 — thin), Morocco +0.9pp. Iran/Korea/Canada sit
  at 2.0–2.3% model prob with 200:1+ odds — just above the tail floor, treat skeptically.
- Model fades vs market: France −7.5pp, Portugal −5.4pp (known underrating pattern).
- Match-level: 216 outcome rows, all model_estimated (EV ≡ 0) until real lines are entered
  in `data/raw/wc2026_match_odds.csv` (template auto-created, keyed by match_id).
