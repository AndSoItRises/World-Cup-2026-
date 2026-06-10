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
| 3 | `outputs/quant_dashboard.html` — interactive research dashboard | ⬜ |
| 4 | `src/models/uncertainty.py` — aleatoric/epistemic quantification | ⬜ |
| 5 | `src/models/clv_tracker.py` — closing line value tracking | ⬜ |
| 6 | Signal tests for 5 new candidate features (orthogonality gate) | ⬜ |

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

## Phase 1 Results (2026-06-10, pre-tournament — real futures odds, 17.5% vig, Shin de-vig)
- Credible (non-tail) positive-EV futures: Mexico +6.2pp edge (≤ known bias!), Japan +4.7pp,
  USA +2.9pp, Spain +1.9pp (EV +0.014 at 5.50 — thin), Morocco +0.9pp. Iran/Korea/Canada sit
  at 2.0–2.3% model prob with 200:1+ odds — just above the tail floor, treat skeptically.
- Model fades vs market: France −7.5pp, Portugal −5.4pp (known underrating pattern).
- Match-level: 216 outcome rows, all model_estimated (EV ≡ 0) until real lines are entered
  in `data/raw/wc2026_match_odds.csv` (template auto-created, keyed by match_id).
