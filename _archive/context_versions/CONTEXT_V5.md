# WC2026 Model — V5 Context Document

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
World Cup 2026 match prediction + bracket simulation. V1 baseline → V2 made it intelligent (ELO,
opponent-weighted form, draw fix, tuning) → V3 fixed data bugs → **V4 added genuinely new signal:
squad VALUE + DEPTH from FIFA/EA ratings, the first model gain since V2.** V5's mandate is open —
pick the next-highest-leverage lever (see seeds below).

---

## Inherited State (V4 — tagged v4.0, closed)

### What V4 delivered (see CONTEXT_V4.md DL-02 + End-of-Version Review)
- **Squad strength + depth feature**, leakage-safe, from FIFA/EA video-game ratings (2005→FC26).
  `src/features/build_squad_strength.py` builds `data/processed/squad_strength_by_year.csv`;
  `feature_engineering.add_squad_strength` merges it (asof backward) + a `squad_both_covered` flag.
- 7 squad model features adopted (strength/depth per-side + diffs + coverage flag). XGB CV log loss
  0.8701→0.8668 (+0.0033); ensemble wins 3/4 validate metrics; WC2022 finals LL −0.016; squad features
  at XGB importance rank 3 & 5.
- **Biases reduced** in the live forecast: model↔market correlation 0.665→0.838; Mexico edge
  +10.0→+6.1pp, France −10.0→−7.7pp. WC2026 top: Spain 18.5%, England 8.9%, Mexico 7.1%, France 7.1%.
- Models: v4 = v3 + squad. Prod retrained to v4 (old prod → *_prod_v3.*). v1/v2/v3 untouched.
- **Quant learning stack built** (Jake's cowork track): `signal_test.py` (orthogonality/IC, Step 6),
  `calibration.py` (Step 1), `backtest.py` (walk-forward, Step 3), `market_backtest.py` (IC vs market, Step 2).

### Known limitations carried into V5
1. **Squad coverage ~34% of all competitive matches** — FIFA omits minnows; the gain is concentrated in
   real-nation matches (incl. every WC2026 tie). Filling FIFA23/FC24/FC25 editions + missing nations
   (Jordan/Uzbekistan have <11 rated players) would raise coverage and likely the overall test-LL gain.
2. **macro_conf_ll dipped slightly** (0.8628→0.8653) even as the bracket improved — worth understanding.
3. Biases reduced, not closed — ELO still dominates; squad value is a complementary prior.

---

## V5 Goals (DRAFT — refine in brainstorm)
Candidate levers, each ⇒ validate-or-cut with the V4 tooling (`signal_test.py` is a one-step orthogonality gate):
- **Confederation / schedule-strength feature** — opponent-confederation average ELO; the obvious next new
  signal (the training ranking CSVs carry a `confederation` column). Distinct from the cut conf_match_pct.
- **Extend squad coverage** — add FIFA23/FC24/FC25 editions + thin-coverage nations; re-test whether the
  squad gain widens on the full match set (not just covered).
- **Ensemble reweight** — XGB improved in V4; the fixed 0.275/0.275/0.45 blend may now be suboptimal. Re-run
  a stacked/weight search (stacking.py exists) — but validate-or-cut (V3 found fixed weights near-optimal).
- **P4 honest edge test** — acquire historical bookmaker odds to backtest model-vs-market IC on WC2018/2022
  before any betting work (gated by data, like V4's squad history was — FIFA ratings were the unlock there).

DEFERRED until a clean out-of-sample edge is demonstrated: Kelly bet sizing, betting product, sim engine.

---

## V5 Decision Log

### DL-01 — V5 mandate: test the remaining levers to settle "is this the final model?"
After V4 shipped the squad value+depth gain, the open question was whether more accessible signal
exists. V5 ran the four candidate levers each under validate-or-cut. The pattern of results is itself
the answer (see DL-02..05 and the End-of-Version verdict).

### DL-02 — Lever 2 (extend squad coverage 2023–2025): DATA-BLOCKED
Goal: de-stale the test window (2023–25 currently uses FIFA22 carried forward) + add nations by
ingesting FIFA23/FC24/FC25 editions. Outcome: those editions are Kaggle-auth-gated or sofifa
scrape-only — not cleanly/reproducibly fetchable (FIFA22 via abineshta and FC26 via EAFC26-DataHub
were the only raw-hosted sofifa mirrors found). Did NOT scrape (brittle/ToS-gray for marginal gain on
an already-validated feature). Structural note: more editions wouldn't fix the ~66% of competitive
matches that involve non-FIFA minnows — that coverage gap is inherent to FIFA data. Anchors stand at
2020 (lbenz730) / 2022 (FIFA22) / 2026 (FC26).

### DL-03 — Lever 1 (confederation / schedule-strength feature): CUT (redundant)
Built `add_confederation_strength` (prior-year confederation average ELO, leakage-safe) as the "how
strong is the region you farm results in?" prior. `signal_test` (now judging candidates against the
full v4 base incl. squad): incremental IC ≈ 0 / negative, ΔLL −0.0005 (all) / −0.0023 (covered) — it
HURTS slightly. **Why:** squad value already measures true quality independent of schedule, so it
subsumes the regional proxy. Cut from the pipeline (function retained for reference, like conf_match_pct).

### DL-04 — Lever 3 (ensemble reweight): KEEP fixed weights
`reweight_v5.py` — honest two-fold held-out search (tune weights on one test half, score the other).
Tuned weights were unstable across folds (one fold zeroed LGBM — overfitting) and the mean held-out
gain was +0.0010, below the adopt threshold. Kept the fixed 0.275/0.275/0.45 blend — consistent with
V3's stacking finding that the fixed blend is near-optimal.

### DL-05 — Lever 4 (historical-odds edge backtest, P4): DATA-GATED
Goal: backtest model-vs-market IC on past matches/tournaments (WC2018/2022) — the gate for any betting
work. Outcome: free historical odds sources are domestic-club-only (football-data.co.uk, xgabora) or
results-only (openfootball, jfjelstul); historical INTERNATIONAL 1x2 odds are effectively paid/scrape.
`market_backtest.py` provides the IC harness + WC2026 cross-section now; the match-level backtest drops
in the moment international historical odds are obtained. Honest, not closed.

---

## V5 Phase Status

| Lever | Description | Status |
|---|---|---|
| L2 | Extend squad coverage (FIFA23/FC24/FC25) | ⬛ data-blocked (DL-02) |
| L1 | Confederation/schedule-strength feature | ✂️ CUT — redundant (DL-03) |
| L3 | Ensemble reweight | ➖ kept fixed (DL-04) |
| L4 | Historical-odds edge backtest | ⬛ data-gated (DL-05) |

**The model did not change in V5** — every accessible lever came back CUT / flat / data-blocked.
That convergence is the evidence the model is at its practical ceiling (see verdict).

---

## ══ END-OF-VERSION REVIEW ══

**Verdict: v4 is the final VALIDATED model for what's rigorously achievable with accessible data.**

The case (every accessible lever, tested under validate-or-cut, came back negative):
- The obvious next feature (confederation-strength) is **redundant** — squad value already subsumes it.
- Reweighting the ensemble does **not** robustly help (overfits).
- Extending squad coverage is **data-blocked**, and wouldn't fix the structural minnow gap anyway.
- The model now **largely agrees with the market** (model↔market corr 0.838) — limited "free" edge left.
- Proving/refuting a real edge needs **historical international odds** (paid) — a data problem, not a
  modelling one. Same shape as V4's squad-history block; if that data arrives, `market_backtest.py` +
  a `bet_sim.py` (Kelly) close it out.

So further gains require new *data* (full-universe historical squad values; historical odds), not new
model cleverness. Within accessible data, the lever set is exhausted. **This is the practical ceiling.**

What V5 leaves behind: a sharper `signal_test` (judges new features against the live model), a
reusable honest reweight harness, a documented redundant feature (so it's not re-tried), and a clear
data shopping list for any future unlock. Model artifacts unchanged from v4 (prod = v4).

Future-unlock candidates (all data-gated, not cleverness-gated): historical international odds → edge
test + Kelly; full-universe historical squad/market values → lift squad coverage past ~34%.
