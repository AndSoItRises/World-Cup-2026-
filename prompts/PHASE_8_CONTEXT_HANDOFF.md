# Phase 8 — Context Handoff (session checkpoint, 2026-06-22)

> Written deliberately at a clean stopping point so the next session starts cold from this file.
> Pairs with the spec `prompts/PHASE_8_QUANT_DESK_PRODUCT.md` and decision log `CONTEXT_V6.md` (DL-19, DL-20).
> **For the next window: open in PLAN MODE, review §4 (decisions for Jake), get approval, then build.**

---

## 1. What this session accomplished (backend foundation — shipped + committed)

Both headline trackers from Phase 8 §2 are built, run cleanly, and are verified. They are
**standalone Python**; no dashboard or pipeline wiring yet (that's the next session, by design —
Jake wants to approve the product/IA surface in plan mode).

### 2A — Model Performance Tracker `src/models/model_performance.py` (DL-19)
- **The honest, REALIZED scorecard**: "is the model actually good?" Settles real results, not a sim.
- Universe = **every** value game from `prediction_ledger.csv` (model + Shin-fair market 1X2 for
  every match), best value side per match, gated on config thresholds. Real book line where it
  exists (flag `real`), else de-vigged fair price (flag `fair`) — **this fixes "not enough games."**
- 4 staking strategies (Flat / ¼ / ½ / Full Kelly, capped 5%), each with a **plain-English
  explainer** (Jake's ask), reporting ROI%, net $, hit-rate, max drawdown, per-bet Sharpe-ish, CLV alignment.
- Outputs: `data/processed/model_performance.json` + `model_performance_ledger.csv`.
- **First run:** 40 value bets, 23 settled (21 real + 2 fair). **Model +16.6% realized (flat)**;
  flat has best Sharpe; Kelly grows more (+30–59%) at ~24% drawdown. Run: `python -m src.models.model_performance`.

### 2B — MAIN $500-from-today Bankroll Sim `src/models/bankroll_sim.py` (DL-20)
- **The hero number**: "$500 today → ?" Starts $500 at a **frozen** `start_match_id` (=41, last
  played +1), persisted so earlier games are never retro-credited.
- **Selective** (model may pass): 3 variants — **BET only**, **BET + LEAN**, **BET + select LEAN**
  (LEAN score ≥ 4.5). ½-Kelly cap 5%. Realized curve (flat $500 until knockouts land) + forward
  **Monte-Carlo projection cone** (P5/P50/P95, seed-fixed).
- Outputs: `data/processed/bankroll_500.json` + `bankroll_500_ledger.csv` (idempotent — re-runs
  hold start_id=41, 20 ledger rows). Run: `python -m src.models.bankroll_sim`.
- **First run:** BET-only 1 pick (P50 $475, P(profit) 42%); BET+LEAN 15 (P50 $675, 79%);
  BET+select-LEAN 4 (P50 $510, 59%).

Decisions logged as **DL-19 / DL-20** in `CONTEXT_V6.md`; HANDOFF §7 item 12 updated; committed.

---

## 2. What still needs to be done (Phase 8 remaining — the product surface)

In rough priority order:

1. **Wire into the pipeline** — add `model_performance.py` + `bankroll_sim.py` to
   `_active_scripts/refresh_all.ps1` (after `insurance_tracker`), add their JSONs to the push
   triggers, and extend `health_check.py` to cover the two new outputs (staleness / schema).
2. **DESK CALLS landing draws a conclusion** (spec §3) — a top **verdict banner**
   ("Model +16.6% realized · CLV +Z% · $500 → \$___ · today's best call: ___"), fold the $500
   hero value + sparkline into the landing, tighten "Next 5 games."
3. **Surface the trackers on the dashboard** (`src/models/build_dashboard.py`) — load the two new
   JSONs into the payload; add a **Track Record** view (2A: strategy comparison chart +
   explainers + verdict) and a **$500 Bankroll** hero (2B: realized curve + projection cone, all
   three variants). **Inline-SVG only — no CDNs** (mirror the existing insurance equity renderer
   ~`build_dashboard.py:409`). Replace/augment the old `runBankroll` (real-only, line ~784).
4. **Tab audit / leaner IA** (spec §4) — propose keep/merge/cut, every kept tab leads with a
   plain-English conclusion. **Get Jake's eyes before deleting anything** (spec §4 hard rule).
5. **JS desk-rule mirror** — if any verdict logic changes, keep the JS mirror in `build_dashboard.py`
   in sync with `desk_call.py` (HANDOFF §8).
6. **Future/research note** (spec §6) — separate doc: agentic-loop architecture, observability.
   Design only; do not build.

---

## 3. Key facts / gotchas for the next agent (save yourself the rediscovery)

- **48-team World Cup → group stage is match_id 1–72** (12 groups × 4). Results land through
  match_id **40**; matches 41–72 are remaining group games, knockouts are 73+.
- Settlement: reuse `settle_bets.load_results()` → `{match_id: {result_side, ...}}` and `_result_side`.
  Already imported by both new scripts and `insurance_tracker.py`.
- `value_bets.csv.market_source` is currently **always `'real'`** — there is no separate fair-priced
  file. Fair coverage comes from `prediction_ledger.csv` (`mkt_home/draw/away` = Shin-fair). That's
  why 2A reads the ledger, not value_bets, for its universe.
- Dashboard data load pattern: `load()` / `_load_json()` near `build_dashboard.py:75`; payload
  assembled in `build_payload()` ~line 98; rebuild with `python -m src.models.build_dashboard`.
- Console is cp1252 — always `$env:PYTHONUTF8="1"` (scripts print box-drawing chars).
- Constraints still in force: V4 frozen; single settler (`settle_bets`); self-contained/offline
  dashboard; never present a number without its caveat (fair-odds P&L is research-grade).

---

## 4. Decisions / considerations for Jake (please weigh in — plan mode next session)

**Product**
- **$500 headline variant.** BET-only is currently **1 pick** (only match 43 is a forward BET until
  knockouts get priced), so its projection cone is degenerate (single win/lose). Options: (a) make
  **BET+LEAN** the headline for now, (b) show all three side-by-side equally, (c) keep BET-only but
  add a "thin board — fills as fixtures price" note. *My lean: (b) show all three, label BET-only
  "highest conviction (currently 1 pick)."*
- **Track Record framing.** 2A's honest result is that **flat staking beats Kelly on risk-adjusted
  terms** so far. Good credibility story (we're not overselling Kelly) — surface it plainly?
- **Projection odds source.** Currently fair-de-vig where no real line (research-grade, slightly
  optimistic). Fine for now; revisit if/when live knockout book odds are stored.

**Information architecture (spec §4 — needs your sign-off before any tab is cut)**
- Proposed lean IA: **Verdict/Desk → $500 Bankroll → Track Record → Today's Board → Lab**
  (DIVERGENCE/UNCERTAINTY/MOVEMENT+ARB/NOTES demoted behind one "Lab" area). MOVEMENT+ARB needs a
  2nd book to mean anything; NOTES is the raw handoff doc. **Confirm what you rely on before I cut.**

**Administrative / professional (dev best-practice, since you're learning)**
- **Pipeline wiring is a real commit boundary** — adding scripts to `refresh_all.ps1` changes what
  auto-runs and auto-pushes every 30 min. Worth doing deliberately (and watching `refresh.log` once)
  rather than bundling silently.
- **Branch vs main.** This session committed straight to `main` (project convention). For the larger
  dashboard/IA overhaul, consider a short-lived feature branch so you can preview the new dashboard
  before it goes live on GitHub Pages.
- **Health coverage = trust.** Before the agentic-loop vision (spec §6), every new output should be
  under `health_check.py` so automation is observable. Cheap to add now, prerequisite later.

---

## 5. Exact run commands (next session quick-start)
```powershell
cd "C:\Users\jakeh\OneDrive\Documents\Claude\Projects\World Cup 2026 Model"
$env:PYTHONUTF8 = "1"
git status
.\venv\Scripts\python.exe -m src.models.model_performance     # 2A → model_performance.json
.\venv\Scripts\python.exe -m src.models.bankroll_sim          # 2B → bankroll_500.json
.\venv\Scripts\python.exe -m src.models.build_dashboard       # rebuild dashboard (after wiring)
```
