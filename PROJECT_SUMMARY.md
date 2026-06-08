# World Cup 2026 Model — Project Summary & Retrospective

*A plain-English walkthrough of what we built, why we built it that way, what paid off, and
why v4 is the final validated model. Written for a quick read and a GitHub reviewer.*

---

## TL;DR

We built a World Cup 2026 match-prediction + bracket-simulation model and improved it across five
versions. The headline arc: **V1** got a baseline working, **V2** made it genuinely intelligent,
**V3** proved it was data-correct and at its ceiling for the existing features, **V4** broke that
ceiling by adding a genuinely new signal (**squad value + depth** from FIFA/EA ratings), and **V5**
tested every remaining lever and confirmed there's no more *accessible* signal to add.

**Final model = v4.** It predicts at ~62% accuracy / 0.846 log loss, and — the part that matters —
its WC2026 forecast now agrees with the betting market at **0.84 correlation** (up from 0.66), with
its two worst historical biases (overrating CONCACAF, underrating Euro powers) materially corrected.

**Is it "final"?** Yes, for what's rigorously achievable with freely available data. Every V5 lever
came back redundant, not-robust, or data-blocked. Further gains need new *data* (historical odds;
full-universe squad values), not cleverer modelling.

---

## The roadmap: what we did each version, and why

| Version | What we did | Why | Result |
|---|---|---|---|
| **V1** | Baseline XGBoost on FIFA rank + simple form | Get an honest, scored baseline before adding complexity | 60.6% acc / 0.8605 LL |
| **V2** | Added ELO, opponent-quality-weighted form, a draw fix, hyperparameter tuning; built XGB+LGBM+Dixon-Coles ensemble | A baseline that can't see opponent quality or draws is leaving signal on the table | 62.1% acc / 0.8458 LL — the big jump |
| **V3** | Fixed data bugs (corrupt FIFA-rank column, team-name mismatches → silent NaNs), proved the model was at its feature ceiling | You can't trust a model built on dirty joins; prove correctness before chasing more accuracy | Metrics held; **proved reweighting existing features was exhausted** |
| **V4** | **Added squad value + DEPTH** from FIFA/EA video-game ratings, leakage-safe; validated and adopted | V3 showed the only lever left was *new signal* — something that sees a team's true talent independent of who it farmed results against | **First gain since V2:** CV +0.0033, biases corrected, model↔market corr 0.66→0.84 |
| **V5** | Tested 4 remaining levers (extend squad coverage, confederation-strength, ensemble reweight, historical-odds edge test) | Settle whether the model is genuinely finished or has more in it | **All CUT / flat / data-blocked → practical ceiling confirmed** |

---

## The payoff strategy (the core idea)

Two principles drove every decision:

**1. You are not predicting reality — you are trying to be less wrong than a price.**
The betting market already embeds squad value, injuries, and reputation. So "the model is accurate"
was never the goal; "the model sees something the price doesn't, out of sample" was. This is why we
scored with **log loss** (which only rewards honest probabilities), not accuracy (which rewards
confident guessing), and why the market is our benchmark, not a stopwatch.

**2. Validate-or-cut: a new idea earns its place only by beating the baseline out of sample.**
Every feature was put through the same gate: does it improve cross-validated log loss **and** move the
WC2026 bracket the right way? If not, it was cut — no exceptions, no "but it feels useful." Things we
*cut* this way: a draw classifier (V3), a confederation-match-% feature (V3), time-decay weighting
(V3), confederation-strength (V5), ensemble reweighting (V5). The discipline is the product.

**Where the payoff actually came from:** the single highest-leverage move was identifying that the
model's errors weren't random — they were a *specific bias* (CONCACAF teams overrated, Euro powers
underrated) caused by a *specific blindness* (the model couldn't see squad talent, only results). That
diagnosis told us exactly what new signal to hunt for. Squad value was the cure because it measures
quality independent of schedule — which is precisely what the biased teams' records hid.

---

## Working techniques (the reusable quant methods)

These are the methods that did the heavy lifting — each is now a script in the repo:

- **Orthogonality / incremental IC** (`signal_test.py`) — the make-or-break test. A new feature is only
  worth adding if it carries signal *the existing features don't already have*. We regress the
  candidate on the current model's features, take the residual, and measure the residual's correlation
  with outcomes. Squad value passed (incremental IC +0.12 on covered matches); confederation-strength
  failed (≈0). **Raw correlation with winning lies; incremental information doesn't.**
- **The coverage flag trick** — squad data only exists for ~⅓ of matches (FIFA omits minnows). Naively
  this diluted the signal to zero. Adding a `squad_both_covered` flag let the model *gate* on whether
  the squad data is real, recovering the full signal where it exists. A small idea with a big payoff.
- **Leakage-safe joins (`merge_asof` backward)** — every time-varying feature (FIFA rank, squad rating)
  is joined so a match only ever sees data available *before* it. FIFA editions stamped with a release
  date; prediction uses the latest pre-tournament edition. No future information ever leaks backward.
- **Name audits before every merge** — a non-negotiable after V3 found a team-name mismatch silently
  turning real ranks into a "150" sentinel. Every new data source is audited against the canonical
  names before joining (V4: 46/48 WC teams matched, the other 2 deliberately sentinel'd).
- **Market as benchmark + Information Coefficient** (`market_backtest.py`) — measure the model's
  *ordering* skill (rank correlation vs outcomes) and compare its WC2026 view to the de-vigged market.
  The model↔market correlation (0.84) is our best single proxy for "the model is sane."
- **Walk-forward backtesting** (`backtest.py`) — retrain on the past, test only on the unseen next
  season. Demonstrated (in our own data) how peeking at the test set flatters results — the overfitting
  trap, quantified.
- **Honest reweighting** (`reweight_v5.py`) — tune ensemble weights on one half of the test set,
  evaluate on the other. Stopped us adopting weights that looked good in-sample but didn't generalize.

---

## Every key insight

1. **Accuracy is a trap; log loss is honest.** We never optimized accuracy — a calibrated 55% beats a
   miscalibrated 90%.
2. **The model's errors were biased, not random** — overrating CONCACAF, underrating Euro powers. Naming
   the bias was what made it fixable.
3. **The bias was a blindness:** the model saw *results* (ELO/form) but not *talent*. CONCACAF teams
   farm weak opponents → inflated records; Euro powers have elite squads the model couldn't see.
4. **Squad value cured it** — and **depth** (bench quality, not just the starting XI) carried its own
   weight (XGB importance rank 5). FIFA/EA ratings were the leakage-safe historical proxy that
   Transfermarkt history wasn't.
5. **Signal can be real but coverage-limited.** Squad value adds nothing across all matches (minnows
   dilute it) yet is strongly orthogonal where data exists — the coverage flag bridged that.
6. **Squad value subsumed confederation-strength.** Once the model can see talent directly, "how strong
   is your region" adds nothing — a clean demonstration of orthogonality.
7. **The ensemble blend was already near-optimal.** Re-tuning overfits; the fixed 0.275/0.275/0.45 holds.
8. **The model now mostly agrees with the market (corr 0.84).** That's both a success (it's sane) and a
   ceiling signal (little free edge remains).
9. **The remaining unknowns are data problems, not modelling problems** — historical odds (to prove an
   edge) and full-universe squad data (to lift coverage). Neither is solved by a better algorithm.

---

## Honest limitations (what we did *not* solve)

- **Squad coverage is ~34% of all competitive matches** — FIFA omits minor nations. The gain is
  concentrated in matches between real footballing nations (which includes *every* WC2026 tie), but
  overall test log loss barely moved and the confederation-balanced metric dipped slightly.
- **Biases reduced, not eliminated** — ELO still dominates; squad value is a complementary prior. Mexico
  is still overrated vs the market (+6pp, down from +10), France still underrated (−7.7pp, down from −10).
- **No proven betting edge** — we deliberately did *not* claim one. Proving it needs historical
  international odds we don't have; the harness is built and waiting.

---

## What would unlock more (a data shopping list, not a modelling to-do)

1. **Historical international 1x2 odds** (WC2018/2022, qualifiers) → run the model-vs-market backtest in
   `market_backtest.py`, then Kelly bet-sizing (`bet_sim.py`). This is the real "is there an edge?" test.
2. **Full-universe historical squad/market values** (incl. minor nations, 2023–25 editions) → push squad
   coverage past 34% and likely widen the validated gain.

---

## Repo map (how to run it)

Setup: activate the venv, set `PYTHONUTF8=1` (Windows), run modules from the repo root.

```
python -m src.features.data_cleaning          # clean + split match data
python -m src.features.build_squad_strength   # V4: build squad strength+depth table (downloads FIFA data)
python -m src.features.feature_engineering    # build train/test feature tables
python -m src.models.retrain_all              # train production XGB+LGBM+Dixon-Coles (v4 features)
python -m src.models.predict_wc2026           # per-fixture probabilities
python -m src.models.monte_carlo              # 10k-sim bracket → tournament win probabilities
python -m src.models.market_divergence        # model vs betting market
```

Validation & research: `signal_test.py` (orthogonality), `train_v4.py` / `validate_v4.py` (the V4
adoption), `bracket_v4.py` (bias-correction check), `reweight_v5.py` (ensemble tuning), `backtest.py`
(walk-forward), `calibration.py` (proper scoring).

Per-version detail lives in `CONTEXT_V2..V5.md` (decision logs + phase status). The quant learning
curriculum is in `../CONTEXT_QUANT_STACK.md`.

---

*Models: v1/v2/v3/v4 all preserved at their own paths; production = v4. Tagged `v4.0` (model) and
`v5.0` (ceiling confirmation + this retrospective).*
