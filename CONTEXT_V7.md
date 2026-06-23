# CONTEXT_V7 — Dashboard as a Product

**Scope:** front-end only. The model (V4 ensemble) and V6 calibrator are unchanged — see
[CONTEXT_V6.md](CONTEXT_V6.md) and [HANDOFF.md](HANDOFF.md) for the model/pipeline state. This doc captures
the dashboard product overhaul applied to `outputs/quant_dashboard.html` so the next session inherits the
reasoning, not just the diff.

---

## 1. What changed (Phase 9 — product overhaul)

A pure front-end overhaul of `outputs/quant_dashboard.html`. No Python, data-fetch, de-vig, Kelly, Monte
Carlo, or bias-disclosure logic was touched.

- **Hero glance panel** (above the tabs): Hit Rate, Simulated ROI, Open Exposure, an auto-generated verdict
  sentence, a cumulative-P&L sparkline (inline SVG), and a live "Updated Xs ago" counter. All values are
  derived from the existing `D.clv` settled/pending rows + the $1,000 desk bankroll.
- **STRATEGIES tab** (was SCANNER): three profiles — CONSERVATIVE, VALUE HUNTER, **+0.5 INSURANCE** — each a
  filter over `D.desk` (or `D.insurance.ledger`) with plain-language rationale parsed from the existing
  `why`/`cautions` flags, plus a collapsible "full model detail" per card. The raw EV scanner + futures
  tables are preserved behind a `<details>`.
- **LINE MOVEMENT tab** (was MOVEMENT+ARB): the arbitrage section was removed (see §3) and replaced by the
  working line-movement table with a proper empty state. No more "–" dashes.
- **Liveness:** adaptive polling (60s inside a match window, 4min otherwise), a refresh countdown,
  flash-on-update for the hero stats, ticker hover-pause, and per-module freshness stamps.
- **Extra UX:** `localStorage` + URL-hash routing (active tab / PASS toggle / strategy profile / bankroll
  inputs persist and are shareable), live in-play "● LIVE / FT" badges from ESPN status, local-timezone
  kickoff times with a countdown, jargon tooltips, sticky/scrollable tab nav, `role="tab"` + keyboard
  switching, a hide-PASS-by-default toggle, a sticky disclaimer footer, and a 900px mobile fix.

Implementation note: the file has a single ~193k-token inline data line (`const D = {...}`), which makes the
Read/Edit tools exceed their token cap. Edit it with anchored Python string transforms (see
`$CLAUDE_JOB_DIR/tmp/patch*.py` from the overhaul session) rather than line-based tools. Backup first.

---

## 1b. What changed (Phase 10 — "Next Up" slate overhaul)

Pure front-end again (`outputs/quant_dashboard.html`). The next-slate of games was promoted from a
buried secondary card into the page's actionable centerpiece.

- **New first tab `NEXT UP`** (`#tab-next`, first in the `TABS` array → the default landing tab for new
  visitors). Each match is an **action-first card**: a prominent BET / LEAN / PASS headline (chip +
  selection + odds + **stake** + desk score), *then* the justification. PASS states say the reason plainly.
- **Pick-vs-call confusion resolved.** When the model's highest-probability outcome differs from what the
  desk actually bets (or when the desk passes), a secondary `.nupick` line explains it — e.g. "model leans
  Draw 34%, but the desk passes — draw upweight is documented model bias, corrected at inference."
- **One stacked 3-way bar** (`bar3()`, home=cyan / draw=amber / away=green) with the market-fair split
  overlaid as white ticks, plus a legend of model% (mkt% in parens). Replaces the three separate `bar()` rows
  (kept only for the MATCH PROJECTIONS tab).
- **Live ticking countdown.** The existing 1-second `setInterval` now also refreshes every `[data-ko]` span
  via `koLabel(m)`, so "kicks off in…" ticks down and in-play score/clock updates without a full re-render.
- **Context chips:** line-movement arrow (toward/against model, from `liveMove`/`embMove`), an entropy
  "size-down" flag at ≥1.4 bits (matches `rationale()`'s threshold, not the desk's 1.5 haircut — they agree
  on the card), and a "+0.5 insurance available" chip for backed underdogs.
- **+0.5 insurance inline** (`.nuins`) when the backed side is a dog with a `D.insurance.ledger` match —
  ML + +0.5 legs surfaced with the "also cashes on a draw" framing.
- **Plain-language reasons** reuse `rationale(r)`; **full detail** reuses `deskCardHtml(r)` in a `<details>`;
  a `full projection →` link jumps to the MATCH PROJECTIONS card (`href="#matches"` via the hashchange route).
- **Slate controls** persist to `wc26.slateN` (3/5/10) and `wc26.slateToday` (today-only). **Flash-on-update:**
  `drawNextUp` diffs a per-match signature (call + edges) and `_flash()`es only cards whose odds/edge changed
  after `liveRefresh()`.
- **DESK tab** keeps a **compact** mirror (top-3 one-line rows, `#next5`) that links to the full NEXT UP tab.

Wiring / ordering note (matters if you touch it): `drawNextUp` and its helpers transitively use consts
defined late in the file (`GLOSS`, `liveMove`, `SIDE_COL`…). To avoid a TDZ crash, the initial
`drawDesk(D.desk)` is gated by a `_nuReady` flag (false until the end of the script) and only `drawNext5`
runs on that first pass; the real `drawNextUp(_deskRows)` fires once at the very end after everything is
defined. `drawDesk` sets `_deskRows` and drives both renderers on every live refresh. Validated with
esprima-python (skip the `const D` preamble, `.replace('??','||')`) — backup parses clean too.

---

## 2. Data architecture ceiling (important for future "live" work)

**Everything is baked at build time** into `const D = {...}` by `src/models/build_dashboard.py`. The ONLY
runtime-live data is the ESPN scoreboard re-fetch in `liveRefresh()` — it updates odds, recomputes
edge/EV/Kelly/stakes/verdicts client-side, and now also captures in-play status + kickoff times. Model
probabilities, settled results, CLV, and the bankroll history do **not** change between Python rebuilds.

Consequences:
- Flash-on-update and freshness only meaningfully apply to the odds-derived cells. The hero/track-record
  numbers move only when the pipeline re-runs.
- A true "live terminal" (live model probs, live settlement) would need a backend or scheduled rebuilds —
  it is out of reach for a static GitHub Pages file. Don't over-promise liveness in copy.

---

## 3. Decisions & honesty constraints (do not regress)

- **Draws are NOT double-corrected.** The brief that drove this overhaul predated V6 and said to deflate
  draw probability by ÷1.75. V6's calibrator already removes that inflation at inference (`p(draw)×0.871`).
  The UI shows the V6-calibrated number with a "bias-corrected" note. Any future "1.75×" language in old
  briefs/caveats is stale — the inline caveats were updated to say so.
- **Arbitrage is impossible with one book.** Only DraftKings feeds the odds (`n_books=1`); the vig
  guarantees Σ(1/odds) > 1. Real arb (the brief's Option A) requires adding a second bookmaker to
  `wc2026_match_odds.csv` — a pipeline/data change, deliberately out of scope. We replaced the dead arb
  section with line movement.
- **+0.5 insurance is now a first-class product story** (the Draw Shield profile), tied to DL-18. It shows
  the ML leg + the +0.5 draw-catch leg together and frames the +0.5 as catching the value the moneyline
  throws away.
- **Realized vs simulated P&L are distinct.** The hero Hit Rate / ROI use *realized* settled `D.clv` rows.
  `runBankroll()` is a forward Monte Carlo projection — keep the two separate in any future copy.
- **Preserved disclaimers** (all still visible, just reorganized): 62% accuracy, edge unproven
  out-of-sample (n<40), ¼-Kelly / 5% per-bet / 25% book cap, Shin de-vig, CONCACAF inflation (Mexico ~+6pp
  = model error), tail-risk futures < 2%, fair-odds P&L = research-grade.

---

## 4. Product framing & tie-in

The dashboard is now a product surface, not a raw research dump. Design principles to preserve:
- **Glance test** — a first-time user should know in ≤10s whether the model is making money (the hero).
- **No empty states** — no "–" dashes or placeholder sections; show an explanatory message instead.
- **Live feel** — visible freshness, countdowns, flashes, in-play badges.

This is the prototype / free-tier surface for the planned **Quant Edge Platform** (Jake's freemium
quant/arbitrage product). Lessons here (glance panel, profile-based curation, honest disclaimers) carry over.

---

## 5. UX backlog (not done this pass)

- Lazy-render non-active tabs (all `draw*()` currently run eagerly on load) for faster first paint.
- Favicon + OG/social meta so the GitHub Pages link previews well.
- "New since last visit" badges on calls (localStorage timestamp).
- Responsive table→card collapse on mobile for the wide tables (CLV, movement, record).
- Multi-book arbitrage — needs a 2nd bookmaker feed (pipeline change).
- A more prominent offline/error banner (currently a one-line status on fetch failure).
- Full keyboard-driven command palette.
