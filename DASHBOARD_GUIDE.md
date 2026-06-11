# WC2026 Quant Dashboard — Operator's Guide
Updated: 2026-06-11 · pairs with HANDOFF.md (agent doc) — this one is for the human at the desk.

Two jobs live in this doc:
1. How to read and use `outputs/quant_dashboard.html`
2. How to enter match results into `data/raw/wc2026_live_results.csv` and refresh everything

---

## 1. Opening the dashboard

| Where | How |
|---|---|
| Desktop | double-click `outputs/quant_dashboard.html` — fully offline, no server needed |
| iPhone / iPad | Safari → https://andsoitrises.github.io/World-Cup-2026-/outputs/quant_dashboard.html → Share → **Add to Home Screen** |
| Anyone else | send them the GitHub Pages link above |

The page is one self-contained file. Everything you see was embedded at build time —
the "built" timestamp in the header tells you how fresh it is. **⟳ LIVE ODDS** (or the
**auto 4m** checkbox) pulls current DraftKings lines + live scores from ESPN in-browser
and recomputes odds, EV, and desk verdicts on the spot.

**What live refresh can and cannot do:** it re-prices everything against live lines, but
the model probabilities themselves come from the local 10k-simulation pipeline. After
real results land, the probabilities only change when you run the refresh ritual (§4).

---

## 2. The tabs — what question each one answers

**DESK CALLS** (landing page) — *"So what do we bet?"*
- **BET** = edge survives every documented-bias haircut. **LEAN** = positive but thinner.
  **PASS** = explained at the bottom of the tab, never hidden.
- Every card shows its evidence chain: `+` lines are why, `!` lines are risk haircuts.
- Stakes assume a $1,000 bankroll: ¼-Kelly, 5% per-bet cap, 25% whole-book cap.
  Scale linearly to your bankroll (e.g. $5,000 bankroll → multiply stakes by 5).

**SCANNER** — *"Show me every +EV price right now."*
The raw value-bet table (matches + tournament futures). The default filters hide draw
bets and you can restrict to favorites — those filters exist because draw/longshot
"edges" are mostly documented model bias. Uncheck them to see everything.

**MATCHES** — *"What does the model think about a specific game?"*
Per-match bars: cyan = model probability, amber tick = market fair probability.
The gap between them is the edge.

**GROUPS** — *"Who advances?"*
Advance % per team from 10k simulations, with `@fair odds` next to it — if a book
offers a better (higher) price than the fair number, that's value on the advancement
market. W % = win the whole tournament.

**BRACKET** — *"Stage-by-stage odds for every team."*
Advance → R16 → QF → SF → Final → Winner probabilities, plus fair winner odds.
Compare directly against futures boards.

**DIVERGENCE** — *"Where does the model disagree with the market most?"*
Scatter of model vs market winner probability. Above the diagonal = model likes the
team more than the price. Mexico's gap is documented CONCACAF inflation — read with suspicion.

**BANKROLL** — *"If I bet this list all tournament, what happens to my money?"*
1,000 simulated Kelly paths. The `truth` selector is the honesty dial: `model` =
optimistic bound, `market fair` = pessimistic (every bet loses the vig), `blend` =
the honest middle. Look at P5 (how bad it gets), not just the median.

**MOVEMENT+ARB** — *"Where are the lines going, and where is the arbitrage?"*
Open→current shifts classified toward/against the model. The arbitrage panel gives the
straight answer: with one book there is none (the vig guarantees it) — it shows the
tightest tickets and the gap to riskless. Add a second book's lines to
`data/raw/wc2026_match_odds.csv` (same columns, any source) and arb scanning activates
automatically.

**CLV** — *"Is the desk actually any good?"*
The scoreboard. Every BET/LEAN gets logged once with the line taken; CLV% = taken ÷
closing − 1. Consistently positive CLV is the real test of edge — it shows up long
before win/loss records mean anything. The **dog CLV** KPI is the live verdict on the
model's biggest open question (DL-10): is its underdog disagreement with the market
edge, or error? Closes are provisional until a match settles.

**UNCERTAINTY** — *"Which matches are genuine coin-flips?"*
3-way entropy per match. Above 1.5 bits = volatile tie, stakes get halved automatically.

**NOTES** — the full research handoff doc, embedded at build time.

---

## 3. Entering results — `data/raw/wc2026_live_results.csv`

After each matchday, add one row per finished match. Columns:

```
match_id,stage,group,date,home_team,away_team,home_goals,away_goals,decided_by,winner
```

**Rules that matter:**
- `match_id` comes from `data/raw/wc2026_fixtures.csv` — it is THE key everywhere.
  Easiest workflow: open fixtures, find the match, copy the id/stage/group/date/team
  columns verbatim, then fill in the goals.
- Team names must match the fixtures spelling exactly: `USA`, `South Korea`, `Czechia`,
  `Bosnia and Herzegovina` — not "United States" / "Korea Republic" etc.
- `home_goals` / `away_goals` = the **90-minute score** (ELO and the goal model train
  on 90-minute results).
- `decided_by`: `90min` for normal results. For knockouts that go long: `AET` or `pens`.
- `winner`: leave **blank** for group games and for knockouts decided in 90 minutes
  (it's derived from the goals). **Required** for a knockout that was level after 90 —
  it tells the simulator who actually advanced.

**Example — group game (Mexico 2–1 South Africa):**
```
1,Group Stage,A,2026-06-11,Mexico,South Africa,2,1,90min,
```

**Example — knockout level after 90, decided on penalties:**
```
79,Round of 32,,2026-06-30,Spain,Morocco,1,1,pens,Morocco
```
(Knockout fixtures show `TBD (Winner A)` etc. until the bracket resolves — use the real
teams in the slot order the bracket produced: fixture home slot first. Valid stage names:
`Group Stage`, `Round of 32`, `Round of 16`, `Quarterfinal`, `Semifinal`, `3rd Place`, `Final`.)

---

## 4. The refresh ritual (run after entering results)

PowerShell, from the repo root:

```powershell
cd "C:\Users\jakeh\OneDrive\Documents\Claude\Projects\World Cup 2026 Model"
$env:PYTHONUTF8 = "1"
.\venv\Scripts\python.exe -m src.models.live_update        # ELO update + 10k re-sim (~3 min)
.\venv\Scripts\python.exe -m src.models.predict_wc2026     # calibrated match probs
.\venv\Scripts\python.exe -m src.models.fetch_live_odds    # latest lines (also runs hourly on its own)
.\venv\Scripts\python.exe -m src.models.market_monitor     # line movement + arb scan
.\venv\Scripts\python.exe -m src.models.bet_sim            # edge / EV / Kelly
.\venv\Scripts\python.exe -m src.models.desk_call          # BET/LEAN/PASS verdicts
.\venv\Scripts\python.exe -m src.models.clv_tracker        # log new calls, settle vs closes
.\venv\Scripts\python.exe -m src.models.build_dashboard    # regenerate the HTML
git add . ; git commit -m "results: matchday YYYY-MM-DD" ; git push
```

The `git push` is what updates the GitHub Pages copy (your phone's version) — allow a
minute or two for Pages to redeploy.

**What's already automated:** odds snapshots run hourly via the Windows scheduled task
"WC2026 odds snapshot" (logs to `logs/odds_fetch.log`), keeping closing lines tight for
the CLV tracker. You never need to fetch odds manually — though running one right
before a kickoff you care about sharpens that close.

**What's manual:** entering results (this file is the single source of truth for what
actually happened) and running the ritual above.

---

## 5. Reading the numbers responsibly

- The model is **62% accurate** (log loss 0.8405 calibrated). Its edge vs the market is
  **unproven out-of-sample** — the CLV tab is the experiment that decides it, live,
  during this tournament.
- Draw and longshot "edges" are mostly documented model bias (1.75× draw upweight, ELO
  compression). The desk haircuts them automatically; the scanner filters default to
  hiding them.
- CONCACAF teams (Mexico especially, ~+6pp) are documented model inflation — treat those
  edges as model error, not market error.
- Futures below 2% model probability are Monte Carlo tail noise, flagged TAIL.
- Everything here is research output, not betting advice. If you do bet: the 25%
  book cap and ¼-Kelly sizing exist because being right about edge and going broke
  anyway is a classic failure mode.
