# Cloud Results Pipeline — Architecture & Workflow Review

**Date:** 2026-06-24
**Author:** Claude (debugging session)
**Scope:** the automated results-ingestion + live-refresh pipeline, both its local
(ESPN) and cloud (football-data.org) paths, and the GitHub Actions that drive it.
**Trigger:** the cloud `daily_update.yml` pipeline had never been able to complete a
fresh CI run. This review captures what was wrong, what was fixed (commit `b50b288`),
and what to do next.

---

## 1. Architecture review

### 1.1 The two results-ingestion paths

There are **two independent ways finished-match results enter the model**, writing the
same file, `data/raw/wc2026_live_results.csv` (schema:
`match_id,stage,group,date,home_team,away_team,home_goals,away_goals,decided_by,winner`):

| | **Local primary (ESPN)** | **Cloud (football-data.org)** |
|---|---|---|
| Script | `src/models/fetch_live_results.py` | `src/models/ingest_results.py` |
| Source | ESPN public scoreboard JSON | football-data.org v4 API (`competition=WC`) |
| Driver | `_active_scripts/refresh_all.ps1`, every ~30 min (local task) | `.github/workflows/daily_update.yml`, 08:00 UTC daily |
| match_id | joined from `wc2026_fixtures.csv` | joined from `wc2026_fixtures.csv` |
| home/away | **re-oriented to the fixture** | written in the **API's** orientation |
| date | from `wc2026_fixtures.csv` | from API `utcDate` (UTC day) |
| decided_by | `90min` / `pens` | `90min` / `AET` / `PKs` |
| Exit codes | `10`=changed, `0`=none, `1`=error | `0`=ok, `1`=error |

**Both key the CSV on the internal fixture `match_id` (1..N), not the provider's match
id.** This is the single most important design decision: it makes the merge idempotent
(a re-run updates a row in place; it never appends a duplicate) and lets the two sources
co-exist. It is also why the original spec's "football-data `match.id` → `match_id`"
idea was *not* implemented — it would have appended all 48 API matches as "new" on top of
the existing fixture-keyed rows, duplicating the entire file.

### 1.2 Downstream chain (consumers of the results CSV)

```
ingest_results / fetch_live_results   ──>  data/raw/wc2026_live_results.csv
                                              │
   live_update ── ELO update + 10k Monte-Carlo re-sim ──> data/processed/tournament_probs_live.csv
   settle_bets ── grades pending picks on match_id ─────> clv_report.csv, bet_ledger_settled.csv
   health_check ── canary: 0 missing, 0 pending-after-played ─> data/processed/model_health.json
```

`live_update` loads the production models from `models/` (`xgb_prod.json`,
`lgbm_prod.txt`, `dixon_coles_params_prod.json`) via `monte_carlo.load_models()`, plus
`wc2026_fixtures.csv` and `current_fifa_rankings.csv`. **Every one of these must be
present in a fresh checkout for the cloud job to run** — that was the crux of the
breakage (see §3).

### 1.3 What gets committed vs. ignored (the data-tracking model)

`data/raw/` is ignored *by default* (raw CSVs can be large), with three small inputs
now explicitly tracked so CI has them:

- ✅ tracked: `wc2026_fixtures.csv` (9 KB), `current_fifa_rankings.csv` (5.5 KB),
  `wc2026_live_results.csv` (≈3 KB)
- ❌ still ignored: `wc2026_match_odds.csv` (1.3 MB — not needed by the daily chain),
  `wc2026_market_odds.csv`, `wc2026_squad_values.csv`
- ✅ tracked: `data/processed/*` outputs and `models/*` artifacts (the live forecast,
  ledgers, and trained models the dashboard + sims read)

---

## 2. Workflow review

### 2.1 `daily_update.yml` (the football-data path) — **now active**

- **Trigger:** `schedule: 0 8 * * *` (08:00 UTC) + `workflow_dispatch`. A
  *Tournament window guard* step short-circuits scheduled runs outside
  2026-06-11…2026-07-20 (cron can't express a date range); manual runs always proceed.
- **Steps:** checkout → Python 3.13 → `pip install` → `ingest_results` →
  `live_update` → `settle_bets` → `health_check` → commit/push → *surface-unhealthy* guard.
- **Health-check design:** `health_check` is `continue-on-error: true` so a tripped
  metric never blocks the commit (fresh data still lands). A final step then re-fails the
  job *after* the push if health was `BROKEN`, so GitHub still emails you. This preserves
  the DL-17 "flag me if broken" signal **and** the "always persist data" goal.
- **Secret:** `FOOTBALL_DATA_API_KEY` is scoped to the `ingest` step only (the rest of
  the chain reads local files and needs no key).

### 2.2 `health_monitor.yml` — runs every 6 h, fails/emails on `BROKEN`

This was **also silently broken before this fix**: `health_check` reads
`data/raw/wc2026_fixtures.csv` + `wc2026_live_results.csv`, which were ignored, so the
job would have crashed on a fresh runner. Un-ignoring those inputs fixes it too.

### 2.3 The local loop (`refresh_all.ps1`) — the documented PRIMARY

Every ~30 min it snapshots odds, ingests ESPN results, re-sims on a new result, reprices
picks, and **`git add -A` + push on a material change**. Health-check is non-fatal here
(logs a WARN).

---

## 3. What was broken, and why CI never ran (fixed in `b50b288`)

Three stacked failures, each of which alone killed the run. Found by exporting the exact
git-tracked fileset into a clean room and running the chain (i.e. simulating CI):

1. **Wrong competition code.** `ingest_results` defaulted `COMPETITION="WC2026"`, which
   football-data.org answers with **HTTP 400**. The live code is **`WC`** (verified: 48
   FINISHED matches). → default flipped to `WC`; `FOOTBALL_DATA_COMPETITION` override kept.
2. **Required inputs gitignored.** `data/raw/` was ignored wholesale, so
   `wc2026_fixtures.csv` and `current_fifa_rankings.csv` were absent in a fresh checkout —
   `ingest_results` and `monte_carlo` crashed on `read_csv` *before any network call*. →
   `.gitignore` now tracks the 3 small inputs; the 1.3 MB odds dump stays ignored.
3. **EOL-fragile models.** `lgbm_prod.txt` / `*.npy` corrupt if LF is rewritten to CRLF on
   a Windows checkout ("Model format error, expect a tree here"). ubuntu CI is LF so it was
   fine *there*, but a Windows clone would break. → `.gitattributes` marks them binary.

Plus: `ingest_results` now exits `1` with a clean message (not a traceback) on API
failure, and forces UTF-8 stdout so it runs on a Windows (cp1252) console.

**Verified:** ingest (48 rows) → live_update → settle_bets → health_check **HEALTHY**
runs green in an ubuntu-equivalent (LF) clean room.

---

## 4. Things to know (gotchas)

- **Competition code is `WC`, not `WC2026`.** `WC2026` returns HTTP 400. Override via the
  `FOOTBALL_DATA_COMPETITION` env var if the provider ever renames it.
- **`data/raw/` is mostly ignored.** If a new pipeline step needs a raw input in CI, add a
  `!data/raw/<file>` negation to `.gitignore` and commit the file, or the cloud job will
  crash on a fresh checkout. There is no "fetch inputs" step.
- **Never let git normalize model files.** Keep `models/*.txt` and `*.npy` marked `-text`.
- **Two sources, two orientations.** ESPN rows are fixture-oriented; football-data rows are
  API-oriented (and use the UTC date, which can differ by a day). The *result* is identical
  (team names travel with their goals, and settlement joins on `match_id`, so nothing
  breaks) — but the two loops will **rewrite each other's `home/away` ordering and date**,
  producing churny commits if both run. See §5.1.
- **Knockouts store the 90-min score** with the advancing team in `winner`; AET/PK level
  games are treated as 90-min draws for ELO. Unmatched events get a flagged synthetic
  `match_id` (9000+), never silently dropped.
- **`health_check` exit codes:** `0` = HEALTHY/DEGRADED, non-zero = BROKEN (the email
  trigger). DEGRADED = warnings only, still usable.
- **The local loop now commits the results CSV.** Because `wc2026_live_results.csv` is now
  tracked, `refresh_all.ps1`'s `git add -A` will start pushing results-file changes too —
  expected and good (both loops converge on one tracked file), just new behavior.

---

## 5. Recommendations (prioritized)

**P1 — Decide on a single source of truth for results.** With both loops now writing the
same tracked file, the only remaining friction is orientation/date churn (§4). Two clean
options:
  - **(a) Make `ingest_results` orient to the fixture** (reuse the home/away-swap +
    fixture-date logic already in `fetch_live_results.py`). Then ESPN and football-data
    produce **byte-identical rows** → zero churn, and the cloud path becomes a true drop-in
    backup. *Recommended — small, surgical change.*
  - **(b) Pick one writer per environment:** cloud owns results, local loop drops to
    odds-only (remove its `fetch_live_results` step), or vice-versa.

**P2 — Add a tiny offline CI check.** A job step that runs
`python -m src.models.ingest_results --self-test` (already exists, no network) on every
push would have caught none of these three bugs — but a step that *imports* the pipeline
and asserts the tracked inputs load (fixtures, rankings, models) would have caught #2 and
#3 instantly. Cheap insurance against "works on my machine."

**P3 — Knockout results need a manual check.** Both ingestors approximate AET/PK games as
90-min draws. During the knockout rounds, eyeball `decided_by` / `winner` on level games;
the football-data path *does* expose `EXTRA_TIME`/`PENALTY_SHOOTOUT` duration, so if you
cut over to it as primary you could record true AET/PK outcomes.

**P4 — Watch the first scheduled run.** It fires 08:00 UTC. Confirm green in the Actions
tab, that the commit lands, and that it doesn't fight the local loop (P1). If churn is
annoying before P1 ships, drop the cron to once post-matchday.

---

## 6. Decision-log entry to record (in `CONTEXT_V7.md`)

> **DL-XX (2026-06-24): Activated the cloud football-data daily pipeline.** DL-17 kept it
> manual-only "until the API is verified." Verified: competition code is `WC` (not
> `WC2026` → 400), free tier returns all FINISHED matches, full chain green in a clean-room
> CI sim. Fixed three CI blockers (competition default, gitignored inputs, CRLF-fragile
> models — commit `b50b288`) and enabled the 08:00 schedule with a tournament-window guard.
> Local ESPN loop remains primary; **open item (P1): unify result orientation** so the two
> writers don't churn `wc2026_live_results.csv`.
