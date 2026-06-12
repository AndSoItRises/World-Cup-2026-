"""
V6 — Live RESULTS auto-ingestor (companion to fetch_live_odds.py).

Reads the same ESPN public scoreboard JSON, finds matches that have FINISHED
(status.type.state == "post" / completed == true), and writes them into
data/raw/wc2026_live_results.csv — the single source of truth that live_update.py
consumes for the ELO update + 10k re-sim.

Why this exists: odds were already automated (fetch_odds_task.ps1), but entering
results was the last manual step in the refresh ritual. This closes that gap so
the whole pipeline can run unattended (see scripts/refresh_all.ps1).

Design choices (match the project's non-negotiables):
  - match_id is THE key. Team names / stage / group / date are taken from
    wc2026_fixtures.csv (authoritative spelling the model expects), NOT from ESPN.
    ESPN is used ONLY for the score + completed status, then oriented to the
    fixture's home/away (ESPN flips home/away for some neutral-venue ties).
  - Idempotent: re-running never duplicates. A match is added if missing, updated
    only if the score changed, otherwise left untouched. Lets a human hand-edit a
    tricky knockout row without it being silently clobbered on the next score-match.
  - Loud name audit: unmatched finished events are printed, never dropped.
  - Goals written are the score ESPN shows at full time. Group stage = 90 min.
    Knockouts: shootout → decided_by="pens" + winner set from ESPN's winner flag;
    otherwise "90min". (AET vs 90 isn't reliably distinguishable from this endpoint;
    group stage — the only thing live now — is unaffected. Flagged for knockouts.)

Exit codes (so the orchestrator knows whether to run the expensive sims):
  10 = at least one result was added or changed   → run live_update + predict
   0 = nothing new                                → skip the heavy sim pipeline
   1 = error (uncaught)                           → handled by the caller

Run: python -m src.models.fetch_live_results
"""

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.models.fetch_live_odds import norm, ESPN_TO_FIXTURE, fetch_scoreboard

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_RAW = BASE / "data" / "raw"

FIXTURES_PATH = DATA_RAW / "wc2026_fixtures.csv"
RESULTS_PATH  = DATA_RAW / "wc2026_live_results.csv"

RESULTS_COLUMNS = ["match_id", "stage", "group", "date", "home_team", "away_team",
                   "home_goals", "away_goals", "decided_by", "winner"]


def _to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def main():
    print("═" * 60)
    print("  V6 — Live Results Fetch (ESPN scoreboard → live_results.csv)")
    print("═" * 60)

    fixtures = pd.read_csv(FIXTURES_PATH)
    real = fixtures[~fixtures["home_team"].str.startswith("TBD")
                    & ~fixtures["away_team"].str.startswith("TBD")].copy()
    # norm(home), norm(away) -> match_id  (same matcher as the odds fetcher)
    lookup = {(norm(r["home_team"]), norm(r["away_team"])): int(r["match_id"])
              for _, r in real.iterrows()}
    # match_id -> fixtures row (authoritative stage/group/date/team spelling)
    by_id = {int(r["match_id"]): r for _, r in real.iterrows()}

    dates = pd.to_datetime(real["date"])
    d1, d2 = dates.min().strftime("%Y%m%d"), dates.max().strftime("%Y%m%d")
    print(f"\n  Fetching {d1}–{d2} ({len(real)} schedulable fixtures)...")
    events = fetch_scoreboard(d1, d2)
    print(f"  Events returned: {len(events)}")

    ingested, unmatched, finished = {}, [], 0
    for ev in events:
        comp = ev["competitions"][0]
        st = (comp.get("status") or {}).get("type") or {}
        if st.get("state") != "post" or not st.get("completed"):
            continue   # not finished yet (pre / in-play)
        finished += 1

        cmap = {c["homeAway"]: c for c in comp["competitors"]}
        eh, ea = cmap.get("home"), cmap.get("away")
        if not eh or not ea:
            continue
        home = ESPN_TO_FIXTURE.get(eh["team"]["displayName"], eh["team"]["displayName"])
        away = ESPN_TO_FIXTURE.get(ea["team"]["displayName"], ea["team"]["displayName"])

        mid, swapped = lookup.get((norm(home), norm(away))), False
        if mid is None:
            mid, swapped = lookup.get((norm(away), norm(home))), True
        if mid is None:
            unmatched.append(ev.get("name", f"{home} vs {away}"))
            continue

        hg_e, ag_e = _to_int(eh.get("score")), _to_int(ea.get("score"))
        if hg_e is None or ag_e is None:
            unmatched.append(ev.get("name") + " (no score)")
            continue

        # Orient goals to the FIXTURE's home/away (what the model probs are keyed to)
        if swapped:
            hg_e, ag_e = ag_e, hg_e
            eh, ea = ea, eh

        fx = by_id[mid]
        is_pens = (eh.get("shootoutScore") is not None
                   or ea.get("shootoutScore") is not None)
        is_group = str(fx["stage"]).strip() == "Group Stage"

        decided_by = "pens" if is_pens else "90min"
        winner = ""
        if not is_group:
            # knockouts: record who advanced (needed if level after 90 / on pens)
            if eh.get("winner"):
                winner = fx["home_team"]
            elif ea.get("winner"):
                winner = fx["away_team"]

        grp = fx["group"]
        ingested[mid] = {
            "match_id": mid,
            "stage": fx["stage"],
            "group": "" if pd.isna(grp) else grp,
            "date": fx["date"],
            "home_team": fx["home_team"],
            "away_team": fx["away_team"],
            "home_goals": hg_e,
            "away_goals": ag_e,
            "decided_by": decided_by,
            "winner": winner,
        }

    # ── Name audit ──
    print(f"\n  ── Audit ──")
    print(f"  Finished events on ESPN: {finished} | matched to fixtures: {len(ingested)}")
    if unmatched:
        print(f"  ⚠️  Unmatched finished events (extend ESPN_TO_FIXTURE): {unmatched}")

    # ── Idempotent merge into the results CSV ──
    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0:
        existing = pd.read_csv(RESULTS_PATH)
    else:
        existing = pd.DataFrame(columns=RESULTS_COLUMNS)
    rows = {int(r["match_id"]): r.to_dict() for _, r in existing.iterrows()} \
        if len(existing) else {}

    added, changed = [], []
    for mid, row in ingested.items():
        if mid not in rows:
            rows[mid] = row
            added.append(mid)
        else:
            old = rows[mid]
            if (_to_int(old.get("home_goals")) != row["home_goals"]
                    or _to_int(old.get("away_goals")) != row["away_goals"]):
                rows[mid] = row
                changed.append(mid)

    out = (pd.DataFrame([rows[m] for m in sorted(rows)], columns=RESULTS_COLUMNS)
           if rows else pd.DataFrame(columns=RESULTS_COLUMNS))
    out.to_csv(RESULTS_PATH, index=False)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if added:
        for m in sorted(added):
            r = rows[m]
            print(f"  ✅ NEW   match {m}: {r['home_team']} {r['home_goals']}–"
                  f"{r['away_goals']} {r['away_team']} ({r['decided_by']})")
    if changed:
        for m in sorted(changed):
            r = rows[m]
            print(f"  ✏️  CHANGED match {m}: now {r['home_team']} {r['home_goals']}–"
                  f"{r['away_goals']} {r['away_team']}")
    n_new = len(added) + len(changed)
    print(f"\n✅ {len(out)} results on file ({len(added)} new, {len(changed)} changed) "
          f"→ {RESULTS_PATH.name}  [{stamp}]")

    sys.exit(10 if n_new else 0)


if __name__ == "__main__":
    main()
