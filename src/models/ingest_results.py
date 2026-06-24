"""
Ingest actual WC2026 results from the football-data.org API.

Pulls completed matches, normalizes team names to the fixtures convention,
joins each to the model's internal match_id via wc2026_fixtures.csv, and
merge-writes data/raw/wc2026_live_results.csv in the exact schema that
live_update.load_live() already consumes:

    match_id, stage, group, date, home_team, away_team,
    home_goals, away_goals, decided_by, winner

Design notes
------------
* This is the FIRST step of the daily pipeline (see live_update orchestration).
  It only produces the results file; settlement / ELO / sim run downstream.
* The write is a MERGE keyed on match_id: existing rows are preserved and
  updated in place, new FINISHED matches are appended. Nothing is dropped, so
  manually-entered rows survive an API pull that doesn't return them.
* Knockout scores are stored as the 90-minute (regular-time) score with the
  actual advancing team in `winner`, matching live_update's ELO/known-result
  convention (AET/PK level games are treated as 90-min draws).
* Network is isolated in fetch_matches(); parse_matches()/normalize_rows() are
  pure and unit-testable offline (see --self-test).

Run:
    python -m src.models.ingest_results              # live API pull
    python -m src.models.ingest_results --self-test  # offline parse/join check (no network)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.features.team_name_map import to_fixture_name

BASE = Path(__file__).resolve().parents[2]
DATA_RAW = BASE / "data" / "raw"

FIXTURES_PATH = DATA_RAW / "wc2026_fixtures.csv"
LIVE_PATH = DATA_RAW / "wc2026_live_results.csv"

API_BASE = "https://api.football-data.org/v4"
# Competition code is config-driven. football-data.org's code for the FIFA World
# Cup is "WC" (verified working against the live v4 API — "WC2026" returns 400).
# Override via env without touching code if the provider ever renames it.
COMPETITION = os.environ.get("FOOTBALL_DATA_COMPETITION", "WC")
API_KEY_ENV = "FOOTBALL_DATA_API_KEY"

# Output column order — MUST match live_update.load_live().
LIVE_COLUMNS = [
    "match_id", "stage", "group", "date", "home_team", "away_team",
    "home_goals", "away_goals", "decided_by", "winner",
]

# football-data.org stage code -> our stage label. load_live only distinguishes
# "Group Stage" from everything else, but readable knockout labels help humans.
STAGE_MAP = {
    "GROUP_STAGE": "Group Stage",
    "LAST_32": "Round of 32",
    "LAST_16": "Round of 16",
    "QUARTER_FINALS": "Quarter-finals",
    "QUARTER_FINAL": "Quarter-finals",
    "SEMI_FINALS": "Semi-finals",
    "SEMI_FINAL": "Semi-finals",
    "THIRD_PLACE": "Third place",
    "FINAL": "Final",
}

DURATION_TO_DECIDED_BY = {
    "REGULAR": "90min",
    "EXTRA_TIME": "AET",
    "PENALTY_SHOOTOUT": "PKs",
    "PENALTIES": "PKs",
}


# ── Network (isolated) ─────────────────────────────────────────────────────────
def fetch_matches(api_key: str | None = None, status: str = "FINISHED",
                  competition: str = COMPETITION, timeout: int = 30) -> dict:
    """GET competition matches from football-data.org. Returns the raw JSON dict.

    Imported lazily so the module loads even where `requests` is absent and so
    offline tests never touch the network.
    """
    import requests  # local import: only needed for live pulls

    api_key = api_key or os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"No API key. Set the {API_KEY_ENV} environment variable "
            f"(GitHub Actions secret of the same name in CI)."
        )
    url = f"{API_BASE}/competitions/{competition}/matches"
    params = {"status": status} if status else {}
    resp = requests.get(url, headers={"X-Auth-Token": api_key},
                        params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ── Parsing (pure) ─────────────────────────────────────────────────────────────
def _score_pair(score: dict, key: str) -> tuple[int, int] | None:
    block = (score or {}).get(key) or {}
    h, a = block.get("home"), block.get("away")
    if h is None or a is None:
        return None
    return int(h), int(a)


def parse_matches(payload: dict, only_finished: bool = True) -> list[dict]:
    """Pure transform: football-data.org JSON -> list of normalized result dicts.

    Each dict uses fixtures-convention team names and the 90-minute score, with
    the advancing team resolved in `winner` for level knockouts.
    """
    rows = []
    for m in payload.get("matches", []):
        status = m.get("status")
        if only_finished and status != "FINISHED":
            continue

        score = m.get("score", {}) or {}
        # Prefer the 90-min (regular) score; fall back to fullTime for group
        # games where regularTime may be omitted.
        reg = _score_pair(score, "regularTime")
        full = _score_pair(score, "fullTime")
        if reg is not None:
            hg, ag = reg
        elif full is not None:
            hg, ag = full
        else:
            continue  # no usable score — skip

        duration = (score.get("duration") or "REGULAR").upper()
        decided_by = DURATION_TO_DECIDED_BY.get(duration, "90min")

        home = to_fixture_name((m.get("homeTeam") or {}).get("name"))
        away = to_fixture_name((m.get("awayTeam") or {}).get("name"))

        # Resolve advancing team (matters when 90-min was level).
        winner_code = score.get("winner")
        if winner_code == "HOME_TEAM":
            winner = home
        elif winner_code == "AWAY_TEAM":
            winner = away
        else:
            winner = ""  # DRAW or unknown -> blank, like the existing live file

        utc = m.get("utcDate") or ""
        date = utc[:10] if len(utc) >= 10 else ""

        stage_code = (m.get("stage") or "").upper()
        stage = STAGE_MAP.get(stage_code, stage_code.replace("_", " ").title() or "Group Stage")

        grp = m.get("group")
        group = grp.replace("GROUP_", "").strip() if isinstance(grp, str) else ""

        rows.append({
            "api_id": m.get("id"),
            "stage": stage,
            "group": group,
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "decided_by": decided_by,
            "winner": winner if (hg == ag) else "",  # only need winner on level 90-min
        })
    return rows


# ── Join to internal match_id ────────────────────────────────────────────────
def build_fixture_index(fixtures: pd.DataFrame) -> dict[tuple[str, str], int]:
    """(home_team, away_team) -> match_id, for fixtures with concrete teams.

    Group-stage and any decided knockout pairings are unique on ordered
    (home, away), so this resolves cleanly. TBD knockout rows are skipped.
    """
    idx = {}
    for _, r in fixtures.iterrows():
        h, a = str(r["home_team"]), str(r["away_team"])
        if h.startswith("TBD") or a.startswith("TBD"):
            continue
        idx[(h, a)] = int(r["match_id"])
    return idx


def attach_match_ids(rows: list[dict], fixtures: pd.DataFrame) -> list[dict]:
    """Fill `match_id` on each parsed row by joining on (home, away).

    Falls back to a date+stage match against still-TBD knockout fixtures, then
    to a clearly-flagged synthetic id so nothing is silently mis-joined.
    """
    pair_idx = build_fixture_index(fixtures)

    # Knockout fixtures still showing TBD, grouped by (date, stage) for fallback.
    tbd = fixtures[
        fixtures["home_team"].astype(str).str.startswith("TBD")
        | fixtures["away_team"].astype(str).str.startswith("TBD")
    ]
    tbd_by_key: dict[tuple[str, str], list[int]] = {}
    for _, r in tbd.iterrows():
        key = (str(r["date"])[:10], str(r["stage"]))
        tbd_by_key.setdefault(key, []).append(int(r["match_id"]))
    for k in tbd_by_key:
        tbd_by_key[k].sort()

    out = []
    unresolved = 0
    for row in rows:
        h, a = row["home_team"], row["away_team"]
        mid = pair_idx.get((h, a))
        if mid is None:
            mid = pair_idx.get((a, h))  # tolerate flipped home/away
        if mid is None:
            # knockout fallback: consume a TBD slot for this date+stage
            bucket = tbd_by_key.get((row["date"], row["stage"]))
            if bucket:
                mid = bucket.pop(0)
                print(f"  [ingest] knockout join by date+stage: "
                      f"{h} vs {a} ({row['date']}, {row['stage']}) -> match_id {mid}")
        if mid is None:
            unresolved += 1
            mid = 9000 + unresolved  # synthetic, flagged below
            print(f"  [ingest] ⚠️  no fixture match for {h} vs {a} "
                  f"({row['date']}) — synthetic match_id {mid}")
        r = dict(row)
        r["match_id"] = mid
        out.append(r)

    if unresolved:
        print(f"  [ingest] ⚠️  {unresolved} match(es) could not be joined to fixtures.")
    return out


# ── Merge-write ────────────────────────────────────────────────────────────────
def _read_existing() -> pd.DataFrame:
    if LIVE_PATH.exists() and LIVE_PATH.stat().st_size > 0:
        df = pd.read_csv(LIVE_PATH)
        # Guard against a stray header-only / malformed file.
        if "match_id" in df.columns:
            return df
    return pd.DataFrame(columns=LIVE_COLUMNS)


def merge_and_write(rows: list[dict], dry_run: bool = False) -> pd.DataFrame:
    """Merge parsed rows into wc2026_live_results.csv on match_id, preserving
    any pre-existing rows the API pull did not return."""
    new_df = pd.DataFrame(rows)
    if not new_df.empty:
        new_df = new_df[LIVE_COLUMNS]  # drop helper cols (api_id), enforce order

    existing = _read_existing()
    if not existing.empty:
        # keep existing schema, fill any missing cols
        for c in LIVE_COLUMNS:
            if c not in existing.columns:
                existing[c] = ""
        existing = existing[LIVE_COLUMNS]

    combined = pd.concat([existing, new_df], ignore_index=True)
    # New rows win on duplicate match_id (API is source of truth for results).
    combined = combined.drop_duplicates(subset="match_id", keep="last")
    combined["match_id"] = combined["match_id"].astype(int)
    combined = combined.sort_values("match_id").reset_index(drop=True)

    if not dry_run:
        combined.to_csv(LIVE_PATH, index=False)
    return combined


# ── Orchestration ──────────────────────────────────────────────────────────────
def ingest_results(api_key: str | None = None, payload: dict | None = None,
                   dry_run: bool = False) -> pd.DataFrame:
    """Fetch (or accept an offline payload), parse, join, and merge-write.

    Pass `payload` to skip the network entirely (used by --self-test and any
    caller that already has the JSON).
    """
    if payload is None:
        print(f"  [ingest] fetching FINISHED matches: competition={COMPETITION}")
        payload = fetch_matches(api_key=api_key, status="FINISHED")

    fixtures = pd.read_csv(FIXTURES_PATH)
    parsed = parse_matches(payload, only_finished=True)
    print(f"  [ingest] parsed {len(parsed)} finished match(es) from payload")

    joined = attach_match_ids(parsed, fixtures)
    combined = merge_and_write(joined, dry_run=dry_run)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    action = "would write" if dry_run else "wrote"
    print(f"  [ingest] {action} {len(combined)} total rows -> {LIVE_PATH.name} "
          f"({len(joined)} from this pull) @ {stamp}")
    return combined


# ── Offline self-test ───────────────────────────────────────────────────────────
def _self_test() -> int:
    """Validate parse + join + merge with a synthetic payload — no network.

    Mirrors a handful of real WC2026 group matches (including a name that needs
    mapping and a level draw) and asserts the output matches the load_live schema.
    """
    sample = {
        "matches": [
            {   # straightforward home win
                "id": 500001, "utcDate": "2026-06-11T16:00:00Z",
                "status": "FINISHED", "stage": "GROUP_STAGE", "group": "GROUP_A",
                "homeTeam": {"name": "Mexico"}, "awayTeam": {"name": "South Africa"},
                "score": {"winner": "HOME_TEAM", "duration": "REGULAR",
                          "fullTime": {"home": 2, "away": 0},
                          "regularTime": {"home": 2, "away": 0}},
            },
            {   # name mapping: "United States" -> "USA"
                "id": 500002, "utcDate": "2026-06-12T20:00:00Z",
                "status": "FINISHED", "stage": "GROUP_STAGE", "group": "GROUP_D",
                "homeTeam": {"name": "United States"}, "awayTeam": {"name": "Paraguay"},
                "score": {"winner": "HOME_TEAM", "duration": "REGULAR",
                          "fullTime": {"home": 4, "away": 1}},
            },
            {   # level draw -> winner blank
                "id": 500003, "utcDate": "2026-06-13T19:00:00Z",
                "status": "FINISHED", "stage": "GROUP_STAGE", "group": "GROUP_B",
                "homeTeam": {"name": "Qatar"}, "awayTeam": {"name": "Switzerland"},
                "score": {"winner": "DRAW", "duration": "REGULAR",
                          "fullTime": {"home": 1, "away": 1}},
            },
            {   # not finished -> must be skipped
                "id": 500004, "utcDate": "2026-07-01T19:00:00Z",
                "status": "SCHEDULED", "stage": "GROUP_STAGE", "group": "GROUP_C",
                "homeTeam": {"name": "Brazil"}, "awayTeam": {"name": "Morocco"},
                "score": {"winner": None, "duration": "REGULAR",
                          "fullTime": {"home": None, "away": None}},
            },
        ]
    }

    fixtures = pd.read_csv(FIXTURES_PATH)
    parsed = parse_matches(sample, only_finished=True)
    assert len(parsed) == 3, f"expected 3 finished, got {len(parsed)}"
    joined = attach_match_ids(parsed, fixtures)

    by_pair = {(r["home_team"], r["away_team"]): r for r in joined}
    # USA name mapping + join to fixtures match_id 4
    usa = by_pair[("USA", "Paraguay")]
    assert usa["match_id"] == 4, f"USA/Paraguay -> match_id {usa['match_id']} (want 4)"
    # Mexico vs South Africa -> match_id 1
    assert by_pair[("Mexico", "South Africa")]["match_id"] == 1
    # Draw row keeps winner blank
    qat = by_pair[("Qatar", "Switzerland")]
    assert qat["winner"] == "" and qat["home_goals"] == 1 and qat["away_goals"] == 1

    # Merge in dry-run mode (does NOT touch the real file) and check schema.
    combined = merge_and_write(joined, dry_run=True)
    assert list(combined.columns) == LIVE_COLUMNS, "schema drift vs load_live"

    print("  [self-test] PASS — parse, name-map, match_id join, and schema OK")
    print("  [self-test] (dry-run: wc2026_live_results.csv NOT modified)")
    return 0


def main() -> int:
    # Windows consoles default to cp1252 and choke on the box-drawing/emoji
    # output below; force UTF-8 so local runs match CI (ubuntu, already UTF-8).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="Ingest WC2026 results from football-data.org")
    ap.add_argument("--self-test", action="store_true",
                    help="run offline parse/join/schema checks (no network, no writes)")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and parse but do not write the CSV")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    print("═" * 60)
    print("  WC2026 Results Ingest (football-data.org)")
    print("═" * 60)
    try:
        ingest_results(dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001 — top-level guard: clean message, no traceback
        import requests  # local: only for the isinstance check
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            r = exc.response
            print(f"\n❌ football-data.org API error: HTTP {r.status_code} for "
                  f"{r.url}\n   competition={COMPETITION!r}. "
                  f"Check the competition code and that {API_KEY_ENV} is set and valid.",
                  file=sys.stderr)
        else:
            print(f"\n❌ Results ingest failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
