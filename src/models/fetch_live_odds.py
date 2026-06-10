"""
V6 — Live match odds fetcher (light scrape of ESPN's public scoreboard JSON).

Source:   https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard
Provider: DraftKings 3-way moneylines via ESPN's betting integration. Each event
          carries OPEN and CURRENT (ESPN calls it 'close') American odds for
          home / draw / away — so one fetch gives both the live line and the
          opening line for movement tracking.

Appends snapshots to data/raw/wc2026_match_odds.csv (the bet_sim input):
  - one 'opening' row per (match_id, book) — written only if not already present
  - one 'current' row per run — accumulating a line history over the tournament

Team names are matched to data/raw/wc2026_fixtures.csv via normalization + an
explicit alias map, keyed to match_id. Unmatched events are printed loudly
(name-audit non-negotiable), never silently dropped.

Run: python -m src.models.fetch_live_odds
"""

import json
import unicodedata
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_RAW = BASE / "data" / "raw"

FIXTURES_PATH   = DATA_RAW / "wc2026_fixtures.csv"
MATCH_ODDS_PATH = DATA_RAW / "wc2026_match_odds.csv"

SCOREBOARD_URL = ("https://site.api.espn.com/apis/site/v2/sports/soccer/"
                  "fifa.world/scoreboard?dates={d1}-{d2}&limit=300")

# ESPN displayName → fixtures name, where normalization alone can't bridge
ESPN_TO_FIXTURE = {
    "United States": "USA",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Türkiye": "Turkey",
    "Czech Republic": "Czechia",
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Curaçao": "Curacao",
    "Congo DR": "DR Congo",
}

ODDS_COLUMNS = ["match_id", "book", "snapshot", "odds_format",
                "home_odds", "draw_odds", "away_odds", "fetched_at"]


def norm(name: str) -> str:
    """Accent-strip + lowercase + alpha-only, so 'Bosnia-Herzegovina' ==
    'Bosnia and Herzegovina' after removing the connective noise."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = "".join(c for c in s.lower() if c.isalpha())
    return s.replace("and", "")


def fetch_scoreboard(d1: str, d2: str) -> list:
    req = urllib.request.Request(
        SCOREBOARD_URL.format(d1=d1, d2=d2),
        headers={"User-Agent": "Mozilla/5.0 (wc2026-research)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode()).get("events", [])


def extract_moneyline(event):
    """(book, {side: (open, current)}) American odds, or None if no 3-way ML."""
    comps = event.get("competitions") or []
    if not comps:
        return None
    for o in comps[0].get("odds") or []:
        ml = o.get("moneyline")
        if not ml:
            continue
        book = (o.get("provider") or {}).get("displayName", "unknown")
        quotes = {}
        for side in ("home", "draw", "away"):
            node = ml.get(side) or {}
            op = (node.get("open") or {}).get("odds")
            cur = (node.get("close") or {}).get("odds")  # ESPN 'close' = current
            if op is None and cur is None:
                return None
            quotes[side] = (op, cur if cur is not None else op)
        return book, quotes
    return None


def main():
    print("═" * 60)
    print("  V6 — Live Odds Fetch (ESPN scoreboard / DraftKings 3-way)")
    print("═" * 60)

    fixtures = pd.read_csv(FIXTURES_PATH)
    real = fixtures[~fixtures["home_team"].str.startswith("TBD")
                    & ~fixtures["away_team"].str.startswith("TBD")].copy()
    lookup = {(norm(r["home_team"]), norm(r["away_team"])): int(r["match_id"])
              for _, r in real.iterrows()}

    dates = pd.to_datetime(real["date"])
    d1, d2 = dates.min().strftime("%Y%m%d"), dates.max().strftime("%Y%m%d")
    print(f"\n  Fetching {d1}–{d2} ({len(real)} schedulable fixtures)...")
    events = fetch_scoreboard(d1, d2)
    print(f"  Events returned: {len(events)}")

    existing = pd.read_csv(MATCH_ODDS_PATH) if MATCH_ODDS_PATH.exists() \
        else pd.DataFrame(columns=ODDS_COLUMNS)
    have_opening = {(int(r["match_id"]), r["book"])
                    for _, r in existing[existing["snapshot"] == "opening"].iterrows()}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new_rows, unmatched, no_odds = [], [], []
    for ev in events:
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c["team"]["displayName"] for c in comp["competitors"]}
        home = ESPN_TO_FIXTURE.get(teams.get("home"), teams.get("home"))
        away = ESPN_TO_FIXTURE.get(teams.get("away"), teams.get("away"))
        # ESPN and the fixtures file disagree on home/away for some neutral-venue
        # ties — try both orientations, re-orient odds to the FIXTURE's home/away
        # (that's what the model probs are keyed to)
        mid, swapped = lookup.get((norm(home), norm(away))), False
        if mid is None:
            mid, swapped = lookup.get((norm(away), norm(home))), True
        if mid is None:
            unmatched.append(ev.get("name", f"{home} vs {away}"))
            continue
        ml = extract_moneyline(ev)
        if ml is None:
            no_odds.append(ev.get("name"))
            continue
        book, q = ml
        if swapped:
            q = {"home": q["away"], "draw": q["draw"], "away": q["home"]}
        if (mid, book) not in have_opening and all(q[s][0] is not None for s in q):
            new_rows.append({"match_id": mid, "book": book, "snapshot": "opening",
                             "odds_format": "american",
                             "home_odds": q["home"][0], "draw_odds": q["draw"][0],
                             "away_odds": q["away"][0], "fetched_at": now})
        new_rows.append({"match_id": mid, "book": book, "snapshot": "current",
                         "odds_format": "american",
                         "home_odds": q["home"][1], "draw_odds": q["draw"][1],
                         "away_odds": q["away"][1], "fetched_at": now})

    print(f"\n  ── Name audit ──")
    print(f"  Matched to fixtures: {len({r['match_id'] for r in new_rows})} matches")
    if unmatched:
        print(f"  ⚠️  Unmatched ESPN events (extend ESPN_TO_FIXTURE): {unmatched}")
    if no_odds:
        print(f"  ⚠️  No 3-way moneyline (in-play/settled?): {no_odds}")

    out = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    out.to_csv(MATCH_ODDS_PATH, index=False)
    n_open = sum(1 for r in new_rows if r["snapshot"] == "opening")
    print(f"\n✅ Appended {len(new_rows)} rows ({n_open} opening, "
          f"{len(new_rows) - n_open} current) → {MATCH_ODDS_PATH.name} "
          f"({len(out)} total)")


if __name__ == "__main__":
    main()
