"""
Settle pending picks against actual WC2026 results.

This is the fix for the CRITICAL reconciliation bug: live_update refreshes
probabilities but never settles bets, so picks sit at status=pending / pnl=0
even after their match has been played.

For every pending pick in clv_report.csv and bet_ledger.csv, this looks up the
match outcome (by match_id, with a (date, home, away) fuzzy fallback), sets
status to WON / LOST / VOID, and computes pnl_usd:

    WON  ->  (taken_decimal - 1) * stake_usd
    LOST ->  -stake_usd
    VOID ->  0.0

Outputs
-------
* data/processed/clv_report.csv          (updated IN PLACE — only status/pnl_usd
                                           of newly-settled pending rows change;
                                           existing CLV data is preserved)
* data/processed/bet_ledger_settled.csv  (bet_ledger.csv + status/pnl_usd)
* data/processed/settlement_log.csv      (append-only audit trail, one row per
                                           settlement with a timestamp)
* data/processed/group_standings.csv     (extended with actual played/pts/gf/ga/gd;
                                           the existing probability columns are kept)

Settlement convention: 1X2 (match-result) bets settled on the 90-minute score,
matching how live_update records results (AET/PK level games are 90-min draws).

Idempotent: rows already WON/LOST/VOID are skipped, so re-running is a no-op.
Independently importable/runnable:

    python -m src.models.settle_bets
    python -m src.models.settle_bets --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.models.monte_carlo import normalize  # fixtures-name -> internal model name

BASE = Path(__file__).resolve().parents[2]
DATA_RAW = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"

LIVE_PATH = DATA_RAW / "wc2026_live_results.csv"
CLV_PATH = DATA_PROC / "clv_report.csv"
LEDGER_PATH = DATA_PROC / "bet_ledger.csv"
LEDGER_SETTLED_PATH = DATA_PROC / "bet_ledger_settled.csv"
SETTLE_LOG_PATH = DATA_PROC / "settlement_log.csv"
STANDINGS_PATH = DATA_PROC / "group_standings.csv"

PENDING = "pending"
WON, LOST, VOID = "WON", "LOST", "VOID"

SETTLE_LOG_COLUMNS = [
    "settled_at", "match_id", "date", "label", "selection", "outcome",
    "home_team", "away_team", "home_goals", "away_goals", "result_side",
    "status", "taken_decimal", "stake_usd", "pnl_usd", "ledger",
]


# ── Results lookup ──────────────────────────────────────────────────────────────
def _result_side(hg: int, ag: int) -> str:
    if hg > ag:
        return "home"
    if ag > hg:
        return "away"
    return "draw"


def load_results() -> tuple[dict[int, dict], list[dict]]:
    """Return (by_match_id, rows). Each result carries the 90-min score, the
    derived winning side, teams and date — enough to settle and to cross-check."""
    if not LIVE_PATH.exists() or LIVE_PATH.stat().st_size == 0:
        return {}, []
    live = pd.read_csv(LIVE_PATH)
    if "match_id" not in live.columns or len(live) == 0:
        return {}, []

    by_id, rows = {}, []
    for _, r in live.iterrows():
        try:
            hg, ag = int(r["home_goals"]), int(r["away_goals"])
        except (ValueError, TypeError):
            continue  # unplayed / malformed -> not settleable
        rec = {
            "match_id": int(r["match_id"]),
            "date": str(r.get("date", ""))[:10],
            "home_team": str(r["home_team"]),
            "away_team": str(r["away_team"]),
            "home_goals": hg,
            "away_goals": ag,
            "stage": str(r.get("stage", "")),
            "result_side": _result_side(hg, ag),
        }
        by_id[rec["match_id"]] = rec
        rows.append(rec)
    return by_id, rows


def _fuzzy_find(pick: pd.Series, results_rows: list[dict]) -> dict | None:
    """Fallback when match_id is absent/unmatched: match on date + teams parsed
    from the pick label ("Home vs Away")."""
    label = str(pick.get("label", ""))
    if " vs " not in label:
        return None
    home, away = (s.strip() for s in label.split(" vs ", 1))
    date = str(pick.get("date", ""))[:10]
    for rec in results_rows:
        if rec["date"] == date and rec["home_team"] == home and rec["away_team"] == away:
            return rec
    return None


# ── Per-pick settlement ──────────────────────────────────────────────────────────
def _settle_pick(pick: pd.Series, result: dict) -> tuple[str, float]:
    """Return (status, pnl_usd) for one pick given its match result."""
    bet_side = str(pick.get("outcome", "")).strip().lower()  # home / away / draw

    # Light integrity cross-check: the pick label teams should match the result.
    label = str(pick.get("label", ""))
    if " vs " in label:
        lh, la = (s.strip() for s in label.split(" vs ", 1))
        if (lh, la) != (result["home_team"], result["away_team"]):
            print(f"  ⚠️  match_id {result['match_id']}: label '{label}' != "
                  f"result '{result['home_team']} vs {result['away_team']}' — settling VOID")
            return VOID, 0.0

    if bet_side not in ("home", "away", "draw"):
        print(f"  ⚠️  match_id {result['match_id']}: unrecognized outcome "
              f"'{bet_side}' — settling VOID")
        return VOID, 0.0

    try:
        decimal = float(pick["taken_decimal"])
        stake = float(pick["stake_usd"])
    except (KeyError, ValueError, TypeError):
        return VOID, 0.0

    if bet_side == result["result_side"]:
        return WON, round((decimal - 1.0) * stake, 2)
    return LOST, round(-stake, 2)


def _settle_frame(df: pd.DataFrame, by_id: dict[int, dict], rows: list[dict],
                  ledger_name: str) -> tuple[pd.DataFrame, list[dict]]:
    """Settle all pending rows of a ledger frame in place. Returns (df, log_rows).

    Adds status/pnl_usd columns if missing (bet_ledger.csv has none). Rows whose
    match has no result yet stay pending; already-settled rows are untouched.
    """
    df = df.copy()
    if "status" not in df.columns:
        df["status"] = PENDING
    if "pnl_usd" not in df.columns:
        df["pnl_usd"] = 0.0
    df["status"] = df["status"].fillna(PENDING).replace("", PENDING)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_rows = []

    for i, pick in df.iterrows():
        if str(pick["status"]).strip().lower() != PENDING:
            continue  # idempotent: skip already-settled

        result = by_id.get(int(pick["match_id"])) if pd.notna(pick.get("match_id")) else None
        if result is None:
            result = _fuzzy_find(pick, rows)
        if result is None:
            continue  # no result yet -> leave pending

        status, pnl = _settle_pick(pick, result)
        df.at[i, "status"] = status
        df.at[i, "pnl_usd"] = pnl

        log_rows.append({
            "settled_at": now,
            "match_id": int(pick["match_id"]),
            "date": str(pick.get("date", ""))[:10],
            "label": pick.get("label", ""),
            "selection": pick.get("selection", ""),
            "outcome": pick.get("outcome", ""),
            "home_team": result["home_team"],
            "away_team": result["away_team"],
            "home_goals": result["home_goals"],
            "away_goals": result["away_goals"],
            "result_side": result["result_side"],
            "status": status,
            "taken_decimal": pick.get("taken_decimal", ""),
            "stake_usd": pick.get("stake_usd", ""),
            "pnl_usd": pnl,
            "ledger": ledger_name,
        })

    return df, log_rows


# ── Group standings (actuals) ────────────────────────────────────────────────────
def compute_actual_standings(rows: list[dict]) -> pd.DataFrame:
    """Accumulate actual played/pts/gf/ga/gd per team from group-stage results.

    Team names are normalized to internal model names so they join onto
    group_standings.csv (which uses internal names)."""
    acc: dict[str, dict] = {}

    def bump(team, gf, ga, pts):
        t = normalize(team)
        s = acc.setdefault(t, {"played": 0, "actual_pts": 0, "actual_gf": 0, "actual_ga": 0})
        s["played"] += 1
        s["actual_gf"] += gf
        s["actual_ga"] += ga
        s["actual_pts"] += pts

    for r in rows:
        if r.get("stage") != "Group Stage":
            continue
        hg, ag = r["home_goals"], r["away_goals"]
        hp, ap = (3, 0) if hg > ag else (0, 3) if ag > hg else (1, 1)
        bump(r["home_team"], hg, ag, hp)
        bump(r["away_team"], ag, hg, ap)

    out = pd.DataFrame([{"team": t, **s} for t, s in acc.items()])
    if not out.empty:
        out["actual_gd"] = out["actual_gf"] - out["actual_ga"]
    return out


def update_group_standings(rows: list[dict], dry_run: bool = False) -> pd.DataFrame:
    """Extend group_standings.csv with actual columns (preserving probabilities)."""
    if not STANDINGS_PATH.exists():
        print("  ⚠️  group_standings.csv missing — skipping standings update")
        return pd.DataFrame()

    standings = pd.read_csv(STANDINGS_PATH)
    actual = compute_actual_standings(rows)

    actual_cols = ["played", "actual_pts", "actual_gf", "actual_ga", "actual_gd"]
    # Drop any prior actual columns so re-runs refresh rather than duplicate.
    standings = standings.drop(columns=[c for c in actual_cols if c in standings.columns],
                               errors="ignore")
    if actual.empty:
        for c in actual_cols:
            standings[c] = 0
    else:
        standings = standings.merge(actual, on="team", how="left")
        for c in actual_cols:
            standings[c] = standings[c].fillna(0).astype(int)

    if not dry_run:
        standings.to_csv(STANDINGS_PATH, index=False)
    return standings


# ── Orchestration ────────────────────────────────────────────────────────────────
def _seed_prior_status(df: pd.DataFrame, settled_path: Path) -> pd.DataFrame:
    """Carry over status/pnl_usd from a previously-written settled file so that
    re-runs are idempotent. bet_ledger.csv itself has no status column, so
    without this the settled copy would re-settle (and re-log) every run.

    Keyed on (match_id, selection, logged_at) — unique per pick. New picks added
    to bet_ledger.csv (absent from the prior settled file) stay pending.
    """
    df = df.copy()
    if not settled_path.exists() or settled_path.stat().st_size == 0:
        return df
    prior = pd.read_csv(settled_path)
    if "status" not in prior.columns:
        return df
    key = ["match_id", "selection", "logged_at"]
    if not all(k in df.columns and k in prior.columns for k in key):
        return df

    def _k(row):
        return tuple(str(row[c]) for c in key)

    prior_map = {_k(r): (r["status"], r.get("pnl_usd", 0.0)) for _, r in prior.iterrows()}
    if "status" not in df.columns:
        df["status"] = PENDING
    if "pnl_usd" not in df.columns:
        df["pnl_usd"] = 0.0
    for i, row in df.iterrows():
        hit = prior_map.get(_k(row))
        if hit is not None:
            df.at[i, "status"] = hit[0]
            df.at[i, "pnl_usd"] = hit[1]
    return df


def _append_log(log_rows: list[dict], dry_run: bool) -> None:
    if not log_rows or dry_run:
        return
    new = pd.DataFrame(log_rows)[SETTLE_LOG_COLUMNS]
    if SETTLE_LOG_PATH.exists() and SETTLE_LOG_PATH.stat().st_size > 0:
        new.to_csv(SETTLE_LOG_PATH, mode="a", header=False, index=False)
    else:
        new.to_csv(SETTLE_LOG_PATH, index=False)


def settle_bets(dry_run: bool = False) -> dict:
    """Settle clv_report.csv (in place) and bet_ledger.csv (-> _settled), update
    standings, append the audit log. Returns a summary dict."""
    by_id, rows = load_results()
    if not by_id:
        print("  No settleable results found — nothing to do.")
        return {"settled": 0, "won": 0, "lost": 0, "void": 0, "net_pnl": 0.0}

    summary = {"settled": 0, "won": 0, "lost": 0, "void": 0, "net_pnl": 0.0}
    all_log: list[dict] = []

    # clv_report.csv — update in place, preserve existing columns/data.
    if CLV_PATH.exists():
        clv = pd.read_csv(CLV_PATH)
        clv, clv_log = _settle_frame(clv, by_id, rows, "clv_report")
        if not dry_run:
            clv.to_csv(CLV_PATH, index=False)
        all_log += clv_log
        print(f"  clv_report.csv: settled {len(clv_log)} pending pick(s)")

    # bet_ledger.csv — write settled copy (original has no status/pnl columns).
    if LEDGER_PATH.exists():
        ledger = pd.read_csv(LEDGER_PATH)
        ledger = _seed_prior_status(ledger, LEDGER_SETTLED_PATH)  # idempotency
        ledger, ledger_log = _settle_frame(ledger, by_id, rows, "bet_ledger")
        if not dry_run:
            ledger.to_csv(LEDGER_SETTLED_PATH, index=False)
        all_log += ledger_log
        print(f"  bet_ledger_settled.csv: settled {len(ledger_log)} pending pick(s)")

    _append_log(all_log, dry_run)
    update_group_standings(rows, dry_run=dry_run)

    # Summarize on the clv ledger (canonical pick log with CLV data).
    clv_log = [r for r in all_log if r["ledger"] == "clv_report"]
    summary["settled"] = len(clv_log)
    summary["won"] = sum(1 for r in clv_log if r["status"] == WON)
    summary["lost"] = sum(1 for r in clv_log if r["status"] == LOST)
    summary["void"] = sum(1 for r in clv_log if r["status"] == VOID)
    summary["net_pnl"] = round(sum(r["pnl_usd"] for r in clv_log), 2)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Settle WC2026 pending picks against results")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute settlements but do not write any files")
    args = ap.parse_args()

    print("═" * 60)
    print("  WC2026 Bet Settlement" + ("  (DRY RUN)" if args.dry_run else ""))
    print("═" * 60)
    s = settle_bets(dry_run=args.dry_run)
    print("─" * 60)
    print(f"  Newly settled: {s['settled']}  |  WON {s['won']}  LOST {s['lost']}  VOID {s['void']}")
    print(f"  Net P&L on newly-settled picks: ${s['net_pnl']:+.2f}")
    if args.dry_run:
        print("  (dry run — no files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
