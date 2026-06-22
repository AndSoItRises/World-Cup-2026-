"""
Pipeline health check — the "flag me if anything is broken" monitor.

Runs a battery of data-integrity checks over the pipeline outputs and classifies
overall health as HEALTHY / DEGRADED / BROKEN. Writes data/processed/model_health.json
(read by the dashboard Model-Health panel) and prints a readable report.

Exit codes (so automation can flag a human):
    0  HEALTHY or DEGRADED   (DEGRADED = warnings only; pipeline still usable)
    1  BROKEN                 (an error-level check failed — needs attention)

Wired into both runners:
  * .github/workflows/daily_update.yml runs this as a step; a non-zero exit FAILS
    the workflow, and GitHub emails the repo owner automatically — that is the
    no-extra-infra "flag me" signal (works with your machine off).
  * _active_scripts/refresh_all.ps1 runs it each cycle and logs a loud WARN.

Each check maps to a specific risk introduced by the Phase 7 results/settlement
work (DL-14/DL-15), so this doubles as the monitoring spec for that version update.

Run:
    python -m src.models.health_check            # report + model_health.json
    python -m src.models.health_check --strict   # exit 1 on DEGRADED too
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DATA_RAW = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"

FIXTURES = DATA_RAW / "wc2026_fixtures.csv"
LIVE = DATA_RAW / "wc2026_live_results.csv"
CLV = DATA_PROC / "clv_report.csv"
LEDGER_SETTLED = DATA_PROC / "bet_ledger_settled.csv"
SETTLE_LOG = DATA_PROC / "settlement_log.csv"
HEALTH_OUT = DATA_PROC / "model_health.json"

LIVE_SCHEMA = {"match_id", "stage", "group", "date", "home_team",
               "away_team", "home_goals", "away_goals", "decided_by", "winner"}

ERROR, WARN, INFO = "error", "warn", "info"


class Report:
    """Collects issues and derives an overall status."""

    def __init__(self):
        self.issues: list[dict] = []
        self.metrics: dict = {}

    def add(self, level: str, code: str, detail: str):
        self.issues.append({"level": level, "code": code, "detail": detail})

    def err(self, code, detail): self.add(ERROR, code, detail)
    def warn(self, code, detail): self.add(WARN, code, detail)
    def info(self, code, detail): self.add(INFO, code, detail)

    @property
    def status(self) -> str:
        if any(i["level"] == ERROR for i in self.issues):
            return "BROKEN"
        if any(i["level"] == WARN for i in self.issues):
            return "DEGRADED"
        return "HEALTHY"


def _read_csv(path: Path, r: Report, *, required: bool):
    """Read a CSV, recording an issue if missing/unparseable. Returns df or None."""
    if not path.exists() or path.stat().st_size == 0:
        (r.err if required else r.warn)("file_missing", f"{path.name} is missing or empty")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:  # noqa: BLE001 — surface any parse failure as an issue
        r.err("file_unparseable", f"{path.name} failed to parse: {e}")
        return None


# ── Individual checks (each maps to a monitored risk) ────────────────────────────
def check_live_results(r: Report) -> pd.DataFrame | None:
    """Risk (DL-14): ingest writes a bad/empty/wrong-schema results file, or the
    cloud API returns nothing (silent blackout)."""
    live = _read_csv(LIVE, r, required=True)
    if live is None:
        return None
    missing = LIVE_SCHEMA - set(live.columns)
    if missing:
        r.err("live_schema", f"wc2026_live_results.csv missing columns {sorted(missing)} "
                             f"(breaks live_update.load_live)")
    n = len(live)
    r.metrics["live_results_rows"] = n
    if n == 0:
        r.warn("live_empty", "wc2026_live_results.csv has 0 rows — ingest may be failing "
                             "(check the API competition code / key)")
    return live


def check_coverage(live: pd.DataFrame | None, r: Report):
    """Risk: a match was played but never ingested (the class of bug that left
    picks stuck pending)."""
    fixtures = _read_csv(FIXTURES, r, required=True)
    if fixtures is None or live is None:
        return
    today = datetime.now(timezone.utc).date()
    fx = fixtures.copy()
    fx["d"] = pd.to_datetime(fx["date"], errors="coerce").dt.date
    # fixtures that should be finished (date strictly before today) with concrete teams
    due = fx[(fx["d"].notna()) & (fx["d"] < today)
             & (~fx["home_team"].astype(str).str.startswith("TBD"))]
    have = set(live["match_id"].astype(int)) if "match_id" in live.columns else set()
    missing_ids = sorted(set(due["match_id"].astype(int)) - have)
    r.metrics["fixtures_due"] = int(len(due))
    r.metrics["results_missing"] = len(missing_ids)
    if missing_ids:
        sample = missing_ids[:8]
        r.warn("results_stale", f"{len(missing_ids)} match(es) past their date have no result "
                                f"ingested (ids {sample}{'...' if len(missing_ids) > 8 else ''}) "
                                f"— ingest may be behind or down")


def check_settlement(live: pd.DataFrame | None, r: Report):
    """Risks (DL-15): (a) the CRITICAL bug regresses — a played pick stays pending;
    (b) settlement math is wrong; (c) status/pnl inconsistency."""
    clv = _read_csv(CLV, r, required=True)
    if clv is None:
        return
    if "status" not in clv.columns or "pnl_usd" not in clv.columns:
        r.err("clv_schema", "clv_report.csv missing status/pnl_usd columns")
        return

    result_ids = set(live["match_id"].astype(int)) if (live is not None and "match_id" in live) else set()
    status = clv["status"].astype(str).str.strip()

    # (a) pending-after-played — the original CRITICAL bug
    pending_played = clv[(status.str.lower() == "pending") & (clv["match_id"].isin(result_ids))]
    r.metrics["pending_total"] = int((status.str.lower() == "pending").sum())
    r.metrics["pending_after_played"] = int(len(pending_played))
    if len(pending_played):
        ids = sorted(pending_played["match_id"].astype(int))[:10]
        r.err("pending_after_played",
              f"{len(pending_played)} pick(s) have a played result but status=pending "
              f"(settlement not running) — match_ids {ids}")

    # (b/c) math + sign consistency on settled rows
    bad_math, bad_sign = [], []
    for _, p in clv.iterrows():
        st = str(p["status"]).strip().upper()
        try:
            pnl = float(p["pnl_usd"])
            dec = float(p["taken_decimal"])
            stake = float(p["stake_usd"])
        except (ValueError, TypeError):
            continue
        if st == "WON":
            if abs(pnl - round((dec - 1) * stake, 2)) > 0.02:
                bad_math.append(int(p["match_id"]))
        elif st == "LOST":
            if abs(pnl - round(-stake, 2)) > 0.02:
                bad_math.append(int(p["match_id"]))
        elif st in ("VOID", "PENDING"):
            if abs(pnl) > 0.001:
                bad_sign.append(int(p["match_id"]))
    if bad_math:
        r.err("settlement_math", f"pnl_usd does not match (decimal-1)*stake / -stake for "
                                 f"match_ids {sorted(set(bad_math))[:10]}")
    if bad_sign:
        r.err("pnl_nonzero", f"VOID/pending picks have non-zero pnl_usd: "
                             f"match_ids {sorted(set(bad_sign))[:10]}")

    settled = clv[status.str.upper().isin(["WON", "LOST", "VOID"])]
    r.metrics["settled_total"] = int(len(settled))
    r.metrics["net_pnl_usd"] = round(float(settled["pnl_usd"].sum()), 2)


def check_ledger_consistency(r: Report):
    """Risk: the two ledgers (clv_report vs bet_ledger_settled) disagree on how many
    picks are settled — a sign one settler ran and the other didn't."""
    clv = _read_csv(CLV, r, required=False)
    ledger = _read_csv(LEDGER_SETTLED, r, required=False)
    if clv is None or ledger is None or "status" not in clv or "status" not in ledger:
        return
    def settled(df):
        return int(df["status"].astype(str).str.upper().isin(["WON", "LOST", "VOID"]).sum())
    c, l = settled(clv), settled(ledger)
    r.metrics["clv_settled"] = c
    r.metrics["ledger_settled"] = l
    # ledger may settle a few more (rows clv had pre-settled); only flag a large gap
    if abs(c - l) > 5:
        r.warn("ledger_divergence",
               f"clv_report settled={c} vs bet_ledger_settled settled={l} — ledgers diverging")


# ── Orchestration ────────────────────────────────────────────────────────────────
def run_checks() -> Report:
    r = Report()
    live = check_live_results(r)
    check_coverage(live, r)
    check_settlement(live, r)
    check_ledger_consistency(r)
    return r


def health_check(write: bool = True) -> Report:
    r = run_checks()
    payload = {
        "status": r.status,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issues": r.issues,
        "metrics": r.metrics,
    }
    if write:
        HEALTH_OUT.write_text(json.dumps(payload, indent=2))
    return r


def _print(r: Report):
    icon = {"HEALTHY": "✅", "DEGRADED": "⚠️", "BROKEN": "❌"}[r.status]
    print("═" * 60)
    print(f"  PIPELINE HEALTH: {icon} {r.status}")
    print("═" * 60)
    if r.metrics:
        print("  metrics:")
        for k, v in r.metrics.items():
            print(f"    {k:<22} {v}")
    errs = [i for i in r.issues if i["level"] == ERROR]
    warns = [i for i in r.issues if i["level"] == WARN]
    if errs:
        print(f"\n  ❌ ERRORS ({len(errs)}) — needs attention:")
        for i in errs:
            print(f"    [{i['code']}] {i['detail']}")
    if warns:
        print(f"\n  ⚠️  WARNINGS ({len(warns)}):")
        for i in warns:
            print(f"    [{i['code']}] {i['detail']}")
    if not errs and not warns:
        print("\n  All checks passed.")
    print(f"\n  wrote {HEALTH_OUT.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="WC2026 pipeline health check")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on DEGRADED as well as BROKEN")
    args = ap.parse_args()

    r = health_check(write=True)
    _print(r)

    if r.status == "BROKEN":
        return 1
    if r.status == "DEGRADED" and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
