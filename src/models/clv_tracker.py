"""
V6 — Closing Line Value tracker (queue #1; the decisive edge experiment).

CLV: did the line we bet beat the market's final (closing) price? Bettors who
consistently beat the close are long-run profitable; results are noise on a
72-match sample, CLV is signal. This is also how we settle DL-10: the model
genuinely disagrees with the market on underdogs — if our dog-side calls beat
the close, that disagreement was edge; if the close steamrolls them, it wasn't.

Mechanics:
  LEDGER (append-only)  data/processed/bet_ledger.csv — every match-kind
    BET/LEAN desk call is logged ONCE (key: match_id+outcome) with the line at
    log time. Re-runs never modify logged rows; the desk changing its mind
    later doesn't rewrite history. Futures excluded (no close until July).
  CLOSING LINE          last odds snapshot per match in wc2026_match_odds.csv
    (ESPN's moneyline disappears at kickoff, so the final fetch ≈ the close —
    fetch shortly before kickoff for best fidelity).
  SETTLEMENT            realized outcomes from data/raw/wc2026_live_results.csv.

Metrics per bet:  clv_pct = taken_decimal / closing_decimal − 1  (+ = beat close)
Aggregates: by category (fav = fair ≥ 40% | dog | draw), by verdict, realized P&L.

Outputs: data/processed/bet_ledger.csv (state), data/processed/clv_report.csv
Run: python -m src.models.clv_tracker   (after desk_call; before build_dashboard)
"""

import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.models.market_ingestion import parse_odds_row

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_RAW = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"

DESK_PATH    = DATA_PROC / "desk_calls.csv"
ODDS_PATH    = DATA_RAW / "wc2026_match_odds.csv"
RESULTS_PATH = DATA_RAW / "wc2026_live_results.csv"
LEDGER_PATH  = DATA_PROC / "bet_ledger.csv"
REPORT_PATH  = DATA_PROC / "clv_report.csv"

SIDE_COL = {"home": "home_odds", "draw": "draw_odds", "away": "away_odds"}


def category(row) -> str:
    if row["outcome"] == "draw":
        return "draw"
    return "fav" if row["market_implied"] >= 0.40 else "dog"


def update_ledger() -> pd.DataFrame:
    """Append new BET/LEAN match calls; never touch existing rows."""
    desk = pd.read_csv(DESK_PATH)
    calls = desk[(desk["kind"] == "match") & (desk["verdict"] != "PASS")].copy()
    ledger = pd.read_csv(LEDGER_PATH) if LEDGER_PATH.exists() else pd.DataFrame()
    have = (set(zip(ledger["match_id"].astype(int), ledger["outcome"]))
            if len(ledger) else set())

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new = []
    for _, r in calls.iterrows():
        key = (int(r["match_id"]), r["outcome"])
        if key in have:
            continue
        new.append({
            "match_id": key[0], "outcome": r["outcome"], "date": r["date"],
            "label": r["label"], "selection": r["selection"],
            "verdict": r["verdict"], "category": category(r),
            "model_prob": r["model_prob"], "market_implied": r["market_implied"],
            "taken_decimal": r["decimal_odds"], "stake_usd": r["stake_usd"],
            "logged_at": now,
        })
    ledger = pd.concat([ledger, pd.DataFrame(new)], ignore_index=True) if new else ledger
    if len(ledger):
        ledger.to_csv(LEDGER_PATH, index=False)
    print(f"  Ledger: {len(new)} new bets logged, {len(ledger)} total")
    return ledger


def closing_lines() -> dict:
    """match_id → {outcome: closing decimal} from the last snapshot per match."""
    odds = pd.read_csv(ODDS_PATH)
    odds = odds[odds["snapshot"] != "opening"].sort_values("fetched_at")
    out = {}
    for mid, grp in odds.groupby("match_id"):
        r = grp.iloc[-1]
        dec = parse_odds_row([r[SIDE_COL["home"]], r[SIDE_COL["draw"]],
                              r[SIDE_COL["away"]]], r["odds_format"])
        out[int(mid)] = {"home": dec[0], "draw": dec[1], "away": dec[2],
                         "as_of": r["fetched_at"]}
    return out


def settled_outcomes() -> dict:
    """match_id → realized outcome ('home'|'draw'|'away') for finished matches."""
    if not RESULTS_PATH.exists():
        return {}
    res = pd.read_csv(RESULTS_PATH)
    out = {}
    for _, r in res.iterrows():
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        out[int(r["match_id"])] = "home" if hg > ag else "away" if ag > hg else "draw"
    return out


def build_report(ledger: pd.DataFrame) -> pd.DataFrame:
    close = closing_lines()
    results = settled_outcomes()
    rows = []
    for _, b in ledger.iterrows():
        mid = int(b["match_id"])
        c = close.get(mid, {})
        closing = c.get(b["outcome"])
        clv = (b["taken_decimal"] / closing - 1.0) if closing else None
        settled = mid in results
        won = settled and results[mid] == b["outcome"]
        pnl = (b["stake_usd"] * (b["taken_decimal"] - 1) if won else -b["stake_usd"]) \
            if settled else 0.0
        rows.append({
            **b, "closing_decimal": round(closing, 3) if closing else "",
            "closing_as_of": c.get("as_of", ""),
            # until the match settles, "closing" is just the latest snapshot —
            # provisional CLV. Only final closes feed desk_call's confidence input.
            "close_is_final": settled,
            "clv_pct": round(clv * 100, 2) if clv is not None else "",
            "beat_close": "" if clv is None else bool(clv > 0),
            "status": ("WON" if won else "LOST") if settled else "pending",
            "pnl_usd": round(pnl, 2),
        })
    rep = pd.DataFrame(rows)
    rep.to_csv(REPORT_PATH, index=False)
    return rep


def print_summary(rep: pd.DataFrame):
    have_clv = rep[rep["clv_pct"] != ""].copy() if len(rep) else rep
    print(f"\n── CLV summary ({len(rep)} tracked bets) ──")
    if not len(have_clv):
        print("  No closing lines yet.")
        return
    have_clv["clv_pct"] = have_clv["clv_pct"].astype(float)
    n_final = int(have_clv["close_is_final"].sum())
    print(f"  Avg CLV: {have_clv['clv_pct'].mean():+.2f}% | beating close: "
          f"{(have_clv['clv_pct'] > 0).mean():.0%} "
          f"({n_final} final closes, {len(have_clv) - n_final} provisional)")
    print(f"\n  {'category':<10}{'n':>4}{'avg CLV':>10}{'beat%':>8}"
          f"{'settled':>9}{'P&L':>9}")
    print(f"  {'-' * 50}")
    for cat in ("fav", "dog", "draw"):
        g = have_clv[have_clv["category"] == cat]
        if not len(g):
            continue
        st = g[g["status"] != "pending"]
        print(f"  {cat:<10}{len(g):>4}{g['clv_pct'].mean():>+9.2f}%"
              f"{(g['clv_pct'] > 0).mean():>8.0%}{len(st):>9}"
              f"{st['pnl_usd'].sum():>+9.2f}")
    settled = rep[rep["status"] != "pending"]
    if len(settled):
        print(f"\n  Settled: {len(settled)} | record "
              f"{(settled['status'] == 'WON').sum()}-{(settled['status'] == 'LOST').sum()}"
              f" | P&L ${settled['pnl_usd'].sum():+.2f}")
    print(f"\n  Read: positive avg CLV on dogs ⇒ the model's market disagreement")
    print(f"  (DL-10) is edge; negative ⇒ the market knew better. Verdict needs ~2 weeks.")


def main():
    print("═" * 60)
    print("  V6 — Closing Line Value Tracker")
    print("═" * 60)
    ledger = update_ledger()
    if not len(ledger):
        print("  Nothing tracked yet — run desk_call first.")
        return
    rep = build_report(ledger)
    print_summary(rep)
    print(f"\n✅ Saved: {LEDGER_PATH}")
    print(f"✅ Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
