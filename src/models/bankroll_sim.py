"""
MAIN $500-from-today bankroll simulator (Phase 8, 2B — DL-20).

The hero number a user actually cares about: "if I gave this model $500 TODAY,
what happens?"  Unlike the Model Performance Tracker (which scores the model's
whole realized history), this starts fresh at $500 on the current date, does NOT
retro-credit games already played, and is SELECTIVE — the model only backs games
it likes (desk verdicts), and is allowed to pass on everything else.

Three selectivity variants are tracked side by side so we can see how much the
'pass discipline' matters:

    BET only            — only the desk's highest-conviction calls (verdict BET).
    BET + LEAN          — every actionable call (BET and LEAN).
    BET + select LEAN   — all BETs plus only the higher-rated LEANs (score >= cut).

Each variant:
  * starts at $500 on as-of date (the last day with a settled result),
  * stakes each pick at 1/2-Kelly capped 5% of the CURRENT bankroll,
  * tracks a REALIZED bankroll as fixtures are played (empty until knockout
    results land — group stage is already settled), and
  * carries a forward Monte-Carlo PROJECTION cone (P5/P50/P95) over the picks not
    yet played, priced off the desk's odds and the model's win probabilities.

Odds: the desk's logged decimal_odds (a real book line where one exists, else the
de-vigged fair price).  Fair-priced legs make the projection research-grade —
directionally honest, slightly optimistic vs a real book.

State persists (append-only ledger) so the realized record accumulates over time:
    data/processed/bankroll_500_ledger.csv  — every placed pick, settled as results land
    data/processed/bankroll_500.json        — realized curves + projection cones + KPIs

Run:
    python -m src.models.bankroll_sim
    python -m src.models.bankroll_sim --start 500 --paths 2000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

from src.models.settle_bets import load_results

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"

CONFIG_PATH = DATA_PROC / "config.json"
DESK_CALLS = DATA_PROC / "desk_calls.csv"
LEDGER_OUT = DATA_PROC / "bankroll_500_ledger.csv"
SUMMARY_OUT = DATA_PROC / "bankroll_500.json"

KELLY_FRACTION = 0.50   # 1/2-Kelly
KELLY_CAP = 0.05        # 5% of bankroll per bet
SELECT_LEAN_CUT = 4.5   # a LEAN must score >= this to make the 'select' variant
SEED = 42               # reproducible projection cone (stable unless data changes)

VARIANTS = [
    ("bet_only", "BET only",
     "Only the desk's strongest calls (verdict BET). Fewest bets, highest average "
     "conviction — the model passes on everything it isn't sure about."),
    ("bet_lean", "BET + LEAN",
     "Every actionable call — strong BETs plus softer LEANs. More games on the "
     "board, lower average conviction per bet."),
    ("bet_select_lean", "BET + select LEAN",
     f"All BETs, plus only the higher-rated LEANs (score ≥ {SELECT_LEAN_CUT}). A "
     "middle ground: more action than BET-only, but it skips the weakest leans."),
]

LEDGER_COLUMNS = [
    "variant", "match_id", "date", "label", "selection", "side", "verdict", "score",
    "model_prob", "decimal_odds", "kelly_full", "kelly_fraction_used",
    "status", "result_side", "settled",
]


def load_config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text())
    return {"unit_size_usd": float(cfg.get("unit_size_usd", 15.0))}


def _kelly_full(p: float, decimal: float) -> float:
    b = decimal - 1.0
    return max(0.0, (b * p - (1.0 - p)) / b) if b > 0 else 0.0


def _variant_filter(verdict: str, score: float, variant: str) -> bool:
    v = verdict.strip().upper()
    if variant == "bet_only":
        return v == "BET"
    if variant == "bet_lean":
        return v in ("BET", "LEAN")
    if variant == "bet_select_lean":
        return v == "BET" or (v == "LEAN" and score >= SELECT_LEAN_CUT)
    return False


def build_picks(variant: str, results: dict, start_id: int) -> list[dict]:
    """Desk picks for this variant in the 'from-today' universe — matches with
    match_id >= start_id (the frozen cutoff: everything played before the sim
    started is history and is NOT retro-credited). As these matches are played they
    settle into the realized leg; until then they feed the projection."""
    if not DESK_CALLS.exists():
        return []
    desk = pd.read_csv(DESK_CALLS)
    desk = desk[desk["kind"] == "match"]
    picks: list[dict] = []
    for _, r in desk.iterrows():
        verdict = str(r.get("verdict", ""))
        try:
            score = float(r.get("score", 0.0))
        except (ValueError, TypeError):
            score = 0.0
        if not _variant_filter(verdict, score, variant):
            continue
        try:
            mid = int(r["match_id"])
            if mid < start_id:
                continue  # earlier tournament game — not retro-credited
            side = str(r["outcome"]).strip().lower()
            p = float(r["model_prob"])
            decimal = float(r["decimal_odds"])
        except (KeyError, ValueError, TypeError):
            continue
        if decimal <= 1.0 or p <= 0:
            continue

        kf = _kelly_full(p, decimal)
        pick = {
            "variant": variant,
            "match_id": mid,
            "date": str(r.get("date", ""))[:10],
            "label": str(r.get("label", "")),
            "selection": str(r.get("selection", "")),
            "side": side,
            "verdict": verdict.strip().upper(),
            "score": round(score, 2),
            "model_prob": round(p, 4),
            "decimal_odds": round(decimal, 4),
            "kelly_full": round(kf, 4),
            "kelly_fraction_used": round(min(kf * KELLY_FRACTION, KELLY_CAP), 4),
            "status": "pending", "result_side": "", "settled": False,
        }
        res = results.get(mid)
        if res is not None:
            rs = res["result_side"]
            pick["result_side"], pick["settled"] = rs, True
            pick["status"] = "WON" if rs == side else "LOST"
        picks.append(pick)
    # de-dup: one pick per (match_id, side); keep the highest-scored.
    best: dict[tuple[int, str], dict] = {}
    for pk in picks:
        key = (pk["match_id"], pk["side"])
        if key not in best or pk["score"] > best[key]["score"]:
            best[key] = pk
    return sorted(best.values(), key=lambda p: (p["date"], p["match_id"]))


def realized_curve(picks: list[dict], start: float) -> dict:
    """Compound the bankroll over picks already PLAYED (chronological). Returns the
    realized curve + current value. Empty board -> flat at start."""
    bank = start
    curve = [{"date": "today", "bank": round(bank, 2)}]
    settled = [p for p in picks if p["settled"]]
    wins = 0
    for p in settled:
        frac = p["kelly_fraction_used"]
        stake = frac * bank
        won = (p["result_side"] == p["side"])
        bank += stake * (p["decimal_odds"] - 1.0) if won else -stake
        wins += int(won)
        curve.append({"date": p["date"], "bank": round(bank, 2)})
    return {
        "curve": curve,
        "current_value": round(bank, 2),
        "n_settled": len(settled),
        "n_wins": wins,
        "realized_roi_pct": round((bank - start) / start * 100, 2) if start else 0.0,
    }


def projection_cone(picks: list[dict], start: float, n_paths: int) -> dict:
    """Monte-Carlo the bankroll forward over UNPLAYED picks. Each path samples each
    pick's outcome from the model's win prob and stakes 1/2-Kelly (cap 5%) of the
    running bankroll. Returns a P5/P50/P95 cone over the pick sequence."""
    rng = random.Random(SEED)
    open_picks = [p for p in picks if not p["settled"]]
    realized_now = realized_curve(picks, start)["current_value"]
    if not open_picks:
        return {"enabled": True, "n_open": 0, "cone": [], "final": {}, "prob_profit_pct": None}

    steps = len(open_picks)
    paths = []  # each path: list of bank values after each open pick
    for _ in range(n_paths):
        bank = realized_now
        row = []
        for p in open_picks:
            frac = p["kelly_fraction_used"]
            stake = frac * bank
            if rng.random() < p["model_prob"]:
                bank += stake * (p["decimal_odds"] - 1.0)
            else:
                bank -= stake
            row.append(bank)
        paths.append(row)

    def pctile(vals: list[float], q: float) -> float:
        s = sorted(vals)
        idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return s[idx]

    cone = [{"step": 0, "date": "today", "p5": round(realized_now, 2),
             "p50": round(realized_now, 2), "p95": round(realized_now, 2)}]
    for i in range(steps):
        col = [path[i] for path in paths]
        cone.append({
            "step": i + 1,
            "date": open_picks[i]["date"],
            "p5": round(pctile(col, 0.05), 2),
            "p50": round(pctile(col, 0.50), 2),
            "p95": round(pctile(col, 0.95), 2),
        })
    finals = [path[-1] for path in paths]
    prob_profit = round(sum(1 for f in finals if f > start) / len(finals) * 100, 1)
    return {
        "enabled": True,
        "n_open": steps,
        "cone": cone,
        "final": {
            "p5": round(pctile(finals, 0.05), 2),
            "p50": round(pctile(finals, 0.50), 2),
            "p95": round(pctile(finals, 0.95), 2),
        },
        "prob_profit_pct": prob_profit,
    }


def _as_of(results_rows: list[dict]) -> str:
    dates = [r["date"] for r in results_rows if r.get("date")]
    return max(dates) if dates else "today"


def _frozen_start_id(results_rows: list[dict]) -> int:
    """The 'today' cutoff: picks with match_id >= this are the from-today board.
    Frozen on first run (persisted in the summary JSON) so it never drifts forward
    and the realized record accumulates as those matches are played."""
    if SUMMARY_OUT.exists():
        try:
            prev = json.loads(SUMMARY_OUT.read_text())
            sid = prev.get("config", {}).get("start_match_id")
            if sid is not None:
                return int(sid)
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass
    played = [r["match_id"] for r in results_rows]
    return (max(played) + 1) if played else 1


def _persist_ledger(all_picks: list[dict]) -> None:
    """Persist the ledger so the realized record accumulates. Keyed on
    (variant, match_id, side): freshly-computed picks (which carry up-to-date
    settlement from current results) replace their prior version, and any prior-only
    rows (e.g. a pick later dropped from the desk before it settled) are preserved."""
    new = pd.DataFrame([{k: p[k] for k in LEDGER_COLUMNS} for p in all_picks])
    key = ["variant", "match_id", "side"]
    if LEDGER_OUT.exists() and LEDGER_OUT.stat().st_size > 0:
        prior = pd.read_csv(LEDGER_OUT)
        new_keys = {tuple(str(r[k]) for k in key) for _, r in new.iterrows()}
        keep = prior[~prior.apply(lambda r: tuple(str(r[k]) for k in key) in new_keys, axis=1)]
        out = pd.concat([new, keep], ignore_index=True)
    else:
        out = new
    out.to_csv(LEDGER_OUT, index=False)


def run(start: float = 500.0, n_paths: int = 2000, write: bool = True) -> dict:
    cfg = load_config()
    results, rows = load_results()
    as_of = _as_of(rows)
    start_id = _frozen_start_id(rows)

    variants_out = {}
    all_picks: list[dict] = []
    for key, label, blurb in VARIANTS:
        picks = build_picks(key, results, start_id)
        all_picks += picks
        realized = realized_curve(picks, start)
        projection = projection_cone(picks, start, n_paths)
        variants_out[key] = {
            "label": label,
            "explainer": blurb,
            "n_picks": len(picks),
            "n_open": sum(1 for p in picks if not p["settled"]),
            "realized": realized,
            "projection": projection,
            "picks": picks,
        }

    summary = {
        "config": {
            "start_bankroll_usd": start,
            "as_of": as_of,
            "start_match_id": start_id,
            "kelly_fraction": KELLY_FRACTION,
            "kelly_cap": KELLY_CAP,
            "select_lean_cut": SELECT_LEAN_CUT,
            "mc_paths": n_paths,
            "projection_default_on": True,
        },
        "headline_variant": "bet_only",
        "variants": variants_out,
        "caveat": (
            "Starts at $500 today — earlier tournament games are NOT retro-credited. "
            "The realized curve is empty until knockout results land; the projection "
            "is a forward Monte-Carlo (model probs + desk odds), research-grade where "
            "fair odds are used. Selective: the model passes on games it doesn't like."
        ),
    }
    if write:
        SUMMARY_OUT.write_text(json.dumps(summary, indent=2, default=str))
        _persist_ledger(all_picks)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="WC2026 $500-from-today bankroll simulator")
    ap.add_argument("--start", type=float, default=500.0, help="starting bankroll (default 500)")
    ap.add_argument("--paths", type=int, default=2000, help="Monte-Carlo paths (default 2000)")
    args = ap.parse_args()

    print("═" * 66)
    print("  WC2026 MAIN Bankroll Simulator — $500 from today, selective")
    print("═" * 66)
    s = run(start=args.start, n_paths=args.paths, write=True)
    print(f"  as-of {s['config']['as_of']}  ·  start ${s['config']['start_bankroll_usd']:.0f}"
          f"  ·  {s['config']['mc_paths']} MC paths")
    print("─" * 66)
    print(f"  {'variant':<20}{'picks':>6}{'open':>6}{'realized$':>11}{'proj P50':>10}{'P(profit)':>11}")
    for key, label, _b in VARIANTS:
        v = s["variants"][key]
        proj = v["projection"]["final"].get("p50", "—")
        pp = v["projection"].get("prob_profit_pct")
        pp_s = f"{pp:.0f}%" if pp is not None else "—"
        proj_s = f"{proj:.0f}" if isinstance(proj, (int, float)) else "—"
        print(f"  {label:<20}{v['n_picks']:>6}{v['n_open']:>6}"
              f"{v['realized']['current_value']:>11.2f}{proj_s:>10}{pp_s:>11}")
    print("─" * 66)
    print(f"  wrote {SUMMARY_OUT.name} + {LEDGER_OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
