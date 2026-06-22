"""
Model Performance Tracker (Phase 8, 2A — DL-19).

The honest, REALIZED scorecard that answers one question: "is the model actually
any good?"  Unlike the forward Monte-Carlo bankroll sim, nothing here is
hypothetical — every bet is settled against the ACTUAL result.

Universe (this is the fix for the "not enough games" problem):
    EVERY match the model considered +value — i.e. for each match in
    prediction_ledger.csv we take the model's single best value side (the outcome
    whose model prob most exceeds the Shin-fair market prob), gated on the same
    thresholds the desk uses (config.json: min_edge, min_model_prob).  We do NOT
    restrict to matches with a real book line.  Where a real line exists
    (value_bets.csv) we use that price and flag market_source='real'; otherwise we
    use the de-vigged fair price and flag market_source='fair'.  Fair-priced P&L is
    research-grade — directionally honest, slightly optimistic vs a real book.

For every such bet that has been PLAYED we settle it (settle_bets.load_results)
and compute a per-unit realized return, then replay the same picks under several
STAKING STRATEGIES so we can compare how the edge would have compounded:

    Flat        — same stake every bet (1 unit = unit_size_usd).  The baseline.
    1/4-Kelly   — quarter of the Kelly-optimal fraction (capped 5%).  Cautious.
    1/2-Kelly   — half  of the Kelly-optimal fraction (capped 5%).  The usual pick.
    Full-Kelly  — the full growth-optimal fraction (capped 5%).  Aggressive.

Each strategy reports ROI%, net $, hit-rate, max drawdown, a per-bet Sharpe-ish
ratio, and CLV alignment (how often these picks beat the closing line).  Every
strategy also carries a plain-English explainer so a non-quant reader gets it.

Outputs (NEW files — nothing existing is touched):
    data/processed/model_performance.json   — strategies + curves + verdict (dashboard)
    data/processed/model_performance_ledger.csv — per-bet realized record (audit)

Run:
    python -m src.models.model_performance
    python -m src.models.model_performance --bankroll 1000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

from src.models.settle_bets import load_results

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"

CONFIG_PATH = DATA_PROC / "config.json"
PRED_LEDGER = DATA_PROC / "prediction_ledger.csv"
VALUE_BETS = DATA_PROC / "value_bets.csv"
CLV_REPORT = DATA_PROC / "clv_report.csv"
SUMMARY_OUT = DATA_PROC / "model_performance.json"
LEDGER_OUT = DATA_PROC / "model_performance_ledger.csv"

SIDES = ("home", "draw", "away")

# Staking strategies: (key, label, kelly_fraction or None for flat, plain-English).
# kelly_fraction None  -> flat stake of 1 unit.
STRATEGIES = [
    ("flat", "Flat stake", None,
     "Bet the exact same amount on every pick (1 unit). Simple and steady — it "
     "ignores how strong each edge is, so it never over-commits, but it also "
     "doesn't press the model's best spots."),
    ("kelly_quarter", "1/4 Kelly", 0.25,
     "Kelly is the math-optimal bet size for long-run growth, based on how big the "
     "edge is and the odds. Quarter-Kelly bets a quarter of that — deliberately "
     "cautious, smoothing out the swings at the cost of a little growth."),
    ("kelly_half", "1/2 Kelly", 0.50,
     "Half of the math-optimal bet size. The industry default: most of Kelly's "
     "growth with roughly half the volatility. Usually the best risk-adjusted "
     "balance."),
    ("kelly_full", "Full Kelly", 1.00,
     "The full growth-optimal bet size. Maximises long-run compounding IF the "
     "model's probabilities are exactly right — but it's a wild ride, with deep "
     "drawdowns, and it punishes any over-confidence in the model."),
]

KELLY_CAP = 0.05  # never stake >5% of bankroll on one bet (matches bet_sim.py)


def load_config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text())
    return {
        "unit_size_usd": float(cfg.get("unit_size_usd", 15.0)),
        "min_edge": float(cfg.get("thresholds", {}).get("min_edge", 0.05)),
        "min_model_prob": float(cfg.get("thresholds", {}).get("min_model_prob", 0.30)),
    }


def _real_line_lookup() -> dict[tuple[int, str], dict]:
    """(match_id, outcome) -> {decimal_odds, market_source} from value_bets.csv, so
    we can prefer a real book price over the fair (de-vigged) one when it exists."""
    if not VALUE_BETS.exists():
        return {}
    vb = pd.read_csv(VALUE_BETS)
    out: dict[tuple[int, str], dict] = {}
    for _, r in vb.iterrows():
        try:
            key = (int(r["match_id"]), str(r["outcome"]).strip().lower())
        except (ValueError, TypeError):
            continue
        out[key] = {
            "decimal_odds": float(r["decimal_odds"]),
            "market_source": str(r.get("market_source", "real")),
        }
    return out


def _clv_lookup() -> dict[tuple[int, str], dict]:
    """(match_id, outcome) -> {clv_pct, beat_close} from clv_report.csv for CLV
    alignment (did these picks beat the closing line?)."""
    if not CLV_REPORT.exists():
        return {}
    clv = pd.read_csv(CLV_REPORT)
    out: dict[tuple[int, str], dict] = {}
    for _, r in clv.iterrows():
        try:
            key = (int(r["match_id"]), str(r["outcome"]).strip().lower())
        except (ValueError, TypeError):
            continue
        try:
            beat = str(r.get("beat_close", "")).strip().lower() in ("true", "1")
            out[key] = {"clv_pct": float(r.get("clv_pct", "nan")), "beat_close": beat}
        except (ValueError, TypeError):
            continue
    return out


def build_value_bets(pred: pd.DataFrame, cfg: dict) -> list[dict]:
    """One record per match where the model sees a +value side, with the best side
    chosen, priced (real line if available else fair), and settled if played."""
    real_lines = _real_line_lookup()
    results, _ = load_results()
    pred = pred[pred.get("prob_source", "pre_result") == "pre_result"].copy()
    pred = pred.drop_duplicates(subset="match_id", keep="first")

    bets: list[dict] = []
    for _, m in pred.iterrows():
        mid = int(m["match_id"])
        model_p = {"home": float(m["p_home"]), "draw": float(m["p_draw"]), "away": float(m["p_away"])}
        mkt_p = {"home": float(m["mkt_home"]), "draw": float(m["mkt_draw"]), "away": float(m["mkt_away"])}

        # Best value side = largest (model - market) edge that clears both gates.
        best_side, best_edge = None, -1.0
        for s in SIDES:
            edge = model_p[s] - mkt_p[s]
            if edge >= cfg["min_edge"] and model_p[s] >= cfg["min_model_prob"] and edge > best_edge:
                best_side, best_edge = s, edge
        if best_side is None:
            continue

        p = model_p[best_side]
        mkt = mkt_p[best_side]
        fair_dec = 1.0 / mkt if mkt > 0 else 0.0

        real = real_lines.get((mid, best_side))
        if real and real["decimal_odds"] > 1.0:
            decimal, source = real["decimal_odds"], real.get("market_source", "real")
        else:
            decimal, source = fair_dec, "fair"

        # Kelly-optimal fraction f* = (b*p - q)/b, b = decimal-1, floored at 0.
        b = decimal - 1.0
        kelly_full = max(0.0, (b * p - (1.0 - p)) / b) if b > 0 else 0.0

        team = m["home_team"] if best_side == "home" else m["away_team"] if best_side == "away" else "Draw"
        bet = {
            "match_id": mid,
            "date": str(m["date"])[:10],
            "label": f"{m['home_team']} vs {m['away_team']}",
            "side": best_side,
            "selection": team if best_side != "draw" else f"{m['home_team']} vs {m['away_team']} (draw)",
            "model_prob": round(p, 4),
            "market_prob": round(mkt, 4),
            "edge": round(best_edge, 4),
            "decimal_odds": round(decimal, 4),
            "market_source": source,
            "kelly_full": round(kelly_full, 4),
            # settlement (filled below):
            "status": "pending", "pnl_unit": 0.0, "settled": False, "result_side": "",
        }

        res = results.get(mid)
        if res is not None:
            rs = res["result_side"]
            won = (rs == best_side)
            bet["result_side"] = rs
            bet["settled"] = True
            bet["status"] = "WON" if won else "LOST"
            bet["pnl_unit"] = round((decimal - 1.0) if won else -1.0, 4)  # per 1 unit staked
        bets.append(bet)
    return bets


def _max_drawdown_pct(curve: list[float]) -> float:
    peak, mdd = (curve[0] if curve else 0.0), 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return round(mdd * 100, 2)


def _sharpe(returns: list[float]) -> float:
    """Per-bet Sharpe-ish: mean / std of realized per-bet returns (in bankroll %).
    Not annualised — a unitless risk-adjusted score for comparing strategies."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    sd = math.sqrt(var)
    return round(mean / sd, 3) if sd > 0 else 0.0


def run_strategy(bets_settled: list[dict], frac, start_bank: float, unit: float) -> dict:
    """Replay settled bets chronologically under one staking rule. frac=None -> flat."""
    bank = start_bank
    curve = [{"date": "start", "bank": round(bank, 2)}]
    bank_returns: list[float] = []  # per-bet return as fraction of pre-bet bankroll
    wins = 0

    for bet in bets_settled:
        if frac is None:
            stake = min(unit, bank)  # flat 1 unit, never more than the bank
        else:
            stake = min(bet["kelly_full"] * frac, KELLY_CAP) * bank
        if stake <= 0 or bank <= 0:
            curve.append({"date": bet["date"], "bank": round(bank, 2)})
            continue
        pnl = stake * bet["pnl_unit"]
        pre = bank
        bank += pnl
        bank_returns.append(pnl / pre if pre > 0 else 0.0)
        wins += int(bet["pnl_unit"] > 0)
        curve.append({"date": bet["date"], "bank": round(bank, 2)})

    n = len(bets_settled)
    banks = [p["bank"] for p in curve]
    net = bank - start_bank
    return {
        "final_bankroll": round(bank, 2),
        "net_usd": round(net, 2),
        "roi_pct": round(net / start_bank * 100, 2) if start_bank else 0.0,
        "n_bets": n,
        "hit_rate": round(wins / n * 100, 1) if n else 0.0,
        "max_drawdown_pct": _max_drawdown_pct(banks),
        "sharpe": _sharpe(bank_returns),
        "curve": curve,
    }


def build_summary(bets: list[dict], cfg: dict, start_bank: float) -> dict:
    clv = _clv_lookup()
    settled = sorted([b for b in bets if b["settled"]], key=lambda b: (b["date"], b["match_id"]))

    # CLV alignment over settled bets that have a closing line logged.
    clv_hits, clv_n = 0, 0
    for b in settled:
        c = clv.get((b["match_id"], b["side"]))
        if c is not None and not math.isnan(c.get("clv_pct", float("nan"))):
            clv_n += 1
            clv_hits += int(c["beat_close"])
    clv_alignment = round(clv_hits / clv_n * 100, 1) if clv_n else None

    n_real = sum(1 for b in settled if b["market_source"] == "real")
    n_fair = sum(1 for b in settled if b["market_source"] == "fair")

    strategies = {}
    for key, label, frac, blurb in STRATEGIES:
        s = run_strategy(settled, frac, start_bank, cfg["unit_size_usd"])
        s.update({"label": label, "explainer": blurb, "clv_alignment_pct": clv_alignment})
        strategies[key] = s

    # Verdict: realized edge (flat ROI) + best risk-adjusted strategy (top Sharpe).
    flat = strategies["flat"]
    best_key = max(strategies, key=lambda k: strategies[k]["sharpe"]) if settled else "kelly_half"
    best_label = strategies[best_key]["label"]
    n = len(settled)
    if n == 0:
        verdict = "No settled value bets yet — verdict pending the first results."
    else:
        sign = "+" if flat["roi_pct"] >= 0 else ""
        verdict = (
            f"Model is {sign}{flat['roi_pct']}% realized (flat staking) over {n} value "
            f"bets so far; {best_label} is the best risk-adjusted strategy to date."
        )

    return {
        "config": {
            "start_bankroll_usd": start_bank,
            "unit_size_usd": cfg["unit_size_usd"],
            "min_edge": cfg["min_edge"],
            "min_model_prob": cfg["min_model_prob"],
            "kelly_cap": KELLY_CAP,
        },
        "universe": {
            "n_value_bets": len(bets),
            "n_settled": n,
            "n_open": len(bets) - n,
            "n_settled_real_line": n_real,
            "n_settled_fair_line": n_fair,
            "clv_alignment_pct": clv_alignment,
        },
        "strategies": strategies,
        "verdict": verdict,
        "caveat": (
            "Fair-priced bets (market_source='fair') use de-vigged odds — research-grade "
            "P&L, slightly optimistic vs a real book. Edge is still unproven out-of-sample; "
            "this tournament is the test."
        ),
        "bets": bets,
    }


def run(start_bank: float = 1000.0, write: bool = True) -> dict:
    cfg = load_config()
    if not PRED_LEDGER.exists():
        print(f"  {PRED_LEDGER.name} not found — cannot build performance tracker.")
        return {}
    pred = pd.read_csv(PRED_LEDGER)
    bets = build_value_bets(pred, cfg)
    summary = build_summary(bets, cfg, start_bank)

    if write:
        SUMMARY_OUT.write_text(json.dumps(summary, indent=2, default=str))
        ledger_cols = ["match_id", "date", "label", "side", "selection", "model_prob",
                       "market_prob", "edge", "decimal_odds", "market_source",
                       "kelly_full", "status", "pnl_unit", "settled", "result_side"]
        pd.DataFrame([{k: b[k] for k in ledger_cols} for b in bets]).to_csv(LEDGER_OUT, index=False)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="WC2026 realized model performance tracker")
    ap.add_argument("--bankroll", type=float, default=1000.0,
                    help="starting bankroll for the strategy comparison (default 1000)")
    args = ap.parse_args()

    print("═" * 66)
    print("  WC2026 Model Performance Tracker (realized, multi-strategy)")
    print("═" * 66)
    s = run(start_bank=args.bankroll, write=True)
    if not s:
        return 1
    u = s["universe"]
    print(f"  value bets: {u['n_value_bets']}  (settled {u['n_settled']}, open {u['n_open']})")
    print(f"  settled coverage: {u['n_settled_real_line']} real-line · {u['n_settled_fair_line']} fair-line")
    print("─" * 66)
    print(f"  {'strategy':<14}{'final':>9}{'ROI%':>9}{'bets':>6}{'hit%':>7}{'maxDD%':>8}{'Sharpe':>8}")
    for key, label, _f, _b in STRATEGIES:
        st = s["strategies"][key]
        print(f"  {label:<14}{st['final_bankroll']:>9.1f}{st['roi_pct']:>+9.1f}"
              f"{st['n_bets']:>6}{st['hit_rate']:>7.1f}{st['max_drawdown_pct']:>8.1f}{st['sharpe']:>8.3f}")
    print("─" * 66)
    print(f"  VERDICT: {s['verdict']}")
    print(f"  wrote {SUMMARY_OUT.name} + {LEDGER_OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
