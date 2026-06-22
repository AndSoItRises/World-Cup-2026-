"""
Underdog "+0.5 insurance" tracker (DL-16/DL-18).

When the model backs an underdog, this logs TWO correlated bets on the match —
the moneyline (team to win) and the +0.5 (team to win OR draw, i.e. cashes whenever
the favorite does not win) — sizes them with JOINT Kelly (see insurance_sizing.py),
settles them against actual results, and tracks THREE bankroll curves so we can tell,
honestly, whether the insurance leg helps:

    1. ML-only      — bet just the moneylines (solo-Kelly sized)
    2. +0.5-only    — bet just the win-or-draw legs (solo-Kelly sized)
    3. Combined     — both legs, joint-Kelly sized   ← the strategy we'd actually run

Tiers (config.json → insurance):
    market-implied win ≤ big_dog_threshold (0.30)  → BIG DOG   (ML + insurance)
    big_dog < implied < 0.50                       → TOSS-UP   (track both)
    implied ≥ 0.50                                 → favorite  (skip — no insurance)

A leg is only recommended if its edge clears min_leg_edge. Source probabilities are
the pre-result model + Shin-fair market 1X2 from prediction_ledger.csv; the +0.5 price
is derived (market win-or-draw = mkt_team + mkt_draw). Odds are fair (de-vigged), so
P&L is research-grade — directionally honest, slightly optimistic vs a real book price.

Outputs:
    data/processed/insurance_ledger.csv    — per-match recommendation + settlement
    data/processed/insurance_summary.json  — 3 equity curves + KPIs + open recs (for the HTML)

Run:
    python -m src.models.insurance_tracker
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from src.models.insurance_sizing import joint_kelly, solo_kelly, explain
from src.models.settle_bets import load_results

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"

CONFIG_PATH = DATA_PROC / "config.json"
PRED_LEDGER = DATA_PROC / "prediction_ledger.csv"
LEDGER_OUT = DATA_PROC / "insurance_ledger.csv"
SUMMARY_OUT = DATA_PROC / "insurance_summary.json"

LEDGER_COLUMNS = [
    "match_id", "date", "selection", "opponent", "tier",
    "market_implied_win", "model_win", "model_cover",
    "edge_ml", "edge_dc", "decimal_ml", "decimal_dc",
    "f_ml", "f_dc", "f_ml_solo", "f_dc_solo",
    "status_ml", "status_dc", "result_side", "settled", "rationale",
]


def load_config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text())
    ins = cfg.get("insurance", {})
    return {
        "big_dog": ins.get("big_dog_threshold", 0.30),
        "tossup_hi": ins.get("tossup_band", [0.30, 0.50])[1],
        "min_leg_edge": ins.get("min_leg_edge", 0.02),
        "kelly_fraction": ins.get("kelly_fraction", 0.5),
        "cap": ins.get("per_leg_cap", 0.05),
        "bank0": ins.get("starting_bankroll_units", 100.0),
    }


def build_recommendations(pred: pd.DataFrame, results: dict, cfg: dict) -> list[dict]:
    """One record per qualifying underdog pick, with both legs sized + (if played) settled."""
    pred = pred.drop_duplicates(subset="match_id", keep="first")
    recs = []
    for _, m in pred.iterrows():
        side = str(m.get("model_pick", "")).strip().lower()
        if side not in ("home", "away"):
            continue  # need a team to back (draw picks have nothing to insure)

        if side == "home":
            p_team, p_oppwin, mkt_team = m["p_home"], m["p_away"], m["mkt_home"]
            team, opp = m["home_team"], m["away_team"]
        else:
            p_team, p_oppwin, mkt_team = m["p_away"], m["p_home"], m["mkt_away"]
            team, opp = m["away_team"], m["home_team"]
        p_d, mkt_d = m["p_draw"], m["mkt_draw"]

        # Tier by how the MARKET prices the pick's win.
        if mkt_team >= cfg["tossup_hi"]:
            continue  # favorite — no insurance leg
        tier = "BIG DOG" if mkt_team <= cfg["big_dog"] else "TOSS-UP"

        # Legs: ML (win) and +0.5 (win-or-draw). Fair decimals from de-vigged market.
        d1 = 1.0 / mkt_team if mkt_team > 0 else 0.0
        edge_ml = float(p_team - mkt_team)
        cover_model = float(p_team + p_d)
        cover_mkt = float(mkt_team + mkt_d)
        d2 = 1.0 / cover_mkt if cover_mkt > 0 else 0.0
        edge_dc = cover_model - cover_mkt

        include_ml = edge_ml > cfg["min_leg_edge"]
        include_dc = edge_dc > cfg["min_leg_edge"]
        if not (include_ml or include_dc):
            continue

        f1, f2 = joint_kelly(p_team, p_d, p_oppwin, d1, d2,
                             include_ml=include_ml, include_dc=include_dc,
                             kelly_fraction=cfg["kelly_fraction"], cap=cfg["cap"])
        f1_solo = solo_kelly(p_team, d1, kelly_fraction=cfg["kelly_fraction"], cap=cfg["cap"]) if include_ml else 0.0
        f2_solo = solo_kelly(cover_model, d2, kelly_fraction=cfg["kelly_fraction"], cap=cfg["cap"]) if include_dc else 0.0

        rec = {
            "match_id": int(m["match_id"]),
            "date": str(m["date"])[:10],
            "selection": f"{team} ({side})",
            "opponent": opp,
            "tier": tier,
            "market_implied_win": round(float(mkt_team), 4),
            "model_win": round(float(p_team), 4),
            "model_cover": round(cover_model, 4),
            "edge_ml": round(edge_ml, 4),
            "edge_dc": round(edge_dc, 4),
            "decimal_ml": round(d1, 4),
            "decimal_dc": round(d2, 4),
            "f_ml": round(f1, 4), "f_dc": round(f2, 4),
            "f_ml_solo": round(f1_solo, 4), "f_dc_solo": round(f2_solo, 4),
            "rationale": explain(p_team, p_d, p_oppwin, d1, d2, f1, f2),
            # filled at settlement:
            "status_ml": "", "status_dc": "", "result_side": "", "settled": False,
            # internal (not written): realized per-unit returns
            "_r1": 0.0, "_r2": 0.0, "_incl_ml": include_ml, "_incl_dc": include_dc,
        }

        res = results.get(rec["match_id"])
        if res is not None:
            rs = res["result_side"]
            rec["result_side"], rec["settled"] = rs, True
            if include_ml:
                ml_win = (rs == side)
                rec["status_ml"] = "WON" if ml_win else "LOST"
                rec["_r1"] = (d1 - 1.0) if ml_win else -1.0
            if include_dc:
                dc_win = rs in (side, "draw")
                rec["status_dc"] = "WON" if dc_win else "LOST"
                rec["_r2"] = (d2 - 1.0) if dc_win else -1.0
        recs.append(rec)
    return recs


def _max_drawdown(curve: list[float]) -> float:
    peak, mdd = curve[0] if curve else 0.0, 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return round(mdd * 100, 2)


def compute_streams(recs: list[dict], cfg: dict) -> dict:
    """Compound three bankrolls over settled picks (chronological) and collect curves."""
    settled = sorted([r for r in recs if r["settled"]], key=lambda r: (r["date"], r["match_id"]))
    bank0 = cfg["bank0"]
    streams = {
        "ml_only": {"bank": bank0, "curve": [], "n": 0, "wins": 0, "label": "ML only"},
        "plus_half": {"bank": bank0, "curve": [], "n": 0, "wins": 0, "label": "+0.5 only"},
        "combined": {"bank": bank0, "curve": [], "n": 0, "wins": 0, "label": "Combined (joint Kelly)"},
    }
    pts = {k: [{"date": "start", "bank": round(bank0, 2)}] for k in streams}

    for r in settled:
        # combined: both legs, joint-Kelly fractions
        ret_c = r["f_ml"] * r["_r1"] + r["f_dc"] * r["_r2"]
        streams["combined"]["bank"] *= (1 + ret_c)
        streams["combined"]["n"] += 1
        streams["combined"]["wins"] += int(ret_c > 0)
        # ml-only: solo-Kelly ML leg
        if r["_incl_ml"]:
            ret_m = r["f_ml_solo"] * r["_r1"]
            streams["ml_only"]["bank"] *= (1 + ret_m)
            streams["ml_only"]["n"] += 1
            streams["ml_only"]["wins"] += int(r["_r1"] > 0)
        # +0.5-only: solo-Kelly DC leg
        if r["_incl_dc"]:
            ret_d = r["f_dc_solo"] * r["_r2"]
            streams["plus_half"]["bank"] *= (1 + ret_d)
            streams["plus_half"]["n"] += 1
            streams["plus_half"]["wins"] += int(r["_r2"] > 0)
        for k in streams:
            pts[k].append({"date": r["date"], "bank": round(streams[k]["bank"], 2)})

    out = {}
    for k, s in streams.items():
        banks = [p["bank"] for p in pts[k]]
        out[k] = {
            "label": s["label"],
            "final_bankroll": round(s["bank"], 2),
            "roi_pct": round((s["bank"] - bank0) / bank0 * 100, 2),
            "n_bets": s["n"],
            "win_rate": round(s["wins"] / s["n"] * 100, 1) if s["n"] else 0.0,
            "max_drawdown_pct": _max_drawdown(banks),
            "curve": pts[k],
        }
    return out


def run(write: bool = True) -> dict:
    cfg = load_config()
    if not PRED_LEDGER.exists():
        print(f"  {PRED_LEDGER.name} not found — cannot build insurance recs.")
        return {}
    pred = pd.read_csv(PRED_LEDGER)
    results, _ = load_results()

    recs = build_recommendations(pred, results, cfg)
    streams = compute_streams(recs, cfg)

    # Ledger CSV (drop internal cols).
    ledger = pd.DataFrame([{k: r[k] for k in LEDGER_COLUMNS} for r in recs])
    if write and not ledger.empty:
        ledger.to_csv(LEDGER_OUT, index=False)

    open_recs = [{k: r[k] for k in LEDGER_COLUMNS} for r in recs if not r["settled"]]
    summary = {
        "config": cfg,
        "n_recommendations": len(recs),
        "n_settled": sum(1 for r in recs if r["settled"]),
        "n_open": len(open_recs),
        "streams": streams,
        "open_recs": open_recs,
        "ledger": [{k: r[k] for k in LEDGER_COLUMNS} for r in recs],
    }
    if write:
        SUMMARY_OUT.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    print("═" * 64)
    print("  WC2026 Underdog +0.5 Insurance Tracker")
    print("═" * 64)
    s = run(write=True)
    if not s:
        return 1
    print(f"  recommendations: {s['n_recommendations']}  "
          f"(settled {s['n_settled']}, open {s['n_open']})")
    print("─" * 64)
    print(f"  {'stream':<26}{'final':>8}{'ROI%':>9}{'bets':>6}{'win%':>7}{'maxDD%':>8}")
    for k in ("ml_only", "plus_half", "combined"):
        st = s["streams"][k]
        print(f"  {st['label']:<26}{st['final_bankroll']:>8.1f}{st['roi_pct']:>+9.1f}"
              f"{st['n_bets']:>6}{st['win_rate']:>7.1f}{st['max_drawdown_pct']:>8.1f}")
    print("─" * 64)
    print(f"  wrote {LEDGER_OUT.name} + {SUMMARY_OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
