"""
V6 — Line movement monitor + cross-book arbitrage scanner.

Line movement (spec 1.6): per (match, book), opening vs latest line per outcome.
  - Significant move: ≥ 0.10 decimal-odds change OR ≥ 5pp implied-prob shift
  - Direction vs model: a move is TOWARD the model if the market's fair prob for
    that outcome moved closer to the model's prob (sharp money agreeing with us),
    AGAINST if it moved away. Sharp action moves lines; public money fades back.

Arbitrage scan: take the best (highest) decimal odds per outcome across all books'
latest lines. If Σ 1/best_odds < 1, backing all three outcomes locks a profit of
(1/Σ − 1) regardless of result. With a single book (DraftKings via ESPN) true arbs
are ~impossible — the scanner activates as more books are added to the odds CSV.
Near-arbs (Σ < 1.02) are listed as low-vig opportunities.

Inputs:  data/raw/wc2026_match_odds.csv  (snapshots from fetch_live_odds.py)
         data/processed/wc2026_predictions.csv (model probs, for move direction)
Outputs: data/processed/line_movement.csv
         data/processed/arb_scan.csv

Run: python -m src.models.market_monitor
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.market_ingestion import parse_odds_row, shin_fair

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_RAW = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"

ODDS_PATH        = DATA_RAW / "wc2026_match_odds.csv"
PREDICTIONS_PATH = DATA_PROC / "wc2026_predictions.csv"
MOVEMENT_OUT     = DATA_PROC / "line_movement.csv"
ARB_OUT          = DATA_PROC / "arb_scan.csv"

SIDES = ["home", "draw", "away"]
SIG_DECIMAL = 0.10   # significant move thresholds (spec 1.6)
SIG_IMPLIED = 0.05


def decimals(row) -> np.ndarray:
    return parse_odds_row([row["home_odds"], row["draw_odds"], row["away_odds"]],
                          row["odds_format"])


def build_movement(odds: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, book, outcome): open vs latest line + classification."""
    model = preds.set_index("match_id")
    rows = []
    for (mid, book), grp in odds.groupby(["match_id", "book"]):
        opens = grp[grp["snapshot"] == "opening"]
        lives = grp[grp["snapshot"] != "opening"].sort_values("fetched_at")
        if not len(opens) or not len(lives) or mid not in model.index:
            continue
        d_open, d_now = decimals(opens.iloc[-1]), decimals(lives.iloc[-1])
        fair_open, fair_now = shin_fair(1.0 / d_open), shin_fair(1.0 / d_now)
        m = model.loc[mid]
        p_model = np.array([m["p_home_win"], m["p_draw"], m["p_away_win"]])
        for i, side in enumerate(SIDES):
            d_move = d_now[i] - d_open[i]
            p_move = fair_now[i] - fair_open[i]
            significant = abs(d_move) >= SIG_DECIMAL or abs(p_move) >= SIG_IMPLIED
            toward = abs(p_model[i] - fair_now[i]) < abs(p_model[i] - fair_open[i])
            rows.append({
                "match_id": mid, "match": f"{m['home_team']} vs {m['away_team']}",
                "date": m["date"], "book": book, "outcome": side,
                "open_decimal": round(float(d_open[i]), 3),
                "now_decimal": round(float(d_now[i]), 3),
                "decimal_move": round(float(d_move), 3),
                "open_fair_prob": round(float(fair_open[i]), 4),
                "now_fair_prob": round(float(fair_now[i]), 4),
                "implied_shift": round(float(p_move), 4),
                "model_prob": round(float(p_model[i]), 4),
                "significant": significant,
                "direction_vs_model": "toward" if toward else "against",
                "last_fetched": lives.iloc[-1]["fetched_at"],
            })
    return pd.DataFrame(rows)


def build_arb(odds: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Best line per outcome across books (latest snapshot per book) → arb check."""
    model = preds.set_index("match_id")
    rows = []
    for mid, grp in odds.groupby("match_id"):
        if mid not in model.index:
            continue
        latest = (grp[grp["snapshot"] != "opening"]
                  .sort_values("fetched_at").groupby("book").last())
        if not len(latest):
            continue
        best_dec, best_book = {}, {}
        for _, r in latest.reset_index().iterrows():
            d = decimals(r)
            for i, side in enumerate(SIDES):
                if side not in best_dec or d[i] > best_dec[side]:
                    best_dec[side], best_book[side] = float(d[i]), r["book"]
        inv_sum = sum(1.0 / best_dec[s] for s in SIDES)
        m = model.loc[mid]
        rows.append({
            "match_id": mid, "match": f"{m['home_team']} vs {m['away_team']}",
            "date": m["date"], "n_books": len(latest),
            **{f"best_{s}_decimal": round(best_dec[s], 3) for s in SIDES},
            **{f"best_{s}_book": best_book[s] for s in SIDES},
            "inv_sum": round(inv_sum, 4),
            "arb": inv_sum < 1.0,
            "arb_roi_pct": round(max(0.0, 1.0 / inv_sum - 1.0) * 100, 3),
            "near_arb": inv_sum < 1.02,
        })
    return pd.DataFrame(rows).sort_values("inv_sum").reset_index(drop=True)


def main():
    print("═" * 60)
    print("  V6 — Line Movement Monitor + Arb Scanner")
    print("═" * 60)

    odds = pd.read_csv(ODDS_PATH)
    preds = pd.read_csv(PREDICTIONS_PATH)
    if not len(odds):
        print("  No odds snapshots yet — run src.models.fetch_live_odds first.")
        return

    move = build_movement(odds, preds)
    move.to_csv(MOVEMENT_OUT, index=False)
    sig = move[move["significant"]].copy()
    sig = sig.reindex(sig["implied_shift"].abs().sort_values(ascending=False).index)

    print(f"\n── Line movement: {len(sig)} significant moves "
          f"(of {len(move)} outcome-lines) ──")
    print(f"  {'Match':<32}{'Out':>6}{'Open':>7}{'Now':>7}{'Shift':>8}"
          f"{'Model%':>8}  vs model")
    print(f"  {'-' * 78}")
    for _, r in sig.head(12).iterrows():
        print(f"  {r['match'][:31]:<32}{r['outcome']:>6}{r['open_decimal']:>7.2f}"
              f"{r['now_decimal']:>7.2f}{r['implied_shift']*100:>+7.1f}p"
              f"{r['model_prob']*100:>7.1f}%  {r['direction_vs_model']}")
    if len(sig):
        toward = int((sig["direction_vs_model"] == "toward").sum())
        print(f"\n  Of significant moves: {toward} toward model / "
              f"{len(sig) - toward} against. Moves toward = sharp money on the")
        print(f"  model's side of the number; against = the market disagrees harder.")

    arb = build_arb(odds, preds)
    arb.to_csv(ARB_OUT, index=False)
    n_books = int(arb["n_books"].max()) if len(arb) else 0
    hits = arb[arb["arb"]]
    print(f"\n── Arb scan: {len(hits)} true arbs, "
          f"{int(arb['near_arb'].sum())} near-arbs (Σ<1.02) | books: {n_books} ──")
    if len(hits):
        for _, r in hits.iterrows():
            print(f"  💰 {r['match']}: Σ={r['inv_sum']:.4f} → "
                  f"{r['arb_roi_pct']:.2f}% riskless ROI")
    else:
        best = arb.iloc[0] if len(arb) else None
        if best is not None:
            print(f"  None (expected with {n_books} book). Tightest: {best['match']} "
                  f"Σ={best['inv_sum']:.3f} ({(best['inv_sum']-1)*100:.1f}% vig)")
        print(f"  Scanner activates automatically when more books land in "
              f"{ODDS_PATH.name}.")

    print(f"\n✅ Saved: {MOVEMENT_OUT}  ({len(move)} rows)")
    print(f"✅ Saved: {ARB_OUT}  ({len(arb)} rows)")


if __name__ == "__main__":
    main()
