"""
V6 — Prediction tracker: store every model prediction, score it when the match
settles (correct / incorrect + log-loss vs the market), feed the record back
to the desk as EVIDENCE — not as an automatic model rewrite.

Why a ledger and not "just read wc2026_predictions.csv": that file is
REGENERATED after every result ingest (live_update re-runs ELO + sims), so a
settled match's current row was produced AFTER its own result — scoring it
would be leakage. Same fix as bet_ledger (DL-11): append-only, log once,
never rewrite history.

  LEDGER  data/processed/prediction_ledger.csv — one row per match, logged the
    first time the match appears in wc2026_predictions.csv while still
    unplayed (prob_source='pre_result'). A match that settled before it could
    be logged is recorded with prob_source='post_result' and EXCLUDED from all
    headline metrics (shown, never scored — honest bookkeeping).
    Market fair probs (Shin de-vig, from market_implied_probs.csv) are logged
    alongside so model-vs-market scoring uses prices from the same moment.
  SCOREBOARD  data/processed/prediction_scoreboard.csv — ledger + settlement:
    realized outcome, correct flag (argmax), p(realized), log-loss, and the
    market's log-loss on the same match.

How this "improves the model": results already flow into future predictions
through the ELO update in live_update.py (the model re-rates teams after
every match). What this module adds is the EVIDENCE layer: once n ≥ 40
settled, it prints a tournament reliability check (predicted vs realized by
favorite bucket — the DL-10 diagnostic, live). Any recalibration that check
suggests still has to clear the validate-or-cut bar (mean ΔLL ≥ +0.003,
both folds) before touching the pipeline — n < 40 results are noise, and
auto-refitting on them would chase it (and contaminate the DL-11 CLV
experiment mid-sample).

Outputs: prediction_ledger.csv (state), prediction_scoreboard.csv
Run: python -m src.models.prediction_tracker   (after predict_wc2026 /
     fetch_live_results; before build_dashboard)
"""

import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_RAW = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"

PRED_PATH    = DATA_PROC / "wc2026_predictions.csv"
IMPLIED_PATH = DATA_PROC / "market_implied_probs.csv"
RESULTS_PATH = DATA_RAW / "wc2026_live_results.csv"
LEDGER_PATH  = DATA_PROC / "prediction_ledger.csv"
SCORE_PATH   = DATA_PROC / "prediction_scoreboard.csv"

EPS = 1e-9
OUTCOME = {"home": "p_home_win", "draw": "p_draw", "away": "p_away_win"}
RELIABILITY_MIN_N = 40


def settled_outcomes() -> dict:
    """match_id → realized outcome ('home'|'draw'|'away')."""
    if not RESULTS_PATH.exists():
        return {}
    res = pd.read_csv(RESULTS_PATH)
    out = {}
    for _, r in res.iterrows():
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        out[int(r["match_id"])] = "home" if hg > ag else "away" if ag > hg else "draw"
    return out


def update_ledger(settled: dict) -> pd.DataFrame:
    """Log every not-yet-logged match ONCE; never touch existing rows."""
    preds = pd.read_csv(PRED_PATH)
    implied = pd.read_csv(IMPLIED_PATH) if IMPLIED_PATH.exists() else pd.DataFrame()
    mkt = {int(r["match_id"]): r for _, r in implied.iterrows()} if len(implied) else {}

    ledger = pd.read_csv(LEDGER_PATH) if LEDGER_PATH.exists() else pd.DataFrame()
    have = set(ledger["match_id"].astype(int)) if len(ledger) else set()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new = []
    for _, r in preds.iterrows():
        mid = int(r["match_id"])
        if mid in have:
            continue
        m = mkt.get(mid)
        new.append({
            "match_id": mid, "date": r["date"], "stage": r["stage"],
            "home_team": r["home_team"], "away_team": r["away_team"],
            "p_home": r["p_home_win"], "p_draw": r["p_draw"], "p_away": r["p_away_win"],
            "model_pick": max(OUTCOME, key=lambda s: r[OUTCOME[s]]),
            "mkt_home": round(float(m["home_fair_prob"]), 4) if m is not None else "",
            "mkt_draw": round(float(m["draw_fair_prob"]), 4) if m is not None else "",
            "mkt_away": round(float(m["away_fair_prob"]), 4) if m is not None else "",
            # settled before we could log it ⇒ current probs were re-simmed AFTER
            # the result (leakage) — keep the row, never score it
            "prob_source": "post_result" if mid in settled else "pre_result",
            "logged_at": now,
        })
    if new:
        ledger = pd.concat([ledger, pd.DataFrame(new)], ignore_index=True)
        ledger.to_csv(LEDGER_PATH, index=False)
    n_post = sum(1 for r in new if r["prob_source"] == "post_result")
    print(f"  Ledger: {len(new)} new predictions logged "
          f"({n_post} post-result, excluded from scoring), {len(ledger)} total")
    return ledger


def build_scoreboard(ledger: pd.DataFrame, settled: dict) -> pd.DataFrame:
    rows = []
    for _, b in ledger.iterrows():
        mid = int(b["match_id"])
        out = settled.get(mid)
        p = {"home": float(b["p_home"]), "draw": float(b["p_draw"]),
             "away": float(b["p_away"])}
        scored = out is not None and b["prob_source"] == "pre_result"
        mkt_p = None
        if out and b["mkt_home"] != "" and pd.notna(b["mkt_home"]):
            mkt_p = {"home": float(b["mkt_home"]), "draw": float(b["mkt_draw"]),
                     "away": float(b["mkt_away"])}[out]
        rows.append({
            **b,
            "status": "settled" if out else "pending",
            "realized": out or "",
            "correct": (b["model_pick"] == out) if scored else "",
            "p_realized": round(p[out], 4) if scored else "",
            "log_loss": round(-np.log(max(p[out], EPS)), 4) if scored else "",
            "mkt_p_realized": round(mkt_p, 4) if scored and mkt_p else "",
            "mkt_log_loss": round(-np.log(max(mkt_p, EPS)), 4)
                            if scored and mkt_p else "",
        })
    board = pd.DataFrame(rows)
    board.to_csv(SCORE_PATH, index=False)
    return board


def reliability_check(board: pd.DataFrame):
    """Tournament-live DL-10 diagnostic: predicted vs realized favorite rate.
    Diagnostic ONLY — any fix still has to clear validate-or-cut (DL-09 bar)."""
    g = board[(board["status"] == "settled") & (board["prob_source"] == "pre_result")]
    if len(g) < RELIABILITY_MIN_N:
        print(f"\n  Reliability check: dormant ({len(g)}/{RELIABILITY_MIN_N} "
              f"settled — n < {RELIABILITY_MIN_N} is noise, not signal)")
        return
    p = g[["p_home", "p_draw", "p_away"]].astype(float).values
    fav_col = np.where(p[:, 0] >= p[:, 2], 0, 2)
    pf = p[np.arange(len(g)), fav_col]
    fav_won = np.where(fav_col == 0, g["realized"] == "home", g["realized"] == "away")
    print(f"\n── Tournament reliability (n={len(g)}; diagnostic, not a calibrator) ──")
    for lo, hi in ((0.34, 0.50), (0.50, 0.65), (0.65, 1.01)):
        m = (pf >= lo) & (pf < hi)
        if m.sum() >= 10:
            print(f"  fav {lo:.2f}–{hi:.2f}: n={int(m.sum())} "
                  f"predicted {pf[m].mean():.1%} realized {fav_won[m].mean():.1%} "
                  f"gap {fav_won[m].mean() - pf[m].mean():+.1%}")


def print_summary(board: pd.DataFrame):
    s = board[(board["status"] == "settled") & (board["prob_source"] == "pre_result")]
    print(f"\n── Model record ({len(s)} scored, "
          f"{(board['status'] == 'pending').sum()} pending) ──")
    if not len(s):
        print("  Nothing scored yet.")
        return
    acc = (s["correct"] == True).mean()  # noqa: E712 — CSV bools
    ll = s["log_loss"].astype(float).mean()
    print(f"  Record: {(s['correct'] == True).sum()}-{(s['correct'] != True).sum()} "  # noqa: E712
          f"({acc:.0%}) | avg log-loss {ll:.4f} (test baseline 0.8405)")
    m = s[s["mkt_log_loss"] != ""]
    if len(m):
        mll = m["mkt_log_loss"].astype(float).mean()
        edge = mll - m["log_loss"].astype(float).mean()
        who = "model ahead" if edge > 0 else "market ahead"
        print(f"  vs market (same {len(m)} matches): model LL "
              f"{m['log_loss'].astype(float).mean():.4f} vs market LL {mll:.4f} "
              f"→ {who} by {abs(edge):.4f}")
        print(f"  (n={len(m)} is far below significance — see the edge-vs-error "
              f"memo §3a; CLV is the powered statistic, this is the diary)")


def main():
    print("═" * 60)
    print("  V6 — Prediction Tracker (append-only; score on settle)")
    print("═" * 60)
    settled = settled_outcomes()
    ledger = update_ledger(settled)
    if not len(ledger):
        print("  Nothing to track — run predict_wc2026 first.")
        return
    board = build_scoreboard(ledger, settled)
    print_summary(board)
    reliability_check(board)
    print(f"\n✅ Saved: {LEDGER_PATH}")
    print(f"✅ Saved: {SCORE_PATH}")


if __name__ == "__main__":
    main()
