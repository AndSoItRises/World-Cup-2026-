"""
V6 Phase 1 / Learning Path Step 4 — Expected Value engine + Kelly criterion.

The quant-betting chain, per outcome:

  edge     = model_prob − market_fair_prob          (are we righter than the price?)
  EV       = p·(d−1) − (1−p)                        (per-$1 profit at decimal odds d)
  Kelly %  = (b·p − q) / b   with b = d−1, q = 1−p  (growth-optimal stake)

Kelly is computed on the QUOTED odds (vig included — that's what you're paid),
edge on the FAIR (de-vigged) prob — that's the honest model-vs-market gap.
Full Kelly on a 62%-accurate model is reckless: we report full + 1/4 Kelly and
cap the recommended stake at KELLY_CAP of bankroll.

Flags (from the V6 spec): edge ≥ +3pp value / ≥ +5pp strong; EV > 0 value /
EV > 0.05 strong.

Inputs:  data/processed/wc2026_predictions.csv      (model 3-way probs)
         data/processed/tournament_probs_live.csv   (model p_winner)
         market odds via src.models.market_ingestion (rebuilt fresh each run)
         data/raw/wc2026_live_results.csv           (played matches → excluded)
Outputs: data/processed/value_bets.csv              (match-level, long format)
         data/processed/value_bets_futures.csv      (tournament winner)

Run: python -m src.models.bet_sim [--bankroll 1000] [--kelly-cap 0.05]

CAVEAT (carried from V5 DL-05, printed on every run): the model's edge vs the
market is UNPROVEN out-of-sample. These are research signals, not validated +EV.
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

from src.models.market_ingestion import (
    build_futures_implied, build_match_implied,
    MATCH_IMPLIED_OUT, FUTURES_IMPLIED_OUT,
)

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_RAW = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"

LIVE_RESULTS_PATH = DATA_RAW / "wc2026_live_results.csv"
PREDICTIONS_PATH  = DATA_PROC / "wc2026_predictions.csv"

MATCH_BETS_OUT   = DATA_PROC / "value_bets.csv"
FUTURES_BETS_OUT = DATA_PROC / "value_bets_futures.csv"

KELLY_CAP = 0.05   # never recommend more than 5% of bankroll on one bet

# Futures probs come from 10k Monte Carlo sims: p_winner = 0.01 is only ~100 sims,
# so tail probabilities carry large relative error — at 100:1+ odds that noise
# masquerades as huge EV (favorite-longshot trap). Flag, don't trust.
TAIL_PROB_FLOOR = 0.02


# ── Core math ─────────────────────────────────────────────────────────────────
def expected_value(p: float, decimal_odds: float) -> float:
    """Per-$1 EV: p·(d−1) − (1−p)."""
    return p * (decimal_odds - 1.0) - (1.0 - p)


def kelly_fraction(p: float, decimal_odds: float) -> float:
    """(b·p − q)/b, floored at 0 (never bet a negative-edge line)."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (b * p - (1.0 - p)) / b)


def edge_flag(edge: float) -> str:
    if edge >= 0.05:
        return "STRONG"
    if edge >= 0.03:
        return "value"
    return ""


def ev_flag(ev: float) -> str:
    if ev > 0.05:
        return "STRONG"
    if ev > 0:
        return "value"
    return ""


def grade_bet(row, p_col, odds_col, fair_col, bankroll, kelly_cap):
    """Common edge/EV/Kelly block appended to a bet row dict."""
    p, d, fair = row[p_col], row[odds_col], row[fair_col]
    ev = expected_value(p, d)
    kf = kelly_fraction(p, d)
    edge = p - fair
    stake_pct = min(kf / 4.0, kelly_cap)
    return {
        "edge": round(edge, 4), "edge_flag": edge_flag(edge),
        "ev": round(ev, 4), "ev_flag": ev_flag(ev),
        "kelly_full": round(kf, 4), "kelly_quarter": round(kf / 4.0, 4),
        "recommended_stake_pct": round(stake_pct, 4),
        "recommended_stake_usd": round(stake_pct * bankroll, 2),
    }


# ── Match-level sheet ─────────────────────────────────────────────────────────
def played_match_ids() -> set:
    if not LIVE_RESULTS_PATH.exists():
        return set()
    live = pd.read_csv(LIVE_RESULTS_PATH)
    return set(live["match_id"].astype(int)) if len(live) else set()


def build_match_bets(bankroll: float, kelly_cap: float) -> pd.DataFrame:
    preds = pd.read_csv(PREDICTIONS_PATH)
    implied = build_match_implied()
    df = preds.merge(
        implied.drop(columns=["date", "stage", "group", "home_team", "away_team"]),
        on="match_id",
    )
    played = played_match_ids()

    rows = []
    for _, m in df.iterrows():
        if int(m["match_id"]) in played:
            continue
        match_label = f"{m['home_team']} vs {m['away_team']}"
        for side, p_col, team in [("home", "p_home_win", m["home_team"]),
                                  ("draw", "p_draw", "Draw"),
                                  ("away", "p_away_win", m["away_team"])]:
            row = {
                "match_id": int(m["match_id"]), "date": m["date"], "stage": m["stage"],
                "group": m["group"], "match": match_label, "outcome": side,
                "selection": team,
                "model_prob": round(float(m[p_col]), 4),
                "decimal_odds": float(m[f"{side}_decimal_odds"]),
                "market_implied": float(m[f"{side}_fair_prob"]),
                "market_source": m["market_source"], "book": m["book"],
            }
            row.update(grade_bet(row, "model_prob", "decimal_odds", "market_implied",
                                 bankroll, kelly_cap))
            rows.append(row)

    out = pd.DataFrame(rows).sort_values("ev", ascending=False).reset_index(drop=True)
    return out


# ── Futures sheet ─────────────────────────────────────────────────────────────
def build_futures_bets(bankroll: float, kelly_cap: float) -> pd.DataFrame:
    fut = build_futures_implied()
    fut = fut[fut["p_winner"].notna() & ~fut["eliminated"].astype(bool)].copy()

    rows = []
    for _, r in fut.iterrows():
        row = {
            "market": "tournament_winner", "selection": r["team"],
            "model_prob": float(r["p_winner"]),
            "american_odds": r["american_odds"],
            "decimal_odds": round(float(r["decimal_odds"]), 4),
            "market_implied": round(float(r["market_prob"]), 4),
            "market_source": "real",
        }
        row.update(grade_bet(row, "model_prob", "decimal_odds", "market_implied",
                             bankroll, kelly_cap))
        row["tail_risk"] = row["model_prob"] < TAIL_PROB_FLOOR
        rows.append(row)

    return pd.DataFrame(rows).sort_values("ev", ascending=False).reset_index(drop=True)


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_bet_table(df: pd.DataFrame, title: str, label_col: str, top: int = 10):
    """Edge, EV and Kelly stake are always shown together — never in isolation."""
    print(f"\n── {title} ──")
    print(f"  {'Selection':<30}{'Model%':>8}{'Fair%':>8}{'Edge':>8}"
          f"{'Odds':>7}{'EV':>8}{'K-full':>8}{'K/4':>7}{'Stake$':>8}  Flag")
    print(f"  {'-' * 102}")
    for _, r in df.head(top).iterrows():
        print(f"  {str(r[label_col])[:29]:<30}{r['model_prob']*100:>7.1f}%"
              f"{r['market_implied']*100:>7.1f}%{r['edge']*100:>+7.1f}p"
              f"{r['decimal_odds']:>7.2f}{r['ev']:>+8.3f}"
              f"{r['kelly_full']*100:>7.1f}%{r['kelly_quarter']*100:>6.1f}%"
              f"{r['recommended_stake_usd']:>8.2f}  {r['ev_flag']}")


def main():
    ap = argparse.ArgumentParser(description="EV + Kelly engine (V6 Vegas layer)")
    ap.add_argument("--bankroll", type=float, default=1000.0,
                    help="bankroll in USD for stake sizing (default 1000)")
    ap.add_argument("--kelly-cap", type=float, default=KELLY_CAP,
                    help="max recommended stake as fraction of bankroll (default 0.05)")
    args = ap.parse_args()

    print("═" * 60)
    print("  V6 — Expected Value Engine + Kelly Criterion")
    print(f"  Bankroll: ${args.bankroll:,.0f} | stake cap: {args.kelly_cap:.0%} "
          f"| sizing: 1/4 Kelly (capped)")
    print("═" * 60)

    futures = build_futures_bets(args.bankroll, args.kelly_cap)
    futures.to_csv(FUTURES_BETS_OUT, index=False)

    matches = build_match_bets(args.bankroll, args.kelly_cap)
    matches.to_csv(MATCH_BETS_OUT, index=False)

    n_value = int((futures["ev"] > 0).sum())
    credible = futures[~futures["tail_risk"]]
    print_bet_table(credible, f"Tournament winner futures (model_prob ≥ "
                              f"{TAIL_PROB_FLOOR:.0%}) — {int((credible['ev'] > 0).sum())} "
                              f"positive-EV of {len(credible)}", "selection")
    n_tail = int((futures["tail_risk"] & (futures["ev"] > 0)).sum())
    if n_tail:
        print(f"\n  {n_tail} more positive-EV rows are tail_risk=True (model_prob < "
              f"{TAIL_PROB_FLOOR:.0%} ≈ <200 of 10k sims) — MC tail noise at long")
        print(f"  odds, not edge. Kept in the CSV, excluded from the headline table.")
    print(f"  ({n_value} positive-EV total of {len(futures)} priced teams)")

    real = matches[matches["market_source"] == "real"].copy()
    if len(real):
        real["bet_label"] = real["selection"] + " (" + real["outcome"] + ") " + real["match"]
        print_bet_table(real, f"Match bets (real lines) — "
                              f"{int((real['ev'] > 0).sum())} positive-EV", "bet_label")
        pos = real[real["ev"] > 0]
        by_outcome = pos["outcome"].value_counts().to_dict()
        print(f"\n  Positive-EV by outcome: {by_outcome} of {len(real)} rows.")
        print(f"  ⚠️  Draw/underdog-heavy +EV is the model's DOCUMENTED draw upweight")
        print(f"  (1.75×) and ELO compression on lopsided ties — model bias, not")
        print(f"  market error. Trust favorite-side edges more than draw/dog edges.")
    else:
        print(f"\n── Match bets ──")
        print(f"  No real match lines yet → all {len(matches)} match-outcome rows are")
        print(f"  model_estimated (EV ≡ 0 by construction). Fill "
              f"data/raw/wc2026_match_odds.csv to activate.")

    print(f"\n✅ Saved: {FUTURES_BETS_OUT}  ({len(futures)} rows)")
    print(f"✅ Saved: {MATCH_BETS_OUT}  ({len(matches)} rows)")
    print(f"✅ Saved: {FUTURES_IMPLIED_OUT.name} / {MATCH_IMPLIED_OUT.name} (via ingestion)")

    print("\n  ⚠️  CAVEAT: model accuracy is 62% and its edge vs the market is")
    print("  UNPROVEN out-of-sample (V5 DL-05 — needs historical odds). Treat these")
    print("  as research signals with uncertainty, not validated +EV bets.")
    print("  ⚠️  Known model biases overlap the edges: CONCACAF inflation (Mexico")
    print("  ~+6pp vs market) is documented model error, not market error.")


if __name__ == "__main__":
    main()
