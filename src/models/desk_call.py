"""
V6 — Desk Call engine: turn the Vegas-layer data into actual recommendations.

The scanner answers "where is the model ahead of the price?" — this module
answers the question a desk actually gets asked: "so what do we bet?"
Rule-based and fully transparent — every verdict carries its evidence chain
and every haircut is a documented model bias, not a vibe:

  PASS outright : EV ≤ 0 · draw bets (1.75× draw upweight = model bias) ·
                  futures tail noise (model_prob < 2%)
  Score the rest: edge size (capped) + favorite-side trust + line movement
                  toward/against the model − longshot haircut − CONCACAF
                  inflation haircut ± realized CLV by category (the desk's own
                  track record vs FINAL closing lines, once n ≥ 8 — queue #1)
  Size          : ¼-Kelly capped at 5% bankroll; HALVED on coin-flip matches
                  (3-way entropy > 1.5 bits)
  Verdict       : score ≥ 6 BET · ≥ 3 LEAN · else PASS (reason given)

Inputs:  data/processed/value_bets.csv, value_bets_futures.csv, line_movement.csv
Outputs: data/processed/desk_calls.csv (matches + futures, with verdicts + why)

Run: python -m src.models.desk_call [--bankroll 1000]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"

BETS_PATH     = DATA_PROC / "value_bets.csv"
FUTURES_PATH  = DATA_PROC / "value_bets_futures.csv"
MOVEMENT_PATH = DATA_PROC / "line_movement.csv"
CLV_PATH      = DATA_PROC / "clv_report.csv"
OUT_PATH      = DATA_PROC / "desk_calls.csv"

BET_SCORE, LEAN_SCORE = 6.0, 3.0
ENTROPY_COINFLIP = 1.5            # bits; max 3-way entropy = 1.585
CONCACAF = {"Mexico", "USA", "Canada", "Panama", "Haiti", "Curacao", "Curaçao"}
CLV_MIN_N, CLV_THRESH, CLV_POINTS = 8, 2.0, 1.5   # confidence input gates


def clv_confidence() -> dict:
    """category → (score_adj, evidence line) from the desk's own realized CLV.

    Uses ONLY final closes (settled matches) — pre-kickoff "closes" are just the
    latest snapshot and would feed the desk its own current line. Gated on
    n ≥ CLV_MIN_N per category and |avg CLV| ≥ CLV_THRESH so early noise can't
    swing verdicts. This is queue #1's feedback loop: if our dog calls beat the
    close, the DL-10 disagreement was edge — trust it more; if the close
    steamrolls them, the market knew better — haircut harder.
    """
    if not CLV_PATH.exists():
        return {}
    rep = pd.read_csv(CLV_PATH)
    if "close_is_final" not in rep.columns:
        return {}
    rep = rep[rep["close_is_final"] == True]  # noqa: E712 — CSV bools
    rep["clv_pct"] = pd.to_numeric(rep["clv_pct"], errors="coerce")
    rep = rep[rep["clv_pct"].notna()]
    adj = {}
    for cat, g in rep.groupby("category"):
        if len(g) < CLV_MIN_N:
            continue
        avg = g["clv_pct"].mean()
        if avg >= CLV_THRESH:
            adj[cat] = (CLV_POINTS,
                        f"desk's {cat} calls are beating final closes "
                        f"({avg:+.1f}% avg CLV, n={len(g)}) — market confirms this lane")
        elif avg <= -CLV_THRESH:
            adj[cat] = (-CLV_POINTS,
                        f"desk's {cat} calls are losing to final closes "
                        f"({avg:+.1f}% avg CLV, n={len(g)}) — market keeps beating us here")
    return adj


def bet_category(market_implied: float, outcome: str) -> str:
    """Same buckets as clv_tracker: draw | fav (fair ≥ 40%) | dog."""
    if outcome == "draw":
        return "draw"
    return "fav" if market_implied >= 0.40 else "dog"


def match_entropy(bets: pd.DataFrame) -> dict:
    """3-way entropy (bits) per match from the model's outcome probs."""
    ent = {}
    for mid, g in bets.groupby("match_id"):
        p = g.set_index("outcome")["model_prob"].reindex(
            ["home", "draw", "away"]).clip(lower=1e-9).values
        ent[mid] = float(-(p * np.log2(p)).sum())
    return ent


def call_match_bet(b, move_row, entropy, clv_adj=None):
    """One bet row → (verdict, score, size_down, why[], cautions[])."""
    why, cautions = [], []
    if b["ev"] <= 0:
        return "PASS", 0.0, False, ["negative EV — the price is better than the model"], []
    if b["outcome"] == "draw":
        return "PASS", 0.0, False, [
            "draw bet — model upweights draws 1.75× by design; this 'edge' is model bias"], []

    score = min(b["edge"] * 100, 10.0) * 0.6
    why.append(f"model {b['model_prob']:.0%} vs market fair {b['market_implied']:.0%} "
               f"= +{b['edge']*100:.1f}pp edge, EV {b['ev']:+.2f}/$1 at {b['decimal_odds']:.2f}")

    if b["market_implied"] >= 0.40:
        score += 2.0
        why.append("favorite side — the zone where the model is most trustworthy")
    elif b["market_implied"] < 0.15:
        score -= 3.0
        cautions.append("longshot — ELO compression inflates underdog probs (model bias)")

    if move_row is not None and bool(move_row["significant"]):
        if move_row["direction_vs_model"] == "toward":
            score += 2.0
            why.append(f"line moved TOWARD model since open "
                       f"({move_row['open_decimal']:.2f}→{move_row['now_decimal']:.2f}) — "
                       f"sharp money on our side of the number")
        else:
            score -= 2.0
            cautions.append(f"line moved AGAINST model since open "
                            f"({move_row['open_decimal']:.2f}→{move_row['now_decimal']:.2f}) — "
                            f"the market is hardening the other way")

    size_down = entropy > ENTROPY_COINFLIP
    if size_down:
        cautions.append(f"coin-flip match (entropy {entropy:.2f} bits) — stake halved")

    if b["selection"] in CONCACAF:
        score -= 2.0
        cautions.append("CONCACAF selection — model inflation documented (Mexico ~+6pp); "
                        "edge partly model error")

    if clv_adj:
        cat = bet_category(b["market_implied"], b["outcome"])
        if cat in clv_adj:
            pts, line = clv_adj[cat]
            score += pts
            (why if pts > 0 else cautions).append(line)

    verdict = "BET" if score >= BET_SCORE else "LEAN" if score >= LEAN_SCORE else "PASS"
    if verdict == "PASS":
        why = [why[0]] if why else []
        cautions.append("signal too weak after bias haircuts")
    return verdict, round(score, 2), size_down, why, cautions


def call_future(f):
    why, cautions = [], []
    if bool(f["tail_risk"]):
        return "PASS", 0.0, False, [
            f"model {f['model_prob']:.1%} < 2% = under ~200 of 10k sims — Monte Carlo "
            f"tail noise at long odds, not edge"], []
    if f["ev"] <= 0:
        return "PASS", 0.0, False, ["negative EV — market price is fair or better"], []

    score = min(f["edge"] * 100, 10.0) * 0.8
    why.append(f"model {f['model_prob']:.1%} vs market fair {f['market_implied']:.1%} "
               f"= +{f['edge']*100:.1f}pp edge, EV {f['ev']:+.2f}/$1 at {f['american_odds']}")
    if f["selection"] in CONCACAF:
        score -= 3.0
        cautions.append("CONCACAF — documented model inflation; treat most of this edge "
                        "as model error")
    if f["model_prob"] < 0.05:
        score -= 1.0
        cautions.append("sub-5% winner prob — thin sim support, keep stake token-sized")

    verdict = "BET" if score >= BET_SCORE else "LEAN" if score >= LEAN_SCORE else "PASS"
    if verdict == "PASS":
        cautions.append("signal too weak after bias haircuts")
    return verdict, round(score, 2), False, why, cautions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bankroll", type=float, default=1000.0)
    args = ap.parse_args()

    print("═" * 64)
    print("  V6 — DESK CALLS  (rule-based; every haircut is a documented bias)")
    print("═" * 64)

    bets = pd.read_csv(BETS_PATH)
    bets = bets[bets["market_source"] == "real"]
    futures = pd.read_csv(FUTURES_PATH)
    move = pd.read_csv(MOVEMENT_PATH)
    move_idx = {(r["match_id"], r["outcome"]): r for _, r in move.iterrows()}
    entropy = match_entropy(bets)
    clv_adj = clv_confidence()
    if clv_adj:
        print("\n  CLV feedback active (realized track record vs final closes):")
        for cat, (pts, line) in clv_adj.items():
            print(f"    {cat}: {pts:+.1f} pts — {line}")
    else:
        print("\n  CLV feedback: dormant (needs ≥ 8 settled bets per category)")

    rows = []
    for _, b in bets.iterrows():
        mr = move_idx.get((b["match_id"], b["outcome"]))
        verdict, score, size_down, why, cautions = call_match_bet(
            b, mr, entropy.get(b["match_id"], 0.0), clv_adj)
        stake = b["recommended_stake_usd"] * (0.5 if size_down else 1.0)
        rows.append({
            "kind": "match", "match_id": int(b["match_id"]), "outcome": b["outcome"],
            "date": b["date"], "label": b["match"],
            "selection": f"{b['selection']} ({b['outcome']})",
            "verdict": verdict, "score": score,
            "model_prob": b["model_prob"], "market_implied": b["market_implied"],
            "edge": b["edge"], "ev": b["ev"], "decimal_odds": b["decimal_odds"],
            "stake_usd": round(stake if verdict != "PASS" else 0.0, 2),
            "why": " | ".join(why), "cautions": " | ".join(cautions),
        })
    for _, f in futures.iterrows():
        verdict, score, _, why, cautions = call_future(f)
        rows.append({
            "kind": "futures", "match_id": "", "outcome": "",
            "date": "", "label": "Tournament winner",
            "selection": f["selection"], "verdict": verdict, "score": score,
            "model_prob": f["model_prob"], "market_implied": f["market_implied"],
            "edge": f["edge"], "ev": f["ev"], "decimal_odds": f["decimal_odds"],
            "stake_usd": round(f["recommended_stake_usd"] if verdict != "PASS" else 0.0, 2),
            "why": " | ".join(why), "cautions": " | ".join(cautions),
        })

    out = pd.DataFrame(rows).sort_values(
        ["verdict", "score"], ascending=[True, False])  # BET < LEAN < PASS alphabetically

    # Portfolio concentration cap: Kelly sizes bets in isolation, but 20+
    # simultaneous group-stage positions compound — cap total book exposure
    # at 25% of bankroll, scaling every stake proportionally. stake_raw_usd
    # keeps the pre-cap number so the dashboard can re-apply the same cap
    # after recomputing stakes on live lines.
    out["stake_raw_usd"] = out["stake_usd"]
    cap_total = 0.25 * args.bankroll
    raw_total = out.loc[out["verdict"] != "PASS", "stake_usd"].sum()
    scaled = raw_total > cap_total
    if scaled:
        out["stake_usd"] = (out["stake_usd"] * cap_total / raw_total).round(2)
    out.to_csv(OUT_PATH, index=False)

    picks = out[out["verdict"] != "PASS"]
    total = picks["stake_usd"].sum()
    print(f"\n  {len(picks[picks['verdict']=='BET'])} BET / "
          f"{len(picks[picks['verdict']=='LEAN'])} LEAN / "
          f"{len(out) - len(picks)} PASS — total stake ${total:.0f} "
          f"({total/args.bankroll:.1%} of ${args.bankroll:,.0f} bankroll)")
    if scaled:
        print(f"  Stakes scaled ×{cap_total/raw_total:.2f}: raw Kelly book was "
              f"${raw_total:.0f} ({raw_total/args.bankroll:.0%}) — capped at 25% "
              f"portfolio exposure")
    for _, r in picks.iterrows():
        print(f"\n  [{r['verdict']}] {r['selection']} — {r['label']}"
              f"{(' · ' + str(r['date'])) if r['date'] else ''}  →  ${r['stake_usd']:.0f}")
        for w in str(r["why"]).split(" | "):
            if w:
                print(f"     + {w}")
        for c in str(r["cautions"]).split(" | "):
            if c:
                print(f"     ! {c}")

    print(f"\n✅ Saved: {OUT_PATH}  ({len(out)} rows)")
    print("\n  ⚠️  Standing caveat: 62% model, edge unproven out-of-sample (V5 DL-05).")
    print("  Desk calls are research conclusions drawn from the data above — not advice.")


if __name__ == "__main__":
    main()
