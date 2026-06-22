"""
Stake sizing for the underdog "+0.5 insurance" strategy (DL-16/DL-18).

The core idea, in plain terms
-----------------------------
When the model backs an underdog (say Senegal over Argentina) we can take TWO bets
on the same match:
  • ML  — Senegal to WIN              (pays only if Senegal wins)
  • +0.5 — Senegal to WIN OR DRAW     (pays whenever Argentina does NOT win)

These two bets share the "Senegal wins" outcome, so they are CORRELATED. Sizing each
one on its own with Kelly and adding them would double-count that shared win and
over-bet. The correct method is JOINT (simultaneous) Kelly over the three
mutually-exclusive match states — win / draw / loss — picking the two stakes
together to maximize expected log-growth of the bankroll:

    maximize  p_w·ln(1 + f1·(d1-1) + f2·(d2-1))      # underdog wins: both legs pay
            + p_d·ln(1 - f1        + f2·(d2-1))      # draw: ML loses, +0.5 pays
            + p_l·ln(1 - f1        - f2)             # favorite wins: both legs lose
       over  f1 (ML stake), f2 (+0.5 stake)

We then apply the project's standard fractional-Kelly multiplier (½) and a per-leg
cap (5%), matching how every other stake in this repo is sized.

`joint_kelly` returns bankroll FRACTIONS (f1, f2). `solo_kelly` is the single-bet
version, used to size the ML-only and +0.5-only benchmark streams for honest
attribution (does adding the +0.5 leg actually help risk-adjusted return?).
"""

from __future__ import annotations

from scipy.optimize import minimize


def solo_kelly(p_win: float, decimal: float, *, kelly_fraction: float = 0.5,
               cap: float = 0.05) -> float:
    """Single-bet fractional Kelly. f = edge*d/(d-1), where edge = p_win - 1/d.

    Matches the repo's existing convention (prompt §3). Returns a bankroll fraction
    in [0, cap]; 0 if the bet has no positive edge."""
    if decimal <= 1.0 or not (0.0 < p_win < 1.0):
        return 0.0
    edge = p_win - (1.0 / decimal)
    if edge <= 0:
        return 0.0
    f = edge * decimal / (decimal - 1.0)        # full Kelly
    f *= kelly_fraction                          # fractional
    return float(max(0.0, min(cap, f)))


def joint_kelly(p_w: float, p_d: float, p_l: float, d1: float, d2: float, *,
                include_ml: bool = True, include_dc: bool = True,
                kelly_fraction: float = 0.5, cap: float = 0.05) -> tuple[float, float]:
    """Joint Kelly for the correlated (ML, +0.5) pair on one match.

    p_w/p_d/p_l : underdog win / draw / loss probabilities (should sum ~1)
    d1          : decimal odds for the ML (underdog win)
    d2          : decimal odds for the +0.5 (underdog win-or-draw)
    include_*   : drop a leg (fix its fraction at 0) when it fails the edge gate

    Returns (f1, f2) as bankroll fractions, fractional-Kelly scaled and per-leg capped.
    """
    # Guard: a leg with non-positive payout carries no value.
    if d1 <= 1.0:
        include_ml = False
    if d2 <= 1.0:
        include_dc = False
    if not include_ml and not include_dc:
        return 0.0, 0.0

    b1, b2 = d1 - 1.0, d2 - 1.0  # net odds

    def neg_log_growth(f):
        f1, f2 = (f[0] if include_ml else 0.0), (f[1] if include_dc else 0.0)
        w_win = 1.0 + f1 * b1 + f2 * b2   # underdog wins
        w_draw = 1.0 - f1 + f2 * b2       # draw
        w_loss = 1.0 - f1 - f2            # favorite wins
        eps = 1e-9
        if min(w_win, w_draw, w_loss) <= eps:
            return 1e6  # ruin — strongly penalize
        return -(p_w * _ln(w_win) + p_d * _ln(w_draw) + p_l * _ln(w_loss))

    # Solve full Kelly with a no-ruin constraint (keep some bankroll in every state).
    x0 = [0.02, 0.02]
    bounds = [(0.0, 0.95), (0.0, 0.95)]
    cons = [{"type": "ineq", "fun": lambda f: 0.98 - (f[0] + f[1])}]  # f1+f2 <= 0.98
    res = minimize(neg_log_growth, x0, method="SLSQP", bounds=bounds, constraints=cons)

    f1_full = res.x[0] if include_ml else 0.0
    f2_full = res.x[1] if include_dc else 0.0

    # Fractional Kelly + per-leg cap.
    f1 = max(0.0, min(cap, f1_full * kelly_fraction))
    f2 = max(0.0, min(cap, f2_full * kelly_fraction))
    return float(f1), float(f2)


def _ln(x: float) -> float:
    from math import log
    return log(x)


def explain(p_w: float, p_d: float, p_l: float, d1: float, d2: float,
            f1: float, f2: float) -> str:
    """One-line plain-English rationale for a sized recommendation (for the ledger
    and the dashboard cards — keeps the strategy legible, not a black box)."""
    cover = p_w + p_d
    parts = []
    if f1 > 0:
        parts.append(f"ML {f1*100:.1f}% of bank (model {p_w*100:.0f}% win vs "
                     f"{100/d1:.0f}% priced)")
    if f2 > 0:
        parts.append(f"+0.5 {f2*100:.1f}% of bank (model {cover*100:.0f}% not-lose vs "
                     f"{100/d2:.0f}% priced)")
    if not parts:
        return "no positive-edge leg — no bet"
    return " + ".join(parts)


if __name__ == "__main__":
    # Worked example: a Senegal-over-Argentina style dog where the model likes the
    # draw too. Shows joint Kelly favoring the +0.5 leg.
    p_w, p_d, p_l = 0.33, 0.30, 0.37
    d1 = 1 / 0.27   # market prices the dog win at 27%
    d2 = 1 / 0.55   # market prices win-or-draw at 55%
    f1, f2 = joint_kelly(p_w, p_d, p_l, d1, d2)
    print(f"joint Kelly  -> ML f1={f1:.4f}  +0.5 f2={f2:.4f}")
    print(f"solo ML      -> {solo_kelly(p_w, d1):.4f}")
    print(f"solo +0.5    -> {solo_kelly(p_w + p_d, d2):.4f}")
    print(explain(p_w, p_d, p_l, d1, d2, f1, f2))
