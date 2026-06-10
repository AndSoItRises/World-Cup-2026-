"""
V6 — Underdog-tilt experiments (stage 2, stacked on the adopted DL-09 calibrator).

DL-09 fixed the draw inflation; the residual question is ELO compression: in
lopsided matches the model appears to give underdogs too much win probability
(diagnosed vs the MARKET). This script asks the test set directly:

  Diagnostic — bucket matches by the model's favorite-win prob and compare
  predicted vs REALIZED favorite win rate. If realized > predicted in strong-
  favorite buckets, the compression is real (not just market disagreement).

Candidates (fit on one chronological half of test, scored on the other, swap;
all applied ON TOP of log_pool + draw_shrink, i.e. the current live pipeline):
  dog_shrink   p'_dog ∝ κ·p_dog (the weaker side's WIN prob)        1 param
  fav_logit    logit-remap of p_fav: σ(α + β·logit(p_fav)),
               draw+dog rescaled proportionally                      2 params
  cond_temp    temperature linear in lopsidedness:
               T(s) = a + b·(s − ⅓), s = max(p)                      2 params

Adopt bar (same as DL-09): mean held-out ΔLL ≥ +0.003 AND both folds positive.
If adopted → "stage2" block in models/calibrator_v6.json (calibrator.py applies
it after stage 1; delete the block to revert).

Run: python -m src.models.calibrate_tilt
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from src.models.calibrate_v6 import regen_components, log_pool

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"
MODELS = BASE / "models"

ADOPT_GAIN = 0.003
EPS = 1e-12
# columns: [away_win, draw, home_win]
AWAY, DRAW, HOME = 0, 1, 2


# ── stage-2 transforms ────────────────────────────────────────────────────────
def _dog_fav_masks(p):
    """Boolean masks for which WIN column (away/home) is the dog vs favorite."""
    home_fav = p[:, HOME] >= p[:, AWAY]
    fav_col = np.where(home_fav, HOME, AWAY)
    dog_col = np.where(home_fav, AWAY, HOME)
    return fav_col, dog_col


def apply_dog_shrink(p, params):
    k = params[0]
    fav_col, dog_col = _dog_fav_masks(p)
    q = p.copy()
    q[np.arange(len(q)), dog_col] *= k
    return q / q.sum(axis=1, keepdims=True)


def apply_fav_logit(p, params):
    a, b = params
    fav_col, _ = _dog_fav_masks(p)
    idx = np.arange(len(p))
    pf = np.clip(p[idx, fav_col], EPS, 1 - EPS)
    pf2 = 1.0 / (1.0 + np.exp(-(a + b * np.log(pf / (1 - pf)))))
    q = p.copy()
    scale = (1.0 - pf2) / np.clip(1.0 - pf, EPS, 1)
    q *= scale[:, None]
    q[idx, fav_col] = pf2
    return q / q.sum(axis=1, keepdims=True)


def apply_cond_temp(p, params):
    a, b = params
    s = p.max(axis=1)
    t = np.clip(a + b * (s - 1.0 / 3.0), 0.2, 5.0)
    q = np.power(np.clip(p, EPS, 1), 1.0 / t[:, None])
    return q / q.sum(axis=1, keepdims=True)


METHODS = {
    "dog_shrink": (apply_dog_shrink, [1.0], [(0.3, 2.0)]),
    "fav_logit":  (apply_fav_logit, [0.0, 1.0], [(-2.0, 2.0), (0.3, 3.0)]),
    "cond_temp":  (apply_cond_temp, [1.0, 0.0], [(0.3, 3.0), (-2.0, 2.0)]),
}


def fit(name, p, y):
    f, x0, bounds = METHODS[name]
    return minimize(lambda x: log_loss(y, f(p, x), labels=[0, 1, 2]),
                    x0=x0, bounds=bounds, method="L-BFGS-B").x


# ── current live pipeline probs (stage 1 applied) ─────────────────────────────
def stage1_probs(test):
    cfg = json.load(open(MODELS / "calibrator_v6.json"))
    p_xgb, p_lgb, p_dc = regen_components(test)
    p = log_pool([p_xgb, p_lgb, p_dc], np.array(cfg["component_weights"]))
    p[:, DRAW] *= cfg["params"][0]
    return p / p.sum(axis=1, keepdims=True)


def reliability(p, y, edges=(0.34, 0.45, 0.55, 0.65, 0.75, 1.01)):
    """Favorite-strength buckets: predicted vs realized favorite win rate."""
    fav_col, _ = _dog_fav_masks(p)
    idx = np.arange(len(p))
    pf = p[idx, fav_col]
    won = (y == fav_col).astype(float)
    rows, lo = [], 0.34
    print(f"\n  {'fav prob bucket':<18}{'n':>6}{'predicted':>11}{'realized':>10}{'gap':>8}")
    print(f"  {'-' * 55}")
    prev = 0.0
    for hi in edges:
        m = (pf >= prev) & (pf < hi)
        if m.sum() >= 30:
            gap = won[m].mean() - pf[m].mean()
            print(f"  {f'{prev:.2f}–{hi:.2f}':<18}{int(m.sum()):>6}"
                  f"{pf[m].mean():>10.1%}{won[m].mean():>10.1%}{gap:>+8.1%}")
            rows.append(gap)
        prev = hi
    return rows


def main():
    print("═" * 70)
    print("  V6 — Underdog Tilt: diagnose, then validate-or-cut (stage 2)")
    print("═" * 70)

    test = pd.read_csv(DATA_PROC / "test_features.csv", parse_dates=["date"])
    y = test["result"].astype(int).values
    p0 = stage1_probs(test)
    base_ll = log_loss(y, p0, labels=[0, 1, 2])

    print(f"\n  Baseline = live pipeline (log_pool + draw_shrink): LL {base_ll:.4f}")
    print(f"\n── Diagnostic: is the compression real vs REALIZED outcomes? ──")
    print(f"  (realized > predicted in high buckets ⇒ favorites underpriced by model)")
    reliability(p0, y)

    order = np.argsort(test["date"].values, kind="stable")
    mid = len(order) // 2
    halves = (order[:mid], order[mid:])
    base_by_fold = {tuple(h[:3]): log_loss(y[h], p0[h], labels=[0, 1, 2])
                    for h in halves}

    print(f"\n  {'method':<14}{'foldA→B':>10}{'foldB→A':>10}{'mean ΔLL':>10}   params")
    print(f"  {'-' * 66}")
    results = {}
    for name in METHODS:
        gains, fitted = [], []
        for fi, ei in (halves, halves[::-1]):
            params = fit(name, p0[fi], y[fi])
            f = METHODS[name][0]
            ll = log_loss(y[ei], f(p0[ei], params), labels=[0, 1, 2])
            gains.append(base_by_fold[tuple(ei[:3])] - ll)
            fitted.append(np.round(params, 3).tolist())
        mean_g = float(np.mean(gains))
        results[name] = {"gains": gains, "mean": mean_g, "fitted": fitted}
        print(f"  {name:<14}{gains[0]:>+10.4f}{gains[1]:>+10.4f}{mean_g:>+10.4f}   "
              f"{fitted[0]} / {fitted[1]}")

    print(f"\n── Verdicts (bar: mean ≥ +{ADOPT_GAIN}, both folds positive) ──")
    best_name, best = None, None
    for name, r in results.items():
        ok = r["mean"] >= ADOPT_GAIN and all(g > 0 for g in r["gains"])
        print(f"  {name:<14}{'✅ PASSES' if ok else '✂️ cut':<12} mean {r['mean']:+.4f}")
        if ok and (best is None or r["mean"] > best["mean"]):
            best_name, best = name, r

    if best is None:
        print(f"\n  Verdict: CUT — no stage-2 transform clears the bar on realized")
        print(f"  outcomes. The dog tilt vs the MARKET is a disagreement, not a")
        print(f"  proven miscalibration; the desk-call haircuts remain the handling.")
        return

    final_params = fit(best_name, p0, y).tolist()
    f = METHODS[best_name][0]
    new_ll = log_loss(y, f(p0, final_params), labels=[0, 1, 2])
    cfg = json.load(open(MODELS / "calibrator_v6.json"))
    cfg["stage2"] = {"method": best_name,
                     "params": [round(x, 4) for x in final_params],
                     "held_out_mean_gain": round(best["mean"], 4),
                     "test_ll_after_stage2": round(new_ll, 4)}
    json.dump(cfg, open(MODELS / "calibrator_v6.json", "w"), indent=2)
    print(f"\n  ADOPTED stage 2: {best_name} {final_params} "
          f"(held-out {best['mean']:+.4f}; full-test LL {base_ll:.4f}→{new_ll:.4f})")
    print(f"✅ Updated: {MODELS / 'calibrator_v6.json'}")
    print(f"\n── Post-fix reliability ──")
    reliability(f(p0, final_params), y)


if __name__ == "__main__":
    main()
