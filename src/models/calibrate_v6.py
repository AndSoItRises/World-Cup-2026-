"""
V6 — Probability-level refinement experiments (the "is there anything left?" suite).

V5 settled that no accessible FEATURE lever improves the model. This tests a
different class: POST-HOC probability refinements. No new features, no
retraining — the v4 components stay byte-identical (non-negotiable). The hook:
the 1.75× draw upweight that buys draw recall also INFLATES p(draw) at
inference, and the Vegas layer prices edges straight off those probs.
Calibration is accuracy, for a bettor.

Candidates (all fit by max-likelihood on a fold, scored on the OTHER fold —
the honest two-fold protocol from reweight_v5 / DL-04):
  temperature  p' ∝ p^(1/T)                    1 param — confidence rescale
  draw_shrink  p' ∝ p·[1, κ, 1]                1 param — undo draw inflation
  class_weight p' ∝ p·[w_a, w_d, 1]            2 params
  vector       p' ∝ p^(1/T_c)·w_c              5 params (overfit risk — folds judge)
  log_pool     p ∝ Πp_k^{w_k}, fixed weights   0 params — geometric vs linear blend
  log_pool + draw_shrink                        1 param on the better pool

Adopt bar: mean held-out ΔLL ≥ +0.003 AND positive on both folds.
If adopted → models/calibrator_v6.json (refit on full test; WC2026 is the true
out-of-sample). v1–v4 artifacts untouched.

Run: python -m src.models.calibrate_v6
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"
MODELS = BASE / "models"

ADOPT_GAIN = 0.003
WEIGHTS = np.array([0.275, 0.275, 0.45])   # xgb, lgb, dc (fixed since V3)
EPS = 1e-12


# ── Calibrator transforms (vectorized, renormalizing) ─────────────────────────
def apply_temperature(p, params):
    t = params[0]
    q = np.power(np.clip(p, EPS, 1), 1.0 / t)
    return q / q.sum(axis=1, keepdims=True)


def apply_draw_shrink(p, params):
    w = np.array([1.0, params[0], 1.0])
    q = p * w
    return q / q.sum(axis=1, keepdims=True)


def apply_class_weight(p, params):
    w = np.array([params[0], params[1], 1.0])
    q = p * w
    return q / q.sum(axis=1, keepdims=True)


def apply_vector(p, params):
    t = params[:3]
    w = np.array([params[3], params[4], 1.0])
    q = np.power(np.clip(p, EPS, 1), 1.0 / t) * w
    return q / q.sum(axis=1, keepdims=True)


METHODS = {
    "temperature":  (apply_temperature, [1.0],            [(0.3, 3.0)]),
    "draw_shrink":  (apply_draw_shrink, [1.0],            [(0.2, 2.0)]),
    "class_weight": (apply_class_weight, [1.0, 1.0],      [(0.2, 3.0)] * 2),
    "vector":       (apply_vector, [1.0, 1.0, 1.0, 1.0, 1.0],
                     [(0.3, 3.0)] * 3 + [(0.2, 3.0)] * 2),
}


def fit(method, p, y):
    f, x0, bounds = METHODS[method]
    res = minimize(lambda x: log_loss(y, f(p, x), labels=[0, 1, 2]),
                   x0=x0, bounds=bounds, method="L-BFGS-B")
    return res.x


def log_pool(components, w=WEIGHTS):
    """Geometric blend: p ∝ Π p_k^w_k. Sharper than the linear pool where
    components agree, softer where they disagree."""
    logp = sum(wk * np.log(np.clip(pk, EPS, 1)) for wk, pk in zip(w, components))
    q = np.exp(logp - logp.max(axis=1, keepdims=True))
    return q / q.sum(axis=1, keepdims=True)


# ── Component regeneration (v4 models, byte-identical, read-only) ─────────────
def regen_components(test):
    import lightgbm as lgb
    import xgboost as xgb
    from src.models.ensemble import get_dc_proba, dc_predict_match  # noqa: F401

    feats = json.load(open(MODELS / "training_report_v4.json"))["features"]
    # replicate train_v4.py's NaN handling exactly: fill with TRAIN medians
    train = pd.read_csv(DATA_PROC / "train_features.csv")
    test = test.copy()
    for c in feats:
        if c in train.columns and train[c].isna().any() or test[c].isna().any():
            test[c] = test[c].fillna(train[c].median())
    Xte = test[feats]
    xm = xgb.XGBClassifier()
    xm.load_model(str(MODELS / "xgb_v4.json"))
    p_xgb = xm.predict_proba(Xte)
    p_lgb = lgb.Booster(model_file=str(MODELS / "lgbm_v4.txt")).predict(Xte.values)

    dc = json.load(open(MODELS / "dixon_coles_params_v3.json"))
    p_dc = np.array([dc_predict_match(r["home_team"], r["away_team"],
                                      bool(r["neutral"]), dc)
                     for _, r in test.iterrows()])
    return p_xgb, p_lgb, p_dc


# ── Evaluation ────────────────────────────────────────────────────────────────
def describe(p, y):
    pred = p.argmax(axis=1)
    return (log_loss(y, p, labels=[0, 1, 2]), accuracy_score(y, pred),
            float(((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())))


def two_fold(name, p, y, halves, base_ll_by_fold):
    """Fit on one chronological half, score the other; return mean held-out gain."""
    gains, params_used = [], []
    for fit_idx, eval_idx in (halves, halves[::-1]):
        params = fit(name, p[fit_idx], y[fit_idx])
        f = METHODS[name][0]
        ll = log_loss(y[eval_idx], f(p[eval_idx], params), labels=[0, 1, 2])
        gains.append(base_ll_by_fold[tuple(eval_idx[:3])] - ll)
        params_used.append(np.round(params, 3).tolist())
    return gains, params_used


def main():
    print("═" * 70)
    print("  V6 — Probability-Level Refinement Suite (validate-or-cut)")
    print("═" * 70)

    test = pd.read_csv(DATA_PROC / "test_features.csv", parse_dates=["date"])
    y = test["result"].astype(int).values
    p_ens = np.load(MODELS / "ensemble_test_proba_v4.npy")
    assert len(test) == len(p_ens), "proba/test misalignment"

    order = np.argsort(test["date"].values, kind="stable")
    mid = len(order) // 2
    h1, h2 = order[:mid], order[mid:]
    halves = (h1, h2)

    base_ll, base_acc, base_dr = describe(p_ens, y)
    base_by_fold = {tuple(h[:3]): log_loss(y[h], p_ens[h], labels=[0, 1, 2])
                    for h in halves}
    print(f"\n  Test: {len(test)} matches ({test['date'].min():%Y-%m}…"
          f"{test['date'].max():%Y-%m}), folds {len(h1)}/{len(h2)}")
    print(f"  Baseline v4 linear ensemble: LL {base_ll:.4f} | acc {base_acc:.4f} "
          f"| draw recall {base_dr:.3f}")
    print(f"  Mean p(draw) emitted: {p_ens[:, 1].mean():.3f} vs realized draw rate "
          f"{(y == 1).mean():.3f}  ← the draw inflation, quantified")

    results = {}

    # ── fitted calibrators on the linear-pool ensemble ──
    print(f"\n  {'method':<26}{'foldA→B':>10}{'foldB→A':>10}{'mean ΔLL':>10}   params")
    print(f"  {'-' * 78}")
    for name in METHODS:
        gains, params = two_fold(name, p_ens, y, halves, base_by_fold)
        mean_g = float(np.mean(gains))
        results[name] = {"gains": gains, "mean": mean_g, "params": params}
        print(f"  {name:<26}{gains[0]:>+10.4f}{gains[1]:>+10.4f}{mean_g:>+10.4f}   "
              f"{params[0]}")

    # ── log-pool (no fitted params → honest on full test directly) ──
    print(f"\n  Regenerating v4 component probabilities for pooling test...")
    p_xgb, p_lgb, p_dc = regen_components(test)
    lin_check = WEIGHTS[0] * p_xgb + WEIGHTS[1] * p_lgb + WEIGHTS[2] * p_dc
    drift = np.abs(lin_check - p_ens).max()
    # ≤1e-3 = float/library-version tolerance (verdicts stable to ±0.0001 across
    # reruns); larger means a data-prep mismatch — fix before trusting pooling rows
    print(f"  Reconstruction check vs saved npy: max |Δ| = {drift:.2e} "
          f"{'OK (within tolerance)' if drift < 1e-3 else '⚠️ MISMATCH — investigate before trusting'}")

    p_logpool = log_pool([p_xgb, p_lgb, p_dc])
    lp_ll, lp_acc, lp_dr = describe(p_logpool, y)
    results["log_pool"] = {"mean": base_ll - lp_ll, "gains": None,
                           "params": "fixed 0.275/0.275/0.45"}
    print(f"  {'log_pool (0 params)':<26}{'—':>10}{'—':>10}{base_ll - lp_ll:>+10.4f}   "
          f"full-test, no fitting")

    # ── best fitted method stacked on log-pool ──
    lp_by_fold = {tuple(h[:3]): log_loss(y[h], p_logpool[h], labels=[0, 1, 2])
                  for h in halves}
    for name in ("draw_shrink", "temperature"):
        gains, params = two_fold(name, p_logpool, y, halves, lp_by_fold)
        mean_total = float(np.mean(gains)) + (base_ll - lp_ll)
        results[f"log_pool+{name}"] = {"gains": gains, "mean": mean_total,
                                       "params": params}
        print(f"  {'log_pool+' + name:<26}{gains[0]:>+10.4f}{gains[1]:>+10.4f}"
              f"{mean_total:>+10.4f}   vs linear baseline; {params[0]}")

    # ── verdicts ──
    print(f"\n── Verdicts (adopt bar: mean ΔLL ≥ +{ADOPT_GAIN} and both folds +) ──")
    best_name, best = None, None
    for name, r in results.items():
        both_pos = r["gains"] is None or all(g > 0 for g in r["gains"])
        ok = r["mean"] >= ADOPT_GAIN and both_pos
        print(f"  {name:<26}{'✅ PASSES' if ok else '✂️ cut':<12} "
              f"mean {r['mean']:+.4f}")
        if ok and (best is None or r["mean"] > best["mean"]):
            best_name, best = name, r

    if best is None:
        print(f"\n  Verdict: NOTHING passes — the ceiling holds at the probability "
              f"level too. No calibrator saved.")
        return

    # refit winner on full test → deploy params (WC2026 = true out-of-sample)
    pool = "log" if best_name.startswith("log_pool") else "linear"
    fitted = best_name.split("+")[-1] if "+" in best_name else (
        best_name if best_name != "log_pool" else None)
    base_p = p_logpool if pool == "log" else p_ens
    final_params = fit(fitted, base_p, y).tolist() if fitted else []
    f = METHODS[fitted][0] if fitted else (lambda p, _: p)
    new_ll, new_acc, new_dr = describe(f(base_p, final_params), y)

    out = {
        "method": best_name, "pool": pool, "calibrator": fitted,
        "params": [round(x, 4) for x in final_params],
        "component_weights": WEIGHTS.tolist(),
        "held_out_mean_gain": round(best["mean"], 4),
        "fold_gains": best["gains"],
        "test_ll_before": round(base_ll, 4), "test_ll_after": round(new_ll, 4),
        "test_acc_before": round(base_acc, 4), "test_acc_after": round(new_acc, 4),
        "draw_recall_before": round(base_dr, 4), "draw_recall_after": round(new_dr, 4),
        "mean_p_draw_after": round(float(f(base_p, final_params)[:, 1].mean()), 4),
        "note": "applies on top of untouched v4 components at inference",
    }
    with open(MODELS / "calibrator_v6.json", "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\n  ADOPTED: {best_name}  (held-out mean ΔLL {best['mean']:+.4f})")
    print(f"  Full-test refit: LL {base_ll:.4f}→{new_ll:.4f} | acc {base_acc:.4f}→"
          f"{new_acc:.4f} | draw recall {base_dr:.3f}→{new_dr:.3f}")
    print(f"✅ Saved: {MODELS / 'calibrator_v6.json'}")


if __name__ == "__main__":
    main()
