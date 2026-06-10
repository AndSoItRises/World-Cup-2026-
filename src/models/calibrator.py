"""
V6 — Adopted probability calibrator (see calibrate_v6.py for the validation).

log-pool + draw-shrink: blend XGB/LGBM/DC geometrically (p ∝ Π p_k^w_k) instead
of arithmetically, then shrink p(draw) by κ and renormalize. Validated under the
two-fold held-out protocol: mean ΔLL +0.0049 vs the v4 linear blend (test LL
0.8461 → 0.8405), and emitted draw rate drops to match the realized rate —
removing the 1.75× draw-upweight inflation that was polluting Vegas-layer edges.

Trade-off, eyes open: draw RECALL falls ~9% → ~1% (argmax rarely lands on draw).
Draw classification was already structurally capped; the probabilities are what
the betting layer prices off, and they are strictly better calibrated.

Behavior is gated on models/calibrator_v6.json: if the file is absent, callers
fall back to the v4 linear blend — delete/rename the file to revert the whole
pipeline. v1–v4 model artifacts are untouched.

All arrays use column order [p_away_win, p_draw, p_home_win].
"""

import json
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parents[2] / "models"
CALIBRATOR_PATH = MODELS / "calibrator_v6.json"
EPS = 1e-12

_cfg = None


def config():
    """Calibrator params, or None → callers should use the v4 linear blend."""
    global _cfg
    if _cfg is None and CALIBRATOR_PATH.exists():
        _cfg = json.load(open(CALIBRATOR_PATH))
    return _cfg


def pool_and_calibrate(xgb_p, lgb_p, dc_p, weights=(0.275, 0.275, 0.45)):
    """(n,3) or (3,) component probs → calibrated ensemble probs (same shape)."""
    cfg = config()
    comps = [np.atleast_2d(np.asarray(p, dtype=float)) for p in (xgb_p, lgb_p, dc_p)]
    if cfg is None:
        q = sum(w * p for w, p in zip(weights, comps))
    else:
        logp = sum(w * np.log(np.clip(p, EPS, 1))
                   for w, p in zip(cfg["component_weights"], comps))
        q = np.exp(logp - logp.max(axis=1, keepdims=True))
        if cfg.get("calibrator") == "draw_shrink":
            q[:, 1] *= cfg["params"][0]
        q = q / q.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return q[0] if np.asarray(xgb_p).ndim == 1 else q


def banner():
    cfg = config()
    return (f"calibrator v6 ACTIVE: {cfg['method']} κ={cfg['params'][0]:.3f} "
            f"(held-out ΔLL {cfg['held_out_mean_gain']:+.4f})"
            if cfg else "calibrator v6 not found — v4 linear blend")
