"""
V4 validation: v4 (squad-augmented) vs v3 on the same four acceptance metrics
used for V3 (see validate_v3). v4 is accepted on the metrics only if it wins > 2
of 4 — but the binding V4 decision ALSO requires the WC2026 bracket to move the
right way (DL-08 lesson: log loss alone missed a bracket regression). Bracket
check is done separately via the prod retrain + market_divergence.

  1. Test log loss (overall)        — lower better
  2. Macro confederation log loss   — lower better (the bias-balance metric)
  3. WC2022 finals log loss         — lower better (out-of-sample tournament)
  4. Draw recall                    — higher better

Run: python -m src.models.validate_v4
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, classification_report

from src.features.data_cleaning import standardize_name
from src.models.validate_v3 import conf_map, draw_recall, macro_conf_ll, metrics

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"
MODELS = BASE / "models"


def main():
    print("═" * 60)
    print("  V4 validation: v4 vs v3")
    print("═" * 60)

    test = pd.read_csv(DATA_PROC / "test_features.csv", parse_dates=["date"]).reset_index(drop=True)
    y = test["result"].astype(int).values
    cm = conf_map()
    conf = test["home_team"].map(cm).fillna("OTHER")
    wc_mask = test["tournament"].astype(str).str.fullmatch("FIFA World Cup").fillna(False).values
    print(f"\n  Test matches: {len(y)} | WC2022 finals: {wc_mask.sum()}")

    v3 = np.load(MODELS / "ensemble_test_proba_v3.npy")
    v4 = np.load(MODELS / "ensemble_test_proba_v4.npy")
    m3 = metrics(v3, y, conf, wc_mask)
    m4 = metrics(v4, y, conf, wc_mask)

    lower_better = {"test_ll": True, "macro_conf_ll": True, "wc2022_ll": True, "draw_recall": False}
    print(f"\n  {'metric':<16} {'V3':>10} {'V4':>10} {'winner':>8}")
    print(f"  {'-'*46}")
    v4_wins = 0
    for k in ["test_ll", "macro_conf_ll", "wc2022_ll", "draw_recall"]:
        a, b = m3[k], m4[k]
        if a is None or b is None:
            print(f"  {k:<16} {str(a):>10} {str(b):>10} {'n/a':>8}")
            continue
        win = (b < a) if lower_better[k] else (b > a)
        v4_wins += int(win)
        print(f"  {k:<16} {a:>10.4f} {b:>10.4f} {'V4' if win else 'V3':>8}")

    accept = v4_wins > 2
    print(f"\n  V4 wins {v4_wins}/4 metrics → {'metrics favour V4' if accept else 'metrics flat/favour V3'}")
    print(f"  NOTE: binding decision also needs the WC2026 bracket to move right (DL-08).")

    out = {"v3": m3, "v4": m4, "v4_wins": v4_wins, "accept_on_metrics": accept}
    with open(MODELS / "validation_v4_vs_v3.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved: {MODELS / 'validation_v4_vs_v3.json'}")


if __name__ == "__main__":
    main()
