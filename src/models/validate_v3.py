"""
V3 P5: V3-vs-V2 validation.
Compares the V2 and V3 ensemble test probabilities head-to-head on the four
acceptance metrics. V3 is accepted only if it beats V2 on > 2 of 4:
  1. Test log loss (overall)         — lower better
  2. Macro confederation log loss    — lower better (balanced across confeds)
  3. WC2022 finals log loss          — lower better (out-of-sample tournament)
  4. Draw recall                     — higher better

Run with (after the V3 ensemble has been generated):
  python -m src.models.validate_v3
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from sklearn.metrics import log_loss, classification_report

from src.features.data_cleaning import standardize_name

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"
DATA_RAW  = BASE / "data" / "raw"
MODELS = BASE / "models"
RANK_DIR = DATA_RAW / "FIFA_rankings_training"


def conf_map():
    frames = []
    for f in sorted(RANK_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        if {"country_full", "confederation"}.issubset(df.columns):
            frames.append(df[["country_full", "confederation"]])
    allc = pd.concat(frames, ignore_index=True).dropna()
    allc["country_full"] = allc["country_full"].apply(standardize_name)
    return allc.groupby("country_full")["confederation"].agg(lambda s: s.mode().iloc[0]).to_dict()


def draw_recall(p, y):
    rep = classification_report(y, np.argmax(p, axis=1),
                                target_names=["a", "d", "h"], output_dict=True, zero_division=0)
    return rep["d"]["recall"]


def macro_conf_ll(p, y, conf):
    lls = []
    for c in ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC"]:
        m = (conf == c).values
        if m.sum() >= 10:
            lls.append(log_loss(y[m], p[m], labels=[0, 1, 2]))
    return float(np.mean(lls))


def metrics(p, y, conf, wc_mask):
    return {
        "test_ll":      round(log_loss(y, p, labels=[0, 1, 2]), 4),
        "macro_conf_ll": round(macro_conf_ll(p, y, conf), 4),
        "wc2022_ll":    round(log_loss(y[wc_mask], p[wc_mask], labels=[0, 1, 2]), 4) if wc_mask.sum() >= 10 else None,
        "draw_recall":  round(draw_recall(p, y), 4),
    }


def main():
    print("═" * 60)
    print("  V3 P5: V3 vs V2 Validation")
    print("═" * 60)

    test = pd.read_csv(TEST := DATA_PROC / "test_features.csv", parse_dates=["date"]).reset_index(drop=True)
    y = test["result"].astype(int).values
    cm = conf_map()
    conf = test["home_team"].map(cm).fillna("OTHER")
    wc_mask = test["tournament"].astype(str).str.fullmatch("FIFA World Cup").fillna(False).values
    print(f"\n  Test matches: {len(y)} | WC2022 finals: {wc_mask.sum()}")

    v2 = np.load(MODELS / "ensemble_test_proba_v2.npy")
    v3 = np.load(MODELS / "ensemble_test_proba_v3.npy")

    m2 = metrics(v2, y, conf, wc_mask)
    m3 = metrics(v3, y, conf, wc_mask)

    # direction: True = lower is better
    lower_better = {"test_ll": True, "macro_conf_ll": True, "wc2022_ll": True, "draw_recall": False}
    print(f"\n  {'metric':<16} {'V2':>10} {'V3':>10} {'winner':>8}")
    print(f"  {'-'*46}")
    v3_wins = 0
    for k in ["test_ll", "macro_conf_ll", "wc2022_ll", "draw_recall"]:
        a, b = m2[k], m3[k]
        if a is None or b is None:
            print(f"  {k:<16} {str(a):>10} {str(b):>10} {'n/a':>8}")
            continue
        win = (b < a) if lower_better[k] else (b > a)
        if win:
            v3_wins += 1
        print(f"  {k:<16} {a:>10.4f} {b:>10.4f} {'V3' if win else 'V2':>8}")

    accept = v3_wins > 2
    print(f"\n  V3 wins {v3_wins}/4 metrics → {'ACCEPT V3' if accept else 'KEEP V2 (V3 not better)'}")

    out = {"v2": m2, "v3": m3, "v3_wins": v3_wins, "accept_v3": accept}
    with open(MODELS / "validation_v3_vs_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved: {MODELS / 'validation_v3_vs_v2.json'}")


if __name__ == "__main__":
    main()
