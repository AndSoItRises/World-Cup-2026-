"""
V4 bracket inspection (DL-08 gate): does squad value move the WC2026 forecast the
RIGHT way — deflate the schedule-inflated CONCACAF teams (Mexico/USA/Canada) and
lift the talent-rich, underrated Euro powers (France/England/Portugal/Spain…)?

Inspection only — trains v3 and v4 prod-style models in memory and compares each
team's mean single-match win probability across their WC2026 fixtures. Writes
nothing over the live prod models; saves a comparison CSV to outputs/.

Run: python -m src.models.bracket_v4
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from src.models.train_xgb import FEATURE_COLS, XGB_PARAMS, DRAW_CLASS_WEIGHT
from src.models.train_lgbm import LGBM_PARAMS
from src.models.train_v4 import FEATURE_COLS_V4, SQUAD_FEATURES
from src.features.feature_engineering import SQUAD_SENTINEL
from src.models import predict_wc2026 as P

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"
OUT = BASE / "outputs"

CONCACAF = ["Mexico", "USA", "Canada", "Panama", "Haiti", "Curaçao"]
EURO = ["France", "England", "Portugal", "Spain", "Germany", "Netherlands",
        "Croatia", "Belgium", "Switzerland", "Austria"]


def squad_lookup():
    """Latest (most recent edition) squad row per team."""
    sq = pd.read_csv(DATA_PROC / "squad_strength_by_year.csv", parse_dates=["availability_date"])
    sq = sq.sort_values("availability_date").groupby("team").tail(1).set_index("team")
    return sq


def train_prod(feats):
    """Train XGB+LGBM on ALL data with the given feature set (no early stopping)."""
    tr = pd.read_csv(DATA_PROC / "train_features.csv", parse_dates=["date"])
    te = pd.read_csv(DATA_PROC / "test_features.csv", parse_dates=["date"])
    df = pd.concat([tr, te], ignore_index=True).sort_values("date").reset_index(drop=True)
    for c in feats:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    df["neutral"] = df["neutral"].astype(int)
    X, y = df[feats], df["result"]
    w = np.ones(len(df)); w[y.values == 1] *= DRAW_CLASS_WEIGHT
    xp = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    xm = xgb.XGBClassifier(**xp); xm.fit(X, y, sample_weight=w, verbose=False)
    lm = lgb.LGBMClassifier(**LGBM_PARAMS); lm.fit(X, y, sample_weight=w)
    return xm, lm


def main():
    print("═" * 60)
    print("  V4 bracket inspection — bias correction check")
    print("═" * 60)

    fixtures, rankings, all_data = P.load_all()
    rank_lookup = P.build_rank_lookup(rankings)
    form_lookup = P.build_form_lookup(all_data)
    h2h_lookup = P.build_h2h_lookup(all_data)
    defaults = P.compute_defaults(all_data)
    sq = squad_lookup()

    with open(P.DC_PARAMS_PATH) as f:
        import json
        dc = json.load(f)

    xm3, lm3 = train_prod(FEATURE_COLS)
    xm4, lm4 = train_prod(FEATURE_COLS_V4)
    print("  trained v3 + v4 prod-style models")

    predictable = fixtures[
        ~fixtures["home_team"].str.startswith("TBD") &
        ~fixtures["away_team"].str.startswith("TBD")
    ].copy()

    def squad_fields(home, away):
        h = sq.loc[home] if home in sq.index else None
        a = sq.loc[away] if away in sq.index else None
        hs = h["squad_strength"] if h is not None else SQUAD_SENTINEL["squad_strength"]
        as_ = a["squad_strength"] if a is not None else SQUAD_SENTINEL["squad_strength"]
        hd = h["squad_depth"] if h is not None else SQUAD_SENTINEL["squad_depth"]
        ad = a["squad_depth"] if a is not None else SQUAD_SENTINEL["squad_depth"]
        return {
            "home_squad_strength": hs, "away_squad_strength": as_,
            "squad_strength_diff": hs - as_,
            "home_squad_depth": hd, "away_squad_depth": ad,
            "squad_depth_diff": hd - ad,
            "squad_both_covered": int(h is not None and a is not None),
        }

    rows3, rows4, meta = [], [], []
    for _, fr in predictable.iterrows():
        home, away = P.normalize(fr["home_team"]), P.normalize(fr["away_team"])
        base = P.build_feature_row(fr, rank_lookup, form_lookup, h2h_lookup, defaults)
        rows3.append(base)
        rows4.append({**base, **squad_fields(home, away)})
        meta.append((home, away))

    X3 = pd.DataFrame(rows3, columns=FEATURE_COLS)
    X4 = pd.DataFrame(rows4, columns=FEATURE_COLS_V4)

    def ensemble(xm, lm, X, feats):
        xp = xm.predict_proba(X)
        lp = lm.booster_.predict(X.values)
        dcp = np.array([P.dc_predict(h, a, True, dc) for h, a in meta])
        return 0.275 * xp + 0.275 * lp + 0.45 * dcp

    e3 = ensemble(xm3, lm3, X3, FEATURE_COLS)
    e4 = ensemble(xm4, lm4, X4, FEATURE_COLS_V4)

    # Per-team mean P(win) across their fixtures (side-aware)
    def team_winprob(ens):
        acc = {}
        for i, (h, a) in enumerate(meta):
            acc.setdefault(h, []).append(ens[i, 2])  # home win
            acc.setdefault(a, []).append(ens[i, 0])  # away win
        return {t: float(np.mean(v)) for t, v in acc.items()}

    w3, w4 = team_winprob(e3), team_winprob(e4)
    rows = []
    for t in sorted(set(w3) | set(w4)):
        rows.append({"team": t, "v3_winprob": w3.get(t, np.nan),
                     "v4_winprob": w4.get(t, np.nan),
                     "delta_pp": (w4.get(t, np.nan) - w3.get(t, np.nan)) * 100})
    cmp = pd.DataFrame(rows)

    def show(group, title):
        sub = cmp[cmp["team"].isin(group)].sort_values("delta_pp")
        print(f"\n── {title} (mean single-match P(win), v3 → v4) ──")
        print(f"  {'team':<16}{'v3%':>7}{'v4%':>7}{'Δpp':>8}")
        for _, r in sub.iterrows():
            print(f"  {r['team']:<16}{r['v3_winprob']*100:>6.1f}%{r['v4_winprob']*100:>6.1f}%{r['delta_pp']:>+8.2f}")
        return sub["delta_pp"].mean()

    d_con = show(CONCACAF, "CONCACAF (expect ↓ — schedule-inflated)")
    d_eur = show(EURO, "Euro powers (expect ↑ — talent-rich, underrated)")

    print(f"\n{'─'*60}")
    print(f"  Mean Δ CONCACAF: {d_con:+.2f}pp   |   Mean Δ Euro: {d_eur:+.2f}pp")
    right_way = d_con < 0 and d_eur > 0
    print(f"  Bracket moves the RIGHT way (CONCACAF↓ & Euro↑): {'YES ✓' if right_way else 'NO'}")

    OUT.mkdir(exist_ok=True)
    cmp.sort_values("delta_pp").to_csv(OUT / "bracket_v4_vs_v3.csv", index=False)
    print(f"\n✅ Saved: {OUT / 'bracket_v4_vs_v3.csv'}")


if __name__ == "__main__":
    main()
