"""
V3 P1: Team resolution audit.
Verifies every WC2026 fixture team resolves to a real FIFA rank and ELO — i.e.
no team silently falls back to a sentinel (rank 80/150) or default ELO (1500).
Catches the name-mismatch bugs that gave Iran rank 150 and DR Congo default form.

Run with:
  python -m src.features.verify_teams
"""

import pandas as pd
from pathlib import Path

from src.features.data_cleaning import standardize_name
from src.models.predict_wc2026 import NAME_MAP, build_rank_lookup

BASE      = Path(__file__).resolve().parents[2]
DATA_RAW  = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"


def normalize(name):
    return NAME_MAP.get(name, name)


def main():
    print("═" * 64)
    print("  V3 P1: WC2026 Team Resolution Audit")
    print("═" * 64)

    fixtures = pd.read_csv(DATA_RAW / "wc2026_fixtures.csv")
    rankings = pd.read_csv(DATA_RAW / "current_fifa_rankings.csv")
    train    = pd.read_csv(DATA_PROC / "train_features.csv")
    test     = pd.read_csv(DATA_PROC / "test_features.csv")
    hist     = pd.concat([train, test], ignore_index=True)

    rank_lookup = build_rank_lookup(rankings)
    elo_teams   = set(hist["home_team"]) | set(hist["away_team"])

    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    teams = [t for t in teams if not str(t).startswith("TBD")]

    rank_fail, elo_fail = [], []
    print(f"\n  {'Fixture team':<24} {'→ model name':<22} {'rank':>5} {'ELO?':>5}")
    print(f"  {'-'*60}")
    for t in teams:
        m = normalize(t)
        rank = rank_lookup.get(m)
        has_elo = m in elo_teams
        rank_str = str(rank) if rank is not None else "MISS"
        elo_str  = "ok" if has_elo else "MISS"
        if rank is None: rank_fail.append(t)
        if not has_elo:  elo_fail.append(t)
        flag = "" if (rank is not None and has_elo) else "  ◄ CHECK"
        print(f"  {t:<24} {m:<22} {rank_str:>5} {elo_str:>5}{flag}")

    print(f"\n── Summary ──")
    print(f"  Teams: {len(teams)} | rank misses: {len(rank_fail)} | ELO misses: {len(elo_fail)}")
    if rank_fail: print(f"  Rank unresolved: {rank_fail}")
    if elo_fail:  print(f"  ELO unresolved:  {elo_fail}")
    if not rank_fail and not elo_fail:
        print("  ✅ All WC2026 teams resolve to a real rank and ELO.")
    else:
        print("  ❌ Unresolved teams remain — fix name maps before proceeding.")
    return len(rank_fail) + len(elo_fail)


if __name__ == "__main__":
    raise SystemExit(main())
