"""
Phase 3: Data Cleaning Pipeline
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[2]
RAW  = ROOT / "data" / "raw"
OUT  = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ── STEP 1: Competitive match filter ──────────────────────────────────────────

EXCLUDE_KEYWORDS = [
    "Friendly", "Island Games", "Merdeka", "King's Cup",
    "Nordic Championship", "British Home Championship",
    "CECAFA Cup", "AFF Championship", "Asian Games",
    "Gulf Cup", "CFU Caribbean Cup", "COSAFA Cup",
    "Kirin Cup", "Confederations Cup",
]

def is_competitive(tournament: str) -> bool:
    t = str(tournament)
    for ex in EXCLUDE_KEYWORDS:
        if ex.lower() in t.lower():
            return False
    competitive_terms = [
        "world cup", "euro", "copa america", "african cup",
        "asian cup", "gold cup", "nations league",
        "qualification", "qualifier",
    ]
    return any(term in t.lower() for term in competitive_terms)

def load_results() -> pd.DataFrame:
    df = pd.read_csv(RAW / "international_results" / "results.csv", parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[df["tournament"].apply(is_competitive)].copy()
    df = df[df["date"] >= "2002-01-01"].copy()
    return df.reset_index(drop=True)

# ── STEP 2: Team name standardization ─────────────────────────────────────────

TEAM_NAME_MAP = {
    "South Korea":                      "Korea Republic",
    "North Korea":                      "Korea DPR",
    "DPR Korea":                        "Korea DPR",
    "Czech Republic":                   "Czechia",
    "DR Congo":                         "Congo DR",
    "Republic of Ireland":              "Ireland",
    "IR Iran":                          "Iran",
    "Cape Verde":                       "Cape Verde Islands",
    "Ivory Coast":                      "Côte d'Ivoire",
    "Cote d'Ivoire":                    "Côte d'Ivoire",
    "United States":                    "USA",
    "Brunei":                           "Brunei Darussalam",
    "Kyrgyzstan":                       "Kyrgyz Republic",
    "Swaziland":                        "Eswatini",
    "Republic of Kosovo":               "Kosovo",
    "Curacao":                          "Curaçao",
    "Bosnia and Herzegovina":           "Bosnia-Herzegovina",
    "Bosnia & Herzegovina":             "Bosnia-Herzegovina",
    "Saint Kitts and Nevis":            "St. Kitts and Nevis",
    "Saint Lucia":                      "St. Lucia",
    "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",
}

def standardize_name(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)

# ── STEP 3: Temporal train/test split ─────────────────────────────────────────

TRAIN_CUTOFF = pd.Timestamp("2022-11-20")

# ── STEP 4: Exponential decay weighting ───────────────────────────────────────

DECAY_HALF_LIFE_DAYS = 730

def compute_decay_weights(dates: pd.Series, reference_date: pd.Timestamp) -> pd.Series:
    lam = np.log(2) / DECAY_HALF_LIFE_DAYS
    days_ago = (reference_date - dates).dt.days.clip(lower=0)
    weights = np.exp(-lam * days_ago)
    weights = weights / weights.mean()
    return weights

# ── STEP 5: Categorical encoding ──────────────────────────────────────────────

def assign_tournament_tier(tournament: str) -> int:
    t = tournament.lower()
    if "fifa world cup" in t and "qualif" not in t:
        return 4
    if any(x in t for x in ["uefa euro", "copa america", "african cup of nations",
                              "afc asian cup", "gold cup"]) and "qualif" not in t:
        return 3
    if any(x in t for x in ["qualif", "nations league"]):
        return 2
    return 1

def encode_result(home_score: float, away_score: float) -> int:
    if home_score > away_score:
        return 2
    elif home_score == away_score:
        return 1
    else:
        return 0

# ── STEP 6: Class weights ──────────────────────────────────────────────────────

def get_class_weights(results: pd.Series) -> dict:
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=results)
    return dict(zip(classes.tolist(), weights.tolist()))

# ── STEP 7: Save to disk ───────────────────────────────────────────────────────

if __name__ == "__main__":
    matches = load_results()
    matches["home_team"] = matches["home_team"].apply(standardize_name)
    matches["away_team"] = matches["away_team"].apply(standardize_name)

    matches["tournament_tier"] = matches["tournament"].apply(assign_tournament_tier)
    matches["result"] = matches.apply(
        lambda r: encode_result(r["home_score"], r["away_score"]), axis=1
    )

    train = matches[matches["date"] <  TRAIN_CUTOFF].copy()
    test  = matches[matches["date"] >= TRAIN_CUTOFF].copy()

    ref_date = train["date"].max()
    train["sample_weight"] = compute_decay_weights(train["date"], ref_date)

    class_weights = get_class_weights(train["result"])

    # Save CSVs
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv",   index=False)

    # Save metadata
    metadata = {
        "class_weights":        class_weights,
        "train_cutoff":         str(TRAIN_CUTOFF.date()),
        "decay_half_life_days": DECAY_HALF_LIFE_DAYS,
        "train_rows":           len(train),
        "test_rows":            len(test),
    }
    with open(OUT / "cleaning_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved train.csv:             {len(train):,} rows")
    print(f"Saved test.csv:              {len(test):,} rows")
    print(f"Saved cleaning_metadata.json")
    print(f"\nFiles are in: {OUT}")