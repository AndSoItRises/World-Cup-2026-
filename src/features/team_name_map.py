"""
Team-name bridge: football-data.org  →  WC2026 fixtures convention.

football-data.org returns full English (sometimes localized) team names
(e.g. "United States", "Korea Republic", "Côte d'Ivoire"). The model's
fixtures / live-results files use a shorter convention (e.g. "USA",
"South Korea", "Ivory Coast"). This module maps the former to the latter.

IMPORTANT — two-layer naming:
  football-data.org name --(this module)--> fixtures name --(monte_carlo.normalize)--> internal model name

This module only does the FIRST hop. It deliberately targets the *fixtures*
convention (the names already in data/raw/wc2026_fixtures.csv and
data/raw/wc2026_live_results.csv) so the output stays compatible with
live_update.load_live(), which applies normalize() afterwards. Do not map
straight to internal names here.

Usage:
    from src.features.team_name_map import to_fixture_name
    to_fixture_name("United States")        # -> "USA"
    to_fixture_name("Korea Republic")        # -> "South Korea"
    to_fixture_name("Some New Name")         # -> fuzzy-matched fixture name (+ warning)
"""

from __future__ import annotations

import difflib
import unicodedata

# The 48 canonical fixture names (data/raw/wc2026_fixtures.csv). Single source
# of truth for fuzzy fallback; keep in sync if the fixture list ever changes.
FIXTURE_TEAMS = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde", "Colombia",
    "Croatia", "Curacao", "Czechia", "DR Congo", "Ecuador", "Egypt",
    "England", "France", "Germany", "Ghana", "Haiti", "Iran", "Iraq",
    "Ivory Coast", "Japan", "Jordan", "Mexico", "Morocco", "Netherlands",
    "New Zealand", "Norway", "Panama", "Paraguay", "Portugal", "Qatar",
    "Saudi Arabia", "Scotland", "Senegal", "South Africa", "South Korea",
    "Spain", "Sweden", "Switzerland", "Tunisia", "Turkey", "USA", "Uruguay",
    "Uzbekistan",
]

# Explicit football-data.org → fixtures-name overrides. Only names that differ
# from the fixture spelling need an entry; identical names pass through.
# Several spelling variants are included defensively (football-data.org has
# changed some labels over time, e.g. "Czech Republic" → "Czechia").
API_TO_FIXTURE = {
    "United States":            "USA",
    "USA":                      "USA",
    "Korea Republic":           "South Korea",
    "Republic of Korea":        "South Korea",
    "South Korea":              "South Korea",
    "Côte d'Ivoire":            "Ivory Coast",
    "Cote d'Ivoire":            "Ivory Coast",
    "Ivory Coast":              "Ivory Coast",
    "Bosnia and Herzegovina":   "Bosnia and Herzegovina",
    "Bosnia-Herzegovina":       "Bosnia and Herzegovina",
    "Cabo Verde":               "Cape Verde",
    "Cape Verde":               "Cape Verde",
    "Cape Verde Islands":       "Cape Verde",
    "Congo DR":                 "DR Congo",
    "DR Congo":                 "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Curaçao":                  "Curacao",
    "Curacao":                  "Curacao",
    "Czech Republic":           "Czechia",
    "Czechia":                  "Czechia",
    "Türkiye":                  "Turkey",
    "Turkiye":                  "Turkey",
    "Turkey":                   "Turkey",
    "IR Iran":                  "Iran",
    "Iran":                     "Iran",
    "Saudi Arabia":             "Saudi Arabia",
    "KSA":                      "Saudi Arabia",
}

# Build a normalized-key index for accent/whitespace-insensitive exact lookup.
def _key(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().replace("-", " ").split())

_EXACT_INDEX = {_key(k): v for k, v in API_TO_FIXTURE.items()}
_EXACT_INDEX.update({_key(t): t for t in FIXTURE_TEAMS})  # fixture names map to themselves


def to_fixture_name(api_name: str, *, warn: bool = True) -> str:
    """Map a football-data.org team name to the WC2026 fixtures convention.

    Resolution order:
      1. exact (accent/case/hyphen-insensitive) lookup in the override + fixture index
      2. fuzzy match against the 48 fixture names (cutoff 0.8)
      3. fall back to the original string (and warn) so nothing is silently dropped
    """
    if api_name is None:
        return api_name
    k = _key(api_name)
    if k in _EXACT_INDEX:
        return _EXACT_INDEX[k]

    # Fuzzy fallback against fixture names (compare on normalized keys).
    fixture_keys = {_key(t): t for t in FIXTURE_TEAMS}
    match = difflib.get_close_matches(k, list(fixture_keys), n=1, cutoff=0.8)
    if match:
        resolved = fixture_keys[match[0]]
        if warn:
            print(f"  [team_name_map] fuzzy: '{api_name}' -> '{resolved}'")
        return resolved

    if warn:
        print(f"  [team_name_map] ⚠️  no match for '{api_name}' — passing through unchanged")
    return api_name


if __name__ == "__main__":
    # Quick self-check.
    samples = [
        "United States", "Korea Republic", "Côte d'Ivoire", "Cabo Verde",
        "Congo DR", "Curaçao", "Czech Republic", "Türkiye", "IR Iran",
        "Brazil", "Bosnia and Herzegovina", "Mexico",
    ]
    print("football-data.org name            -> fixture name")
    print("-" * 52)
    for s in samples:
        print(f"  {s:<32} -> {to_fixture_name(s, warn=False)}")
