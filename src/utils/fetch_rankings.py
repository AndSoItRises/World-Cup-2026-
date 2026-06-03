"""
fetch_rankings.py
Scrapes current FIFA rankings from football-ranking.com
Saves to data/raw/current_fifa_rankings.csv

Usage:
    python src/utils/fetch_rankings.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

BASE_URL = "https://football-ranking.com/fifa-rankings"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
OUTPUT_PATH = os.path.join("data", "raw", "current_fifa_rankings.csv")

def scrape_rankings_page(page: int = 1) -> list[dict]:
    """Scrape a single page of rankings. Returns list of dicts."""
    url = BASE_URL if page == 1 else f"{BASE_URL}?page={page}"
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    rows = []
    for tr in table.find_all("tr")[1:]:  # skip header row
        cols = tr.find_all("td")
        if len(cols) < 4:
            continue
        rank_text = cols[0].get_text(strip=True)
        team_text = cols[1].get_text(strip=True)
        points_text = cols[2].get_text(strip=True)

        # Skip separator rows (empty rank)
        if not rank_text or not team_text:
            continue

        # Clean rank (remove movement arrows like ↑1)
        rank_clean = ''.join(filter(str.isdigit, rank_text))
        if not rank_clean:
            continue

        # Extract team name and code e.g. "France (FRA)" -> name=France, code=FRA
        if "(" in team_text and ")" in team_text:
            name = team_text[:team_text.rfind("(")].strip()
            code = team_text[team_text.rfind("(")+1:team_text.rfind(")")]
        else:
            name = team_text.strip()
            code = ""

        # Clean points (remove commas, bold markers)
        # Strip commas, then take only the number before any parenthesis
        points_clean = points_text.replace(",", "").split("(")[0].strip() if points_text else ""

        rows.append({
            "rank": int(rank_clean),
            "team_name": name,
            "team_code": code,
            "points": float(points_clean) if points_clean else None,
        })

    return rows


def fetch_all_rankings(total_pages: int = 5) -> pd.DataFrame:
    """Scrape all pages and return a combined DataFrame."""
    all_rows = []
    for page in range(1, total_pages + 1):
        print(f"Scraping page {page}/{total_pages}...")
        rows = scrape_rankings_page(page)
        all_rows.extend(rows)
        time.sleep(1)  # be polite to the server

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["rank"]).sort_values("rank").reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Fetching current FIFA rankings...")
    df = fetch_all_rankings(total_pages=5)
    print(f"Fetched {len(df)} teams")
    print(df.head(10).to_string())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
