#!/usr/bin/env python3
"""
Understat xG Scraper — P7-1 Top5 (E0/SP1/D1/I1/F1)
Fetches team rolling xG/xGA for last 10 matches per team.
Free, no key, 1 req/2s, cache to kbet/data/understat_cache/*.json

Usage:
  python kbet/engine/scrapers/understat_scraper.py --season 2025
  python kbet/engine/scrapers/understat_scraper.py --team Chelsea --season 2025

Storage: kbet/data/team_context.db -> team_xg_form (team, date, xg_for, xg_against, goals_for, goals_against)
"""
from __future__ import annotations

import json
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import requests
import pandas as pd

log = logging.getLogger("understat")

BASE = "https://understat.com"
LEAGUE_MAP = {
    "E0":  ("EPL", "Premier League"),
    "SP1": ("La Liga", "La Liga"),
    "D1":  ("Bundesliga", "Bundesliga"),
    "I1":  ("Serie A", "Serie A"),
    "F1":  ("Ligue 1", "Ligue 1"),
}
CACHE_DIR = Path(__file__).parent.parent / "data" / "understat_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(__file__).parent.parent / "data" / "team_context.db"

HEADERS = {"User-Agent": "Mozilla/5.0 KBet/1.0"}

def _cache_path(league: str, season: str) -> Path:
    return CACHE_DIR / f"{league}_{season}.html"

def fetch_league_page(league: str, season: str, use_cache: bool = True) -> Optional[str]:
    """Fetch Understat league page HTML with cache."""
    p = _cache_path(league, season)
    if use_cache and p.exists() and (time.time() - p.stat().st_mtime) < 86400:
        return p.read_text(encoding="utf-8")
    under = LEAGUE_MAP[league][0]
    url = f"{BASE}/league/{under}/{season}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            log.warning(f"Understat {league} {season} -> {r.status_code}")
            return None
        p.write_text(r.text, encoding="utf-8")
        time.sleep(2)  # respectful
        return r.text
    except Exception as e:
        log.warning(f"Understat fetch {league} {season}: {e}")
        return None

def parse_xg_json(html: str) -> List[Dict]:
    """Parse embedded JSON.parse('...') for team xG data."""
    # Understat embeds data as: JSON.parse('...')
    # Extract all such payloads
    matches = re.findall(r"JSON\.parse\('(.+?)'\)", html)
    out = []
    for m in matches:
        try:
            decoded = m.encode().decode("unicode_escape")
            data = json.loads(decoded)
            # Look for team history structure: list of dicts with xG
            if isinstance(data, dict) and "history" in data:
                out.append(data)
            elif isinstance(data, list) and data and isinstance(data[0], dict) and "xG" in str(data[0]):
                out.extend(data)
        except Exception:
            continue
    return out

def scrape_league(league: str, season: str = "2025") -> pd.DataFrame:
    """Scrape one league and return DataFrame team, date, xg_for, xg_against."""
    html = fetch_league_page(league, season)
    if not html:
        return pd.DataFrame()
    # Extract team blocks: simpler — parse dates table
    # Fallback: extract from teamsData
    m = re.search(r"teamsData\s*=\s*JSON\.parse\('(.+?)'\)", html)
    if not m:
        log.warning(f"No teamsData for {league} {season}")
        return pd.DataFrame()
    try:
        raw = json.loads(m.group(1).encode().decode("unicode_escape"))
        # raw is dict team_id -> {title, history: [{h_a, xG, xGA, date, goals, ...}]}
        rows = []
        for tid, tdata in raw.items():
            title = tdata.get("title", tid)
            for h in tdata.get("history", []):
                rows.append({
                    "league": league,
                    "team": title,
                    "date": h.get("date"),
                    "h_a": h.get("h_a"),
                    "xg_for": float(h.get("xG", 0) or 0),
                    "xg_against": float(h.get("xGA", 0) or 0),
                    "goals_for": int(h.get("scored", 0) or 0),
                    "goals_against": int(h.get("missed", 0) or 0),
                    "result": h.get("result"),
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        log.warning(f"Parse {league} {season}: {e}")
        return pd.DataFrame()

def save_to_db(df: pd.DataFrame):
    """Append to team_context.db -> team_xg_form."""
    if df.empty:
        return
    import sqlite3
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS team_xg_form (
            league TEXT, team TEXT, date TEXT,
            h_a TEXT, xg_for REAL, xg_against REAL,
            goals_for INTEGER, goals_against INTEGER, result TEXT,
            PRIMARY KEY (team, date)
        )""")
        df.to_sql("team_xg_form", con, if_exists="replace", index=False)
        log.info(f"Saved {len(df)} xG rows to {DB_PATH}")

def rolling_xg(team: str, as_of: str, n: int = 10) -> Optional[Dict]:
    """Get rolling xG last n for team as_of date."""
    import sqlite3
    if not DB_PATH.exists():
        return None
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql("SELECT * FROM team_xg_form WHERE team=? AND date < ? ORDER BY date DESC LIMIT ?", con, params=(team, as_of, n))
    if df.empty:
        return None
    return {"xg_for_avg": df["xg_for"].mean(), "xg_against_avg": df["xg_against"].mean(), "n": len(df)}

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Understat xG scraper Top5")
    p.add_argument("--season", default="2025", help="Understat season e.g. 2025")
    p.add_argument("--league", default=None, help="E0/SP1/D1/I1/F1 or all")
    p.add_argument("--team", default=None, help="Filter team e.g. Chelsea")
    args = p.parse_args()
    leagues = [args.league] if args.league else list(LEAGUE_MAP.keys())
    for lg in leagues:
        df = scrape_league(lg, args.season)
        print(f"{lg} {args.season}: {len(df)} rows")
        if args.team:
            df = df[df["team"].str.contains(args.team, case=False, na=False)]
            print(df.head(10).to_string())
        save_to_db(df)
