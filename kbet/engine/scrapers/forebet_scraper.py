#!/usr/bin/env python3
"""
Forebet Consensus Scraper — P8 Contrarian Divergence ±0.12
Fetches Forebet math predictions for Top5, stores consensus, used as contrarian signal (not direct weight).

Why contrarian: Forebet ≈ bookmaker consensus. Direct averaging kills edge.
Divergence = model - forebet
  >+0.12 → model finds value market misses → confidence boost
  <-0.12 → market knows something → reduce confidence / flag review

Usage:
  python kbet/engine/scrapers/forebet_scraper.py --date 2026-09-12
  python kbet/engine/scrapers/forebet_scraper.py --team Chelsea

Storage: kbet/data/team_context.db -> consensus_predictions
"""
from __future__ import annotations

import re
import time
import logging
from pathlib import Path
from typing import Dict, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd

log = logging.getLogger("forebet")

BASE = "https://www.forebet.com"
CACHE_DIR = Path(__file__).parent.parent / "data" / "forebet_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(__file__).parent.parent / "data" / "team_context.db"

HEADERS = {"User-Agent": "Mozilla/5.0 KBet/1.0"}

DIVERGENCE_THRESHOLD = 0.12
CONTRARIAN_BOOST = 0.30  # EV boost 30% when model finds value market misses (O/U only cautious 0.35)

def fetch_forebet_page(date: str = None) -> Optional[str]:
    """Fetch Forebet predictions page. No key, 1 req/3s."""
    url = f"{BASE}/en/football-predictions/"
    if date:
        url = f"{BASE}/en/football-predictions/predictions-1x2/{date}"
    cache = CACHE_DIR / f"forebet_{date or 'today'}.html"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 3600:
        return cache.read_text(encoding="utf-8")
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            log.warning(f"Forebet {r.status_code}")
            return None
        cache.write_text(r.text, encoding="utf-8")
        time.sleep(3)
        return r.text
    except Exception as e:
        log.warning(f"Forebet fetch: {e}")
        return None

def parse_forebet(html: str) -> pd.DataFrame:
    """Parse Forebet prediction cards: team vs team, 1/2/3 prob, score, over/under."""
    if not html:
        return pd.DataFrame()
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    # Forebet structure: .rcnt .predictions -> .tr_0/.tr_1 rows
    for tr in soup.select("tr.tr_0, tr.tr_1"):
        try:
            teams = tr.select_one("td.prediction a")
            if not teams:
                continue
            txt = teams.get_text(strip=True)
            # "Chelsea - Crystal Palace"
            if " - " not in txt:
                continue
            home, away = [s.strip() for s in txt.split(" - ", 1)]
            # Probabilities: spans with % or data
            probs = tr.select("td.predictions span, td.fprc span")
            # Fallback: extract numbers
            nums = re.findall(r"(\d+)%", tr.get_text())
            h = d = a = None
            if len(nums) >= 3:
                h, d, a = map(lambda x: int(x)/100, nums[:3])
            rows.append({"home": home, "away": away, "forebet_h": h, "forebet_d": d, "forebet_a": a})
        except Exception:
            continue
    return pd.DataFrame(rows)

def get_consensus(home: str, away: str, date: str = None) -> Optional[Dict]:
    """Get Forebet consensus for a match. Returns None if not found."""
    df = parse_forebet(fetch_forebet_page(date))
    if df.empty:
        return None
    # Fuzzy match
    def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
    nh, na = norm(home), norm(away)
    for _, r in df.iterrows():
        rh, ra = norm(r["home"]), norm(r["away"])
        if (nh[:6] in rh or rh[:6] in nh) and (na[:6] in ra or ra[:6] in na):
            return {"h": r["forebet_h"], "d": r["forebet_d"], "a": r["forebet_a"]}
    return None

def divergence_signal(model_prob: float, forebet_prob: Optional[float]) -> str:
    """Contrarian signal."""
    if forebet_prob is None:
        return "NO_CONSENSUS"
    div = model_prob - forebet_prob
    if div > DIVERGENCE_THRESHOLD:
        return "MODEL_FINDS_VALUE"
    if div < -DIVERGENCE_THRESHOLD:
        return "MARKET_KNOWS_SOMETHING"
    return "CONSENSUS"

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Forebet scraper")
    p.add_argument("--date", default=None, help="YYYY-MM-DD")
    p.add_argument("--team", default=None, help="Filter team")
    args = p.parse_args()
    html = fetch_forebet_page(args.date)
    df = parse_forebet(html or "")
    print(f"Forebet {args.date or 'today'}: {len(df)} rows")
    if args.team:
        df = df[df["home"].str.contains(args.team, case=False, na=False) | df["away"].str.contains(args.team, case=False, na=False)]
    print(df.head(10).to_string(index=False))
