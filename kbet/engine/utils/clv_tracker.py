"""
CLV Tracker — Week 3
====================
Tracks Closing Line Value per bet after match settlement.

CLV = our_entry_implied_prob - closing_implied_prob
    = 1/our_entry_odds_devigged - 1/closing_odds_devigged

Positive CLV means we were on the right side of the market.
CLV is the gold standard for assessing long-run bettor skill —
a positive CLV bettor WILL show ROI over large samples regardless
of short-term variance.

Pipeline:
    1. Bet placed at T-48h entry odds
    2. Match settles → CLV tracker fetches final closing odds
    3. CLV computed and stored in kbet.db
    4. Daily summary report shows: avg CLV, Sharpe of CLV, ROI vs CLV tracking

The Odds API usage:
    - Closing odds for settled bets: 1 request per sport per day
    - Free tier (500 req/month): budget ~15 sports-per-day calls
    - Cache 30 days before purging
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

BASE_URL      = "https://api.the-odds-api.com/v4"
_db_env = os.environ.get("DB_PATH")
if _db_env:
    DB_PATH = Path(_db_env)
else:
    _data_dir = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent.parent / "data")))
    DB_PATH = _data_dir / "kbet.db"
    # legacy path fallback
    if not DB_PATH.exists() and (Path(__file__).parent.parent.parent / "data" / "kbet.db").exists():
        DB_PATH = Path(__file__).parent.parent.parent / "data" / "kbet.db"
CACHE_DIR     = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent.parent / "data"))) / "closing_odds_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SPORT_KEYS = {
    "E0":  "soccer_epl",
    "SP1": "soccer_spain_la_liga",
    "D1":  "soccer_germany_bundesliga",
    "I1":  "soccer_italy_serie_a",
    "F1":  "soccer_france_ligue_one",
    "E1":  "soccer_england_league1",
    "N1":  "soccer_netherlands_eredivisie",
    "P1":  "soccer_portugal_primeira_liga",
    "B1":  "soccer_belgium_first_div",
    "G1":  "soccer_greece_super_league",
}


# ── De-vig helpers ────────────────────────────────────────────────────────────

def devig_1x2(h: float, d: float, a: float) -> Optional[Tuple[float, float, float]]:
    """Return de-vigged (H, D, A) probabilities or None if invalid."""
    try:
        if any(math.isnan(v) or v <= 1.0 for v in (h, d, a)):
            return None
        inv = 1/h + 1/d + 1/a
        return (1/h)/inv, (1/d)/inv, (1/a)/inv
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def devig_ou(over: float, under: float) -> Optional[Tuple[float, float]]:
    try:
        if any(math.isnan(v) or v <= 1.0 for v in (over, under)):
            return None
        inv = 1/over + 1/under
        return (1/over)/inv, (1/under)/inv
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# ── The Odds API closing-odds fetcher ─────────────────────────────────────────

class ClosingOddsFetcher:
    """
    Fetches closing odds from The Odds API for settled bets.
    Uses the historical endpoint with a timestamp just before match kickoff.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        self.requests_remaining: Optional[int] = None

    def _get_historical(self, sport_key: str, timestamp_iso: str) -> List[Dict]:
        """Fetch odds at a specific timestamp (just before kickoff)."""
        if not self.api_key:
            return []
        url = f"{BASE_URL}/sports/{sport_key}/odds-history"
        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": "h2h,totals",
            "bookmakers": "pinnacle",
            "date": timestamp_iso,
            "dateFormat": "iso",
            "oddsFormat": "decimal",
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            self.requests_remaining = int(resp.headers.get("x-requests-remaining", -1))
            if resp.status_code == 200:
                data = resp.json()
                return data.get("data", []) if isinstance(data, dict) else data
            else:
                log.warning(f"Odds API [{resp.status_code}]: {resp.text[:200]}")
                return []
        except Exception as e:
            log.warning(f"Closing odds fetch error: {e}")
            return []

    def get_closing_odds(
        self,
        league: str,
        home_team: str,
        away_team: str,
        match_date: str,  # YYYY-MM-DD
    ) -> Optional[Dict]:
        """
        Retrieve closing Pinnacle odds for a specific match.
        Returns dict with h/d/a and over/under if found, else None.
        """
        sport_key = SPORT_KEYS.get(league)
        if not sport_key:
            log.warning(f"Unknown league: {league}")
            return None

        # Closing = T-5min before scheduled kickoff (assume 15:00 UTC default)
        import pandas as pd
        match_dt = pd.Timestamp(f"{match_date}T14:55:00Z")
        timestamp_iso = match_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        events = self._get_historical(sport_key, timestamp_iso)

        for evt in events:
            if _team_match(evt.get("home_team", ""), home_team) and \
               _team_match(evt.get("away_team", ""), away_team):
                result = {"found": True, "bookmaker": "pinnacle"}
                for bm in evt.get("bookmakers", []):
                    if bm.get("key") != "pinnacle":
                        continue
                    for mkt in bm.get("markets", []):
                        if mkt["key"] == "h2h":
                            oc = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                            result["h2h_h"] = oc.get(evt["home_team"])
                            result["h2h_d"] = oc.get("Draw")
                            result["h2h_a"] = oc.get(evt["away_team"])
                        elif mkt["key"] == "totals":
                            for o in mkt.get("outcomes", []):
                                if o.get("point") == 2.5:
                                    if o["name"] == "Over":
                                        result["ou_over_2_5"] = o["price"]
                                    elif o["name"] == "Under":
                                        result["ou_under_2_5"] = o["price"]
                return result

        return None


def _team_match(api_name: str, our_name: str) -> bool:
    """Fuzzy team name matching (handles slight naming differences)."""
    import re
    def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
    a, b = norm(api_name), norm(our_name)
    return a == b or a[:6] == b[:6] or (len(a) >= 4 and a[:4] in b) or (len(b) >= 4 and b[:4] in a)


# ── CLV Calculator ────────────────────────────────────────────────────────────

def compute_clv(
    market: str,
    pick: str,
    entry_prob: float,
    closing_odds_data: Optional[Dict],
) -> Optional[float]:
    """
    Compute CLV for a settled bet.

    CLV = our_model_prob - closing_devigged_prob
    Positive = we were smarter than the closing line.

    Parameters
    ----------
    market          : "1x2", "ou"
    pick            : "H", "D", "A", "O2.5", "U2.5"
    entry_prob      : our model's probability at bet placement
    closing_odds_data : dict from ClosingOddsFetcher.get_closing_odds()

    Returns
    -------
    float CLV or None if closing odds unavailable
    """
    if closing_odds_data is None or not closing_odds_data.get("found"):
        return None

    if market == "1x2":
        h = closing_odds_data.get("h2h_h")
        d = closing_odds_data.get("h2h_d")
        a = closing_odds_data.get("h2h_a")
        if any(v is None for v in (h, d, a)):
            return None
        dv = devig_1x2(float(h), float(d), float(a))
        if dv is None:
            return None
        closing_prob = {"H": dv[0], "D": dv[1], "A": dv[2]}.get(pick)
        if closing_prob is None:
            return None
        return entry_prob - closing_prob

    elif market == "ou":
        over = closing_odds_data.get("ou_over_2_5")
        under = closing_odds_data.get("ou_under_2_5")
        if any(v is None for v in (over, under)):
            return None
        dv = devig_ou(float(over), float(under))
        if dv is None:
            return None
        closing_prob = {"O2.5": dv[0], "U2.5": dv[1]}.get(pick)
        if closing_prob is None:
            return None
        return entry_prob - closing_prob

    return None


# ── Database operations ───────────────────────────────────────────────────────

def update_bet_clv(db_path: Path, bet_id: int, clv: float) -> None:
    """Write computed CLV back to the bets table."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE bets SET clv = ? WHERE id = ?", (clv, bet_id))
        conn.commit()


def get_pending_clv_bets(db_path: Path) -> List[Dict]:
    """Return settled bets that don't yet have CLV computed."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, league, home_team, away_team, match_date,
                   market, pick, entry_prob
            FROM bets
            WHERE result IS NOT NULL
              AND clv IS NULL
              AND entry_prob IS NOT NULL
        """).fetchall()
    return [dict(r) for r in rows]


# ── CLV settlement runner ─────────────────────────────────────────────────────

def settle_clv(db_path: Optional[Path] = None) -> Dict:
    """
    Main CLV settlement job.
    - Fetches pending settled bets from DB
    - Retrieves closing odds for each
    - Updates CLV column

    Returns summary dict.
    """
    if db_path is None:
        db_path = DB_PATH

    if not db_path.exists():
        log.info("No DB found — no CLV to settle")
        return {"settled": 0, "missing": 0}

    fetcher = ClosingOddsFetcher()
    if not fetcher.api_key:
        log.warning("ODDS_API_KEY not set — CLV settlement skipped")
        return {"settled": 0, "missing": 0, "reason": "no_api_key"}

    pending = get_pending_clv_bets(db_path)
    log.info(f"CLV settlement: {len(pending)} bets pending")

    settled = 0
    missing = 0

    for bet in pending:
        closing = fetcher.get_closing_odds(
            league=bet["league"],
            home_team=bet["home_team"],
            away_team=bet["away_team"],
            match_date=bet["match_date"],
        )
        clv = compute_clv(
            market=bet["market"],
            pick=bet["pick"],
            entry_prob=bet["entry_prob"],
            closing_odds_data=closing,
        )
        if clv is not None:
            update_bet_clv(db_path, bet["id"], clv)
            settled += 1
        else:
            missing += 1
            log.debug(f"  No closing odds: {bet['home_team']} v {bet['away_team']} [{bet['market']}]")

        time.sleep(0.2)  # Polite rate limiting

    log.info(f"CLV settlement complete: {settled} settled, {missing} missing")
    return {"settled": settled, "missing": missing}


# ── CLV report ────────────────────────────────────────────────────────────────

def clv_report(db_path: Optional[Path] = None) -> Dict:
    """
    Generate a CLV performance summary from all tracked bets.
    Returns dict with avg_clv, sharpe_clv, correlation with result, etc.
    """
    import numpy as np
    if db_path is None:
        db_path = DB_PATH

    if not db_path.exists():
        return {"error": "no_db"}

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT clv, result, pnl, market, pick, book_odds
            FROM bets
            WHERE clv IS NOT NULL AND result IS NOT NULL
        """).fetchall()

    if not rows:
        return {"n_bets": 0, "avg_clv": None}

    clvs  = np.array([r["clv"]  for r in rows], dtype=float)
    pnls  = np.array([r["pnl"]  for r in rows], dtype=float)
    won   = np.array([1 if r["result"] == "W" else 0 for r in rows], dtype=float)

    avg_clv   = float(np.mean(clvs))
    std_clv   = float(np.std(clvs))
    sharpe    = float(avg_clv / std_clv) if std_clv > 0 else 0.0
    corr      = float(np.corrcoef(clvs, won)[0, 1]) if len(clvs) > 2 else 0.0
    avg_roi   = float(np.mean(pnls))

    return {
        "n_bets":        len(rows),
        "avg_clv":       round(avg_clv,  4),
        "std_clv":       round(std_clv,  4),
        "sharpe_clv":    round(sharpe,   3),
        "clv_win_corr":  round(corr,     3),
        "avg_roi":       round(avg_roi,  4),
        "positive_clv_pct": round((clvs > 0).mean(), 3),
        "by_market": {
            "1x2": {"n": int(sum(1 for r in rows if r["market"] == "1x2")),
                    "avg_clv": float(np.mean([r["clv"] for r in rows if r["market"] == "1x2"]) if any(r["market"] == "1x2" for r in rows) else 0)},
            "ou":  {"n": int(sum(1 for r in rows if r["market"] == "ou")),
                    "avg_clv": float(np.mean([r["clv"] for r in rows if r["market"] == "ou"]) if any(r["market"] == "ou" for r in rows) else 0)},
        }
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("CLV Tracker — testing devig functions")
    r1 = devig_1x2(2.10, 3.40, 3.60)
    print(f"  devig(2.10, 3.40, 3.60) = {r1}")
    r2 = devig_ou(1.90, 1.95)
    print(f"  devig_ou(1.90, 1.95) = {r2}")
    # CLV test
    clv = compute_clv("1x2", "H", 0.52, {"found": True, "h2h_h": 2.10, "h2h_d": 3.40, "h2h_a": 3.60})
    print(f"  CLV (entry=0.52, closing=devig(2.10,3.40,3.60)[H]) = {clv:.4f}")
    print("  ✅ CLV tracker OK")
