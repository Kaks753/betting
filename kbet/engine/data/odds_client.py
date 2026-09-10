"""
The Odds API client — Week 2.
Provides:
  - Live odds for today's matches (pre-closing, genuine spread)
  - Historical odds snapshots for backtest (if historical endpoint available on plan)
  - Quota-aware: tracks requests remaining and raises OddsAPIQuotaError

API docs: https://the-odds-api.com/liveapi/guides/v4/
Free tier: 500 requests/month. Each sport query = 1 request per market per bookmaker page.

Usage (live):
    client = OddsAPIClient(api_key="YOUR_KEY")
    games = client.get_today_odds(leagues=["soccer_epl","soccer_spain_la_liga"])
    for g in games:
        print(g.home, g.away, g.h2h, g.totals)

Usage (backtest simulation — from cached snapshots):
    client = OddsAPIClient(api_key=None)  # read-only from cache
    games = client.load_cached_snapshot(date="2024-09-14")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# League mapping: football-data.co.uk code → The Odds API sport key
# ---------------------------------------------------------------------------
LEAGUE_TO_SPORT_KEY: Dict[str, str] = {
    "E0":  "soccer_epl",
    "E1":  "soccer_england_league1",
    "SP1": "soccer_spain_la_liga",
    "D1":  "soccer_germany_bundesliga",
    "I1":  "soccer_italy_serie_a",
    "F1":  "soccer_france_ligue_one",
    "N1":  "soccer_netherlands_eredivisie",
    "P1":  "soccer_portugal_primeira_liga",
    "B1":  "soccer_belgium_first_div",
    "G1":  "soccer_greece_super_league",
}

ALL_SPORT_KEYS = list(LEAGUE_TO_SPORT_KEY.values())

# Target bookmakers in preference order (Pinnacle first = sharpest reference)
BOOKMAKERS = ["pinnacle", "betfair_ex_eu", "bet365", "unibet", "williamhill"]

BASE_URL = "https://api.the-odds-api.com/v4"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OddsH2H:
    """Home/Draw/Away moneyline odds from a single bookmaker."""
    bookmaker: str
    home: float
    draw: float
    away: float
    last_update: str = ""


@dataclass
class OddsBTTS:
    """Both Teams To Score odds from a single bookmaker."""
    bookmaker: str
    yes: float
    no: float
    last_update: str = ""


@dataclass
class OddsTotals:
    """Over/Under total goals odds from a single bookmaker."""
    bookmaker: str
    line: float          # e.g. 2.5
    over: float
    under: float
    last_update: str = ""


@dataclass
class MatchOdds:
    """All odds data for a single match."""
    match_id: str
    sport_key: str
    league_code: str          # football-data.co.uk code (E0, SP1, …)
    home_team: str
    away_team: str
    commence_time: str        # ISO-8601 UTC
    h2h: List[OddsH2H] = field(default_factory=list)
    totals: List[OddsTotals] = field(default_factory=list)
    btts: List[OddsBTTS] = field(default_factory=list)
    snapshot_time: str = ""   # When this snapshot was taken

    # --- Derived helpers ------------------------------------------------

    def best_h2h(self, prefer: str = "pinnacle") -> Optional[OddsH2H]:
        """Return the sharpest available H2H odds (Pinnacle > others)."""
        by_book = {o.bookmaker: o for o in self.h2h}
        for bk in [prefer] + BOOKMAKERS:
            if bk in by_book:
                return by_book[bk]
        return self.h2h[0] if self.h2h else None

    def max_h2h(self) -> Optional[OddsH2H]:
        """Return the best available price per outcome (market max)."""
        if not self.h2h:
            return None
        best_home = max(self.h2h, key=lambda o: o.home)
        best_draw = max(self.h2h, key=lambda o: o.draw)
        best_away = max(self.h2h, key=lambda o: o.away)
        return OddsH2H(
            bookmaker="MAX",
            home=best_home.home,
            draw=best_draw.draw,
            away=best_away.away,
        )

    def pinnacle_h2h(self) -> Optional[OddsH2H]:
        for o in self.h2h:
            if o.bookmaker == "pinnacle":
                return o
        return None

    def best_totals(self, line: float = 2.5) -> Optional[OddsTotals]:
        candidates = [t for t in self.totals if abs(t.line - line) < 0.01]
        if not candidates:
            return None
        best_over  = max(candidates, key=lambda t: t.over)
        best_under = max(candidates, key=lambda t: t.under)
        return OddsTotals(
            bookmaker="MAX",
            line=line,
            over=best_over.over,
            under=best_under.under,
        )

    def pinnacle_totals(self, line: float = 2.5) -> Optional[OddsTotals]:
        for t in self.totals:
            if t.bookmaker == "pinnacle" and abs(t.line - line) < 0.01:
                return t
        return None

    def devig_pinnacle_h2h(self) -> Optional[Tuple[float, float, float]]:
        """Return de-vigged (home_p, draw_p, away_p) from Pinnacle H2H."""
        pin = self.pinnacle_h2h()
        if pin is None:
            return None
        inv = 1/pin.home + 1/pin.draw + 1/pin.away
        return (1/pin.home)/inv, (1/pin.draw)/inv, (1/pin.away)/inv

    def devig_pinnacle_ou(self, line: float = 2.5) -> Optional[Tuple[float, float]]:
        """Return de-vigged (over_p, under_p) from Pinnacle totals."""
        pin = self.pinnacle_totals(line)
        if pin is None:
            return None
        inv = 1/pin.over + 1/pin.under
        return (1/pin.over)/inv, (1/pin.under)/inv

    def best_btts(self) -> Optional[OddsBTTS]:
        if not self.btts:
            return None
        best_yes = max(self.btts, key=lambda o: o.yes)
        best_no = max(self.btts, key=lambda o: o.no)
        return OddsBTTS(bookmaker="MAX", yes=best_yes.yes, no=best_no.no)

    def pinnacle_btts(self) -> Optional[OddsBTTS]:
        for o in self.btts:
            if o.bookmaker == "pinnacle":
                return o
        return None

    def devig_pinnacle_btts(self) -> Optional[Tuple[float, float]]:
        pin = self.pinnacle_btts()
        if pin is None:
            return None
        inv = 1/pin.yes + 1/pin.no
        return (1/pin.yes)/inv, (1/pin.no)/inv


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class OddsAPIError(Exception):
    pass

class OddsAPIQuotaError(OddsAPIError):
    """Raised when the monthly API quota is exhausted."""
    pass

class OddsAPIKeyMissingError(OddsAPIError):
    pass


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OddsAPIClient:
    """
    Thin wrapper around The Odds API v4.

    Parameters
    ----------
    api_key : str or None
        Your Odds API key. If None, only cached snapshot loading works.
    cache_dir : str
        Directory for caching API responses to disk (prevents re-fetching).
    rate_limit_s : float
        Minimum seconds between API requests (default 0.5 = 2 req/s).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = None,
        rate_limit_s: float = 0.5,
    ):
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        if cache_dir is None:
            base = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent.parent / "data")))
            cache_dir = str(base / "odds_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_s = rate_limit_s
        self._last_request_time = 0.0
        self._requests_remaining: Optional[int] = None
        self._requests_used: Optional[int] = None
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_today_odds(
        self,
        leagues: Optional[List[str]] = None,
        markets: List[str] = None,
        regions: str = "eu",
        use_cache: bool = True,
    ) -> List[MatchOdds]:
        """
        Fetch live odds for today's matches across specified leagues.

        Parameters
        ----------
        leagues : list of football-data.co.uk league codes, e.g. ["E0","SP1"]
                  If None, fetches all 10 supported leagues.
        markets : ["h2h", "totals", "btts"] by default (btts needs paid plan, falls back gracefully)
        regions : bookmaker regions (eu recommended for Pinnacle coverage)
        use_cache : serve from disk cache if < 15 minutes old

        Returns
        -------
        list of MatchOdds, sorted by commence_time ascending
        """
        if not self.api_key:
            raise OddsAPIKeyMissingError(
                "ODDS_API_KEY not set. Export it or pass api_key= to OddsAPIClient()."
            )
        if markets is None:
            markets = ["h2h", "totals", "btts"]

        sport_keys = [
            LEAGUE_TO_SPORT_KEY[lc]
            for lc in (leagues or list(LEAGUE_TO_SPORT_KEY.keys()))
            if lc in LEAGUE_TO_SPORT_KEY
        ]

        all_matches: List[MatchOdds] = []
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        for sport_key in sport_keys:
            cache_key = f"live_{sport_key}_{today_str}"
            cached = self._load_cache(cache_key, max_age_minutes=15) if use_cache else None

            if cached is not None:
                raw = cached
            else:
                raw = self._fetch_odds(sport_key, markets=markets, regions=regions)
                if raw is not None:
                    self._save_cache(cache_key, raw)

            if raw:
                league_code = self._sport_key_to_league(sport_key)
                matches = self._parse_odds_response(raw, sport_key, league_code)
                all_matches.extend(matches)

        # Sort by kick-off time
        all_matches.sort(key=lambda m: m.commence_time)
        logger.info(f"Fetched {len(all_matches)} matches from The Odds API")
        return all_matches

    def get_historical_snapshot(
        self,
        date: str,
        leagues: Optional[List[str]] = None,
        markets: List[str] = None,
        regions: str = "eu",
        use_cache: bool = True,
    ) -> List[MatchOdds]:
        """
        Fetch historical odds snapshot for a given date (requires paid plan).

        date : "YYYY-MM-DD" — snapshot is taken at T-24h before matches on that date.
        """
        if not self.api_key:
            raise OddsAPIKeyMissingError("API key required for historical snapshots.")
        if markets is None:
            markets = ["h2h", "totals", "btts"]

        sport_keys = [
            LEAGUE_TO_SPORT_KEY[lc]
            for lc in (leagues or list(LEAGUE_TO_SPORT_KEY.keys()))
            if lc in LEAGUE_TO_SPORT_KEY
        ]

        # Historical: snapshot at 12:00 UTC on the date (24-36h pre-closing typically)
        snapshot_iso = f"{date}T12:00:00Z"
        all_matches: List[MatchOdds] = []

        for sport_key in sport_keys:
            cache_key = f"hist_{sport_key}_{date}"
            cached = self._load_cache(cache_key, max_age_minutes=999999) if use_cache else None

            if cached is not None:
                raw = cached
            else:
                raw = self._fetch_historical_odds(
                    sport_key, snapshot_iso, markets=markets, regions=regions
                )
                if raw is not None:
                    self._save_cache(cache_key, raw)

            if raw:
                league_code = self._sport_key_to_league(sport_key)
                matches = self._parse_odds_response(raw, sport_key, league_code,
                                                     snapshot_time=snapshot_iso)
                all_matches.extend(matches)

        all_matches.sort(key=lambda m: m.commence_time)
        return all_matches

    def load_cached_snapshot(self, date: str) -> List[MatchOdds]:
        """Load previously cached odds for a given date (no API call)."""
        sport_keys = list(LEAGUE_TO_SPORT_KEY.values())
        all_matches: List[MatchOdds] = []
        for sport_key in sport_keys:
            cache_key = f"hist_{sport_key}_{date}"
            raw = self._load_cache(cache_key, max_age_minutes=999999)
            if raw:
                league_code = self._sport_key_to_league(sport_key)
                matches = self._parse_odds_response(raw, sport_key, league_code)
                all_matches.extend(matches)
        return all_matches

    def get_quota_status(self) -> Dict:
        """Return remaining/used API request counts from last call headers."""
        return {
            "requests_remaining": self._requests_remaining,
            "requests_used": self._requests_used,
        }

    # -----------------------------------------------------------------------
    # Private — HTTP
    # -----------------------------------------------------------------------

    def _fetch_odds(self, sport_key: str, markets: List[str], regions: str) -> Optional[List]:
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
            "bookmakers": ",".join(BOOKMAKERS),
        }
        url = f"{BASE_URL}/sports/{sport_key}/odds"
        return self._get(url, params)

    def _fetch_historical_odds(
        self, sport_key: str, snapshot_iso: str, markets: List[str], regions: str
    ) -> Optional[List]:
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
            "bookmakers": ",".join(BOOKMAKERS),
            "date": snapshot_iso,
        }
        url = f"{BASE_URL}/sports/{sport_key}/odds-history"
        return self._get(url, params)

    def _get(self, url: str, params: Dict) -> Optional[List]:
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_s:
            time.sleep(self.rate_limit_s - elapsed)
        self._last_request_time = time.time()

        try:
            r = self.session.get(url, params=params, timeout=15)
        except requests.RequestException as e:
            logger.error(f"Odds API request failed: {e}")
            return None

        # Capture quota headers
        if "x-requests-remaining" in r.headers:
            self._requests_remaining = int(r.headers["x-requests-remaining"])
        if "x-requests-used" in r.headers:
            self._requests_used = int(r.headers["x-requests-used"])

        if r.status_code == 401:
            raise OddsAPIKeyMissingError("Invalid or missing Odds API key.")
        if r.status_code == 429:
            raise OddsAPIQuotaError("Odds API monthly quota exhausted.")
        if r.status_code == 422:
            logger.warning(f"No odds available for {url.split('/')[-2]}")
            return []
        if r.status_code != 200:
            logger.warning(f"Odds API returned {r.status_code} for {url}")
            return None

        return r.json()

    # -----------------------------------------------------------------------
    # Private — Parse
    # -----------------------------------------------------------------------

    def _parse_odds_response(
        self,
        raw: List,
        sport_key: str,
        league_code: str,
        snapshot_time: str = "",
    ) -> List[MatchOdds]:
        matches = []
        if not snapshot_time:
            snapshot_time = datetime.now(timezone.utc).isoformat()

        for event in raw:
            m = MatchOdds(
                match_id=event.get("id", ""),
                sport_key=sport_key,
                league_code=league_code,
                home_team=event.get("home_team", ""),
                away_team=event.get("away_team", ""),
                commence_time=event.get("commence_time", ""),
                snapshot_time=snapshot_time,
            )
            for bk_data in event.get("bookmakers", []):
                bk_name = bk_data.get("key", "")
                last_upd = bk_data.get("last_update", "")
                for mkt in bk_data.get("markets", []):
                    outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                    if mkt["key"] == "h2h":
                        h = outcomes.get(event["home_team"])
                        d = outcomes.get("Draw")
                        a = outcomes.get(event["away_team"])
                        if h and d and a:
                            m.h2h.append(OddsH2H(bk_name, h, d, a, last_upd))
                    elif mkt["key"] == "totals":
                        over_price = under_price = None
                        line = 2.5
                        for o in mkt.get("outcomes", []):
                            if o["name"] == "Over":
                                over_price = o["price"]
                                line = o.get("point", 2.5)
                            elif o["name"] == "Under":
                                under_price = o["price"]
                        if over_price and under_price:
                            m.totals.append(
                                OddsTotals(bk_name, line, over_price, under_price, last_upd)
                            )
                    elif mkt["key"] == "btts":
                        o_map = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                        y = o_map.get("Yes")
                        n = o_map.get("No")
                        if y and n:
                            m.btts.append(OddsBTTS(bk_name, y, n, last_upd))
            matches.append(m)
        return matches

    def _sport_key_to_league(self, sport_key: str) -> str:
        inv = {v: k for k, v in LEAGUE_TO_SPORT_KEY.items()}
        return inv.get(sport_key, "??")

    # -----------------------------------------------------------------------
    # Private — Cache
    # -----------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str, max_age_minutes: int = 15) -> Optional[List]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        age_minutes = (time.time() - p.stat().st_mtime) / 60
        if age_minutes > max_age_minutes:
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, key: str, data: List) -> None:
        p = self._cache_path(key)
        try:
            with open(p, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Could not save cache {p}: {e}")
