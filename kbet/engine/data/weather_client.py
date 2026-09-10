"""
Open-Meteo weather client — Week 2.
Free, no API key required. Fetches precipitation and temperature
for a stadium location on match day.

Why weather matters:
  - Heavy rain (≥5mm) → fewer goals, fewer corners, more cards (slippery surface)
  - Very cold (<5°C)  → slight reduction in open play, more set pieces
  - Wind (>30 km/h)   → reduced over rate on corners/totals

Usage:
    wc = WeatherClient()
    cond = wc.get_match_conditions("Wembley", "London", 51.556, -0.280, "2024-09-14")
    print(cond.precipitation_mm, cond.temp_max_c, cond.wind_max_kmh)
    print(cond.weather_tag)   # "DRY" | "LIGHT_RAIN" | "HEAVY_RAIN" | "COLD" | "WINDY"

Stadium coordinates for all 10 leagues are pre-loaded (avg per city).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.open-meteo.com/v1"

# ---------------------------------------------------------------------------
# Stadium / city coordinates for supported leagues
# Fallback per city; real implementation resolves per home team stadium.
# ---------------------------------------------------------------------------

# league_code -> (city, lat, lon)
LEAGUE_CITY_COORDS: Dict[str, Tuple[str, float, float]] = {
    "E0":  ("London",     51.5074,  -0.1278),
    "E1":  ("Birmingham", 52.4862,  -1.8904),
    "SP1": ("Madrid",     40.4168,  -3.7038),
    "D1":  ("Berlin",     52.5200,  13.4050),
    "I1":  ("Milan",      45.4642,   9.1900),
    "F1":  ("Paris",      48.8566,   2.3522),
    "N1":  ("Amsterdam",  52.3702,   4.8952),
    "P1":  ("Lisbon",     38.7223,  -9.1393),
    "B1":  ("Brussels",   50.8503,   4.3517),
    "G1":  ("Athens",     37.9838,  23.7275),
}

# Team-level stadium coords (best-effort, major clubs only)
TEAM_STADIUM_COORDS: Dict[str, Tuple[float, float]] = {
    "Man City":        (53.4831, -2.2004),
    "Man United":      (53.4631, -2.2913),
    "Arsenal":         (51.5549, -0.1084),
    "Chelsea":         (51.4816, -0.1910),
    "Liverpool":       (53.4308, -2.9608),
    "Tottenham":       (51.6042, -0.0665),
    "Newcastle":       (54.9756, -1.6218),
    "Aston Villa":     (52.5090, -1.8846),
    "West Ham":        (51.5386, -0.0164),
    "Real Madrid":     (40.4531, -3.6883),
    "Barcelona":       (41.3809,  2.1228),
    "Atletico Madrid": (40.4361, -3.5996),
    "Bayern Munich":   (48.2188, 11.6248),
    "Borussia Dortmund": (51.4926, 7.4518),
    "Juventus":        (45.1096, 7.6413),
    "Inter":           (45.4781, 9.1240),
    "Milan":           (45.4781, 9.1240),
    "PSG":             (48.8414, 2.2530),
    "Ajax":            (52.3143, 4.9411),
    "Benfica":         (38.7519, -9.1845),
    "Porto":           (41.1612, -8.5832),
}


@dataclass
class WeatherConditions:
    """Weather forecast for a match."""
    date: str
    lat: float
    lon: float
    precipitation_mm: float   # Total daily precipitation
    temp_max_c: float
    temp_min_c: float
    wind_max_kmh: float
    weather_tag: str          # DRY | LIGHT_RAIN | HEAVY_RAIN | COLD | WINDY

    # ---------------------------------------------------------------------------
    # Adjustment factors for probability models
    # ---------------------------------------------------------------------------
    @property
    def goals_adj(self) -> float:
        """Additive adjustment to expected goals based on weather."""
        adj = 0.0
        if self.precipitation_mm >= 5:
            adj -= 0.12   # Heavy rain reduces goals
        elif self.precipitation_mm >= 2:
            adj -= 0.05   # Light rain, slight reduction
        if self.temp_max_c < 3:
            adj -= 0.05   # Very cold reduces open play
        if self.wind_max_kmh > 35:
            adj -= 0.08   # High wind disrupts play
        return adj

    @property
    def corners_adj(self) -> float:
        """Additive adjustment to expected corners."""
        adj = 0.0
        if self.precipitation_mm >= 5:
            adj -= 0.40   # Wet pitch → fewer attacking plays
        if self.wind_max_kmh > 35:
            adj += 0.30   # Wind → more ball out of play → more corners oddly
        return adj

    @property
    def cards_adj(self) -> float:
        """Additive adjustment to expected yellow cards."""
        adj = 0.0
        if self.precipitation_mm >= 5:
            adj += 0.20   # Slippery surface → more fouls → more cards
        if self.wind_max_kmh > 35:
            adj += 0.10   # Frustration factor in wind
        return adj

    @property
    def is_adverse(self) -> bool:
        """True if conditions are meaningfully adverse (reduces bet confidence)."""
        return (self.precipitation_mm >= 5 or
                self.wind_max_kmh > 35 or
                self.temp_max_c < 3)


class WeatherClient:
    """
    Fetches weather data from Open-Meteo (free, no key).

    Parameters
    ----------
    cache_dir : directory for caching weather responses (24h TTL).
    """

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            base = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent.parent / "data")))
            cache_dir = str(base / "weather_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self._last_request = 0.0

    def get_match_conditions(
        self,
        home_team: str,
        league_code: str,
        match_date: str,  # "YYYY-MM-DD"
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> Optional[WeatherConditions]:
        """
        Get weather conditions for a match.

        Coordinate resolution priority:
        1. Explicit lat/lon
        2. Team-level stadium coordinates (TEAM_STADIUM_COORDS)
        3. League-level city coordinates (LEAGUE_CITY_COORDS)

        Returns None if API call fails.
        """
        # Resolve coordinates
        if lat is None or lon is None:
            if home_team in TEAM_STADIUM_COORDS:
                lat, lon = TEAM_STADIUM_COORDS[home_team]
            elif league_code in LEAGUE_CITY_COORDS:
                _, lat, lon = LEAGUE_CITY_COORDS[league_code]
            else:
                logger.warning(f"No coordinates for {home_team} / {league_code}")
                return None

        cache_key = f"{match_date}_{lat:.2f}_{lon:.2f}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        result = self._fetch(lat, lon, match_date)
        if result is not None:
            self._save_cache(cache_key, result)
        return result

    def get_bulk_conditions(
        self,
        matches: list,  # list of (home_team, league_code, date)
    ) -> Dict[str, Optional[WeatherConditions]]:
        """
        Fetch weather for multiple matches.
        Returns dict keyed by f"{home_team}_{date}".
        """
        results = {}
        seen_coords: Dict[str, WeatherConditions] = {}

        for home_team, league_code, match_date in matches:
            key = f"{home_team}_{match_date}"
            # Deduplicate by coordinates (many teams in same city)
            if home_team in TEAM_STADIUM_COORDS:
                lat, lon = TEAM_STADIUM_COORDS[home_team]
            elif league_code in LEAGUE_CITY_COORDS:
                _, lat, lon = LEAGUE_CITY_COORDS[league_code]
            else:
                results[key] = None
                continue

            coord_key = f"{match_date}_{lat:.2f}_{lon:.2f}"
            if coord_key in seen_coords:
                results[key] = seen_coords[coord_key]
            else:
                cond = self.get_match_conditions(home_team, league_code, match_date, lat, lon)
                results[key] = cond
                if cond is not None:
                    seen_coords[coord_key] = cond

        return results

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _fetch(self, lat: float, lon: float, date: str) -> Optional[WeatherConditions]:
        """Call Open-Meteo daily forecast API."""
        elapsed = time.time() - self._last_request
        if elapsed < 0.2:
            time.sleep(0.2 - elapsed)
        self._last_request = time.time()

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,wind_speed_10m_max",
            "timezone": "Europe/London",
            "start_date": date,
            "end_date": date,
        }
        try:
            r = self.session.get(f"{BASE_URL}/forecast", params=params, timeout=10)
            if r.status_code != 200:
                logger.warning(f"Open-Meteo returned {r.status_code}")
                return None
            data = r.json()
            daily = data.get("daily", {})
            if not daily.get("time"):
                return None

            precip = daily.get("precipitation_sum", [0])[0] or 0.0
            t_max  = daily.get("temperature_2m_max",  [15])[0] or 15.0
            t_min  = daily.get("temperature_2m_min",  [10])[0] or 10.0
            wind   = daily.get("wind_speed_10m_max",  [10])[0] or 10.0

            tag = self._classify(precip, t_max, wind)
            return WeatherConditions(
                date=date, lat=lat, lon=lon,
                precipitation_mm=precip,
                temp_max_c=t_max, temp_min_c=t_min,
                wind_max_kmh=wind,
                weather_tag=tag,
            )
        except Exception as e:
            logger.warning(f"Weather fetch failed for {lat},{lon} on {date}: {e}")
            return None

    @staticmethod
    def _classify(precip: float, t_max: float, wind: float) -> str:
        if precip >= 8:
            return "HEAVY_RAIN"
        if wind > 35:
            return "WINDY"
        if t_max < 3:
            return "COLD"
        if precip >= 2:
            return "LIGHT_RAIN"
        return "DRY"

    def _cache_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe}.json"

    def _load_cache(self, key: str) -> Optional[WeatherConditions]:
        import json
        p = self._cache_path(key)
        if not p.exists():
            return None
        # Weather cache is valid for 12 hours
        age_h = (time.time() - p.stat().st_mtime) / 3600
        if age_h > 12:
            return None
        try:
            with open(p) as f:
                d = json.load(f)
            return WeatherConditions(**d)
        except Exception:
            return None

    def _save_cache(self, key: str, cond: WeatherConditions) -> None:
        import json, dataclasses
        p = self._cache_path(key)
        try:
            with open(p, "w") as f:
                json.dump(dataclasses.asdict(cond), f)
        except Exception as e:
            logger.warning(f"Could not cache weather: {e}")
