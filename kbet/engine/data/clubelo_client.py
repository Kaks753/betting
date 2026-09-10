"""
ClubElo client — Week 2.
Provides pre-match Elo ratings for teams as priors for the Dixon-Coles model.

ClubElo CSV API: http://api.clubelo.com/{ClubName}
Format: Rank,Club,Country,Level,Elo,From,To

The Elo difference between home and away team encodes expected outcome:
  E(home win) = 1 / (1 + 10^((elo_away - elo_home - HOME_ADV) / 400))

The home advantage in Elo terms is ~65 points (from ClubElo documentation).

Usage:
    client = ClubEloClient()
    elo = client.get_elo("Arsenal", as_of="2024-08-17")
    print(elo)  # e.g. 1842.3

    probs = client.get_match_probs("Arsenal", "Chelsea", as_of="2024-08-17")
    print(probs)  # {"home": 0.45, "draw": 0.28, "away": 0.27}

Fallback: If ClubElo API is unavailable (sandbox/network restriction),
returns None gracefully. Dixon-Coles then uses its own ratings without prior.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
import pandas as pd

logger = logging.getLogger(__name__)

# ClubElo home advantage constant in Elo points
HOME_ELO_ADVANTAGE = 65.0

# Approximate Elo → 1X2 probability function (Bradley-Terry with draw zone)
# Using the Elo Soccer model (Hvattum & Arntzen 2010):
#   dr = (elo_home + HOME_ADV) - elo_away
#   P(home win) = 1 / (1 + 10^(-dr/400))
#   Draw probability modelled via empirical logistic fit on dr

CLUBELO_BASE = "http://api.clubelo.com"


class ClubEloClient:
    """
    Fetches ClubElo ratings with local caching.

    Parameters
    ----------
    cache_dir : path for CSV caches (one file per club, refreshed weekly)
    timeout   : HTTP timeout in seconds
    """

    def __init__(
        self,
        cache_dir: str = None,
        timeout: int = 12,
    ):
        if cache_dir is None:
            base = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent.parent / "data")))
            cache_dir = str(base / "elo_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "KBet/2.0 Research"})
        self._last_request = 0.0
        self._available = True   # Set False after repeated failures

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_elo(self, team_name: str, as_of: str) -> Optional[float]:
        """
        Return Elo rating for `team_name` on `as_of` date (YYYY-MM-DD).
        Returns None if team not found or API unavailable.
        """
        df = self._load_ratings(team_name)
        if df is None or df.empty:
            return None

        as_of_dt = pd.Timestamp(as_of)
        # Find row where as_of_dt falls within [From, To)
        mask = (df["From"] <= as_of_dt) & (df["To"] > as_of_dt)
        row = df[mask]
        if row.empty:
            # Use the most recent entry before as_of
            before = df[df["To"] <= as_of_dt]
            if before.empty:
                return None
            return float(before.iloc[-1]["Elo"])
        return float(row.iloc[-1]["Elo"])

    def get_match_probs(
        self,
        home_team: str,
        away_team: str,
        as_of: str,
    ) -> Optional[Dict[str, float]]:
        """
        Return 1X2 probabilities based on Elo difference.

        Returns dict {"home": p, "draw": p, "away": p} or None.
        """
        elo_h = self.get_elo(home_team, as_of)
        elo_a = self.get_elo(away_team, as_of)
        if elo_h is None or elo_a is None:
            return None
        return self._elo_to_probs(elo_h, elo_a)

    def get_elo_pair(
        self, home_team: str, away_team: str, as_of: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return (elo_home, elo_away) or (None, None)."""
        return self.get_elo(home_team, as_of), self.get_elo(away_team, as_of)

    def preload_bulk(self, teams: list, as_of: str) -> Dict[str, Optional[float]]:
        """Preload Elo ratings for many teams at once (avoids repeated API calls)."""
        result = {}
        for team in teams:
            result[team] = self.get_elo(team, as_of)
        return result

    # -----------------------------------------------------------------------
    # Elo → Probability conversion
    # -----------------------------------------------------------------------

    @staticmethod
    def _elo_to_probs(elo_home: float, elo_away: float) -> Dict[str, float]:
        """
        Convert Elo ratings to 1X2 probabilities.

        Uses the method from Hvattum & Arntzen (2010) extended with
        an empirical draw model based on Elo difference magnitude.

        Draw probability peaks when teams are equal (dr ≈ 0) and
        decreases as the skill gap widens.
        """
        dr = (elo_home + HOME_ELO_ADVANTAGE) - elo_away
        p_home_raw = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))

        # Draw probability: logistic model fitted on European football data
        # draw_p ≈ 0.285 - 0.0004 * |dr|  (peaks at ~28.5% for equal teams)
        draw_p = max(0.05, 0.285 - 0.0004 * abs(dr))

        # Redistribute remaining probability between home and away
        remaining = 1.0 - draw_p
        p_home = p_home_raw * remaining
        p_away = (1.0 - p_home_raw) * remaining

        # Renormalise
        total = p_home + draw_p + p_away
        return {
            "home": round(p_home / total, 4),
            "draw": round(draw_p / total, 4),
            "away": round(p_away / total, 4),
        }

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def _load_ratings(self, team_name: str) -> Optional[pd.DataFrame]:
        """Load ratings from cache or fetch from ClubElo API."""
        cache_path = self._cache_path(team_name)

        if cache_path.exists():
            age_days = (time.time() - cache_path.stat().st_mtime) / 86400
            if age_days < 7:
                return self._read_cache(cache_path)

        if not self._available:
            # API previously failed; try cache anyway even if stale
            if cache_path.exists():
                return self._read_cache(cache_path)
            return None

        df = self._fetch(team_name)
        if df is not None:
            self._write_cache(cache_path, df)
        elif cache_path.exists():
            # Fetch failed but stale cache exists — use it
            df = self._read_cache(cache_path)
        return df

    def _fetch(self, team_name: str) -> Optional[pd.DataFrame]:
        """Fetch CSV from ClubElo API."""
        # ClubElo uses URL-safe team slugs: spaces → %20
        slug = team_name.replace(" ", "%20")
        url = f"{CLUBELO_BASE}/{slug}"

        elapsed = time.time() - self._last_request
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_request = time.time()

        try:
            r = self.session.get(url, timeout=self.timeout)
            if r.status_code == 200 and "Rank" in r.text[:100]:
                df = pd.read_csv(io.StringIO(r.text))
                df["From"] = pd.to_datetime(df["From"])
                df["To"]   = pd.to_datetime(df["To"])
                return df
            elif r.status_code == 200:
                # Got HTML (website, not API) — endpoint blocked
                logger.warning(f"ClubElo API returned HTML for {team_name} — likely rate limited or blocked")
                self._available = False
                return None
            else:
                logger.warning(f"ClubElo returned {r.status_code} for {team_name}")
                return None
        except Exception as e:
            logger.warning(f"ClubElo fetch failed for {team_name}: {e}")
            self._available = False
            return None

    def _cache_path(self, team_name: str) -> Path:
        safe = team_name.replace(" ", "_").replace("/", "_").lower()
        return self.cache_dir / f"{safe}.csv"

    def _read_cache(self, path: Path) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(path)
            df["From"] = pd.to_datetime(df["From"])
            df["To"]   = pd.to_datetime(df["To"])
            return df
        except Exception as e:
            logger.warning(f"Cache read failed {path}: {e}")
            return None

    def _write_cache(self, path: Path, df: pd.DataFrame) -> None:
        try:
            df.to_csv(path, index=False)
        except Exception as e:
            logger.warning(f"Cache write failed {path}: {e}")
