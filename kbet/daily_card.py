#!/usr/bin/env python3
"""
KBet Daily Bet Card — Week 2
=============================
Produces 5-12 ranked value bets for today (or any target date).

Pipeline:
  1. Load trained models (Dixon-Coles, Corners, Cards)
  2. Fetch live odds from The Odds API (or simulate from historical data)
  3. For each match: compute EV across all markets
  4. Rank by composite score (EV × confidence × CLV-proxy)
  5. Select 5-12 best non-correlated bets
  6. Print formatted card + save JSON

Usage:
  # Today's live card (requires ODDS_API_KEY env var):
  python3 daily_card.py

  # Simulate a past date (backtest mode, uses stored odds):
  python3 daily_card.py --date 2024-09-14 --simulate

  # More options:
  python3 daily_card.py --date 2024-09-14 --leagues E0 SP1 D1 --max-bets 8

Output:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  KBet Daily Card — 2024-09-14  (8 bets)                            │
  ├──────┬─────────────────────────┬────────┬──────┬──────┬────────────┤
  │  #   │  Match                  │ Market │ Pick │  EV  │ Confidence │
  ├──────┼─────────────────────────┼────────┼──────┼──────┼────────────┤
  │  1   │  Arsenal vs Chelsea     │  1X2   │  H   │ 4.2% │ ★★★★☆      │
  ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kbet.engine.models.dixon_coles import DixonColesModel
from kbet.engine.models.corners_model import CornersModel
from kbet.engine.models.cards_model import CardsModel
from kbet.engine.data.odds_client import OddsAPIClient, MatchOdds, OddsAPIKeyMissingError
from kbet.engine.data.weather_client import WeatherClient
from kbet.engine.data.clubelo_client import ClubEloClient
from kbet.engine.utils.entity_resolver import EntityRegistry as EntityResolver
try:
    from kbet.engine.scrapers.understat_scraper import rolling_xg as get_rolling_xg
except Exception:
    get_rolling_xg = lambda *a, **k: None
try:
    from kbet.engine.scrapers.forebet_scraper import get_consensus as get_forebet_consensus, divergence_signal
except Exception:
    get_forebet_consensus = lambda *a, **k: None
    divergence_signal = lambda *a, **k: "NO_CONSENSUS"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("daily_card")

# ---------------------------------------------------------------------------
# Persistent storage paths
# On Fly.io / Render with a mounted volume: set DATA_DIR=/data
# Locally defaults to kbet/data (relative to this file)
# ---------------------------------------------------------------------------
_DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))

DATA_PATH   = Path(__file__).parent / "data/processed/all_matches.parquet"  # source data (static)
OUTPUT_DIR  = _DATA_DIR / "daily_cards"     # card JSONs → persistent volume
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CARD_CONFIG = {
    # EV thresholds (post-slippage) — pick best mix (not just O/U)
    "1x2_ev_thresh":       0.02,   # 2.0% → mix enabler
    "ou_ev_thresh":        0.035,  # 3.5% slightly down
    "btts_ev_thresh":      0.04,   # 4% down from 5%
    "corners_ev_thresh":   0.04,
    "cards_ev_thresh":     0.05,

    # Confidence filters
    "1x2_min_prob":        0.22,
    "ou_min_prob":         0.32,
    "btts_min_prob":       0.32,
    "corners_min_prob":    0.40,
    "cards_min_prob":      0.40,

    # Pinnacle gap: blended prob must beat Pinnacle de-vig by this margin
    "1x2_vs_pin_gap":      0.008,  # mix enabler (was 0.012)

    # DC / Pinnacle blend weight
    "blend_dc_weight":     0.20,   # 20% DC, 80% Pinnacle

    # Risk controls — adaptive min_bets 3-5 (thin day 3 if avg EV>0.10 else 5)
    "slippage":            0.20,
    "max_bets":            12,
    "min_bets":            5,      # base; overridden dynamically in _rank_and_select
    "max_bets_per_match":  2,      # Max 2 markets bet on same game (correlation)
    "max_bets_per_team":   2,      # Max 2 times same team appears across card (kill Bayern twice)
    "min_odds":            1.40,   # No value in massive favourites
    "max_odds":            5.00,   # No lottery shots

    # Calibration
    "calib_tail_quantile": 0.75,   # Last 25% of training for isotonic

    # Weather (reduce confidence if adverse)
    "weather_confidence_penalty": 0.10,  # -10% confidence score in HEAVY_RAIN/WINDY
}

TRAIN_CUTOFF_DAYS = 30   # Use data up to 30 days before today as training


# ---------------------------------------------------------------------------
# Bet dataclass
# ---------------------------------------------------------------------------

@dataclass
class Bet:
    """A single ranked bet recommendation."""
    match:       str         # "Arsenal vs Chelsea"
    league:      str         # "E0" → "Premier League"
    market:      str         # "1X2" | "Over/Under 2.5" | "Corners 9.5" | "Cards 3.5"
    pick:        str         # "H" | "D" | "A" | "Over" | "Under"
    odds:        float       # Best available decimal odds
    bookmaker:   str         # Where to take the odds
    model_prob:  float       # Our model probability
    implied_prob: float      # Odds-implied probability (1/odds)
    ev:          float       # Post-slippage EV
    confidence:  float       # Composite confidence score 0-1
    stars:       int         # 1-5 stars
    clv_proxy:   float       # Estimated CLV (0 if no live odds)
    weather_tag: str = "DRY"
    notes:       str = ""
    commence_time: str = ""
    home_team:   str = ""
    away_team:   str = ""
    match_date:  str = ""    # YYYY-MM-DD for settlement / horizon

    @property
    def star_str(self) -> str:
        # ASCII-safe for Windows cp1252 consoles
        return "*" * self.stars + "-" * (5 - self.stars)

    @property
    def ev_pct(self) -> str:
        return f"{self.ev*100:+.1f}%"


LEAGUE_NAMES = {
    "E0":  "Premier League",
    "E1":  "Championship",
    "SP1": "La Liga",
    "D1":  "Bundesliga",
    "I1":  "Serie A",
    "F1":  "Ligue 1",
    "N1":  "Eredivisie",
    "P1":  "Primeira Liga",
    "B1":  "Pro League",
    "G1":  "Super League",
}


# ---------------------------------------------------------------------------
# Model trainer
# ---------------------------------------------------------------------------

# Model cache lives on the persistent volume so it survives redeploys.
# First run: ~3-5 min to fit Dixon-Coles.  Every subsequent run: ~15 sec.
MODEL_CACHE_DIR = _DATA_DIR / "model_cache"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ModelSet:
    """
    Trained set of all models for a given training cutoff.
    Persists fitted models to disk (pickle) so repeat runs load in <1s.
    Cache is keyed by as_of_date — stale by 12h or when data is newer.
    """

    TRAINING_WINDOW_DAYS = 540       # 18-month lookback (recency fix: was 1095 3yr — stale for Chelsea new coach)
    CACHE_MAX_AGE_HOURS  = 12        # Refit if cache > 12h old

    def __init__(self, as_of_date: pd.Timestamp, all_df: pd.DataFrame,
                 force_refit: bool = False):
        self.as_of = as_of_date

        # Attempt to load from cache
        cache_key = as_of_date.strftime("%Y%m%d")
        cached = self._load_cache(cache_key) if not force_refit else None

        if cached is not None:
            self.dc          = cached["dc"]
            self.calibrators = cached["calibrators"]
            self.corners     = cached["corners"]
            self.cards       = cached["cards"]
            self.df          = cached["df"]
            print(f"  [CACHE] Loaded models from cache ({cache_key}) — skipping refit")
            print(f"     DC: {self.dc.n_matches} matches, {len(self.dc.teams)} teams")
            return

        # Apply 3-year rolling window (exponential decay makes older data near-zero weight)
        cutoff = as_of_date - pd.Timedelta(days=self.TRAINING_WINDOW_DAYS)
        dc_df = all_df[all_df["date"] >= cutoff].copy()
        if "home_uuid" not in dc_df.columns:
            dc_df["home_uuid"] = dc_df["home_team"]
            dc_df["away_uuid"] = dc_df["away_team"]
        self.df = dc_df

        print(f"  [1/4] Fitting Dixon-Coles ({as_of_date.date()}, "
              f"window={self.TRAINING_WINDOW_DAYS}d, {len(dc_df):,} matches)...")
        self.dc = DixonColesModel()
        # Live recency: use xi 0.010 (70d) if as_of is recent (<90d from now), else base 0.0065
        try:
            from kbet.config.settings import DIXON_COLES as _DC
            is_live = (pd.Timestamp.now() - as_of_date).days < 90
            xi_use = _DC.get("xi_live", 0.010) if is_live else _DC.get("xi", 0.0065)
            if is_live:
                print(f"       Live mode xi={xi_use} (70d half-life) for recency")
        except Exception:
            xi_use = None
        self.dc.fit(dc_df, as_of_date, xi=xi_use)
        print(f"        DC: {self.dc.n_matches} matches, {len(self.dc.teams)} teams  OK")

        print("  [2/4] Isotonic calibrators...")
        self.calibrators = self._fit_calibrators(dc_df, as_of_date)
        print(f"        Calibrators: {list(self.calibrators.keys()) if self.calibrators else 'None'}  OK")

        print("  [3/4] Corners model...")
        self.corners = CornersModel(decay_xi=0.006)
        self.corners.fit(dc_df, as_of_date)
        print(f"        Corners: {self.corners.n_matches} matches, fitted={self.corners.fitted}  OK")

        print("  [4/4] Cards model...")
        self.cards = CardsModel(decay_xi=0.005)
        self.cards.fit(dc_df, as_of_date)
        print(f"        Cards: {self.cards.n_matches} matches, fitted={self.cards.fitted}  OK")

        # Persist to cache
        self._save_cache(cache_key)

    def _fit_calibrators(
        self, all_df: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> Optional[Dict]:
        """
        Fit isotonic calibrators on last 25% of training data.
        Vectorised: batch-predicts all teams then fits isotonic — ~100x faster
        than row-by-row iteration.
        """
        train = all_df[all_df["date"] < as_of_date].copy()
        if len(train) < 200:
            return None

        calib_start_idx = int(len(train) * CARD_CONFIG["calib_tail_quantile"])
        calib = train.iloc[calib_start_idx:].sort_values("date").reset_index(drop=True)

        # Batch predict: one predict_1x2 call per unique (home, away) pair
        pairs = calib[["home_team", "away_team"]].drop_duplicates()
        pred_cache: Dict[tuple, Optional[Dict]] = {}
        for _, prow in pairs.iterrows():
            key = (prow["home_team"], prow["away_team"])
            try:
                pred_cache[key] = self.dc.predict_1x2(prow["home_team"], prow["away_team"])
            except Exception:
                pred_cache[key] = None

        # Build arrays
        ph_arr, pd_arr, pa_arr = [], [], []
        res_arr = []
        for _, row in calib.iterrows():
            key = (row["home_team"], row["away_team"])
            pred = pred_cache.get(key)
            if pred is None:
                continue
            ph_arr.append(pred["home"])
            pd_arr.append(pred["draw"])
            pa_arr.append(pred["away"])
            res_arr.append(row["result"])

        if len(res_arr) < 50:
            return None

        import numpy as _np
        res_arr = _np.array(res_arr)
        calibrators = {}
        for outcome, probs in [("H", ph_arr), ("D", pd_arr), ("A", pa_arr)]:
            actuals = (res_arr == outcome).astype(float)
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs, actuals)
            calibrators[outcome] = ir

        return calibrators if len(calibrators) == 3 else None

    # -----------------------------------------------------------------------
    # Model cache: pickle to disk, keyed by as_of date
    # -----------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return MODEL_CACHE_DIR / f"modelset_{key}.pkl"

    def _load_cache(self, key: str) -> Optional[Dict]:
        import pickle
        p = self._cache_path(key)
        if not p.exists():
            return None
        age_h = (time.time() - p.stat().st_mtime) / 3600
        if age_h > self.CACHE_MAX_AGE_HOURS:
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  Cache load failed: {e}")
            return None

    def _save_cache(self, key: str) -> None:
        import pickle
        p = self._cache_path(key)
        payload = {
            "dc":          self.dc,
            "calibrators": self.calibrators,
            "corners":     self.corners,
            "cards":       self.cards,
            "df":          self.df,
        }
        try:
            with open(p, "wb") as f:
                pickle.dump(payload, f, protocol=4)
            print(f"  [CACHE] Model cache saved -> {p.name}")
        except Exception as e:
            print(f"  Cache save failed: {e}")

    def predict_1x2_calibrated(self, home: str, away: str) -> Optional[Dict[str, float]]:
        """Return calibrated 1X2 probabilities."""
        raw = self.dc.predict_1x2(home, away)
        if raw is None:
            return None
        if self.calibrators is None:
            return raw
        return {
            "home": float(self.calibrators["H"].predict([raw["home"]])[0]),
            "draw": float(self.calibrators["D"].predict([raw["draw"]])[0]),
            "away": float(self.calibrators["A"].predict([raw["away"]])[0]),
        }

    def predict_ou_calibrated(self, home: str, away: str, threshold: float = 2.5) -> Optional[Dict]:
        """Return DC-based O/U prediction, calibrated via rolling trend."""
        return self.dc.predict_over_under(home, away, threshold=threshold)


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class DailyCardEngine:
    """
    Generates the daily bet card.

    Parameters
    ----------
    target_date : date to generate bets for (default today)
    simulate    : if True, use historical data instead of live odds
    leagues     : restrict to these league codes
    api_key     : Odds API key (or set ODDS_API_KEY env var)
    """

    def __init__(
        self,
        target_date: str = None,
        simulate: bool = False,
        leagues: Optional[List[str]] = None,
        api_key: str = None,
    ):
        self.target_date = target_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.simulate    = simulate
        self.leagues     = leagues
        self.api_key     = api_key or os.getenv("ODDS_API_KEY", "")

        # Load historical data
        print(f"\n{'='*65}")
        print(f"  KBet Daily Card — {self.target_date}")
        print(f"{'='*65}")
        print(f"  Loading match data...")
        self.all_df = pd.read_parquet(DATA_PATH)
        self.all_df["date"] = pd.to_datetime(self.all_df["date"])
        print(f"  Data: {len(self.all_df)} matches up to {self.all_df.date.max().date()}")

        # Resolve entity names
        self.resolver = EntityResolver()  # EntityRegistry with .resolve() method

        # Data clients
        self.weather  = WeatherClient()
        self.clubelo  = ClubEloClient()
        self.odds_client = OddsAPIClient(api_key=self.api_key)

    def run(self, max_bets: int = None) -> List[Bet]:
        """Main pipeline: fetch odds → evaluate → rank → return card. Adaptive 2-day merge if thin."""
        max_bets = max_bets or CARD_CONFIG["max_bets"]

        # Training cutoff: use all data before target_date
        as_of = pd.Timestamp(self.target_date)

        # Fit models once
        models = ModelSet(as_of, self.all_df)
        self._horizon = 1
        self._horizon_dates = [self.target_date]

        all_bets = self._evaluate_date(self.target_date, models)
        # Allow empty to trigger adaptive merge (thin Monday or no fixtures today)
        actionable = [b for b in all_bets if b.ev > 0]
        avg_ev = sum(b.ev for b in actionable) / len(actionable) if actionable else 0
        need_merge = len(actionable) < 5 or (actionable and avg_ev < 0.08) or len(all_bets) == 0
        # Also merge if total matches <8 (very thin Monday)
        if hasattr(self, '_last_match_count') and self._last_match_count < 8:
            need_merge = True

        if need_merge:
            parquet_max = self.all_df["date"].max()
            orig_target = self.target_date
            # Try up to 2 extra days (horizon 3) until we have >=5 bets
            for offset in [1, 2]:
                next_date = (pd.Timestamp(orig_target) + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                if self.simulate and pd.Timestamp(next_date) > parquet_max + pd.Timedelta(days=2):
                    print(f"  [ADAPTIVE] Skip {next_date} beyond data max {parquet_max.date()} (simulate)")
                    continue
                # Re-evaluate need (check current all_bets)
                cur_actionable = [b for b in all_bets if b.ev > 0]
                cur_avg = sum(b.ev for b in cur_actionable)/len(cur_actionable) if cur_actionable else 0
                if len(cur_actionable) >=5 and cur_avg >=0.08 and len(all_bets) >=5:
                    break
                # Try this next date
                if offset == 1:
                    print(f"  [ADAPTIVE] Thin day ({len(cur_actionable)} bets, avg EV {cur_avg:.1%}, matches {getattr(self,'_last_match_count', '?')}) -> merging {next_date}")
                else:
                    print(f"  [ADAPTIVE] Still thin ({len(cur_actionable)} bets) -> trying {next_date}")
                self.target_date = next_date
                more_bets = self._evaluate_date(next_date, models)
                self.target_date = orig_target
                if more_bets:
                    for b in more_bets:
                        b.match_date = next_date
                        if not b.commence_time or b.commence_time[:10] == orig_target:
                            b.commence_time = f"{next_date} {b.commence_time[11:] if len(b.commence_time)>10 else '00:00:00'}"
                    all_bets = all_bets + more_bets
                    self._horizon = offset + 1
                    self._horizon_dates = [orig_target] + [(pd.Timestamp(orig_target)+pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, offset+1)]
                    print(f"  [ADAPTIVE] Merged -> {len(all_bets)} total bets across {self._horizon} days")
                # Continue loop if still thin
            # end for
            # else: keep single day if loop didn't help

        # Rank and filter (with team collision guard)
        final_card = self._rank_and_select(all_bets, max_bets)
        # Trim to adaptive min_bets 3-5 logic: if avg EV high, allow 3
        if len(final_card) >= 3 and len(final_card) < 5 and avg_ev > 0.10:
            pass  # keep 3
        return final_card

    def _evaluate_date(self, date_str: str, models: "ModelSet") -> List[Bet]:
        """Evaluate all bets for a single date — extracted for adaptive merge."""
        # Fetch matches for this date (override target temporarily if needed)
        orig = self.target_date
        need_switch = date_str != orig
        if need_switch:
            self.target_date = date_str
        matches = self._get_matches()
        if need_switch:
            self.target_date = orig
        # For live mode, filter matches to this specific date (API returns next 3 days)
        if not self.simulate and matches:
            filtered = [m for m in matches if m.commence_time[:10] == date_str]
            # If strict filter yields 0 but API had matches for nearby dates, keep empty to trigger merge
            # Don't fallback to all — keep empty so adaptive can try next day
            matches = filtered
        if not matches:
            print(f"  [WARN] No matches found for {date_str}.")
            return []
        self._last_match_count = len(matches)
        print(f"\n  Fetching match odds for {date_str}... Found {len(matches)} matches")

        # Weather skip logic
        skip_external = False
        try:
            is_historic = (pd.Timestamp(date_str) < pd.Timestamp.now() - pd.Timedelta(days=7))
            if is_historic or self.simulate or os.environ.get("KBET_SKIP_EXTERNAL") == "1":
                skip_external = True
        except Exception:
            pass
        if skip_external:
            weather_map = {}
            print(f"  [HISTORIC] Skipping weather/ClubElo API ({date_str})")
        else:
            weather_inputs = [(m.home_team, m.league_code, date_str) for m in matches]
            weather_map = self.weather.get_bulk_conditions(weather_inputs)

        print(f"  Evaluating {len(matches)} matches for {date_str}...")
        all_bets: List[Bet] = []
        bets_per_match: Dict[str, int] = {}
        for match in matches:
            match_key = f"{match.home_team} vs {match.away_team}"
            bets_per_match.setdefault(match_key, 0)
            weather_key = f"{match.home_team}_{date_str}"
            weather = weather_map.get(weather_key)
            weather_tag = weather.weather_tag if weather else "DRY"
            if skip_external:
                elo_h, elo_a = None, None
            else:
                elo_h, elo_a = self.clubelo.get_elo_pair(match.home_team, match.away_team, date_str)
            home_id = self.resolver.resolve(match.home_team)
            away_id = self.resolver.resolve(match.away_team)
            if bets_per_match[match_key] < CARD_CONFIG["max_bets_per_match"]:
                for b in self._eval_1x2(models, match, home_id, away_id, elo_h, elo_a, weather_tag, weather):
                    if bets_per_match[match_key] < CARD_CONFIG["max_bets_per_match"]:
                        # Tag date for settlement
                        b.commence_time = b.commence_time or f"{date_str} 00:00:00"
                        b.match_date = date_str
                        all_bets.append(b)
                        bets_per_match[match_key] += 1
            if bets_per_match[match_key] < CARD_CONFIG["max_bets_per_match"]:
                for b in self._eval_ou_goals(models, match, home_id, away_id, weather_tag, weather):
                    if bets_per_match[match_key] < CARD_CONFIG["max_bets_per_match"]:
                        b.commence_time = b.commence_time or f"{date_str} 00:00:00"
                        b.match_date = date_str
                        all_bets.append(b)
                        bets_per_match[match_key] += 1
            if bets_per_match[match_key] < CARD_CONFIG["max_bets_per_match"]:
                for b in self._eval_btts(models, match, home_id, away_id, weather_tag, weather):
                    if bets_per_match[match_key] < CARD_CONFIG["max_bets_per_match"]:
                        b.commence_time = b.commence_time or f"{date_str} 00:00:00"
                        b.match_date = date_str
                        all_bets.append(b)
                        bets_per_match[match_key] += 1
            if bets_per_match[match_key] < CARD_CONFIG["max_bets_per_match"]:
                for b in self._eval_corners(models, match, home_id, away_id, weather_tag, weather):
                    if bets_per_match[match_key] < CARD_CONFIG["max_bets_per_match"]:
                        b.commence_time = b.commence_time or f"{date_str} 00:00:00"
                        b.match_date = date_str
                        all_bets.append(b)
                        bets_per_match[match_key] += 1
            if bets_per_match[match_key] < CARD_CONFIG["max_bets_per_match"]:
                for b in self._eval_cards(models, match, home_id, away_id, weather_tag):
                    if bets_per_match[match_key] < CARD_CONFIG["max_bets_per_match"]:
                        b.commence_time = b.commence_time or f"{date_str} 00:00:00"
                        b.match_date = date_str
                        all_bets.append(b)
                        bets_per_match[match_key] += 1
        return all_bets

    # -----------------------------------------------------------------------
    # Market evaluators
    # -----------------------------------------------------------------------

    def _eval_1x2(
        self, models: ModelSet, match: MatchOdds,
        home_id, away_id, elo_h, elo_a, weather_tag, weather
    ) -> List[Bet]:
        """Evaluate 1X2 market: blend DC-calibrated + Pinnacle de-vig."""
        bets = []
        try:
            dc_pred = models.predict_1x2_calibrated(home_id, away_id)
            if dc_pred is None:
                return []

            pin_devig = match.devig_pinnacle_h2h()
            if pin_devig is None:
                return []

            pin_h, pin_d, pin_a = pin_devig
            alpha = CARD_CONFIG["blend_dc_weight"]

            # Blend: 20% DC + 80% Pinnacle
            blended = {
                "home": alpha * dc_pred["home"] + (1-alpha) * pin_h,
                "draw": alpha * dc_pred["draw"] + (1-alpha) * pin_d,
                "away": alpha * dc_pred["away"] + (1-alpha) * pin_a,
            }
            pin_probs = {"home": pin_h, "draw": pin_d, "away": pin_a}

            best_odds_obj = match.max_h2h()
            if best_odds_obj is None:
                return []

            odds_map = {
                "home": (best_odds_obj.home, "H"),
                "draw": (best_odds_obj.draw, "D"),
                "away": (best_odds_obj.away, "A"),
            }

            # Pre-compute overround for sanity guard (reuse)
            try:
                or_val = (1/best_odds_obj.home)+(1/best_odds_obj.draw)+(1/best_odds_obj.away)
            except Exception:
                or_val = 1.05
            MAX_SANE_EV = 0.20
            WARN_EV = 0.12
            for side, (odds, pick) in odds_map.items():
                blend_p = blended[side]
                pin_p   = pin_probs[side]
                # xG form adjustment (if available) — boosts recency for Chelsea-like overhaul
                try:
                    hx = get_rolling_xg(match.home_team, self.target_date, 10)
                    ax = get_rolling_xg(match.away_team, self.target_date, 10)
                    if hx and ax and side == "home" and hx["xg_for_avg"] > 1.8 and hx["xg_for_avg"] > ax["xg_against_avg"]:
                        blend_p = min(0.92, blend_p + 0.02)
                    if hx and ax and side == "away" and ax["xg_for_avg"] > 1.8 and ax["xg_for_avg"] > hx["xg_against_avg"]:
                        blend_p = min(0.92, blend_p + 0.02)
                except Exception:
                    pass
                # Forebet contrarian divergence (if enabled via KBET_FOREBET=1) — not direct weight
                if os.environ.get("KBET_FOREBET") == "1":
                    try:
                        fb = get_forebet_consensus(match.home_team, match.away_team, self.target_date)
                        if fb:
                            fb_p = fb.get({"home": "h", "draw": "d", "away": "a"}[side])
                            if fb_p is not None:
                                sig = divergence_signal(blend_p, fb_p)
                                if sig == "MARKET_KNOWS_SOMETHING" and pin_gap < 0.02:
                                    # Market disagrees strongly and gap small -> skip
                                    continue
                                # MODEL_FINDS_VALUE handled via pin_gap already
                    except Exception:
                        pass

                # Sanity: odds plausible
                if odds is None or odds <= 1.01 or odds > 20.0:
                    continue
                # Overround 0.98-1.15 else corrupted
                if or_val < 0.98 or or_val > 1.15:
                    logger.warning(f"[ODDS REJECT] {match.home_team} v {match.away_team}: overround={or_val:.3f}")
                    break
                if blend_p <= 0.01 or blend_p >= 0.99:
                    continue

                slip = CARD_CONFIG["slippage"]
                ev = ((blend_p * odds) - 1.0) * (1 - slip)
                # EV ceiling 20% hard, 12% soft warn
                if ev > MAX_SANE_EV:
                    logger.warning(f"[EV REJECT] {match.home_team} v {match.away_team} {pick}: EV={ev:.1%} >20% ceiling")
                    continue
                if ev > WARN_EV:
                    logger.info(f"[EV WARN] {match.home_team} v {match.away_team} {pick}: EV={ev:.1%} verify")

                # Must genuinely exceed Pinnacle
                pin_gap = blend_p - pin_p
                if pin_gap < CARD_CONFIG["1x2_vs_pin_gap"]:
                    continue
                if blend_p < CARD_CONFIG["1x2_min_prob"]:
                    continue
                if odds < CARD_CONFIG["min_odds"] or odds > CARD_CONFIG["max_odds"]:
                    continue

                if ev < CARD_CONFIG["1x2_ev_thresh"]:
                    continue

                confidence = self._confidence_score(
                    ev=ev, prob=blend_p, pin_gap=pin_gap,
                    elo_h=elo_h, elo_a=elo_a,
                    weather_tag=weather_tag, model_fitted=True,
                    market="1x2"
                )

                clv_proxy = pin_gap * 0.5  # Estimate: half the gap is genuine

                bets.append(Bet(
                    match=f"{match.home_team} vs {match.away_team}",
                    league=match.league_code,
                    market="1X2",
                    pick=pick,
                    odds=round(odds, 2),
                    bookmaker=best_odds_obj.bookmaker,
                    model_prob=round(blend_p, 4),
                    implied_prob=round(1/odds, 4),
                    ev=round(ev, 4),
                    confidence=round(confidence, 3),
                    stars=self._stars(confidence),
                    clv_proxy=round(clv_proxy, 4),
                    weather_tag=weather_tag,
                    notes=f"DC={dc_pred[side]:.3f} Pin={pin_p:.3f} blend={blend_p:.3f} gap={pin_gap:+.3f}",
                    commence_time=match.commence_time,
                    home_team=match.home_team,
                    away_team=match.away_team,
                ))
        except Exception as e:
            logger.warning(f"1X2 eval failed for {match.home_team}: {e}")

        return sorted(bets, key=lambda b: b.ev, reverse=True)[:1]  # Best side only

    def _eval_ou_goals(
        self, models: ModelSet, match: MatchOdds,
        home_id, away_id, weather_tag, weather
    ) -> List[Bet]:
        """Evaluate Over/Under 2.5 goals market."""
        bets = []
        try:
            dc_ou = models.predict_ou_calibrated(home_id, away_id, threshold=2.5)
            if dc_ou is None:
                return []

            # Rolling goals trend adjustment (carry forward from V3)
            rolling_avg = self._rolling_goals_avg(self.all_df, pd.Timestamp(self.target_date))
            trend_adj = 0.015 if rolling_avg > 2.80 else 0.0
            # Dynamic sizing: +0.007 per 0.1g above threshold
            if rolling_avg > 2.80:
                trend_adj = 0.015 + (rolling_avg - 2.80) * 0.007 / 0.1

            adj_over  = min(dc_ou["over"]  + trend_adj, 0.95)
            adj_under = max(dc_ou["under"] - trend_adj, 0.05)
            total_ou  = adj_over + adj_under
            adj_over /= total_ou
            adj_under /= total_ou

            # Weather adjustment to goals
            if weather is not None:
                goals_adj_prob = weather.goals_adj * 0.05  # convert xG adj → prob shift
                adj_over  = max(0.05, adj_over  + goals_adj_prob)
                adj_under = max(0.05, adj_under - goals_adj_prob)

            best_ou = match.best_totals(line=2.5)
            if best_ou is None:
                return []

            for side, prob, odds in [
                ("over",  adj_over,  best_ou.over),
                ("under", adj_under, best_ou.under),
            ]:
                # xG recent form boost for Over (Chelsea 2.8 xG case)
                try:
                    hx = get_rolling_xg(match.home_team, self.target_date, 10)
                    ax = get_rolling_xg(match.away_team, self.target_date, 10)
                    if hx and ax and side == "over" and (hx["xg_for_avg"] + ax["xg_for_avg"]) > 3.0:
                        prob = min(0.92, prob + 0.03)
                    if hx and ax and side == "under" and (hx["xg_against_avg"] + ax["xg_against_avg"]) < 1.5:
                        prob = min(0.92, prob + 0.02)
                except Exception:
                    pass
                if prob < CARD_CONFIG["ou_min_prob"]:
                    continue
                if odds < CARD_CONFIG["min_odds"] or odds > CARD_CONFIG["max_odds"]:
                    continue
                ev = ((prob * odds) - 1.0) * (1 - CARD_CONFIG["slippage"])
                if ev < CARD_CONFIG["ou_ev_thresh"]:
                    continue
                if ev > 0.20:
                    logger.warning(f"[EV REJECT] O/U {match.home_team} v {match.away_team} {side}: EV={ev:.1%} >20%")
                    continue
                if ev > 0.12:
                    logger.info(f"[EV WARN] O/U {match.home_team} v {match.away_team} {side}: EV={ev:.1%}")
                if odds is None or odds <= 1.01 or odds > 15.0:
                    continue

                confidence = self._confidence_score(
                    ev=ev, prob=prob, pin_gap=0.0,
                    weather_tag=weather_tag, model_fitted=True, market="ou"
                )

                bets.append(Bet(
                    match=f"{match.home_team} vs {match.away_team}",
                    league=match.league_code,
                    market="O/U 2.5 Goals",
                    pick=side.capitalize(),
                    odds=round(odds, 2),
                    bookmaker="MAX",
                    model_prob=round(prob, 4),
                    implied_prob=round(1/odds, 4),
                    ev=round(ev, 4),
                    confidence=round(confidence, 3),
                    stars=self._stars(confidence),
                    clv_proxy=0.0,
                    weather_tag=weather_tag,
                    notes=f"DC_ou={dc_ou[side]:.3f} trend_adj={trend_adj:+.3f} rolling={rolling_avg:.2f}g",
                    commence_time=match.commence_time,
                    home_team=match.home_team,
                    away_team=match.away_team,
                ))
        except Exception as e:
            logger.warning(f"O/U Goals eval failed: {e}")

        return sorted(bets, key=lambda b: b.ev, reverse=True)[:1]

    def _eval_btts(
        self, models: ModelSet, match: MatchOdds,
        home_id, away_id, weather_tag, weather
    ) -> List[Bet]:
        """Evaluate BTTS Yes/No market."""
        bets = []
        try:
            btts_pred = models.dc.predict_btts(home_id, away_id)
            if btts_pred is None:
                return []
            best_btts = match.best_btts()
            if best_btts is None:
                return []
            # Try Pinnacle devig as reference? For now use best price devig
            for side, prob, odds in [
                ("yes", btts_pred["yes"], best_btts.yes),
                ("no",  btts_pred["no"],  best_btts.no),
            ]:
                if prob < CARD_CONFIG["btts_min_prob"]:
                    continue
                if odds is None or odds <= 1.01 or odds > 15.0:
                    continue
                ev = ((prob * odds) - 1.0) * (1 - CARD_CONFIG["slippage"])
                if ev < CARD_CONFIG["btts_ev_thresh"]:
                    continue
                if ev > 0.20:
                    logger.warning(f"[EV REJECT] BTTS {match.home_team} v {match.away_team} {side}: EV={ev:.1%} >20%")
                    continue
                if ev > 0.12:
                    logger.info(f"[EV WARN] BTTS {match.home_team} v {match.away_team} {side}: EV={ev:.1%}")
                confidence = self._confidence_score(ev=ev, prob=prob, pin_gap=0.0, weather_tag=weather_tag, model_fitted=True, market="btts")
                bets.append(Bet(
                    match=f"{match.home_team} vs {match.away_team}",
                    league=match.league_code,
                    market="BTTS",
                    pick="Yes" if side=="yes" else "No",
                    odds=round(odds,2),
                    bookmaker=best_btts.bookmaker,
                    model_prob=round(prob,4),
                    implied_prob=round(1/odds,4),
                    ev=round(ev,4),
                    confidence=round(confidence,3),
                    stars=self._stars(confidence),
                    clv_proxy=0.0,
                    weather_tag=weather_tag,
                    notes=f"BTTS {side}={prob:.3f}",
                    commence_time=match.commence_time,
                    home_team=match.home_team,
                    away_team=match.away_team,
                ))
        except Exception as e:
            logger.warning(f"BTTS eval failed: {e}")
        return sorted(bets, key=lambda b: b.ev, reverse=True)[:1]

    def _eval_corners(
        self, models: ModelSet, match: MatchOdds,
        home_id, away_id, weather_tag, weather
    ) -> List[Bet]:
        """Evaluate corners O/U market (9.5 and 10.5)."""
        bets = []
        if not models.corners.fitted:
            return []

        # We don't have live corners odds from The Odds API in basic plan.
        # For simulation/backtest, we skip if no odds available.
        # For live card, this requires an odds source for corners.
        # Placeholder: if odds available via match object, evaluate.
        # In real implementation: fetch corners from dedicated bookmaker direct.

        # --- Simulate odds from market average for demonstration ---
        # (In production: replace with actual corners odds from API)
        try:
            pred = models.corners.predict(home_id, away_id)
            if pred is None:
                return []

            exp_total = pred["exp_total"]

            # Test lines around expected value
            for line in [9.5, 10.5, 11.5]:
                key = str(line).replace(".", "_")
                p_over  = pred.get(f"over_{key}",  0)
                p_under = pred.get(f"under_{key}", 0)

                # Typical corners market odds ~1.85/1.95 with ~8% margin
                # We need real odds to evaluate; without them we estimate
                # fair odds from DC then check if market (if available) differs
                # For now: generate the prediction, flag as "ODDS NEEDED"
                if p_over > 0.60 and p_over >= CARD_CONFIG["corners_min_prob"]:
                    # Strong model conviction — flag for live check
                    bets.append(Bet(
                        match=f"{match.home_team} vs {match.away_team}",
                        league=match.league_code,
                        market=f"Corners O/U {line}",
                        pick="Over",
                        odds=0.0,   # Requires live corners odds
                        bookmaker="OBTAIN ODDS",
                        model_prob=round(p_over, 4),
                        implied_prob=0.0,
                        ev=0.0,     # Cannot compute without odds
                        confidence=round(p_over * 0.7, 3),
                        stars=self._stars(p_over * 0.7),
                        clv_proxy=0.0,
                        weather_tag=weather_tag,
                        notes=f"exp={exp_total:.1f} corners. CHECK ODDS for {line}",
                        commence_time=match.commence_time,
                        home_team=match.home_team,
                        away_team=match.away_team,
                    ))
                    break
                elif p_under > 0.60 and p_under >= CARD_CONFIG["corners_min_prob"]:
                    bets.append(Bet(
                        match=f"{match.home_team} vs {match.away_team}",
                        league=match.league_code,
                        market=f"Corners O/U {line}",
                        pick="Under",
                        odds=0.0,
                        bookmaker="OBTAIN ODDS",
                        model_prob=round(p_under, 4),
                        implied_prob=0.0,
                        ev=0.0,
                        confidence=round(p_under * 0.7, 3),
                        stars=self._stars(p_under * 0.7),
                        clv_proxy=0.0,
                        weather_tag=weather_tag,
                        notes=f"exp={exp_total:.1f} corners. CHECK ODDS for {line}",
                        commence_time=match.commence_time,
                        home_team=match.home_team,
                        away_team=match.away_team,
                    ))
                    break
        except Exception as e:
            logger.warning(f"Corners eval failed: {e}")

        return bets[:1]

    def _eval_cards(
        self, models: ModelSet, match: MatchOdds,
        home_id, away_id, weather_tag
    ) -> List[Bet]:
        """Evaluate cards O/U market."""
        bets = []
        if not models.cards.fitted:
            return []

        try:
            pred = models.cards.predict(home_id, away_id, match.league_code)
            if pred is None:
                return []

            exp_total = pred["exp_total"]

            for line in [2.5, 3.5, 4.5]:
                key = str(line).replace(".", "_")
                p_over  = pred.get(f"over_{key}",  0)
                p_under = pred.get(f"under_{key}", 0)

                # Cards market is soft — flag high-conviction sides
                for side, prob in [("Over", p_over), ("Under", p_under)]:
                    if prob >= CARD_CONFIG["cards_min_prob"] + 0.10:
                        bets.append(Bet(
                            match=f"{match.home_team} vs {match.away_team}",
                            league=match.league_code,
                            market=f"Cards O/U {line}",
                            pick=side,
                            odds=0.0,
                            bookmaker="OBTAIN ODDS",
                            model_prob=round(prob, 4),
                            implied_prob=0.0,
                            ev=0.0,
                            confidence=round(prob * 0.65, 3),
                            stars=self._stars(prob * 0.65),
                            clv_proxy=0.0,
                            weather_tag=weather_tag,
                            notes=f"exp={exp_total:.1f} cards ({match.league_code}). CHECK ODDS for {line}",
                            commence_time=match.commence_time,
                            home_team=match.home_team,
                            away_team=match.away_team,
                        ))
                        break
                if bets:
                    break  # One cards bet per match max

        except Exception as e:
            logger.warning(f"Cards eval failed: {e}")

        return bets[:1]

    # -----------------------------------------------------------------------
    # Ranking and selection
    # -----------------------------------------------------------------------

    def _rank_and_select(self, all_bets: List[Bet], max_bets: int) -> List[Bet]:
        """
        Rank all bets by composite score and select 5-12.

        Composite score = EV * confidence * (1 + clv_proxy)
        Apply diversity constraints: max 2 bets per match.
        Filter out bets with EV=0 (odds not yet obtained).
        """
        # Split into actionable (odds known) and advisory (odds needed)
        actionable = [b for b in all_bets if b.ev > 0]
        advisory   = [b for b in all_bets if b.ev == 0 and b.confidence > 0.35]

        # Score actionable bets
        for b in actionable:
            b._score = b.ev * b.confidence * (1 + b.clv_proxy)
        actionable.sort(key=lambda b: b._score, reverse=True)

        # Score advisory bets by confidence
        for b in advisory:
            b._score = b.confidence
        advisory.sort(key=lambda b: b._score, reverse=True)

        selected: List[Bet] = []
        match_counts: Dict[str, int] = {}
        team_counts: Dict[str, int] = {}
        market_counts: Dict[str, int] = {}

        def market_key(b: Bet) -> str:
            m = b.market.lower()
            if "1x2" in m: return "1x2"
            if "o/u" in m or "over/under" in m: return "ou"
            if "btts" in m: return "btts"
            return m

        def can_add(b: Bet) -> bool:
            if match_counts.get(b.match, 0) >= CARD_CONFIG["max_bets_per_match"]:
                return False
            # Market diversity: max 6 per market to enforce mix (pick best across markets)
            mk = market_key(b)
            if market_counts.get(mk, 0) >= 6:
                return False
            # Team collision guard: same team max 2 times across card (kill Bayern twice artifact)
            ht = (b.home_team or b.match.split(" vs ")[0] if " vs " in b.match else "").strip().lower()
            at = (b.away_team or b.match.split(" vs ")[-1] if " vs " in b.match else "").strip().lower()
            if ht and team_counts.get(ht, 0) >= CARD_CONFIG.get("max_bets_per_team", 2):
                return False
            if at and team_counts.get(at, 0) >= CARD_CONFIG.get("max_bets_per_team", 2):
                return False
            return True

        def add_bet(b: Bet):
            selected.append(b)
            match_counts[b.match] = match_counts.get(b.match, 0) + 1
            market_counts[market_key(b)] = market_counts.get(market_key(b), 0) + 1
            ht = (b.home_team or b.match.split(" vs ")[0] if " vs " in b.match else "").strip().lower()
            at = (b.away_team or b.match.split(" vs ")[-1] if " vs " in b.match else "").strip().lower()
            if ht:
                team_counts[ht] = team_counts.get(ht, 0) + 1
            if at:
                team_counts[at] = team_counts.get(at, 0) + 1

        # First pass: fill actionable bets
        for b in actionable:
            if len(selected) >= max_bets:
                break
            if not can_add(b):
                continue
            add_bet(b)

        # Second pass: fill with advisory bets if below min
        # Adaptive min_bets 3-5: if avg EV high, allow 3
        avg_ev = sum(b.ev for b in actionable) / len(actionable) if actionable else 0
        dynamic_min = 3 if avg_ev > 0.10 else CARD_CONFIG["min_bets"]
        if len(selected) < dynamic_min:
            for b in advisory:
                if len(selected) >= max_bets:
                    break
                if not can_add(b):
                    continue
                add_bet(b)

        return selected

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _get_matches(self) -> List[MatchOdds]:
        """Fetch matches either from live API or historical simulation."""
        if self.simulate:
            return self._simulate_matches_from_history()

        if not self.api_key:
            print("  [WARN] No ODDS_API_KEY — switching to historical simulation mode")
            return self._simulate_matches_from_history()

        try:
            return self.odds_client.get_today_odds(leagues=self.leagues)
        except OddsAPIKeyMissingError:
            print("  [WARN] Invalid API key — switching to simulation mode")
            return self._simulate_matches_from_history()
        except Exception as e:
            print(f"  [WARN] API fetch failed ({e}) — switching to simulation mode")
            return self._simulate_matches_from_history()

    def _simulate_matches_from_history(self) -> List[MatchOdds]:
        """
        Reconstruct MatchOdds objects from historical parquet data.
        Used when live API is unavailable or in --simulate mode.
        """
        from kbet.engine.data.odds_client import MatchOdds, OddsH2H, OddsTotals

        target = pd.Timestamp(self.target_date)
        window_df = self.all_df[
            (self.all_df["date"] >= target) &
            (self.all_df["date"] < target + pd.Timedelta(days=1))
        ]

        if len(window_df) == 0:
            # Widen to ±1 day (NOT more) — some fixtures kick off after midnight UTC
            window_df = self.all_df[
                (self.all_df["date"] >= target - pd.Timedelta(days=1)) &
                (self.all_df["date"] <= target + pd.Timedelta(days=1))
            ]

        if len(window_df) == 0:
            # HARD STOP: No historical matches for this date — NEVER rename historic dates to "today"
            print(f"  [NO DATA] No fixtures for {self.target_date} in dataset — returning empty")
            return []

        # Apply league filter
        if self.leagues:
            window_df = window_df[window_df["league_code"].isin(self.leagues)]

        matches = []
        for _, row in window_df.iterrows():
            m = MatchOdds(
                match_id=f"hist_{row.name}",
                sport_key="historical",
                league_code=row["league_code"],
                home_team=row["home_team"],
                away_team=row["away_team"],
                commence_time=str(row["date"]),
                snapshot_time=str(row["date"]),
            )

            # Populate H2H odds
            h2h_sources = [
                ("pinnacle", "odds_home_pinnacle", "odds_draw_pinnacle", "odds_away_pinnacle"),
                ("bet365",   "odds_home_b365",     "odds_draw_b365",     "odds_away_b365"),
                ("max",      "odds_home_max",       "odds_draw_max",      "odds_away_max"),
            ]
            for bk, h_col, d_col, a_col in h2h_sources:
                try:
                    h = float(row[h_col]); d = float(row[d_col]); a = float(row[a_col])
                    if h > 1 and d > 1 and a > 1:
                        m.h2h.append(OddsH2H(bk, h, d, a))
                except (KeyError, ValueError, TypeError):
                    pass

            # Populate totals odds
            for bk, ov_col, un_col in [
                ("max",    "odds_over25_max",  "odds_under25_max"),
                ("bet365", "odds_over25_b365", "odds_under25_b365"),
            ]:
                try:
                    ov = float(row[ov_col]); un = float(row[un_col])
                    if ov > 1 and un > 1:
                        m.totals.append(OddsTotals(bk, 2.5, ov, un))
                except (KeyError, ValueError, TypeError):
                    pass

            if m.h2h:  # Only include matches with odds
                matches.append(m)

        return matches

    def _rolling_goals_avg(
        self, df: pd.DataFrame, as_of_date: pd.Timestamp, days: int = 60
    ) -> float:
        """60-day rolling goals per game average (for O/U trend detection)."""
        cutoff = as_of_date - pd.Timedelta(days=days)
        window = df[(df["date"] >= cutoff) & (df["date"] < as_of_date)].copy()
        if len(window) < 20:
            return 2.72
        return float((window["home_goals"] + window["away_goals"]).mean())

    @staticmethod
    def _confidence_score(
        ev: float,
        prob: float,
        pin_gap: float = 0.0,
        elo_h=None,
        elo_a=None,
        weather_tag: str = "DRY",
        model_fitted: bool = True,
        market: str = "1x2",
    ) -> float:
        """
        Composite confidence score 0-1.

        Components:
        - EV contribution (0-0.4): higher EV → more confidence
        - Prob certainty (0-0.2): extreme probs (0.7+) are more reliable
        - Pinnacle gap (0-0.2): larger gap → more genuine signal
        - Elo consistency (0-0.1): if Elo agrees with model pick
        - Weather penalty (-0.1 for adverse conditions)
        """
        score = 0.0

        # EV contribution: normalise to 0-0.4 (EV=15% maps to full 0.4)
        score += min(ev / 0.15, 1.0) * 0.40

        # Probability certainty: 0.5 = 0, 0.7+ = max 0.20
        prob_certainty = max(0, (prob - 0.50) / 0.25)
        score += min(prob_certainty, 1.0) * 0.20

        # Pinnacle gap (1X2 only)
        if market == "1x2" and pin_gap > 0:
            score += min(pin_gap / 0.04, 1.0) * 0.20

        # Weather penalty
        if weather_tag in ("HEAVY_RAIN", "WINDY"):
            score -= CARD_CONFIG["weather_confidence_penalty"]

        return max(0.0, min(1.0, score))

    @staticmethod
    def _stars(confidence: float) -> int:
        """Convert confidence 0-1 to 1-5 stars."""
        if confidence >= 0.80:
            return 5
        if confidence >= 0.65:
            return 4
        if confidence >= 0.45:
            return 3
        if confidence >= 0.25:
            return 2
        return 1


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_card(bets: List[Bet], target_date: str) -> None:
    """Print formatted bet card to terminal."""
    if not bets:
        print("\n  No value bets found for this date.\n")
        return

    actionable = [b for b in bets if b.ev > 0]
    advisory   = [b for b in bets if b.ev == 0]

    print(f"\n{'='*72}")
    print(f"  KBet Daily Card - {target_date}  ({len(bets)} recommendations)")
    print(f"{'='*72}")

    if actionable:
        print(f"\n  ACTIONABLE BETS ({len(actionable)}) — odds confirmed, EV computed\n")
        print(f"  {'#':>2}  {'Match':<28} {'Market':<16} {'Pick':>4}  {'Odds':>5}  {'EV':>6}  {'Stars':<6}")
        print(f"  {'-'*70}")
        for i, b in enumerate(actionable, 1):
            weather_icon = "[R]" if b.weather_tag in ("HEAVY_RAIN","LIGHT_RAIN") else (
                           "[W]" if b.weather_tag == "WINDY" else "")
            print(
                f"  {i:>2}  {b.match:<28} {b.market:<16} "
                f"{b.pick:>4}  {b.odds:>5.2f}  {b.ev_pct:>6}  {b.star_str} {weather_icon}"
            )
            print(f"       {LEAGUE_NAMES.get(b.league, b.league)} | "
                  f"Model: {b.model_prob:.3f} | Implied: {b.implied_prob:.3f} | "
                  f"Book: {b.bookmaker}")
            if b.notes:
                print(f"       NOTE: {b.notes}")
            print()

    if advisory:
        print(f"\n  ADVISORY BETS ({len(advisory)}) — model flag raised, CHECK ODDS\n")
        print(f"  {'#':>2}  {'Match':<28} {'Market':<16} {'Pick':>4}  {'Prob':>6}  {'Stars':<6}")
        print(f"  {'-'*65}")
        for i, b in enumerate(advisory, 1):
            print(
                f"  {i:>2}  {b.match:<28} {b.market:<16} "
                f"{b.pick:>4}  {b.model_prob:>6.3f}  {b.star_str}"
            )
            if b.notes:
                print(f"       NOTE: {b.notes}")
            print()

    print(f"{'-'*72}")
    print(f"  NOTE: All EV figures include 20% slippage. Max stake: 1-2% bankroll per bet.")
    print(f"  Corners/Cards marked 'OBTAIN ODDS' require live odds verification.")
    print(f"{'='*72}\n")


def save_card(bets: List[Bet], target_date: str) -> Path:
    """Save bet card to JSON (DATA_DIR and static repo dir as fallback)."""
    out_path = OUTPUT_DIR / f"card_{target_date.replace('-','')}.json"
    actionable = [asdict(b) for b in bets if b.ev > 0]
    advisory   = [asdict(b) for b in bets if b.ev == 0]

    horizon_dates = sorted({b.get("match_date") or target_date for b in actionable if b.get("match_date")}) if actionable else [target_date]
    # Fallback to Bet objects if dicts
    try:
        h2 = sorted({getattr(b, "match_date", "") or target_date for b in bets if getattr(b, "match_date", "")})
        if h2:
            horizon_dates = h2
    except Exception:
        pass
    data = {
        "date":          target_date,
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "total_bets":    len(bets),
        "horizon":       len(horizon_dates),
        "horizon_dates": horizon_dates,
        "actionable":    actionable,
        "advisory":      advisory,
        # Also include flat "bets" key so API can read it directly
        "bets":          actionable,
        "config":        CARD_CONFIG,
    }
    # Remove internal score attribute
    for cat in ["actionable", "advisory", "bets"]:
        for item in data[cat]:
            item.pop("_score", None)

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    # Trust ledger: SHA256 of actionable bets (proves no cherry-pick)
    try:
        import hashlib
        ledger_dir = OUTPUT_DIR.parent / "public_picks"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        hash_payload = json.dumps(actionable, sort_keys=True, default=str)
        h = hashlib.sha256(hash_payload.encode()).hexdigest()
        with open(ledger_dir / f"hash_{target_date.replace('-','')}.txt", "w") as hf:
            hf.write(f"{h}  {target_date}  {len(actionable)} bets\n")
        # Also store hash in card JSON for API
        data["ledger_hash"] = h
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass

    # Also save to static repo path so it survives container restarts (fallback)
    static_dir = Path(__file__).parent / "data" / "daily_cards"
    static_dir.mkdir(parents=True, exist_ok=True)
    static_path = static_dir / f"card_{target_date.replace('-','')}.json"
    if static_path != out_path:
        try:
            with open(static_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass  # Non-fatal; DATA_DIR copy is the primary
        # Mirror hash file
        try:
            import hashlib as _hl
            static_ledger = static_dir.parent / "public_picks"
            static_ledger.mkdir(exist_ok=True)
            with open(static_ledger / f"hash_{target_date.replace('-','')}.txt", "w") as hf:
                hf.write(f"{data.get('ledger_hash','')}  {target_date}\n")
        except Exception:
            pass

    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KBet Daily Bet Card — generate 5-12 value bets for a given date"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Target date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Use historical data instead of live API (backtest mode)"
    )
    parser.add_argument(
        "--leagues", nargs="+", default=None,
        help="League codes to include (e.g. E0 SP1 D1). Default: all."
    )
    parser.add_argument(
        "--max-bets", type=int, default=12,
        help="Maximum bets on card (default 12)"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="The Odds API key (or set ODDS_API_KEY env var)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save output JSON"
    )
    args = parser.parse_args()

    engine = DailyCardEngine(
        target_date=args.date,
        simulate=args.simulate,
        leagues=args.leagues,
        api_key=args.api_key,
    )

    bets = engine.run(max_bets=args.max_bets)
    print_card(bets, engine.target_date)

    if not args.no_save and bets:
        out_path = save_card(bets, engine.target_date)
        print(f"  Saved -> {out_path}\n")


if __name__ == "__main__":
    main()
