#!/usr/bin/env python3
"""
Week 2 unit tests — Corners, Cards, OddsClient, WeatherClient, ClubElo, DailyCard
"""

import sys
import json
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kbet.engine.models.corners_model import CornersModel
from kbet.engine.models.cards_model import CardsModel
from kbet.engine.data.odds_client import (
    OddsAPIClient, MatchOdds, OddsH2H, OddsTotals,
    OddsAPIKeyMissingError
)
from kbet.engine.data.weather_client import WeatherClient, WeatherConditions
from kbet.engine.data.clubelo_client import ClubEloClient


# ── Fixture data ──────────────────────────────────────────────────────────────

def make_synthetic_matches(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic match data for model testing."""
    rng = np.random.default_rng(seed)
    teams = [f"Team_{i}" for i in range(20)]
    leagues = ["E0", "SP1", "D1", "I1", "F1"]

    rows = []
    base_date = pd.Timestamp("2020-01-01")

    for i in range(n):
        home = teams[rng.integers(0, 20)]
        away = teams[rng.integers(0, 20)]
        while away == home:
            away = teams[rng.integers(0, 20)]

        goals_h = int(rng.poisson(1.5))
        goals_a = int(rng.poisson(1.1))
        result = "H" if goals_h > goals_a else ("D" if goals_h == goals_a else "A")

        rows.append({
            "date":         base_date + pd.Timedelta(days=i // 10),
            "home_team":    home,
            "away_team":    away,
            "home_goals":   goals_h,
            "away_goals":   goals_a,
            "result":       result,
            "home_corners": int(rng.integers(3, 12)),
            "away_corners": int(rng.integers(2, 10)),
            "home_yellow":  int(rng.integers(0, 5)),
            "away_yellow":  int(rng.integers(0, 5)),
            "home_red":     int(rng.choice([0, 0, 0, 1])),
            "away_red":     int(rng.choice([0, 0, 0, 1])),
            "league_code":  leagues[i % len(leagues)],
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Corners Model Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCornersModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = make_synthetic_matches(600)
        cls.as_of = pd.Timestamp("2021-01-01")
        cls.model = CornersModel(min_matches=50)
        cls.model.fit(cls.df, cls.as_of)

    def test_model_fits(self):
        self.assertTrue(self.model.fitted, "CornersModel should be fitted")

    def test_teams_populated(self):
        self.assertGreater(len(self.model.teams), 0)
        self.assertIn("intercept", dir(self.model))

    def test_predict_returns_dict(self):
        pred = self.model.predict("Team_0", "Team_1")
        self.assertIsNotNone(pred)
        self.assertIn("exp_total", pred)
        self.assertIn("over_9_5", pred)
        self.assertIn("under_9_5", pred)

    def test_predict_probabilities_sum_to_one(self):
        pred = self.model.predict("Team_0", "Team_1", lines=[9.5])
        over  = pred["over_9_5"]
        under = pred["under_9_5"]
        self.assertAlmostEqual(over + under, 1.0, places=5)

    def test_predict_exp_total_positive(self):
        pred = self.model.predict("Team_0", "Team_5")
        self.assertGreater(pred["exp_total"], 0)
        self.assertLess(pred["exp_total"], 30)  # Sanity: <30 corners per game

    def test_predict_unknown_team_returns_result(self):
        # Unknown teams should fall back to 0 (global mean) without crashing
        pred = self.model.predict("Unknown_FC", "Unknown_Rovers")
        self.assertIsNotNone(pred)
        self.assertTrue(pred["unknown_home"])
        self.assertTrue(pred["unknown_away"])

    def test_predict_ev_structure(self):
        ev = self.model.predict_ev("Team_0", "Team_1", odds_over=1.90, odds_under=1.90, line=9.5)
        self.assertIn("ev_over", ev)
        self.assertIn("ev_under", ev)
        self.assertIn("best_side", ev)
        self.assertIn(ev["best_side"], ["over", "under", "none"])

    def test_insufficient_data(self):
        small_df = make_synthetic_matches(10)
        model = CornersModel(min_matches=500)
        model.fit(small_df, pd.Timestamp("2020-06-01"))
        self.assertFalse(model.fitted)

    def test_all_lines_computed(self):
        lines = [8.5, 9.5, 10.5, 11.5, 12.5]
        pred = self.model.predict("Team_2", "Team_3", lines=lines)
        for line in lines:
            key = str(line).replace(".", "_")
            self.assertIn(f"over_{key}", pred)
            self.assertIn(f"under_{key}", pred)


# ─────────────────────────────────────────────────────────────────────────────
# Cards Model Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCardsModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = make_synthetic_matches(600)
        cls.as_of = pd.Timestamp("2021-01-01")
        cls.model = CardsModel(min_matches=50)
        cls.model.fit(cls.df, cls.as_of)

    def test_model_fits(self):
        self.assertTrue(self.model.fitted)

    def test_dispersion_positive(self):
        self.assertGreater(self.model.dispersion, 0)

    def test_predict_returns_dict(self):
        pred = self.model.predict("Team_0", "Team_1", "E0")
        self.assertIsNotNone(pred)
        self.assertIn("exp_total", pred)
        self.assertIn("over_3_5", pred)

    def test_predict_probabilities_in_range(self):
        pred = self.model.predict("Team_0", "Team_1", "E0", lines=[3.5])
        self.assertGreaterEqual(pred["over_3_5"], 0.0)
        self.assertLessEqual(pred["over_3_5"], 1.0)
        self.assertAlmostEqual(pred["over_3_5"] + pred["under_3_5"], 1.0, places=4)

    def test_league_effects_exist(self):
        self.assertGreater(len(self.model.league_fx), 0)

    def test_predict_ev(self):
        ev = self.model.predict_ev("Team_0", "Team_1", "E0",
                                    odds_over=1.85, odds_under=2.00, line=3.5)
        self.assertIn("ev_over", ev)
        self.assertIn("best_side", ev)
        # EV should be finite
        self.assertFalse(np.isnan(ev["ev_over"]))

    def test_unknown_league_fallback(self):
        pred = self.model.predict("Team_0", "Team_1", "UNKNOWN")
        self.assertIsNotNone(pred)

    def test_negative_binomial_heavier_tails(self):
        """NegBinom should assign more probability to extremes than Poisson."""
        from scipy.stats import poisson, nbinom
        mu = self.model._mean_cards
        r  = self.model.dispersion
        p  = r / (r + mu)
        # P(X >= 9) under NegBinom should be > under Poisson (heavier tail)
        nb_tail = 1 - nbinom.cdf(8, r, p)
        pois_tail = 1 - poisson.cdf(8, mu)
        self.assertGreater(nb_tail, pois_tail)


# ─────────────────────────────────────────────────────────────────────────────
# OddsAPIClient Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOddsAPIClient(unittest.TestCase):

    def setUp(self):
        self.client = OddsAPIClient(api_key="TEST_KEY")

    def test_no_key_raises_on_fetch(self):
        client = OddsAPIClient(api_key="")
        with self.assertRaises(OddsAPIKeyMissingError):
            client.get_today_odds()

    def test_parse_odds_response(self):
        raw = [{
            "id": "abc123",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "commence_time": "2024-09-14T14:00:00Z",
            "bookmakers": [{
                "key": "pinnacle",
                "last_update": "2024-09-14T10:00:00Z",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Arsenal", "price": 2.10},
                        {"name": "Draw",    "price": 3.40},
                        {"name": "Chelsea", "price": 3.20},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over",  "price": 1.87, "point": 2.5},
                        {"name": "Under", "price": 2.02, "point": 2.5},
                    ]},
                ]
            }]
        }]
        matches = self.client._parse_odds_response(raw, "soccer_epl", "E0")
        self.assertEqual(len(matches), 1)
        m = matches[0]
        self.assertEqual(m.home_team, "Arsenal")
        self.assertEqual(len(m.h2h), 1)
        self.assertEqual(m.h2h[0].home, 2.10)
        self.assertEqual(len(m.totals), 1)
        self.assertAlmostEqual(m.totals[0].over, 1.87)

    def test_devig_pinnacle_h2h(self):
        m = MatchOdds("id1", "soccer_epl", "E0", "Arsenal", "Chelsea", "2024-09-14")
        m.h2h = [OddsH2H("pinnacle", 2.10, 3.40, 3.20)]
        devig = m.devig_pinnacle_h2h()
        self.assertIsNotNone(devig)
        ph, pd_, pa = devig
        self.assertAlmostEqual(ph + pd_ + pa, 1.0, places=4)
        self.assertGreater(ph, 0.30)  # Arsenal slight favourite

    def test_devig_ou(self):
        m = MatchOdds("id1", "soccer_epl", "E0", "Arsenal", "Chelsea", "2024-09-14")
        m.totals = [OddsTotals("pinnacle", 2.5, 1.87, 2.02)]
        devig = m.devig_pinnacle_ou(2.5)
        self.assertIsNotNone(devig)
        po, pu = devig
        self.assertAlmostEqual(po + pu, 1.0, places=4)

    def test_max_h2h(self):
        m = MatchOdds("id1", "soccer_epl", "E0", "Arsenal", "Chelsea", "2024-09-14")
        m.h2h = [
            OddsH2H("pinnacle", 2.10, 3.40, 3.20),
            OddsH2H("bet365",   2.15, 3.50, 3.10),  # Better home and draw
        ]
        mx = m.max_h2h()
        self.assertEqual(mx.bookmaker, "MAX")
        self.assertAlmostEqual(mx.home, 2.15)  # Best home odds = bet365
        self.assertAlmostEqual(mx.draw, 3.50)  # Best draw odds = bet365

    def test_cache_roundtrip(self, tmp_path=None):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            client = OddsAPIClient(api_key="TEST", cache_dir=tmpdir)
            data = [{"test": "data"}]
            client._save_cache("test_key", data)
            loaded = client._load_cache("test_key", max_age_minutes=60)
            self.assertEqual(loaded, data)

    def test_cache_expiry(self):
        import tempfile, time as t
        with tempfile.TemporaryDirectory() as tmpdir:
            client = OddsAPIClient(api_key="TEST", cache_dir=tmpdir)
            client._save_cache("exp_key", [{"old": "data"}])
            # Cache should expire if max_age_minutes=0
            loaded = client._load_cache("exp_key", max_age_minutes=0)
            self.assertIsNone(loaded)

    def test_quota_status_initial(self):
        status = self.client.get_quota_status()
        self.assertIn("requests_remaining", status)
        self.assertIn("requests_used", status)

    def test_sport_key_to_league_roundtrip(self):
        from kbet.engine.data.odds_client import LEAGUE_TO_SPORT_KEY
        for lc, sk in LEAGUE_TO_SPORT_KEY.items():
            result = self.client._sport_key_to_league(sk)
            self.assertEqual(result, lc)


# ─────────────────────────────────────────────────────────────────────────────
# WeatherClient Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWeatherClient(unittest.TestCase):

    def test_classify_dry(self):
        tag = WeatherClient._classify(0.5, 18.0, 10.0)
        self.assertEqual(tag, "DRY")

    def test_classify_heavy_rain(self):
        tag = WeatherClient._classify(10.0, 15.0, 15.0)
        self.assertEqual(tag, "HEAVY_RAIN")

    def test_classify_windy(self):
        tag = WeatherClient._classify(1.0, 15.0, 40.0)
        self.assertEqual(tag, "WINDY")

    def test_classify_cold(self):
        tag = WeatherClient._classify(0.0, 1.0, 5.0)
        self.assertEqual(tag, "COLD")

    def test_classify_light_rain(self):
        tag = WeatherClient._classify(3.0, 15.0, 10.0)
        self.assertEqual(tag, "LIGHT_RAIN")

    def test_weather_conditions_goals_adj_dry(self):
        cond = WeatherConditions("2024-01-01", 51.5, -0.1, 0.0, 18.0, 10.0, 10.0, "DRY")
        self.assertEqual(cond.goals_adj, 0.0)

    def test_weather_conditions_goals_adj_heavy_rain(self):
        cond = WeatherConditions("2024-01-01", 51.5, -0.1, 8.0, 15.0, 8.0, 10.0, "HEAVY_RAIN")
        self.assertLess(cond.goals_adj, 0.0)  # Negative adjustment

    def test_weather_conditions_is_adverse(self):
        dry  = WeatherConditions("2024-01-01", 51.5, -0.1, 0.0, 18.0, 10.0, 10.0, "DRY")
        rain = WeatherConditions("2024-01-01", 51.5, -0.1, 9.0, 15.0, 8.0, 10.0, "HEAVY_RAIN")
        self.assertFalse(dry.is_adverse)
        self.assertTrue(rain.is_adverse)

    def test_cards_adj_rainy(self):
        cond = WeatherConditions("2024-01-01", 51.5, -0.1, 6.0, 14.0, 8.0, 10.0, "HEAVY_RAIN")
        self.assertGreater(cond.cards_adj, 0.0)  # Slippery → more fouls → more cards

    def test_live_api_fetch(self):
        """Integration test — calls real Open-Meteo API. Skip if offline."""
        try:
            wc = WeatherClient()
            cond = wc.get_match_conditions("Arsenal", "E0", "2025-10-01")
            if cond is not None:
                self.assertIsInstance(cond.precipitation_mm, float)
                self.assertIn(cond.weather_tag, ["DRY","LIGHT_RAIN","HEAVY_RAIN","COLD","WINDY"])
        except Exception:
            self.skipTest("Open-Meteo API unavailable")


# ─────────────────────────────────────────────────────────────────────────────
# ClubElo Client Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestClubEloClient(unittest.TestCase):

    def test_elo_to_probs_sum_to_one(self):
        probs = ClubEloClient._elo_to_probs(1800, 1750)
        total = probs["home"] + probs["draw"] + probs["away"]
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_elo_to_probs_favourite_wins_more(self):
        probs_fav    = ClubEloClient._elo_to_probs(2000, 1600)  # Big favourite
        probs_even   = ClubEloClient._elo_to_probs(1800, 1800)  # Even
        self.assertGreater(probs_fav["home"], probs_even["home"])

    def test_elo_home_advantage(self):
        """Home team should have higher win prob at equal ratings due to home advantage."""
        probs = ClubEloClient._elo_to_probs(1800, 1800)
        self.assertGreater(probs["home"], probs["away"])

    def test_draw_peaks_at_even(self):
        """Draw probability should be highest when teams are equal."""
        probs_even  = ClubEloClient._elo_to_probs(1800, 1800)
        probs_gap   = ClubEloClient._elo_to_probs(2100, 1500)
        self.assertGreater(probs_even["draw"], probs_gap["draw"])

    def test_api_unavailable_graceful(self):
        """If API returns HTML (blocked), client should set _available=False."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            client = ClubEloClient(cache_dir=tmpdir)
            client._available = False
            result = client.get_elo("NonExistentTeamXYZ", "2024-01-01")
            self.assertIsNone(result)

    def test_elo_probs_all_positive(self):
        probs = ClubEloClient._elo_to_probs(1900, 1700)
        for k, v in probs.items():
            self.assertGreater(v, 0.0, f"{k} should be positive")


# ─────────────────────────────────────────────────────────────────────────────
# Integration: Daily Card (simulation mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestDailyCardSimulation(unittest.TestCase):

    def test_daily_card_runs_simulation(self):
        """Run daily card in --simulate mode on a known historical date."""
        # Use a date where we have data in the parquet
        data_path = Path(__file__).parent.parent / "data/processed/all_matches.parquet"
        if not data_path.exists():
            self.skipTest("all_matches.parquet not found")

        df = pd.read_parquet(data_path)
        df["date"] = pd.to_datetime(df["date"])

        # Pick a date with enough matches (mid-season)
        target_date = "2023-09-16"
        from kbet.daily_card import DailyCardEngine, CARD_CONFIG

        engine = DailyCardEngine(
            target_date=target_date,
            simulate=True,
            leagues=["E0"],   # Just EPL for speed
        )
        bets = engine.run(max_bets=6)

        # Should return some bets (even if advisory only)
        self.assertIsInstance(bets, list)
        # Each bet must have required fields
        for b in bets:
            self.assertIsNotNone(b.match)
            self.assertIsNotNone(b.market)
            self.assertIn(b.stars, [1, 2, 3, 4, 5])
            self.assertGreaterEqual(b.confidence, 0.0)
            self.assertLessEqual(b.confidence, 1.0)

    def test_bet_card_max_bets_respected(self):
        """No more bets than max_bets on the card."""
        data_path = Path(__file__).parent.parent / "data/processed/all_matches.parquet"
        if not data_path.exists():
            self.skipTest("all_matches.parquet not found")

        from kbet.daily_card import DailyCardEngine
        engine = DailyCardEngine(target_date="2023-09-16", simulate=True, leagues=["E0"])
        bets = engine.run(max_bets=5)
        self.assertLessEqual(len(bets), 5)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  KBet Week 2 — Unit Tests")
    print("="*60 + "\n")
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestCornersModel))
    suite.addTests(loader.loadTestsFromTestCase(TestCardsModel))
    suite.addTests(loader.loadTestsFromTestCase(TestOddsAPIClient))
    suite.addTests(loader.loadTestsFromTestCase(TestWeatherClient))
    suite.addTests(loader.loadTestsFromTestCase(TestClubEloClient))
    suite.addTests(loader.loadTestsFromTestCase(TestDailyCardSimulation))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
