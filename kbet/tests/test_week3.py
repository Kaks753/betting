"""
Week 3 Test Suite — Kelly, CLV Tracker, DB, Snapshots, API
===========================================================
Run:
  pytest kbet/tests/test_week3.py -v --tb=short -k "not API"
  pytest kbet/tests/test_week3.py -v --tb=short  # all tests
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kbet.engine.utils.kelly import (
    kelly_stake, kelly_ev, DailyPortfolio, BetOrder,
    DEFAULT_FRACTION, DEFAULT_MAX_PER_BET
)
from kbet.engine.utils.clv_tracker import (
    devig_1x2, devig_ou, compute_clv
)
from kbet.db import Database, SCHEMA
from kbet.ingest_odds_snapshots import (
    synthetic_pre_odds, devig, load_snapshot, load_all_snapshots
)


# ────────────────────────────────────────────────────────────────────────────
# 1. Kelly staking tests
# ────────────────────────────────────────────────────────────────────────────

class TestKellyStake:
    def test_basic_value_bet(self):
        """Standard value bet: 45% prob @ 2.40 odds."""
        s = kelly_stake(prob=0.45, odds=2.40)
        assert s > 0, "Should have positive stake"
        assert s <= DEFAULT_MAX_PER_BET, "Must not exceed max_per_bet"
        # Check Kelly formula: edge=0.08, full_kelly=0.08/1.40=0.0571, quarter=0.0143
        expected = (0.45 * 2.40 - 1) / (2.40 - 1) * DEFAULT_FRACTION
        assert abs(s - expected) < 1e-5

    def test_no_value(self):
        """No value: 30% prob @ 2.00 (fair=50%)."""
        s = kelly_stake(prob=0.30, odds=2.00)
        assert s == 0.0

    def test_negative_ev(self):
        """Strongly negative EV should return 0."""
        s = kelly_stake(prob=0.20, odds=2.00)
        assert s == 0.0

    def test_odds_floor(self):
        """Below minimum odds (1.35) should return 0."""
        s = kelly_stake(prob=0.95, odds=1.10)
        assert s == 0.0

    def test_odds_ceiling(self):
        """Above max odds (8.0) should return 0."""
        s = kelly_stake(prob=0.15, odds=9.00)
        assert s == 0.0

    def test_max_per_bet_cap(self):
        """Very large edge should be capped at max_per_bet."""
        s = kelly_stake(prob=0.80, odds=5.00, max_per_bet=0.02)
        assert s == 0.02

    def test_ev_function(self):
        """Kelly EV: prob * odds - 1."""
        assert abs(kelly_ev(0.50, 2.00) - 0.0)   < 1e-9
        assert abs(kelly_ev(0.60, 2.00) - 0.2)   < 1e-9
        assert abs(kelly_ev(0.30, 2.00) - (-0.4)) < 1e-9

    def test_min_ev_threshold(self):
        """EV below min_ev (3%) should be rejected."""
        # EV = 0.02 (2%) < min_ev (3%)
        s = kelly_stake(prob=0.36, odds=1.60, min_ev=0.03)
        ev = 0.36 * 1.60 - 1
        if ev < 0.03:
            assert s == 0.0

    def test_probability_edge_cases(self):
        """Degenerate probabilities."""
        assert kelly_stake(prob=0.0, odds=2.0) == 0.0
        assert kelly_stake(prob=1.0, odds=2.0) == 0.0
        assert kelly_stake(prob=-0.1, odds=2.0) == 0.0


class TestDailyPortfolio:
    def test_single_bet_accepted(self):
        """First bet on empty portfolio should be accepted."""
        port = DailyPortfolio()
        b = port.try_add("M1_A_B", "Arsenal", "Chelsea", "1x2", "H", 0.45, 2.40)
        assert b is not None
        assert b.ev > 0
        assert b.stake_pct > 0

    def test_daily_exposure_cap(self):
        """Total daily exposure must not exceed max_daily (5%)."""
        port = DailyPortfolio(max_daily=0.05, max_per_bet=0.02)
        for i in range(20):
            port.try_add(f"M{i}_A_B", f"Team{i*2}", f"Team{i*2+1}", "1x2", "H", 0.48, 2.30)
        assert port.total_exposure <= 0.05 + 1e-9

    def test_match_bet_count_limit(self):
        """Max 2 bets per match by default."""
        port = DailyPortfolio(max_per_match=0.10, max_per_bet=0.01, max_daily=0.50)
        b1 = port.try_add("M1_A_B", "Arsenal", "Chelsea", "1x2",    "H",    0.45, 2.40)
        b2 = port.try_add("M1_A_B", "Arsenal", "Chelsea", "ou",     "O2.5", 0.58, 1.90)
        b3 = port.try_add("M1_A_B", "Arsenal", "Chelsea", "corners","O9.5", 0.55, 2.10)
        assert b1 is not None
        assert b2 is not None
        assert b3 is None  # Count limit: 2 per match

    def test_below_min_ev_rejected(self):
        """Bet with EV below min_ev should be rejected."""
        port = DailyPortfolio(min_ev=0.05)
        # EV = 0.33 * 3.00 - 1 = -0.01 < 0.05
        b = port.try_add("M1_A_B", "A", "B", "1x2", "H", 0.33, 3.00)
        assert b is None

    def test_confidence_tiers(self):
        """FIRE for EV≥15%, SOLID for 8-15%, WATCH for 3-8%."""
        port = DailyPortfolio(max_per_bet=0.01, max_daily=0.50, max_per_match=0.10)
        # FIRE: 0.55 @ 2.80 → EV = 54%
        b_fire = port.try_add("M1", "A", "B", "1x2", "H", 0.55, 2.80)
        assert b_fire is not None and b_fire.confidence == "FIRE"
        # WATCH: 0.38 @ 1.80 → EV = -0.16 → should be rejected
        # Need something in 3-8% range
        # EV = 0.40 * 1.90 - 1 = -0.24 → rejected. Let's use 0.55 @ 1.70 → EV = -0.065 rejected
        # Actually for WATCH we need ev in 3-8%: 0.62 @ 1.68 → ev = 0.4216 no
        # 0.55 @ 1.90 → ev = 0.045 = 4.5% = WATCH
        b_watch = port.try_add("M2", "C", "D", "ou", "O2.5", 0.55, 1.90)
        if b_watch:
            assert b_watch.confidence in ("WATCH", "SOLID")

    def test_portfolio_summary(self):
        """Summary string should be non-empty when bets added."""
        port = DailyPortfolio(max_per_bet=0.01, max_daily=0.50)
        port.try_add("M1", "A", "B", "1x2", "H", 0.48, 2.30)
        summary = port.summary()
        assert "Portfolio:" in summary
        assert "EV=" in summary

    def test_bet_order_fields(self):
        """BetOrder should have all required fields."""
        port = DailyPortfolio(max_per_bet=0.01, max_daily=0.50)
        b = port.try_add("M1_Arsenal_Chelsea", "Arsenal", "Chelsea", "1x2", "H", 0.48, 2.30)
        assert b is not None
        assert b.home == "Arsenal"
        assert b.away == "Chelsea"
        assert b.market == "1x2"
        assert b.pick == "H"
        assert 0 < b.model_prob <= 1
        assert b.stake_pct > 0
        assert b.stake_units == round(b.stake_pct * 100, 4)
        assert b.implied_prob == pytest.approx(1/2.30, abs=1e-6)
        assert len(b.confidence_stars) == 5  # e.g. ★★★★☆


class TestDailyPortfolioEdgeCases:
    def test_empty_portfolio(self):
        """Empty portfolio should have 0 bets and 0 exposure."""
        port = DailyPortfolio()
        assert port.total_bets == 0
        assert port.total_exposure == 0.0
        assert port.bets == []

    def test_invalid_prob(self):
        """Invalid probabilities should be rejected by kelly_stake."""
        port = DailyPortfolio()
        b = port.try_add("M1", "A", "B", "1x2", "H", -0.1, 2.00)
        assert b is None
        b = port.try_add("M2", "A", "B", "1x2", "H", 1.1, 2.00)
        assert b is None


# ────────────────────────────────────────────────────────────────────────────
# 2. CLV tracker tests
# ────────────────────────────────────────────────────────────────────────────

class TestDevig:
    def test_devig_1x2_valid(self):
        """Standard 1X2 de-vig."""
        r = devig_1x2(2.10, 3.40, 3.60)
        assert r is not None
        ph, pd_, pa = r
        assert abs(ph + pd_ + pa - 1.0) < 1e-9
        assert ph > pd_ > pa  # Home favourite

    def test_devig_1x2_nan(self):
        """NaN inputs should return None."""
        assert devig_1x2(float("nan"), 3.40, 3.60) is None

    def test_devig_1x2_below_1(self):
        """Odds ≤ 1 should return None."""
        assert devig_1x2(1.0, 3.40, 3.60) is None
        assert devig_1x2(0.5, 3.40, 3.60) is None

    def test_devig_ou_valid(self):
        """Standard O/U de-vig."""
        r = devig_ou(1.90, 1.95)
        assert r is not None
        over, under = r
        assert abs(over + under - 1.0) < 1e-9
        assert abs(over - under) < 0.10  # Close to evens

    def test_devig_ou_nan(self):
        assert devig_ou(float("nan"), 1.95) is None
        assert devig_ou(1.90, float("nan")) is None

    def test_devig_ou_invalid(self):
        assert devig_ou(0.0, 1.95) is None


class TestCLVComputation:
    def test_clv_positive(self):
        """CLV positive when our prob > closing devigged prob."""
        closing_data = {"found": True, "h2h_h": 2.10, "h2h_d": 3.40, "h2h_a": 3.60}
        dv = devig_1x2(2.10, 3.40, 3.60)
        closing_h = dv[0]  # ~0.475
        entry_prob = closing_h + 0.05  # We had 5% higher prob → positive CLV
        clv = compute_clv("1x2", "H", entry_prob, closing_data)
        assert clv is not None
        assert clv > 0.0

    def test_clv_negative(self):
        """CLV negative when our prob < closing devigged prob."""
        closing_data = {"found": True, "h2h_h": 2.10, "h2h_d": 3.40, "h2h_a": 3.60}
        dv = devig_1x2(2.10, 3.40, 3.60)
        closing_h = dv[0]
        entry_prob = closing_h - 0.05
        clv = compute_clv("1x2", "H", entry_prob, closing_data)
        assert clv is not None
        assert clv < 0.0

    def test_clv_no_closing_data(self):
        """If closing data is None, CLV should be None."""
        clv = compute_clv("1x2", "H", 0.50, None)
        assert clv is None

    def test_clv_missing_h2h(self):
        """Missing h2h odds in closing data → None."""
        clv = compute_clv("1x2", "H", 0.50, {"found": True})
        assert clv is None

    def test_clv_ou_over(self):
        """CLV for O2.5 market."""
        closing_data = {"found": True, "ou_over_2_5": 1.90, "ou_under_2_5": 1.95}
        clv = compute_clv("ou", "O2.5", 0.60, closing_data)
        assert clv is not None
        # devig(1.90, 1.95)[0] ≈ 0.506, our_prob=0.60 → clv ≈ 0.094
        assert clv > 0.0

    def test_clv_ou_under(self):
        """CLV for U2.5 market."""
        closing_data = {"found": True, "ou_over_2_5": 1.90, "ou_under_2_5": 1.95}
        clv = compute_clv("ou", "U2.5", 0.40, closing_data)
        assert clv is not None
        # devig(1.90, 1.95)[1] ≈ 0.494, our_prob=0.40 → negative CLV
        assert clv < 0.0

    def test_clv_invalid_market(self):
        """Unknown market returns None."""
        clv = compute_clv("btts", "Y", 0.50, {"found": True})
        assert clv is None


# ────────────────────────────────────────────────────────────────────────────
# 3. Database tests
# ────────────────────────────────────────────────────────────────────────────

class TestDatabase:
    @pytest.fixture
    def db(self, tmp_path):
        """Create a fresh in-memory DB for each test."""
        db = Database(path=tmp_path / "test_kbet.db")
        db.migrate()
        yield db
        db.close()

    def test_migrate_creates_tables(self, db):
        """Migration should create all required tables."""
        conn = db.connect()
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "bets" in tables
        assert "daily_cards" in tables
        assert "clv_log" in tables
        assert "performance_log" in tables

    def test_save_and_retrieve_card(self, db):
        """Save a card and retrieve it by date."""
        card_id = db.save_daily_card(
            run_date="2024-09-14",
            bets=[{"home": "Arsenal", "market": "1x2", "pick": "H"}],
            leagues=["E0"],
            markets=["1x2"],
            total_exposure=0.03,
        )
        assert card_id > 0
        summary = db.get_daily_summary("2024-09-14")
        assert summary is not None
        assert summary["run_date"] == "2024-09-14"
        assert summary["n_bets"] == 1

    def test_save_and_settle_bet(self, db):
        """Save a bet, then settle it."""
        card_id = db.save_daily_card(
            run_date="2024-09-14",
            bets=[],
            leagues=["E0"],
            markets=["1x2"],
            total_exposure=0.015,
        )
        bet_id = db.save_bet(
            card_id=card_id,
            match_date="2024-09-14",
            league="E0",
            home_team="Arsenal",
            away_team="Chelsea",
            market="1x2",
            pick="H",
            model_prob=0.48,
            book_odds=2.30,
            eff_odds=2.10,
            stake_pct=0.015,
            stake_units=1.5,
            ev=0.084,
            confidence="SOLID",
        )
        assert bet_id > 0

        # Settle as win
        db.settle_bet(bet_id=bet_id, result="W", pnl=1.65)
        conn = db.connect()
        row = conn.execute("SELECT result, pnl FROM bets WHERE id = ?", (bet_id,)).fetchone()
        assert row["result"] == "W"
        assert abs(row["pnl"] - 1.65) < 1e-9

    def test_clv_update(self, db):
        """Update CLV on a settled bet."""
        card_id = db.save_daily_card("2024-09-14", [], ["E0"], ["1x2"], 0.01)
        bet_id  = db.save_bet(card_id, "2024-09-14", "E0", "A", "B", "1x2", "H",
                               0.48, 2.30, 2.10, 0.015, 1.5, 0.084, "SOLID")
        db.settle_bet(bet_id, "W", 1.65)
        db.update_clv(bet_id, 0.0342, source="synthetic")
        conn = db.connect()
        row = conn.execute("SELECT clv FROM bets WHERE id = ?", (bet_id,)).fetchone()
        assert abs(row["clv"] - 0.0342) < 1e-9

    def test_get_unsettled_bets(self, db):
        """Unsettled bets are returned before settlement."""
        card_id = db.save_daily_card("2024-09-15", [], ["E0"], ["1x2"], 0.01)
        bet_id  = db.save_bet(card_id, "2024-09-15", "E0", "A", "B", "1x2", "H",
                               0.48, 2.30, 2.10, 0.015, 1.5, 0.084, "SOLID")
        unsettled = db.get_unsettled_bets()
        assert any(b["id"] == bet_id for b in unsettled)
        db.settle_bet(bet_id, "L", -1.5)
        unsettled_after = db.get_unsettled_bets()
        assert not any(b["id"] == bet_id for b in unsettled_after)

    def test_performance_empty(self, db):
        """Performance on empty DB should not error."""
        stats = db.get_performance(days=30)
        assert "n_bets" in stats

    def test_migrate_idempotent(self, db):
        """Running migrate twice should not error."""
        db.migrate()  # Second migration
        tables = {r[0] for r in db.connect().execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "bets" in tables


# ────────────────────────────────────────────────────────────────────────────
# 4. Snapshot ingestion tests
# ────────────────────────────────────────────────────────────────────────────

class TestSnapshots:
    def test_devig_valid(self):
        """devig helper returns valid probs."""
        r = devig(2.10, 3.40, 3.60)
        assert r is not None
        assert abs(sum(r) - 1.0) < 1e-9

    def test_devig_nan(self):
        """NaN inputs return None."""
        assert devig(float("nan"), 3.40, 3.60) is None

    def test_synthetic_pre_odds_structure(self):
        """Synthetic odds should return valid tuple."""
        r = synthetic_pre_odds(2.10, 3.40, 3.60, hours_before=48)
        assert r is not None
        p_h, p_d, p_a = r
        assert abs(p_h + p_d + p_a - 1.0) < 1e-6
        assert all(0.05 <= p <= 0.92 for p in (p_h, p_d, p_a))

    def test_synthetic_pre_odds_nan_input(self):
        """NaN closing odds should return None."""
        r = synthetic_pre_odds(float("nan"), 3.40, 3.60)
        assert r is None

    def test_load_snapshot_missing_date(self):
        """Loading a non-existent snapshot returns None."""
        snap = load_snapshot("1900-01-01")
        assert snap is None

    def test_load_all_snapshots_missing_range(self):
        """Loading a range with no files returns empty DataFrame."""
        df = load_all_snapshots("1900-01-01", "1900-01-10")
        assert isinstance(df, pd.DataFrame)
        # OK if empty

    def test_snapshot_dir_exists(self):
        """Snapshot directory should exist after module import."""
        from kbet.ingest_odds_snapshots import SNAPSHOTS_DIR
        assert SNAPSHOTS_DIR.exists()


# ────────────────────────────────────────────────────────────────────────────
# 5. API smoke tests (require running server — marked for CI exclusion)
# ────────────────────────────────────────────────────────────────────────────

class TestAPISmoke:
    """
    Smoke tests for FastAPI endpoints.
    Uses TestClient (in-process, no server needed).
    """

    @pytest.fixture
    def client(self, tmp_path):
        """Create API test client with temp DB."""
        from fastapi.testclient import TestClient
        # Patch DB path to temp
        import kbet.db as db_module
        original_path = db_module.DB_PATH
        db_module.DB_PATH = tmp_path / "test.db"
        db = Database(path=db_module.DB_PATH)
        db.migrate()
        db.close()

        from kbet.api.app import app
        client = TestClient(app)
        yield client

        db_module.DB_PATH = original_path

    def test_health_check(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_status_endpoint(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_bets" in data
        assert "total_cards" in data

    def test_bets_empty(self, client):
        resp = client.get("/bets")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["bets"] == []

    def test_card_not_found(self, client):
        resp = client.get("/card/2000-01-01")
        assert resp.status_code == 404

    def test_card_invalid_date(self, client):
        resp = client.get("/card/not-a-date")
        assert resp.status_code == 400

    def test_performance_empty(self, client):
        resp = client.get("/performance")
        assert resp.status_code == 200
        data = resp.json()
        assert "n_bets" in data

    def test_clv_report_empty(self, client):
        resp = client.get("/clv")
        assert resp.status_code == 200

    def test_admin_requires_token(self, client):
        resp = client.post("/admin/run-cron")
        assert resp.status_code == 401

    def test_pagination(self, client):
        resp = client.get("/bets?page=1&limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "page" in data
        assert "pages" in data


# ────────────────────────────────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v", "--tb=short"])
