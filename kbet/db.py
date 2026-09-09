"""
KBet Database — SQLite schema and helpers
==========================================
Single-file SQLite DB for all bet tracking, CLV, and reporting.
Stored at: kbet/data/kbet.db

Tables:
  bets          — all placed bets with entry odds, result, CLV, PnL
  daily_cards   — one row per daily card run (JSON of full card)
  clv_log       — CLV settlement audit log
  snapshots     — odds snapshots metadata (points to parquet files)

Usage:
  from kbet.db import Database
  db = Database()
  db.migrate()
  card_id = db.save_daily_card(date="2024-09-14", bets=[...])
  db.save_bet(card_id=card_id, bet=bet_order)
  db.settle_bet(bet_id=1, result="W", closing_odds={...})
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger(__name__)

# DB_PATH — support env var override for Fly.io persistent volume (/data)
# Default: kbet/data/kbet.db (local dev)
# Fly.io: set DB_PATH=/data/kbet.db in fly.toml → persists across deploys
_db_env = os.environ.get("DB_PATH")
if _db_env:
    DB_PATH = Path(_db_env)
else:
    DB_PATH = Path(__file__).parent / "data" / "kbet.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
-- All placed bets
CREATE TABLE IF NOT EXISTS bets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id         INTEGER REFERENCES daily_cards(id),
    placed_at       TEXT NOT NULL,          -- ISO8601 UTC
    match_date      TEXT NOT NULL,          -- YYYY-MM-DD
    league          TEXT,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    market          TEXT NOT NULL,          -- 1x2, ou, corners, cards
    pick            TEXT NOT NULL,          -- H, D, A, O2.5, U2.5, O9.5...
    model_prob      REAL NOT NULL,          -- our model's probability
    entry_prob      REAL,                   -- de-vigged entry (reference) prob
    book_odds       REAL NOT NULL,          -- decimal odds at entry
    eff_odds        REAL NOT NULL,          -- post-slippage decimal odds
    stake_pct       REAL NOT NULL,          -- fraction of bankroll
    stake_units     REAL NOT NULL,          -- stake_pct * 100
    ev              REAL NOT NULL,          -- expected value at entry
    confidence      TEXT,                   -- FIRE / SOLID / WATCH
    result          TEXT,                   -- W / L / V(oid) — filled post-match
    pnl             REAL,                   -- profit/loss in units
    clv             REAL,                   -- closing line value (filled later)
    closing_odds_h  REAL,
    closing_odds_d  REAL,
    closing_odds_a  REAL,
    settled_at      TEXT,
    notes           TEXT
);

-- Daily card runs
CREATE TABLE IF NOT EXISTS daily_cards (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date        TEXT NOT NULL,          -- YYYY-MM-DD
    generated_at    TEXT NOT NULL,          -- ISO8601 UTC
    n_bets          INTEGER NOT NULL,
    leagues         TEXT,                   -- JSON array
    markets         TEXT,                   -- JSON array
    total_exposure  REAL,
    card_json       TEXT NOT NULL,          -- full card as JSON
    bankroll_start  REAL DEFAULT 1000.0,
    bankroll_end    REAL
);

-- CLV settlement audit
CREATE TABLE IF NOT EXISTS clv_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    settled_at      TEXT NOT NULL,
    bet_id          INTEGER REFERENCES bets(id),
    clv             REAL,
    source          TEXT,                   -- api / synthetic / manual
    notes           TEXT
);

-- Performance snapshots (daily summary)
CREATE TABLE IF NOT EXISTS performance_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date   TEXT NOT NULL,          -- YYYY-MM-DD
    bankroll        REAL,
    daily_pnl       REAL,
    bets_placed     INTEGER,
    bets_won        INTEGER,
    avg_clv         REAL,
    roi_rolling_30d REAL,
    notes           TEXT
);

-- Indices
CREATE INDEX IF NOT EXISTS idx_bets_date    ON bets(match_date);
CREATE INDEX IF NOT EXISTS idx_bets_league  ON bets(league);
CREATE INDEX IF NOT EXISTS idx_bets_market  ON bets(market);
CREATE INDEX IF NOT EXISTS idx_bets_result  ON bets(result);
CREATE INDEX IF NOT EXISTS idx_cards_date   ON daily_cards(run_date);
"""


# ── Database class ────────────────────────────────────────────────────────────

class Database:
    """Thin SQLite wrapper for KBet persistence."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or DB_PATH
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def migrate(self) -> None:
        """Apply schema migrations."""
        conn = self.connect()
        conn.executescript(SCHEMA)
        conn.commit()
        log.info(f"DB migrated: {self.path}")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Daily card ─────────────────────────────────────────────────────────────

    def save_daily_card(
        self,
        run_date: str,
        bets: List[Dict],
        leagues: List[str],
        markets: List[str],
        total_exposure: float,
        bankroll_start: float = 1000.0,
    ) -> int:
        """Save a full daily card and return its ID."""
        conn = self.connect()
        cursor = conn.execute(
            """INSERT INTO daily_cards
               (run_date, generated_at, n_bets, leagues, markets, total_exposure,
                card_json, bankroll_start)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_date,
                datetime.now(timezone.utc).isoformat(),
                len(bets),
                json.dumps(leagues),
                json.dumps(markets),
                total_exposure,
                json.dumps(bets, default=str),
                bankroll_start,
            )
        )
        conn.commit()
        return cursor.lastrowid

    # ── Bets ───────────────────────────────────────────────────────────────────

    def save_bet(
        self,
        card_id: int,
        match_date: str,
        league: str,
        home_team: str,
        away_team: str,
        market: str,
        pick: str,
        model_prob: float,
        book_odds: float,
        eff_odds: float,
        stake_pct: float,
        stake_units: float,
        ev: float,
        confidence: str,
        entry_prob: float = 0.0,
    ) -> int:
        """Insert a new bet and return its ID."""
        conn = self.connect()
        cursor = conn.execute(
            """INSERT INTO bets
               (card_id, placed_at, match_date, league,
                home_team, away_team, market, pick,
                model_prob, entry_prob, book_odds, eff_odds,
                stake_pct, stake_units, ev, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                card_id,
                datetime.now(timezone.utc).isoformat(),
                match_date,
                league,
                home_team,
                away_team,
                market,
                pick,
                model_prob,
                entry_prob,
                book_odds,
                eff_odds,
                stake_pct,
                stake_units,
                ev,
                confidence,
            )
        )
        conn.commit()
        return cursor.lastrowid

    def settle_bet(
        self,
        bet_id: int,
        result: str,  # "W", "L", "V"
        pnl: float,
        closing_odds_h: Optional[float] = None,
        closing_odds_d: Optional[float] = None,
        closing_odds_a: Optional[float] = None,
        clv: Optional[float] = None,
    ) -> None:
        """Mark a bet as settled with result and P&L."""
        conn = self.connect()
        conn.execute(
            """UPDATE bets SET
               result = ?, pnl = ?,
               closing_odds_h = ?, closing_odds_d = ?, closing_odds_a = ?,
               clv = ?, settled_at = ?
               WHERE id = ?""",
            (
                result, pnl,
                closing_odds_h, closing_odds_d, closing_odds_a,
                clv,
                datetime.now(timezone.utc).isoformat(),
                bet_id,
            )
        )
        conn.commit()

    def update_clv(self, bet_id: int, clv: float, source: str = "api") -> None:
        """Update CLV for a specific bet."""
        conn = self.connect()
        conn.execute("UPDATE bets SET clv = ? WHERE id = ?", (clv, bet_id))
        conn.execute(
            "INSERT INTO clv_log (settled_at, bet_id, clv, source) VALUES (?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), bet_id, clv, source)
        )
        conn.commit()

    # ── Query helpers ──────────────────────────────────────────────────────────

    def get_unsettled_bets(self, before_date: Optional[str] = None) -> List[Dict]:
        """Return bets without a result (for settlement runner)."""
        conn = self.connect()
        q = "SELECT * FROM bets WHERE result IS NULL"
        if before_date:
            q += f" AND match_date <= '{before_date}'"
        return [dict(r) for r in conn.execute(q).fetchall()]

    def get_daily_summary(self, run_date: str) -> Optional[Dict]:
        """Return today's bet summary."""
        conn = self.connect()
        row = conn.execute(
            "SELECT * FROM daily_cards WHERE run_date = ? ORDER BY id DESC LIMIT 1",
            (run_date,)
        ).fetchone()
        return dict(row) if row else None

    def get_performance(self, days: int = 30) -> Dict:
        """Return rolling performance stats."""
        import numpy as np
        conn = self.connect()
        rows = conn.execute("""
            SELECT result, pnl, clv, market, stake_units
            FROM bets
            WHERE settled_at IS NOT NULL
              AND match_date >= date('now', '-{} days')
        """.format(days)).fetchall()

        if not rows:
            return {"n_bets": 0, "roi": 0.0, "avg_clv": None}

        pnls  = [r["pnl"]  for r in rows if r["pnl"]  is not None]
        clvs  = [r["clv"]  for r in rows if r["clv"]  is not None]
        stakes = [r["stake_units"] for r in rows]

        total_staked = sum(stakes) if stakes else 1
        roi = sum(pnls) / total_staked if total_staked > 0 else 0.0

        return {
            "n_bets":    len(rows),
            "n_settled": len(pnls),
            "roi":       round(roi, 4),
            "total_pnl": round(sum(pnls), 2),
            "avg_clv":   round(float(np.mean(clvs)), 4) if clvs else None,
            "win_rate":  round(sum(1 for r in rows if r["result"] == "W") / len(rows), 3),
        }

    def all_bets_df(self):
        """Return all bets as a pandas DataFrame."""
        import pandas as pd
        conn = self.connect()
        return pd.read_sql("SELECT * FROM bets ORDER BY match_date, id", conn)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db = Database()
    db.migrate()
    print(f"✅ Database initialised: {DB_PATH}")
    print(f"   Tables: bets, daily_cards, clv_log, performance_log")

    # Quick validation
    conn = db.connect()
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    print(f"   Confirmed tables: {tables}")
    db.close()
