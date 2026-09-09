#!/usr/bin/env python3
"""
KBet Cron Runner — Week 3 Production Pipeline
==============================================
Runs daily at 10:00 UTC to:
  1. Generate today's 5-12 bet card
  2. Save bets to kbet.db
  3. Settle yesterday's bets (fetch results + closing odds)
  4. Update CLV tracking
  5. Log daily performance

Usage:
  # Full daily run:
  python3 kbet/cron_runner.py

  # Generate card only (no settlement):
  python3 kbet/cron_runner.py --card-only

  # Settle only (no new card):
  python3 kbet/cron_runner.py --settle-only

  # Generate for a specific date (backtest mode):
  python3 kbet/cron_runner.py --date 2024-09-14 --simulate

Deploy on Railway/Render:
  Set cron schedule: "0 10 * * *"  (10:00 UTC daily)
  Environment vars: ODDS_API_KEY, BANKROLL (optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from kbet.db import Database, DB_PATH
from kbet.daily_card import DailyCardEngine, Bet as DailyCardBet
from kbet.engine.utils.clv_tracker import settle_clv, ClosingOddsFetcher, compute_clv

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "cron.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("cron")

# ── Config ────────────────────────────────────────────────────────────────────
BANKROLL = float(os.environ.get("BANKROLL", "1000.0"))
MAX_BETS = int(os.environ.get("MAX_BETS", "10"))
MIN_BETS = int(os.environ.get("MIN_BETS", "5"))
DEFAULT_LEAGUES = ["E0", "SP1", "D1", "I1", "F1"]
DEFAULT_MARKETS = ["1x2", "ou"]


# ── Step 1: Generate daily card ───────────────────────────────────────────────

def generate_card(
    target_date: str,
    leagues: List[str],
    markets: List[str],
    simulate: bool,
    max_bets: int,
) -> Optional[Dict]:
    """
    Run the daily card engine for target_date.
    Returns card dict or None on failure.
    """
    log.info(f"[CARD] Generating {target_date} card (simulate={simulate}) ...")

    try:
        engine = DailyCardEngine(
            target_date=target_date,
            simulate=simulate,
            leagues=leagues,
            api_key=os.getenv("ODDS_API_KEY", ""),
        )
        bets: List[DailyCardBet] = engine.run(max_bets=max_bets)

        # Normalise to dict format expected downstream
        card = {
            "date": target_date,
            "total_exposure": sum(getattr(b, "stake_pct", 0.0) for b in bets),
            "bets": [
                {
                    "match_id":   getattr(b, "match_id", ""),
                    "home":       getattr(b, "home", ""),
                    "away":       getattr(b, "away", ""),
                    "league":     getattr(b, "league", ""),
                    "match_date": getattr(b, "match_date", target_date),
                    "market":     getattr(b, "market", ""),
                    "pick":       getattr(b, "pick", ""),
                    "model_prob": getattr(b, "model_prob", 0.0),
                    "book_odds":  getattr(b, "book_odds", 0.0),
                    "ev":         getattr(b, "ev", 0.0),
                    "stake_pct":  getattr(b, "stake_pct", 0.0),
                    "stake_units": getattr(b, "stake_units", 0.0),
                    "confidence": getattr(b, "confidence", "WATCH"),
                }
                for b in bets
            ],
        }
    except Exception as e:
        log.error(f"[CARD] Card generation failed: {e}", exc_info=True)
        return None

    n_bets = len(card.get("bets", []))
    log.info(f"[CARD] Generated {n_bets} bets")
    return card


# ── Step 2: Persist card to DB ────────────────────────────────────────────────

def persist_card(
    db: Database,
    card: Dict,
    target_date: str,
    leagues: List[str],
    markets: List[str],
) -> int:
    """Save card and all bets to DB. Returns card_id."""
    bets = card.get("bets", [])
    total_exposure = card.get("total_exposure", 0.0)

    card_id = db.save_daily_card(
        run_date=target_date,
        bets=bets,
        leagues=leagues,
        markets=markets,
        total_exposure=total_exposure,
        bankroll_start=BANKROLL,
    )

    n_saved = 0
    for bet in bets:
        try:
            db.save_bet(
                card_id=card_id,
                match_date=bet.get("match_date", target_date),
                league=bet.get("league", ""),
                home_team=bet.get("home_team", ""),
                away_team=bet.get("away_team", ""),
                market=bet.get("market", ""),
                pick=bet.get("pick", ""),
                model_prob=float(bet.get("model_prob", 0)),
                book_odds=float(bet.get("book_odds", 0)),
                eff_odds=float(bet.get("eff_odds", bet.get("book_odds", 0))),
                stake_pct=float(bet.get("stake_pct", 0)),
                stake_units=float(bet.get("stake_units", 0)),
                ev=float(bet.get("ev", 0)),
                confidence=bet.get("confidence", "WATCH"),
                entry_prob=float(bet.get("entry_prob", 0)),
            )
            n_saved += 1
        except Exception as e:
            log.warning(f"  Could not save bet: {e}")

    log.info(f"[DB] Saved card #{card_id} with {n_saved} bets")
    return card_id


# ── Step 3: Settle yesterday's bets ──────────────────────────────────────────

def settle_yesterday(db: Database, target_date: str, simulate: bool) -> Dict:
    """
    Settle bets that should have results by now.
    In simulate mode, uses parquet historical results.
    In live mode, would need a results API (Football-Data.co.uk or similar).
    """
    log.info("[SETTLE] Starting settlement for bets before {target_date} ...")

    if simulate:
        # In simulation mode, use historical results from parquet
        return _settle_from_parquet(db, target_date)
    else:
        # In production, would fetch from Football-Data.co.uk API
        log.warning("[SETTLE] Production settlement not yet implemented (needs results API)")
        return {"settled": 0, "failed": 0, "mode": "production_stub"}


def _settle_from_parquet(db: Database, before_date: str) -> Dict:
    """
    Settle bets using historical results from all_matches.parquet.
    Used in simulate/backtest mode.
    """
    import pandas as pd

    parquet_path = Path(__file__).parent / "data/processed/all_matches.parquet"
    if not parquet_path.exists():
        log.warning("[SETTLE] No parquet file — cannot settle")
        return {"settled": 0, "failed": 0}

    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["date_str"] = df["date"].astype(str)

    unsettled = db.get_unsettled_bets(before_date=before_date)
    log.info(f"[SETTLE] {len(unsettled)} bets to settle")

    settled = failed = 0
    for bet in unsettled:
        match_date = bet["match_date"]
        home = bet["home_team"].strip().lower()
        away = bet["away_team"].strip().lower()

        # Find match in parquet
        matches = df[
            (df["date_str"] == match_date) &
            (df["home_team"].str.strip().str.lower() == home) &
            (df["away_team"].str.strip().str.lower() == away)
        ]

        if matches.empty:
            log.debug(f"  No match: {bet['home_team']} v {bet['away_team']} {match_date}")
            failed += 1
            continue

        row = matches.iloc[0]
        actual_result = row.get("result", "")
        home_goals = row.get("home_goals", 0) or 0
        away_goals = row.get("away_goals", 0) or 0
        total_goals = home_goals + away_goals

        # Determine win/loss
        pick = bet["pick"]
        market = bet["market"]
        won = False

        if market == "1x2":
            won = pick == actual_result
        elif market == "ou":
            if pick == "O2.5":
                won = total_goals > 2.5
            elif pick == "U2.5":
                won = total_goals <= 2.5

        result_code = "W" if won else "L"
        eff_odds = bet["eff_odds"] or bet["book_odds"]
        stake_units = bet["stake_units"]
        pnl = stake_units * (eff_odds - 1) if won else -stake_units

        db.settle_bet(
            bet_id=bet["id"],
            result=result_code,
            pnl=pnl,
        )
        settled += 1

    log.info(f"[SETTLE] Done: {settled} settled, {failed} not found")
    return {"settled": settled, "failed": failed}


# ── Step 4: Update CLV ────────────────────────────────────────────────────────

def update_clv(db: Database, simulate: bool) -> Dict:
    """Run CLV settlement."""
    if simulate:
        log.info("[CLV] Simulate mode — skipping live CLV (no API key used)")
        return {"mode": "simulate", "settled": 0}
    return settle_clv(db.path)


# ── Step 5: Log performance snapshot ─────────────────────────────────────────

def log_performance(db: Database, target_date: str) -> None:
    """Write daily performance to performance_log."""
    stats = db.get_performance(days=30)
    conn = db.connect()
    conn.execute("""
        INSERT INTO performance_log (snapshot_date, bets_placed, roi_rolling_30d, avg_clv, notes)
        VALUES (?, ?, ?, ?, ?)
    """, (
        target_date,
        stats.get("n_bets", 0),
        stats.get("roi", 0.0),
        stats.get("avg_clv"),
        json.dumps(stats),
    ))
    conn.commit()
    log.info(f"[PERF] ROI(30d)={stats.get('roi', 0)*100:+.1f}% | "
             f"CLV={stats.get('avg_clv') or 'N/A'} | Bets={stats.get('n_bets', 0)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    t0 = time.time()

    target_date = args.date or date.today().strftime("%Y-%m-%d")
    yesterday   = (datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    simulate    = args.simulate
    leagues     = args.leagues or DEFAULT_LEAGUES
    markets     = args.markets or DEFAULT_MARKETS

    log.info("=" * 60)
    log.info(f"KBet Cron — {target_date} ({'SIMULATE' if simulate else 'LIVE'})")
    log.info("=" * 60)

    # Initialise DB
    db = Database()
    db.migrate()

    # ── Card generation ───────────────────────────────────────────────────────
    if not args.settle_only:
        # Check if card already exists for today
        existing = db.get_daily_summary(target_date)
        if existing and not args.force:
            log.info(f"[CARD] Card already exists for {target_date} (id={existing['id']}) — skip")
            log.info("      Use --force to regenerate")
        else:
            card = generate_card(target_date, leagues, markets, simulate, args.max_bets)
            if card:
                persist_card(db, card, target_date, leagues, markets)
                # Print card summary
                _print_card(card, target_date)
            else:
                log.error("[CARD] Card generation failed")

    # ── Settlement ────────────────────────────────────────────────────────────
    if not args.card_only:
        settle_yesterday(db, yesterday, simulate)
        update_clv(db, simulate)
        log_performance(db, target_date)

    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"  Cron run complete in {elapsed:.1f}s")
    log.info(f"  DB: {DB_PATH}")
    log.info(f"{'='*60}")

    db.close()


def _print_card(card: Dict, target_date: str) -> None:
    """Pretty-print the daily card."""
    bets = card.get("bets", [])
    log.info(f"\n{'='*65}")
    log.info(f"  🎯 KBet Daily Card — {target_date}  ({len(bets)} bets)")
    log.info(f"{'='*65}")
    log.info(f"  {'#':<3} {'Match':<30} {'Mkt':<7} {'Pick':<6} {'EV':>6} {'Stake':>7} {'Conf'}")
    log.info(f"  {'-'*60}")
    for i, bet in enumerate(bets, 1):
        home = bet.get("home_team", "")[:12]
        away = bet.get("away_team", "")[:12]
        match = f"{home} v {away}"
        ev    = bet.get("ev", 0) * 100
        stake = bet.get("stake_units", 0)
        log.info(
            f"  {i:<3} {match:<30} {bet.get('market',''):<7} "
            f"{bet.get('pick',''):<6} {ev:>+5.1f}% {stake:>5.1f}u  "
            f"{bet.get('confidence','')}"
        )
    log.info(f"{'='*65}")
    log.info(f"  Total exposure: {card.get('total_exposure', 0)*100:.1f}% of bankroll")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KBet daily cron runner")
    parser.add_argument("--date",        default=None,   help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--simulate",    action="store_true", help="Use historical data instead of live API")
    parser.add_argument("--card-only",   action="store_true", help="Only generate card (skip settlement)")
    parser.add_argument("--settle-only", action="store_true", help="Only settle bets (skip card)")
    parser.add_argument("--force",       action="store_true", help="Regenerate card if already exists")
    parser.add_argument("--leagues",     nargs="+", default=None,
                        choices=["E0","E1","SP1","D1","I1","F1","N1","P1","B1","G1"])
    parser.add_argument("--markets",     nargs="+", default=None,
                        choices=["1x2","ou","corners","cards"])
    parser.add_argument("--max-bets",    type=int, default=MAX_BETS)
    args = parser.parse_args()
    main(args)
