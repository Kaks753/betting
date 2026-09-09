"""
KBet FastAPI — Production API
==============================
Serves the daily bet card, performance stats, and CLV reports.
Designed to run on Railway / Render free tier with SQLite.

Endpoints:
  GET /                    → Health check + version
  GET /card/{date}         → Today's bet card (or any date)
  GET /card/today          → Today's card (shortcut)
  GET /performance         → Rolling 30d performance summary
  GET /clv                 → CLV tracking report
  GET /bets                → All bets (paginated)
  GET /bets/{id}           → Single bet detail
  POST /admin/run-cron     → Trigger manual cron run (admin only)

Deploy:
  uvicorn kbet.api.app:app --host 0.0.0.0 --port 8000
  OR: railway up / render deploy
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from kbet.db import Database, DB_PATH
from kbet.engine.utils.clv_tracker import clv_report as compute_clv_report

# ── Config ────────────────────────────────────────────────────────────────────
API_VERSION  = "1.0.0"
ADMIN_TOKEN  = os.environ.get("ADMIN_TOKEN", "kbet-admin-2024")  # Override in prod
log = logging.getLogger("api")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="KBet API",
    description="Football bet card with CLV tracking",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── DB dependency ─────────────────────────────────────────────────────────────

def get_db() -> Database:
    db = Database()
    db.migrate()
    return db


# ── Auth dependency ───────────────────────────────────────────────────────────

def require_admin(x_admin_token: str = Header(default="")):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def health() -> Dict:
    """Health check and system info."""
    db_exists = DB_PATH.exists()
    return {
        "status":    "ok",
        "version":   API_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "db":        str(DB_PATH) if db_exists else "not_initialised",
        "db_exists": db_exists,
        "service":   "KBet — Football Betting Analytics",
        "endpoints": ["/card/today", "/card/{date}", "/performance", "/clv", "/bets"],
    }


@app.get("/card/today", tags=["card"])
async def get_today_card(db: Database = Depends(get_db)) -> Dict:
    """Return today's bet card."""
    today = date.today().strftime("%Y-%m-%d")
    return await get_card_by_date(today, db)


@app.get("/card/{run_date}", tags=["card"])
async def get_card_by_date(
    run_date: str,
    db: Database = Depends(get_db),
) -> Dict:
    """
    Return bet card for a specific date.
    Format: YYYY-MM-DD
    """
    # Validate date format
    try:
        datetime.strptime(run_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format — use YYYY-MM-DD")

    summary = db.get_daily_summary(run_date)
    if not summary:
        raise HTTPException(
            status_code=404,
            detail=f"No card for {run_date}. Run cron_runner.py first."
        )

    # Parse card JSON
    try:
        card_bets = json.loads(summary.get("card_json", "[]"))
    except (json.JSONDecodeError, TypeError):
        card_bets = []

    # Get settled results for bets on this date
    conn = db.connect()
    settled = conn.execute(
        "SELECT market, pick, result, pnl, clv, ev FROM bets WHERE card_id = ?",
        (summary["id"],)
    ).fetchall()

    settled_summary = {
        "total":    len(settled),
        "won":      sum(1 for r in settled if r["result"] == "W"),
        "lost":     sum(1 for r in settled if r["result"] == "L"),
        "pending":  sum(1 for r in settled if r["result"] is None),
        "total_pnl": round(sum(r["pnl"] or 0 for r in settled), 2),
        "avg_clv":  None,
    }
    clvs = [r["clv"] for r in settled if r["clv"] is not None]
    if clvs:
        import numpy as np
        settled_summary["avg_clv"] = round(float(np.mean(clvs)), 4)

    return {
        "date":            run_date,
        "generated_at":    summary["generated_at"],
        "n_bets":          summary["n_bets"],
        "leagues":         json.loads(summary.get("leagues", "[]")),
        "markets":         json.loads(summary.get("markets", "[]")),
        "total_exposure":  summary.get("total_exposure", 0),
        "bankroll_start":  summary.get("bankroll_start", 1000),
        "settled_summary": settled_summary,
        "bets":            card_bets,
    }


@app.get("/performance", tags=["analytics"])
async def get_performance(
    days: int = Query(default=30, ge=1, le=365),
    db: Database = Depends(get_db),
) -> Dict:
    """
    Rolling performance summary.
    - ROI
    - Win rate
    - Average CLV
    - P&L breakdown
    """
    stats = db.get_performance(days=days)
    stats["window_days"] = days
    stats["generated_at"] = datetime.now(timezone.utc).isoformat()
    return stats


@app.get("/clv", tags=["analytics"])
async def get_clv_report(db: Database = Depends(get_db)) -> Dict:
    """
    Closing Line Value report.
    Shows average CLV, Sharpe of CLV, correlation with outcomes.
    Positive avg CLV = we consistently have pre-close edge.
    """
    report = compute_clv_report(db.path)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    return report


@app.get("/bets", tags=["bets"])
async def list_bets(
    page:   int   = Query(default=1, ge=1),
    limit:  int   = Query(default=50, ge=1, le=200),
    market: Optional[str] = Query(default=None),
    result: Optional[str] = Query(default=None, pattern="^[WLV]$"),
    db: Database = Depends(get_db),
) -> Dict:
    """List all bets with pagination."""
    offset = (page - 1) * limit
    conn = db.connect()

    where_parts = []
    params = []
    if market:
        where_parts.append("market = ?")
        params.append(market)
    if result:
        where_parts.append("result = ?")
        params.append(result)

    where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    total = conn.execute(f"SELECT COUNT(*) FROM bets {where}", params).fetchone()[0]
    rows  = conn.execute(
        f"SELECT * FROM bets {where} ORDER BY match_date DESC, id DESC LIMIT ? OFFSET ?",
        params + [limit, offset]
    ).fetchall()

    return {
        "total":   total,
        "page":    page,
        "limit":   limit,
        "pages":   (total + limit - 1) // limit,
        "bets":    [dict(r) for r in rows],
    }


@app.get("/bets/{bet_id}", tags=["bets"])
async def get_bet(bet_id: int, db: Database = Depends(get_db)) -> Dict:
    """Get a single bet by ID."""
    conn = db.connect()
    row = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Bet {bet_id} not found")
    return dict(row)


@app.post("/admin/run-cron", tags=["admin"])
async def run_cron(
    target_date: Optional[str] = Query(default=None),
    simulate:    bool          = Query(default=True),
    _: bool = Depends(require_admin),
) -> Dict:
    """
    Manually trigger the cron runner in the background (admin only).
    Returns immediately — card generation takes 3-5 minutes.
    Check /card/{date} or /status after ~5 minutes to see results.
    Requires X-Admin-Token header.
    """
    import subprocess
    date_str = target_date or date.today().strftime("%Y-%m-%d")
    cmd = [sys.executable, str(Path(__file__).parent.parent / "cron_runner.py"),
           "--date", date_str, "--card-only"]
    if simulate:
        cmd.append("--simulate")

    # Launch as background process — don't wait for it (model fitting takes 3-5min)
    log_path = Path(__file__).parent.parent / "logs" / f"cron_{date_str}.log"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    return {
        "status":   "started",
        "pid":      proc.pid,
        "date":     date_str,
        "simulate": simulate,
        "message":  "Card generation started in background. Check /card/{date} in ~5 minutes.",
        "log":      str(log_path),
    }


@app.get("/status", tags=["health"])
async def status(db: Database = Depends(get_db)) -> Dict:
    """Extended status with DB stats."""
    conn = db.connect()
    n_bets   = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
    n_cards  = conn.execute("SELECT COUNT(*) FROM daily_cards").fetchone()[0]
    n_settled = conn.execute("SELECT COUNT(*) FROM bets WHERE result IS NOT NULL").fetchone()[0]
    last_card = conn.execute(
        "SELECT run_date FROM daily_cards ORDER BY id DESC LIMIT 1"
    ).fetchone()

    return {
        "db":            str(DB_PATH),
        "total_bets":    n_bets,
        "settled_bets":  n_settled,
        "total_cards":   n_cards,
        "last_card":     last_card[0] if last_card else None,
        "api_version":   API_VERSION,
        "uptime":        "ok",
    }


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    log.info(f"Starting KBet API on port {port}")
    uvicorn.run(
        "kbet.api.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
