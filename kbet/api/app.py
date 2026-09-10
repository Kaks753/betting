"""
KBet FastAPI — Production API
==============================
Serves the daily bet card, performance stats, and CLV reports.
Designed to run on Railway / Render free tier with SQLite.

Endpoints:
  GET /                    → Dashboard UI (HTML)
  GET /api/health          → Health check JSON
  GET /card/{date}         → Today's bet card (or any date)
  GET /card/today          → Today's card (shortcut)
  GET /performance         → Rolling 30d performance summary
  GET /clv                 → CLV tracking report
  GET /bets                → All bets (paginated)
  GET /bets/{id}           → Single bet detail
  POST /admin/run-cron     → Trigger manual cron run (admin only)
  GET /admin/cron-status   → Check background cron progress

Deploy:
  uvicorn kbet.api.app:app --host 0.0.0.0 --port 8000
  OR: railway up / render deploy
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from kbet.db import Database, DB_PATH
from kbet.engine.utils.clv_tracker import clv_report as compute_clv_report

# ── Config ────────────────────────────────────────────────────────────────────
API_VERSION  = "1.0.0"
ADMIN_TOKEN  = os.environ.get("ADMIN_TOKEN", "kbet-admin-2024")
log = logging.getLogger("api")

# ── Templates dir ─────────────────────────────────────────────────────────────
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_TEMPLATES_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

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
    allow_methods=["GET", "POST", "HEAD"],
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


# ── Startup: auto-generate today's card if missing ───────────────────────────

def _auto_generate_card():
    """Background thread: generate today's card if not in DB yet."""
    import time
    time.sleep(5)  # Let the server start fully first

    today = date.today().strftime("%Y-%m-%d")
    try:
        db = Database()
        db.migrate()
        existing = db.get_daily_summary(today)
        if existing:
            log.info(f"[STARTUP] Card for {today} already exists ({existing['n_bets']} bets) — skip auto-gen")
            return

        log.info(f"[STARTUP] No card for {today} — auto-generating in background...")

        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"cron_{today}.log"

        cmd = [
            sys.executable,
            str(Path(__file__).parent.parent / "cron_runner.py"),
            "--date", today,
            "--card-only",
            "--simulate",   # safe default; use live when ODDS_API_KEY is set
        ]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        app_root = str(Path(__file__).parent.parent.parent)
        env["PYTHONPATH"] = app_root + os.pathsep + env.get("PYTHONPATH", "")

        # Use live mode if API key is set
        api_key = os.environ.get("ODDS_API_KEY", "")
        if api_key and api_key != "your_key_here":
            cmd.remove("--simulate")
            log.info("[STARTUP] ODDS_API_KEY detected — using LIVE mode")
        else:
            log.info("[STARTUP] No ODDS_API_KEY — using SIMULATE mode (historical data)")

        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
        log.info(f"[STARTUP] cron_runner started (pid={proc.pid}), log: {log_path}")

    except Exception as e:
        log.error(f"[STARTUP] Auto-generate failed: {e}", exc_info=True)


@app.on_event("startup")
async def startup_event():
    """Start background card generation on server startup."""
    t = threading.Thread(target=_auto_generate_card, daemon=True)
    t.start()
    log.info("[STARTUP] Background card-gen thread started")


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["ui"])
@app.head("/", tags=["ui"])
async def dashboard(request: Request):
    """Serve the main dashboard UI."""
    template_path = _TEMPLATES_DIR / "dashboard.html"
    if template_path.exists():
        content = template_path.read_text(encoding="utf-8")
        return HTMLResponse(content=content)
    # Fallback minimal HTML
    return HTMLResponse(content="""
<!DOCTYPE html><html><head><title>KBet</title></head><body>
<h1>KBet API</h1>
<p>Dashboard template not found. API is running.</p>
<ul>
  <li><a href="/status">System Status</a></li>
  <li><a href="/card/today">Today's Card (JSON)</a></li>
  <li><a href="/docs">API Docs</a></li>
</ul>
</body></html>""")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["health"])
@app.head("/api/health", tags=["health"])
async def api_health() -> Dict:
    """Health check and system info (JSON)."""
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


# ── Card endpoints ────────────────────────────────────────────────────────────

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

    # 1) Try DB first
    summary = db.get_daily_summary(run_date)

    # 2) Fallback: check local JSON file (from repo or previous run)
    if not summary:
        summary = _load_card_from_json(run_date)
        if summary:
            log.info(f"[CARD] Serving {run_date} from JSON fallback")

    if not summary:
        # 3) Check if a cron is currently running for today
        today = date.today().strftime("%Y-%m-%d")
        log_path = Path(__file__).parent.parent / "logs" / f"cron_{run_date}.log"
        if log_path.exists():
            tail = _tail_log(log_path, 5)
            raise HTTPException(
                status_code=404,
                detail=f"Card for {run_date} is being generated. Check back in 5-10 minutes. Progress: {tail}"
            )
        raise HTTPException(
            status_code=404,
            detail=f"No card for {run_date}. The system auto-generates today's card on startup (~5-8 min)."
        )

    # Parse card JSON
    try:
        card_bets = json.loads(summary.get("card_json", "[]"))
    except (json.JSONDecodeError, TypeError):
        card_bets = []

    # If card_bets is a list of bets (from DB card_json), use directly
    # Otherwise handle nested {"bets": [...]} format
    if isinstance(card_bets, dict):
        card_bets = card_bets.get("bets", [])

    # Get settled results for bets on this date
    settled_summary = {"total": 0, "won": 0, "lost": 0, "pending": 0, "total_pnl": 0, "avg_clv": None}
    try:
        conn = db.connect()
        settled = conn.execute(
            "SELECT market, pick, result, pnl, clv, ev FROM bets WHERE card_id = ?",
            (summary.get("id", 0),)
        ).fetchall()

        settled_summary = {
            "total":     len(settled),
            "won":       sum(1 for r in settled if r["result"] == "W"),
            "lost":      sum(1 for r in settled if r["result"] == "L"),
            "pending":   sum(1 for r in settled if r["result"] is None),
            "total_pnl": round(sum(r["pnl"] or 0 for r in settled), 2),
            "avg_clv":   None,
        }
        clvs = [r["clv"] for r in settled if r["clv"] is not None]
        if clvs:
            settled_summary["avg_clv"] = round(sum(clvs) / len(clvs), 4)
    except Exception:
        pass

    return {
        "date":            run_date,
        "generated_at":    summary.get("generated_at", ""),
        "n_bets":          summary.get("n_bets", len(card_bets)),
        "leagues":         _safe_json(summary.get("leagues", "[]")),
        "markets":         _safe_json(summary.get("markets", "[]")),
        "total_exposure":  summary.get("total_exposure", 0),
        "bankroll_start":  summary.get("bankroll_start", 1000),
        "settled_summary": settled_summary,
        "bets":            card_bets,
    }


def _load_card_from_json(run_date: str) -> Optional[Dict]:
    """Try to load a card from local JSON file (fallback for ephemeral deploys)."""
    # Check DATA_DIR first, then repo data dir
    data_dirs = []
    data_dir_env = os.environ.get("DATA_DIR")
    if data_dir_env:
        data_dirs.append(Path(data_dir_env) / "daily_cards")
    data_dirs.append(Path(__file__).parent.parent / "data" / "daily_cards")

    date_compact = run_date.replace("-", "")  # 20230916

    for d in data_dirs:
        for fname in [f"card_{date_compact}.json", f"card_{run_date}.json"]:
            p = d / fname
            if p.exists():
                try:
                    with open(p) as f:
                        data = json.load(f)
                    # Normalize to DB-style summary dict
                    bets = data.get("bets") or data.get("actionable", [])
                    return {
                        "id": 0,
                        "run_date": run_date,
                        "generated_at": data.get("generated_at", ""),
                        "n_bets": len(bets),
                        "leagues": json.dumps(list({b.get("league","") for b in bets if b.get("league")})),
                        "markets": json.dumps(list({b.get("market","") for b in bets if b.get("market")})),
                        "total_exposure": data.get("total_exposure", 0),
                        "bankroll_start": 1000.0,
                        "card_json": json.dumps(bets),
                    }
                except Exception as e:
                    log.warning(f"Failed to load JSON card {p}: {e}")
    return None


def _safe_json(val):
    if isinstance(val, (list, dict)):
        return val
    try:
        return json.loads(val or "[]")
    except Exception:
        return []


def _tail_log(path: Path, n: int = 10) -> str:
    try:
        lines = path.read_text().splitlines()
        return " | ".join(lines[-n:])
    except Exception:
        return ""


# ── Performance ───────────────────────────────────────────────────────────────

@app.get("/performance", tags=["analytics"])
async def get_performance(
    days: int = Query(default=30, ge=1, le=365),
    db: Database = Depends(get_db),
) -> Dict:
    """Rolling performance summary (ROI, win rate, avg CLV, P&L)."""
    stats = db.get_performance(days=days)
    stats["window_days"] = days
    stats["generated_at"] = datetime.now(timezone.utc).isoformat()
    return stats


# ── CLV ───────────────────────────────────────────────────────────────────────

@app.get("/clv", tags=["analytics"])
async def get_clv_report(db: Database = Depends(get_db)) -> Dict:
    """Closing Line Value report."""
    report = compute_clv_report(db.path)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    return report


# ── Bets ──────────────────────────────────────────────────────────────────────

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
        "total":  total,
        "page":   page,
        "limit":  limit,
        "pages":  (total + limit - 1) // limit,
        "bets":   [dict(r) for r in rows],
    }


@app.get("/bets/{bet_id}", tags=["bets"])
async def get_bet(bet_id: int, db: Database = Depends(get_db)) -> Dict:
    """Get a single bet by ID."""
    conn = db.connect()
    row = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Bet {bet_id} not found")
    return dict(row)


# ── Admin ─────────────────────────────────────────────────────────────────────

@app.post("/admin/run-cron", tags=["admin"])
async def run_cron(
    target_date: Optional[str] = Query(default=None),
    simulate:    bool          = Query(default=True),
    force:       bool          = Query(default=False),
    _: bool = Depends(require_admin),
) -> Dict:
    """
    Manually trigger the cron runner in the background (admin only).
    Returns immediately — card generation takes 3-5 minutes (15s after first run).
    Check /card/{date} or /admin/cron-status after ~5 minutes to see results.
    Use force=true to regenerate if card already exists for the date.
    Requires X-Admin-Token header.
    """
    date_str = target_date or date.today().strftime("%Y-%m-%d")
    cmd = [sys.executable, str(Path(__file__).parent.parent / "cron_runner.py"),
           "--date", date_str, "--card-only"]
    if simulate:
        cmd.append("--simulate")
    if force:
        cmd.append("--force")

    # Launch as background process
    log_path = Path(__file__).parent.parent / "logs" / f"cron_{date_str}.log"
    log_path.parent.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    app_root = str(Path(__file__).parent.parent.parent)
    env["PYTHONPATH"] = app_root + os.pathsep + env.get("PYTHONPATH", "")
    logf = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

    return {
        "status":   "started",
        "pid":      proc.pid,
        "date":     date_str,
        "simulate": simulate,
        "force":    force,
        "message":  "Card generation started in background. Check /admin/cron-status in ~5 min (15s if cache warm).",
        "log":      str(log_path),
    }


@app.get("/admin/cron-status", tags=["admin"])
async def cron_status(
    target_date: Optional[str] = Query(default=None),
    _: bool = Depends(require_admin),
) -> Dict:
    """Check background cron progress for a given date."""
    date_str = target_date or date.today().strftime("%Y-%m-%d")
    log_path = Path(__file__).parent.parent / "logs" / f"cron_{date_str}.log"

    log_tail = ""
    if log_path.exists():
        with open(log_path, "r") as f:
            lines = f.readlines()
            log_tail = "".join(lines[-30:])

    db = get_db()
    summary = db.get_daily_summary(date_str)
    card_ready = summary is not None
    n_bets = summary.get("n_bets", 0) if summary else 0

    data_dir = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent / "data")))
    cache_dir = data_dir / "model_cache"
    cache_files = list(cache_dir.glob("modelset_*.pkl")) if cache_dir.exists() else []
    cache_info = []
    for cf in sorted(cache_files)[-3:]:
        age_h = (datetime.now().timestamp() - cf.stat().st_mtime) / 3600
        cache_info.append({
            "file": cf.name,
            "age_hours": round(age_h, 1),
            "size_mb": round(cf.stat().st_size / 1e6, 1),
        })

    return {
        "date":        date_str,
        "card_ready":  card_ready,
        "n_bets":      n_bets,
        "model_cache": cache_info,
        "log_path":    str(log_path),
        "log_exists":  log_path.exists(),
        "log_tail":    log_tail,
    }


# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/status", tags=["health"])
@app.head("/status", tags=["health"])
async def status(db: Database = Depends(get_db)) -> Dict:
    """Extended status with DB stats."""
    conn = db.connect()
    n_bets    = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
    n_cards   = conn.execute("SELECT COUNT(*) FROM daily_cards").fetchone()[0]
    n_settled = conn.execute("SELECT COUNT(*) FROM bets WHERE result IS NOT NULL").fetchone()[0]
    last_card = conn.execute(
        "SELECT run_date FROM daily_cards ORDER BY id DESC LIMIT 1"
    ).fetchone()
    db_exists = DB_PATH.exists()

    return {
        "db":           str(DB_PATH),
        "db_exists":    db_exists,
        "total_bets":   n_bets,
        "settled_bets": n_settled,
        "total_cards":  n_cards,
        "last_card":    last_card[0] if last_card else None,
        "api_version":  API_VERSION,
        "uptime":       "ok",
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
