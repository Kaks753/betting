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

    # File lock to prevent duplicate concurrent generation on Render free wake-ups
    try:
        import fcntl
        lock_path = Path("/tmp/kbet_card_gen.lock")
        lock_path.parent.mkdir(exist_ok=True)
        lock_file = open(lock_path, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("[STARTUP] Another card-gen is already running — skipping")
            return
    except ImportError:
        # Windows fallback — no fcntl, use simple existence check
        pass
    except Exception:
        pass

    today = date.today().strftime("%Y-%m-%d")
    try:
        db = Database()
        db.migrate()
        existing = db.get_daily_summary(today)
        if existing:
            # Freshness guard: skip if recent (<12h) and has bets
            gen_at = existing.get("generated_at", "") or ""
            try:
                gen_dt = datetime.fromisoformat(gen_at.replace("Z", "+00:00"))
                age_hours = (datetime.now(timezone.utc) - gen_dt).total_seconds() / 3600
                n_bets = int(existing.get("n_bets", 0) or 0)
                if n_bets > 0 and age_hours < 12:
                    log.info(f"[STARTUP] Card for {today} is {age_hours:.1f}h old with {n_bets} bets — skip regen")
                    return
                if n_bets == 0:
                    log.info(f"[STARTUP] Card for {today} exists but 0 bets — regenerating")
                else:
                    log.info(f"[STARTUP] Card for {today} is {age_hours:.1f}h old — regenerating")
            except Exception:
                # Fallback: if can't parse, respect n_bets
                if int(existing.get("n_bets", 0) or 0) > 0:
                    log.info(f"[STARTUP] Card for {today} already exists ({existing['n_bets']} bets) — skip")
                    return

        log.info(f"[STARTUP] No fresh card for {today} — auto-generating in background...")

        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"cron_{today}.log"

        cmd = [
            sys.executable,
            str(Path(__file__).parent.parent / "cron_runner.py"),
            "--date", today,
            "--card-only",
        ]
        api_key = os.environ.get("ODDS_API_KEY", "").strip()
        is_prod = os.environ.get("KBET_ENV", "").lower() == "production"
        if not api_key or api_key == "your_key_here":
            if is_prod:
                log.error("[STARTUP] No ODDS_API_KEY in production — refusing to generate SIMULATED card")
                return
            log.info("[STARTUP] No ODDS_API_KEY — using SIMULATE mode (dev)")
            cmd.append("--simulate")
        else:
            log.info("[STARTUP] ODDS_API_KEY detected — using LIVE mode")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        app_root = str(Path(__file__).parent.parent.parent)
        env["PYTHONPATH"] = app_root + os.pathsep + env.get("PYTHONPATH", "")

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


# ── Mode (live vs simulated) ────────────────────────────────────────────

@app.get("/api/mode", tags=["health"])
async def api_mode() -> Dict:
    """Report whether system is LIVE or SIMULATED — transparency."""
    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    env = os.environ.get("KBET_ENV", "development").lower()
    is_live = bool(api_key) and env == "production"
    db = Database()
    db.migrate()
    today = date.today().strftime("%Y-%m-%d")
    summary = db.get_daily_summary(today)
    age_hours = None
    if summary:
        try:
            gen_at = (summary.get("generated_at") or "").replace("Z", "+00:00")
            gen_dt = datetime.fromisoformat(gen_at)
            age_hours = round((datetime.now(timezone.utc) - gen_dt).total_seconds() / 3600, 1)
        except Exception:
            pass
    return {
        "mode": "LIVE" if is_live else "SIMULATED",
        "environment": env,
        "has_api_key": bool(api_key),
        "today_card": {
            "exists": summary is not None,
            "n_bets": (summary or {}).get("n_bets", 0),
            "age_hours": age_hours,
            "generated_at": (summary or {}).get("generated_at", ""),
        },
        "warning": None if is_live else "SIMULATED MODE: Card from historical data, not for real betting.",
    }


# ── Card endpoints ────────────────────────────────────────────────────────────

@app.get("/card/today", tags=["card"])
async def get_today_card(db: Database = Depends(get_db)) -> Dict:
    """Return today's bet card."""
    today = date.today().strftime("%Y-%m-%d")
    return await get_card_by_date(today, db)


@app.get("/card/latest", tags=["card"])
async def get_latest_card(db: Database = Depends(get_db)) -> Dict:
    """Return the most recent card that has bets (for demo/sample display)."""
    # Try DB first — find most recent card with n_bets > 0
    try:
        conn = db.connect()
        row = conn.execute(
            "SELECT run_date FROM daily_cards WHERE n_bets > 0 ORDER BY run_date DESC LIMIT 1"
        ).fetchone()
        if row:
            card = await get_card_by_date(row[0], db)
            card["demo_mode"] = True
            card["demo_note"] = f"Sample card from {row[0]} (historical backtest)"
            return card
    except Exception:
        pass

    # Try JSON fallback files — check DATA_DIR first, then repo dir
    search_dirs = []
    data_dir_env = os.environ.get("DATA_DIR")
    if data_dir_env:
        search_dirs.append(Path(data_dir_env) / "daily_cards")
    search_dirs.append(Path(__file__).parent.parent / "data" / "daily_cards")
    seen = set()
    for json_dir in search_dirs:
        if not json_dir.exists():
            continue
        card_files = sorted(json_dir.glob("card_*.json"), reverse=True)
        for cf in card_files:
            if cf.resolve() in seen:
                continue
            seen.add(cf.resolve())
            try:
                with open(cf) as f:
                    data = json.load(f)
                # Handle both keys: "bets" (new) and "actionable" (legacy v1)
                bets = data.get("bets")
                if not bets:
                    bets = data.get("actionable", [])
                if not isinstance(bets, list):
                    bets = []
                # Also handle count fields
                if len(bets) == 0 and isinstance(data.get("total_bets"), int) and data["total_bets"] > 0:
                    # Some files use total_bets but empty bets due to partial write — skip
                    continue
                if len(bets) > 0:
                    run_date = data.get("date", cf.stem.replace("card_", ""))
                    if len(run_date) == 8 and "-" not in run_date:
                        run_date = f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:]}"
                    # Normalise bet shape — ensure required keys
                    normed = []
                    for b in bets:
                        if not isinstance(b, dict):
                            continue
                        # Legacy files used "book_odds" vs "odds"
                        if "odds" not in b and "book_odds" in b:
                            b["odds"] = b["book_odds"]
                        normed.append(b)
                    if not normed:
                        continue
                    return {
                        "date":            run_date,
                        "generated_at":    data.get("generated_at", ""),
                        "n_bets":          len(normed),
                        "leagues":         list({b.get("league", "") for b in normed if b.get("league")}),
                        "markets":         list({b.get("market", "") for b in normed if b.get("market")}),
                        "total_exposure":  data.get("total_exposure", 0),
                        "bankroll_start":  1000.0,
                        "settled_summary": {"total": 0, "won": 0, "lost": 0, "pending": 0, "total_pnl": 0, "avg_clv": None},
                        "bets":            normed,
                        "horizon":         data.get("horizon", 1),
                        "horizon_dates":   data.get("horizon_dates", [run_date]),
                        "ledger_hash":     data.get("ledger_hash", ""),
                        "demo_mode":       True,
                        "demo_note":       "Sample card (historical backtest — live card requires ODDS_API_KEY)",
                    }
            except Exception as e:
                log.warning(f"[LATEST] Failed to read {cf}: {e}")
                continue

    raise HTTPException(status_code=404, detail="No cards with bets found.")


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

    # Coerce n_bets to int — DB may store as string in some deploys
    try:
        n_bets_val = int(summary.get("n_bets", len(card_bets)))
    except Exception:
        n_bets_val = len(card_bets)
    # Try to enrich with horizon/ledger_hash from JSON file if present
    horizon = 1
    horizon_dates = [run_date]
    ledger_hash = ""
    try:
        j = _load_card_from_json(run_date)
        if j:
            raw = json.loads(j.get("card_json","[]")) if isinstance(j.get("card_json"), str) else j.get("card_json",[])
            # j is summary dict from JSON file load, not DB summary — need direct file read
            pass
        # Direct file read for horizon
        for d in [Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent / "data"))) / "daily_cards" / f"card_{run_date.replace('-','')}.json", Path(__file__).parent.parent / "data" / "daily_cards" / f"card_{run_date.replace('-','')}.json"]:
            if d.exists():
                with open(d) as f:
                    jd=json.load(f)
                horizon = jd.get("horizon", 1)
                horizon_dates = jd.get("horizon_dates", [run_date])
                ledger_hash = jd.get("ledger_hash", "")
                break
    except Exception:
        pass
    return {
        "date":            run_date,
        "generated_at":    summary.get("generated_at", ""),
        "n_bets":          n_bets_val,
        "leagues":         _safe_json(summary.get("leagues", "[]")),
        "markets":         _safe_json(summary.get("markets", "[]")),
        "total_exposure":  summary.get("total_exposure", 0),
        "bankroll_start":  summary.get("bankroll_start", 1000),
        "settled_summary": settled_summary,
        "bets":            card_bets,
        "horizon":         horizon,
        "horizon_dates":   horizon_dates,
        "ledger_hash":     ledger_hash,
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


@app.get("/performance/history", tags=["analytics"])
async def get_performance_history(
    days: int = Query(default=90, ge=7, le=365),
    db: Database = Depends(get_db),
) -> Dict:
    """Bankroll history for chart: daily P&L + cumulative."""
    conn = db.connect()
    rows = conn.execute("""
        SELECT match_date as date, SUM(pnl) as daily_pnl, COUNT(*) as n
        FROM bets
        WHERE result IS NOT NULL AND match_date IS NOT NULL
          AND match_date >= date('now', '-{} days')
        GROUP BY match_date
        ORDER BY date
    """.format(days)).fetchall()
    # Fallback to all-time if no recent
    if not rows:
        rows = conn.execute("""
            SELECT match_date as date, SUM(pnl) as daily_pnl, COUNT(*) as n
            FROM bets
            WHERE result IS NOT NULL
            GROUP BY match_date
            ORDER BY date
        """).fetchall()
    hist = []
    cum = 1000.0
    for r in rows:
        cum += r["daily_pnl"] or 0
        hist.append({"date": r["date"], "daily_pnl": round(r["daily_pnl"] or 0, 2), "bankroll": round(cum, 2), "n": r["n"]})
    return {"history": hist, "start_bankroll": 1000.0, "current_bankroll": round(cum, 2) if hist else 1000.0}


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
