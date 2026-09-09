#!/usr/bin/env python3
"""
The Odds API — Historical Snapshot Ingestion  (Week 3, Step 1)
==============================================================
Downloads pre-closing odds snapshots for every match in all_matches.parquet
and stores them as date-keyed parquet files.

Key concept:
    The Odds API /historical endpoint returns odds "as of" a given timestamp.
    We target T-48h before match kickoff to capture pre-market prices that
    differ from the closing line → enabling genuine CLV tracking.

Storage layout:
    data/odds_snapshots/
      YYYY-MM-DD.parquet   ← one file per calendar day
      _manifest.json       ← list of dates downloaded + request counts

Columns per row:
    date, home_team, away_team, league,
    pre_h, pre_d, pre_a,          # T-48h Pinnacle devigged probs
    closing_h, closing_d, closing_a,  # closing Pinnacle devigged probs
    pre_odds_h, pre_odds_d, pre_odds_a,  # raw pre odds
    closing_odds_h, closing_odds_d, closing_odds_a  # raw closing odds

Usage:
    # Download all missing dates (will use ~requests_per_run API requests)
    python3 kbet/ingest_odds_snapshots.py

    # Download specific date range
    python3 kbet/ingest_odds_snapshots.py --from 2023-08-01 --to 2024-06-01

    # Dry run — show what would be downloaded (no API calls)
    python3 kbet/ingest_odds_snapshots.py --dry-run

    # Show download status
    python3 kbet/ingest_odds_snapshots.py --status

Environment:
    ODDS_API_KEY — The Odds API key (free tier: 500 requests/month)
                   If not set, the script generates synthetic snapshots
                   using an improved B365 baseline (for backtest development).

IMPORTANT — Quota management:
    Free tier = 500 requests/month.
    Each historical snapshot = 1 request per sport (10 sports = 10 requests/date).
    To cover all 3 backtest rounds (2022-08 → 2025-06) = ~900 matchdays × 10 leagues.
    Strategy: Use Pinnacle data from all_matches.parquet as synthetic closing line;
    simulate pre-closing odds by perturbing with market noise model.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data"
PROCESSED_DIR  = DATA_DIR / "processed"
SNAPSHOTS_DIR  = DATA_DIR / "odds_snapshots"
MANIFEST_PATH  = SNAPSHOTS_DIR / "_manifest.json"
PARQUET_PATH   = PROCESSED_DIR / "all_matches.parquet"

SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_snapshots")

# ── The Odds API config ──────────────────────────────────────────────────────

BASE_URL = "https://api.the-odds-api.com/v4"

SPORT_KEYS = {
    "E0":  "soccer_epl",
    "E1":  "soccer_england_league1",
    "SP1": "soccer_spain_la_liga",
    "D1":  "soccer_germany_bundesliga",
    "I1":  "soccer_italy_serie_a",
    "F1":  "soccer_france_ligue_one",
    "N1":  "soccer_netherlands_eredivisie",
    "P1":  "soccer_portugal_primeira_liga",
    "B1":  "soccer_belgium_first_div",
    "G1":  "soccer_greece_super_league",
}

# Timezone for kickoff timestamp construction
UTC = timezone.utc

# ── Manifest helpers ──────────────────────────────────────────────────────────

def load_manifest() -> Dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"downloaded_dates": [], "total_requests": 0, "synthetic_dates": []}


def save_manifest(manifest: Dict) -> None:
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def is_downloaded(date_str: str, manifest: Dict) -> bool:
    return date_str in manifest.get("downloaded_dates", []) or \
           date_str in manifest.get("synthetic_dates", [])


# ── De-vig ────────────────────────────────────────────────────────────────────

def devig(h: float, d: float, a: float) -> Optional[Tuple[float, float, float]]:
    """Return de-vigged probabilities or None if any input is invalid."""
    if any(math.isnan(v) or v <= 1.0 for v in (h, d, a)):
        return None
    inv = 1/h + 1/d + 1/a
    return (1/h)/inv, (1/d)/inv, (1/a)/inv


# ── The Odds API historical snapshot fetcher ──────────────────────────────────

class OddsAPISnapshotter:
    """
    Fetches historical odds snapshots from The Odds API.
    Handles quota tracking and graceful degradation.
    """

    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.requests_remaining: Optional[int] = None
        self.requests_used: int = 0

    def _get(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Single authenticated GET with quota tracking."""
        if not self.api_key:
            return None
        url = f"{BASE_URL}/{endpoint}"
        params["apiKey"] = self.api_key
        try:
            resp = requests.get(url, params=params, timeout=15)
            self.requests_remaining = int(resp.headers.get("x-requests-remaining", -1))
            self.requests_used += 1
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 422:
                log.warning(f"  [422] No data at that timestamp — skipping")
                return None
            elif resp.status_code == 401:
                log.error("  [401] Invalid API key")
                return None
            elif resp.status_code == 429:
                log.warning("  [429] Rate limited — sleeping 5s")
                time.sleep(5)
                return None
            else:
                log.warning(f"  [{resp.status_code}] Unexpected: {resp.text[:200]}")
                return None
        except Exception as e:
            log.warning(f"  Network error: {e}")
            return None

    def get_historical_snapshot(
        self, sport_key: str, timestamp_iso: str
    ) -> List[Dict]:
        """
        Fetch odds snapshot for a sport at a given ISO-8601 timestamp.
        Returns list of event dicts with h2h odds or empty list.
        """
        data = self._get(
            f"sports/{sport_key}/odds-history",
            {
                "regions": "eu",
                "markets": "h2h",
                "bookmakers": "pinnacle",
                "date": timestamp_iso,
                "dateFormat": "iso",
                "oddsFormat": "decimal",
            }
        )
        if not data:
            return []
        return data.get("data", []) if isinstance(data, dict) else data

    def quota_ok(self, min_remaining: int = 20) -> bool:
        """Check if we have enough quota to continue."""
        if self.requests_remaining is None:
            return True  # No info yet — assume OK
        return self.requests_remaining >= min_remaining


# ── Team name fuzzy-matching ──────────────────────────────────────────────────

def normalise_name(name: str) -> str:
    """Lower-case, strip punctuation for fuzzy team matching."""
    import re
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_teams(
    api_home: str, api_away: str,
    df_home: str, df_away: str,
) -> bool:
    """Check if API team names roughly match parquet team names."""
    a_h = normalise_name(api_home)
    a_a = normalise_name(api_away)
    d_h = normalise_name(df_home)
    d_a = normalise_name(df_away)
    # Exact match first
    if a_h == d_h and a_a == d_a:
        return True
    # Prefix match (handles "Man City" vs "Manchester City")
    if (d_h[:6] in a_h or a_h[:6] in d_h) and (d_a[:6] in a_a or a_a[:6] in d_a):
        return True
    return False


# ── Synthetic snapshot generator (no API key / quota exhausted) ──────────────

NOISE_SIGMA = 0.018   # std of per-game market drift in probability space
NOISE_SEED  = 42      # reproducible for backtesting

rng = random.Random(NOISE_SEED)
np_rng = np.random.default_rng(NOISE_SEED)


def synthetic_pre_odds(
    closing_h: float, closing_d: float, closing_a: float,
    hours_before: int = 48,
) -> Optional[Tuple[float, float, float]]:
    """
    Simulate pre-closing odds by applying market-noise perturbation.

    Research basis: Betfair exchange data shows ~1.8% std in implied probability
    between T-48h and closing for top-6 European leagues. Direction is random;
    magnitude scales with time-to-close.

    This is NOT genuine CLV — it's a synthetic baseline for development.
    Real data from The Odds API historical endpoint replaces this in production.
    """
    base = devig(closing_h, closing_d, closing_a)
    if base is None:
        return None
    bh, bd, ba = base

    # Correlated noise: if home moves up, draw/away move down proportionally
    noise_h = np_rng.normal(0, NOISE_SIGMA)
    noise_d = np_rng.normal(0, NOISE_SIGMA * 0.6)
    noise_a = -(noise_h + noise_d)  # simplex constraint: sum = 1

    pre_h = np.clip(bh + noise_h, 0.05, 0.92)
    pre_d = np.clip(bd + noise_d, 0.05, 0.55)
    pre_a = np.clip(ba + noise_a, 0.05, 0.92)

    # Re-normalise to simplex
    total = pre_h + pre_d + pre_a
    return float(pre_h/total), float(pre_d/total), float(pre_a/total)


def convert_prob_to_odds(p_h: float, p_d: float, p_a: float, margin: float = 0.045) -> Tuple[float, float, float]:
    """Convert fair probabilities back to decimal odds with bookmaker margin."""
    # Add margin proportionally
    raw_h = p_h * (1 + margin)
    raw_d = p_d * (1 + margin)
    raw_a = p_a * (1 + margin)
    return round(1/raw_h, 4), round(1/raw_d, 4), round(1/raw_a, 4)


# ── Core ingestion logic ──────────────────────────────────────────────────────

def ingest_date(
    date_str: str,
    day_df: pd.DataFrame,
    snapshotter: OddsAPISnapshotter,
    use_synthetic: bool,
) -> pd.DataFrame:
    """
    Build a snapshot DataFrame for one matchday.

    For each match in day_df:
      - If real API: fetch T-48h snapshot, match by team name
      - If synthetic: perturb closing Pinnacle odds with noise model
    Returns a DataFrame ready to be saved as parquet.
    """
    records = []
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    pre_dt   = date_obj - timedelta(hours=48)
    pre_iso  = pre_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Real API path ─────────────────────────────────────────────────────────
    api_snapshots: Dict[str, List[Dict]] = {}
    if not use_synthetic and snapshotter.api_key:
        leagues_in_day = day_df["league"].unique()
        for league in leagues_in_day:
            sport_key = SPORT_KEYS.get(league)
            if not sport_key:
                continue
            if not snapshotter.quota_ok():
                log.warning("  ⚠ Quota low — switching to synthetic for remaining")
                use_synthetic = True
                break
            evts = snapshotter.get_historical_snapshot(sport_key, pre_iso)
            api_snapshots[league] = evts
            log.debug(f"    {league}: {len(evts)} events from API")
            time.sleep(0.3)  # Polite delay

    # ── Build records ─────────────────────────────────────────────────────────
    for _, row in day_df.iterrows():
        rec: Dict = {
            "date":       date_str,
            "league":     row.get("league", ""),
            "home_team":  str(row.get("home_team", "")),
            "away_team":  str(row.get("away_team", "")),
            "match_id":   f"{row.get('home_team', '')}_{row.get('away_team', '')}_{date_str}",
            # Closing Pinnacle odds (from all_matches.parquet)
            "closing_odds_h": float(row.get("odds_home_pinnacle", float("nan"))),
            "closing_odds_d": float(row.get("odds_draw_pinnacle", float("nan"))),
            "closing_odds_a": float(row.get("odds_away_pinnacle", float("nan"))),
            # To be filled
            "pre_odds_h": float("nan"),
            "pre_odds_d": float("nan"),
            "pre_odds_a": float("nan"),
            "pre_h":      float("nan"),
            "pre_d":      float("nan"),
            "pre_a":      float("nan"),
            "closing_h":  float("nan"),
            "closing_d":  float("nan"),
            "closing_a":  float("nan"),
            "source":     "missing",
        }

        # Closing devigged probs
        c_dv = devig(rec["closing_odds_h"], rec["closing_odds_d"], rec["closing_odds_a"])
        if c_dv:
            rec["closing_h"], rec["closing_d"], rec["closing_a"] = c_dv

        # ── Try API match ──────────────────────────────────────────────────
        api_matched = False
        if not use_synthetic:
            league_events = api_snapshots.get(row.get("league", ""), [])
            for evt in league_events:
                if match_teams(
                    evt.get("home_team", ""), evt.get("away_team", ""),
                    rec["home_team"], rec["away_team"]
                ):
                    # Extract Pinnacle h2h from bookmakers list
                    for bm in evt.get("bookmakers", []):
                        if bm.get("key") == "pinnacle":
                            for mkt in bm.get("markets", []):
                                if mkt.get("key") == "h2h":
                                    outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                                    # Outcomes keyed by full team name
                                    home_name = evt["home_team"]
                                    away_name = evt["away_team"]
                                    ph = outcomes.get(home_name, float("nan"))
                                    pd_ = outcomes.get("Draw", float("nan"))
                                    pa = outcomes.get(away_name, float("nan"))
                                    if not any(math.isnan(v) for v in (ph, pd_, pa)):
                                        rec["pre_odds_h"] = ph
                                        rec["pre_odds_d"] = pd_
                                        rec["pre_odds_a"] = pa
                                        dv = devig(ph, pd_, pa)
                                        if dv:
                                            rec["pre_h"], rec["pre_d"], rec["pre_a"] = dv
                                        rec["source"] = "api"
                                        api_matched = True
                    break

        # ── Synthetic fallback ─────────────────────────────────────────────
        if not api_matched:
            syn = synthetic_pre_odds(
                rec["closing_odds_h"], rec["closing_odds_d"], rec["closing_odds_a"]
            )
            if syn:
                rec["pre_h"], rec["pre_d"], rec["pre_a"] = syn
                # Back-convert to odds for storage
                o_h, o_d, o_a = convert_prob_to_odds(*syn, margin=0.035)
                rec["pre_odds_h"] = o_h
                rec["pre_odds_d"] = o_d
                rec["pre_odds_a"] = o_a
                rec["source"] = "synthetic"

        records.append(rec)

    return pd.DataFrame(records)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── Load matches ──────────────────────────────────────────────────────────
    if not PARQUET_PATH.exists():
        log.error(f"Matches parquet not found: {PARQUET_PATH}")
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    log.info(f"Loaded {len(df):,} matches from {PARQUET_PATH.name}")

    # ── Filter by date range ──────────────────────────────────────────────────
    if args.from_date:
        from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").date()
        df = df[df["date"] >= from_dt]
    if args.to_date:
        to_dt = datetime.strptime(args.to_date, "%Y-%m-%d").date()
        df = df[df["date"] <= to_dt]

    log.info(f"Filtered to {len(df):,} matches in [{args.from_date or 'all'} → {args.to_date or 'all'}]")

    # Unique matchdays
    matchdays = sorted(df["date"].unique())
    log.info(f"Unique matchdays: {len(matchdays)}")

    # ── Status mode ───────────────────────────────────────────────────────────
    manifest = load_manifest()
    downloaded = set(manifest.get("downloaded_dates", [])) | set(manifest.get("synthetic_dates", []))
    pending = [d for d in matchdays if str(d) not in downloaded]

    if args.status:
        print(f"\n{'='*60}")
        print(f"  Snapshot Status")
        print(f"{'='*60}")
        print(f"  Total matchdays in range : {len(matchdays)}")
        print(f"  Downloaded (real API)    : {len(manifest.get('downloaded_dates', []))}")
        print(f"  Generated (synthetic)    : {len(manifest.get('synthetic_dates', []))}")
        print(f"  Pending                  : {len(pending)}")
        print(f"  Total API requests used  : {manifest.get('total_requests', 0)}")
        print(f"{'='*60}\n")
        return

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        log.info("DRY RUN — no API calls will be made")
        for d in pending[:10]:
            day_df = df[df["date"] == d]
            log.info(f"  Would process: {d} ({len(day_df)} matches)")
        if len(pending) > 10:
            log.info(f"  ... and {len(pending)-10} more dates")
        return

    # ── Real ingestion ────────────────────────────────────────────────────────
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        log.warning("ODDS_API_KEY not set — using synthetic pre-closing odds")
        use_synthetic = True
    else:
        log.info(f"ODDS_API_KEY found — will use real historical snapshots")
        use_synthetic = False

    snapshotter = OddsAPISnapshotter(api_key=api_key)

    total_processed = 0
    total_real = 0
    total_synthetic = 0

    for date in pending:
        date_str = str(date)
        day_df = df[df["date"] == date].copy()
        if day_df.empty:
            continue

        log.info(f"  Processing {date_str} ({len(day_df)} matches) ...")

        snap_df = ingest_date(date_str, day_df, snapshotter, use_synthetic)

        # Save to parquet
        out_path = SNAPSHOTS_DIR / f"{date_str}.parquet"
        snap_df.to_parquet(out_path, index=False)

        # Update manifest
        real_rows = (snap_df["source"] == "api").sum()
        synth_rows = (snap_df["source"] == "synthetic").sum()
        total_real += real_rows
        total_synthetic += synth_rows
        total_processed += len(snap_df)

        if real_rows > 0:
            manifest["downloaded_dates"].append(date_str)
        else:
            manifest["synthetic_dates"].append(date_str)
        manifest["total_requests"] = manifest.get("total_requests", 0) + snapshotter.requests_used
        snapshotter.requests_used = 0  # reset counter per date
        save_manifest(manifest)

        log.info(f"    → {real_rows} real + {synth_rows} synthetic | saved {out_path.name}")

        # Quota check
        if snapshotter.requests_remaining is not None and snapshotter.requests_remaining < 20:
            log.warning(f"  ⚠ Only {snapshotter.requests_remaining} API requests remaining — stopping")
            break

    log.info(
        f"\n{'='*55}\n"
        f"  Ingestion complete\n"
        f"  Dates processed  : {len(pending)} processed\n"
        f"  Matches total    : {total_processed:,}\n"
        f"  Real API rows    : {total_real:,}\n"
        f"  Synthetic rows   : {total_synthetic:,}\n"
        f"{'='*55}"
    )


# ── Load snapshot for a specific date (used by V5 backtest) ─────────────────

def load_snapshot(date_str: str) -> Optional[pd.DataFrame]:
    """Load the pre-closing snapshot for a date. Returns None if not found."""
    path = SNAPSHOTS_DIR / f"{date_str}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def load_all_snapshots(from_date: str, to_date: str) -> pd.DataFrame:
    """
    Load and concatenate all snapshots between two dates.
    Used by V5 backtest to attach pre-closing odds to all matches.
    """
    from_dt = datetime.strptime(from_date, "%Y-%m-%d").date()
    to_dt   = datetime.strptime(to_date,   "%Y-%m-%d").date()

    dfs = []
    missing = []
    current = from_dt
    while current <= to_dt:
        ds = str(current)
        snap = load_snapshot(ds)
        if snap is not None:
            dfs.append(snap)
        else:
            missing.append(ds)
        current += timedelta(days=1)

    if missing:
        # Only log summary — not every missing date (too noisy)
        matchdays_missing = [d for d in missing if Path(SNAPSHOTS_DIR / f"{d}.parquet").parent.exists()]
        log.debug(f"load_all_snapshots: {len(missing)} dates with no snapshot file (no matches = normal)")

    if not dfs:
        log.warning(f"No snapshots found in [{from_date} → {to_date}]. Run ingest first.")
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    log.info(f"Loaded {len(result):,} snapshot rows for [{from_date} → {to_date}]")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest pre-closing odds snapshots")
    parser.add_argument("--from", dest="from_date", default=None,
                        help="Start date YYYY-MM-DD (default: all history)")
    parser.add_argument("--to", dest="to_date", default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without making API calls")
    parser.add_argument("--status", action="store_true",
                        help="Show download progress and exit")
    parser.add_argument("--force", action="store_true",
                        help="Re-download already-processed dates")
    args = parser.parse_args()

    main(args)
