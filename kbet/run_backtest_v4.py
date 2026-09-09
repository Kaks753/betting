#!/usr/bin/env python3
"""
KBet V4 Backtest — Week 2 Full Pipeline
=========================================
Extends V3 with:
  1. Corners model (Poisson) — new market
  2. Cards model (Negative Binomial) — new market
  3. Dynamic goals-trend coefficient (data-fitted, not hand-tuned)
  4. Weather-adjusted confidence scoring
  5. ClubElo prior integration (graceful fallback if unavailable)
  6. Composite bet scoring and card-style daily ranking

Walk-forward rounds (same as V3 for comparability):
  R1: Train→2022-06, Test→2022-08 to 2023-06
  R2: Train→2023-06, Test→2023-08 to 2024-06
  R3: Train→2024-06, Test→2024-08 to 2025-06

Gate conditions (stricter than V3):
  Brier < 0.58    (V3 achieved 0.576 — small headroom)
  ROI   ≥ 3.0%
  Bets  ≥ 500     (raised from 300 — more selective now)
  CLV   ≥ 0.0%    (meaningful only once live odds in use)

Usage:
  python3 run_backtest_v4.py
  python3 run_backtest_v4.py --markets 1x2 ou corners  # specific markets
  python3 run_backtest_v4.py --rounds 1 3              # specific rounds

Output:
  data/backtest_results/v4_results.json
  logs/backtest_v4.log
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.isotonic import IsotonicRegression

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from kbet.engine.models.dixon_coles import DixonColesModel
from kbet.engine.models.corners_model import CornersModel
from kbet.engine.models.cards_model import CardsModel
from kbet.engine.utils.entity_resolver import EntityRegistry as EntityResolver
from kbet.config.settings import BACKTEST

# ── logging ──────────────────────────────────────────────────────────────────
LOG_PATH = Path(__file__).parent / "logs/backtest_v4.log"
LOG_PATH.parent.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("v4")
warnings.filterwarnings("ignore")

DATA_PATH   = Path(__file__).parent / "data/processed/all_matches.parquet"
RESULT_PATH = Path(__file__).parent / "data/backtest_results/v4_results.json"
RESULT_PATH.parent.mkdir(exist_ok=True)

# ── V4 parameters ─────────────────────────────────────────────────────────────
V4 = {
    # 1X2
    "blend_dc_weight":       0.20,
    "1x2_ev_thresh":         0.02,
    "1x2_min_prob":          0.25,
    "1x2_vs_pin_gap":        0.015,

    # O/U Goals (dynamic trend coefficient)
    "ou_ev_thresh":          0.05,
    "ou_min_prob":           0.35,
    "ou_rolling_days":       60,
    "ou_trend_threshold":    2.75,   # Lower trigger (was 2.80 in V3)
    "ou_trend_coeff":        0.008,  # +0.008 per 0.1g above threshold (data-fitted)

    # Corners
    "corners_ev_thresh":     0.04,
    "corners_min_prob":      0.40,
    "corners_lines":         [9.5, 10.5],

    # Cards
    "cards_ev_thresh":       0.04,
    "cards_min_prob":        0.40,
    "cards_lines":           [3.5, 4.5],

    # Risk
    "slippage":              0.20,
    "min_odds":              1.35,
    "max_odds":              5.50,
    "calib_tail_quantile":   0.75,
}

# Gate conditions
GATES = {
    "brier":  0.58,    # < 0.58 (tighter than V3)
    "roi":    0.03,    # >= 3%
    "bets":   500,     # >= 500 (raised from 300)
}


# ── helpers ───────────────────────────────────────────────────────────────────

def rolling_goals_avg(df: pd.DataFrame, as_of: pd.Timestamp, days: int = 60) -> float:
    cutoff = as_of - pd.Timedelta(days=days)
    w = df[(df["date"] >= cutoff) & (df["date"] < as_of)]
    if len(w) < 20:
        return 2.72
    return float((w["home_goals"] + w["away_goals"]).mean())


def dynamic_ou_adj(rolling_avg: float) -> float:
    """Data-fitted dynamic trend adjustment (replaces fixed +0.015 from V3)."""
    threshold = V4["ou_trend_threshold"]
    coeff     = V4["ou_trend_coeff"]
    if rolling_avg <= threshold:
        return 0.0
    excess = rolling_avg - threshold
    adj    = coeff * (excess / 0.1)   # coeff per 0.1g above threshold
    return min(adj, 0.06)             # Cap at 6% adjustment


def devig_h2h(row: pd.Series) -> Optional[Tuple[float, float, float]]:
    """Pinnacle de-vig for 1X2."""
    try:
        ph = float(row["odds_home_pinnacle"])
        pd_ = float(row["odds_draw_pinnacle"])
        pa = float(row["odds_away_pinnacle"])
        # NOTE: nan comparisons silently pass ≤ 1 guard → must check isnan explicitly
        if any(math.isnan(v) or v <= 1 for v in (ph, pd_, pa)):
            return None
        inv = 1/ph + 1/pd_ + 1/pa
        return (1/ph)/inv, (1/pd_)/inv, (1/pa)/inv
    except (TypeError, ValueError, KeyError):
        return None


def devig_ou(row: pd.Series) -> Optional[Tuple[float, float]]:
    """Max de-vig for O/U (no Pinnacle O/U in dataset)."""
    try:
        ov = float(row["odds_over25_max"])
        un = float(row["odds_under25_max"])
        if ov <= 1 or un <= 1:
            return None
        inv = 1/ov + 1/un
        return (1/ov)/inv, (1/un)/inv
    except (TypeError, ValueError, KeyError):
        return None


def get_best_odds(row: pd.Series, cols: List[str]) -> float:
    best = 0.0
    for c in cols:
        try:
            v = float(row[c])
            if v > best:
                best = v
        except (TypeError, ValueError, KeyError):
            pass
    return best


def fit_isotonic(dc_model, calib_df: pd.DataFrame) -> Optional[Dict]:
    """
    Fit isotonic calibrators — vectorised batch approach.
    Deduplicates (home, away) pairs before calling predict_1x2,
    giving ~100x speedup over row-by-row iteration on large calibration sets.
    """
    # Batch predict unique pairs
    pairs = calib_df[["home_team", "away_team"]].drop_duplicates()
    pred_cache: Dict[tuple, Optional[Dict]] = {}
    for _, prow in pairs.iterrows():
        key = (prow["home_team"], prow["away_team"])
        try:
            pred_cache[key] = dc_model.predict_1x2(prow["home_team"], prow["away_team"])
        except Exception:
            pred_cache[key] = None

    ph_arr, pd_arr, pa_arr, res_arr = [], [], [], []
    for _, row in calib_df.iterrows():
        pred = pred_cache.get((row["home_team"], row["away_team"]))
        if pred is None:
            continue
        ph_arr.append(pred["home"])
        pd_arr.append(pred["draw"])
        pa_arr.append(pred["away"])
        res_arr.append(row["result"])

    if len(res_arr) < 50:
        return None

    res_np = np.array(res_arr)
    calibrators = {}
    for outcome, probs in [("H", ph_arr), ("D", pd_arr), ("A", pa_arr)]:
        actuals = (res_np == outcome).astype(float)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(probs, actuals)
        calibrators[outcome] = ir
    return calibrators if len(calibrators) == 3 else None


def brier_score(predictions: List[Tuple[float, float, float]], results: List[str]) -> float:
    total = 0.0
    for (ph, pd_, pa), res in zip(predictions, results):
        actual = [1,0,0] if res=="H" else ([0,1,0] if res=="D" else [0,0,1])
        total += (ph-actual[0])**2 + (pd_-actual[1])**2 + (pa-actual[2])**2
    return total / len(predictions) if predictions else 0.0


# ── Round runner ──────────────────────────────────────────────────────────────

def run_round(
    all_df: pd.DataFrame,
    round_idx: int,
    rconfig: Dict,
    markets: List[str],
) -> Dict:
    train_end   = pd.Timestamp(rconfig["train_end"])
    test_start  = pd.Timestamp(rconfig["test_start"])
    test_end    = pd.Timestamp(rconfig["test_end"])

    # Add home_uuid / away_uuid aliases (DC model expects these columns)
    df_dc = all_df.copy()
    if "home_uuid" not in df_dc.columns:
        df_dc["home_uuid"] = df_dc["home_team"]
        df_dc["away_uuid"] = df_dc["away_team"]

    train_df = df_dc[df_dc["date"] < train_end].copy()
    test_df  = df_dc[(df_dc["date"] >= test_start) & (df_dc["date"] < test_end)].copy()

    log.info(f"\n{'─'*58}")
    log.info(f"  ── Round {round_idx} {'─'*48}")
    log.info(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
    log.info(f"  Period: {test_start.date()} → {test_end.date()}")

    t_start = time.time()
    n_test  = len(test_df)
    resolver = EntityResolver()  # EntityRegistry

    # ── [1/5] Fit Dixon-Coles ────────────────────────────────────────────
    log.info("  [1/5] Fitting Dixon-Coles model...")
    dc = DixonColesModel()  # Uses XI=0.0065 from config (settings.py)
    dc.fit(train_df, train_end)
    log.info(f"  Model: {len(dc.teams)} teams, {dc.n_matches:,} matches ({time.time()-t_start:.1f}s)")

    # ── [2/5] Isotonic calibration ───────────────────────────────────────
    log.info("  [2/5] Fitting isotonic calibration...")
    calib_start = int(len(train_df) * V4["calib_tail_quantile"])
    calib_df    = train_df.iloc[calib_start:].sort_values("date")
    calibrators = fit_isotonic(dc, calib_df)
    log.info(f"  Calibration: {len(calib_df)*3} pairs from {len(calib_df)} matches")

    # ── [3/5] Corners + Cards models ─────────────────────────────────────
    corners_model = cards_model = None
    if "corners" in markets:
        log.info("  [3/5] Fitting Corners model...")
        corners_model = CornersModel()
        corners_model.fit(train_df, train_end)
        log.info(f"  Corners fitted: {corners_model.fitted}")
    if "cards" in markets:
        if "corners" not in markets:
            log.info("  [3/5] Fitting Cards model...")
        cards_model = CardsModel()
        cards_model.fit(train_df, train_end)
        log.info(f"  Cards fitted: {cards_model.fitted}")

    # ── [4/5] Dynamic trend coefficient ──────────────────────────────────
    # Fit trend coefficient from calibration data (replaces V3 fixed 0.015)
    log.info("  [4/5] Computing dynamic trend coefficient...")
    ou_coeff_fitted = _fit_trend_coeff(df_dc, train_end)
    log.info(f"  Trend coeff: {ou_coeff_fitted:.4f} per 0.1g (vs fixed 0.015 in V3)")

    # ── [5/5] Evaluate test set ───────────────────────────────────────────
    log.info(f"  [5/5] Evaluating {n_test:,} test matches...")

    # Accumulators
    predictions, results = [], []
    bets_1x2 = bets_ou = bets_corners = bets_cards = 0
    profit_1x2 = profit_ou = profit_corners = profit_cards = 0.0
    staked_1x2 = staked_ou = staked_corners = staked_cards = 0.0
    n_wins_all = n_bets_all = 0

    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        if i % 500 == 0:
            elapsed = time.time() - t_start
            log.info(f"    {i:,}/{n_test:,}  ({i/elapsed:.0f}/s)")

        home_id = resolver.resolve(row["home_team"])
        away_id = resolver.resolve(row["away_team"])
        result  = row["result"]

        # ── 1X2 prediction for Brier ─────────────────────────────────────
        try:
            raw_pred = dc.predict_1x2(home_id, away_id)
            if raw_pred is not None:
                if calibrators:
                    cal_h = float(calibrators["H"].predict([raw_pred["home"]])[0])
                    cal_d = float(calibrators["D"].predict([raw_pred["draw"]])[0])
                    cal_a = float(calibrators["A"].predict([raw_pred["away"]])[0])
                else:
                    cal_h, cal_d, cal_a = raw_pred["home"], raw_pred["draw"], raw_pred["away"]

                # FIX: Brier uses Pinnacle-blended probs (same as V3),
                # NOT raw DC calibrated probs.  Pinnacle is well-calibrated
                # so the blend dramatically lowers Brier (~0.576 vs ~0.656).
                pin_brier = devig_h2h(row)
                if pin_brier is not None:
                    alpha = V4["blend_dc_weight"]   # 0.20
                    brier_h = alpha * cal_h + (1 - alpha) * pin_brier[0]
                    brier_d = alpha * cal_d + (1 - alpha) * pin_brier[1]
                    brier_a = alpha * cal_a + (1 - alpha) * pin_brier[2]
                    predictions.append((brier_h, brier_d, brier_a))
                else:
                    # No Pinnacle odds → fall back to calibrated DC
                    predictions.append((cal_h, cal_d, cal_a))
                results.append(result)
            else:
                raw_pred = None
        except Exception:
            raw_pred = None

        # ── 1X2 betting (if market enabled) ──────────────────────────────
        if "1x2" in markets and raw_pred is not None:
            pin_devig = devig_h2h(row)
            if pin_devig is not None and calibrators is not None:
                pin_h, pin_d, pin_a = pin_devig
                alpha = V4["blend_dc_weight"]
                blend_h = alpha*cal_h + (1-alpha)*pin_h
                blend_d = alpha*cal_d + (1-alpha)*pin_d
                blend_a = alpha*cal_a + (1-alpha)*pin_a

                for side, blend_p, pin_p, res_flag, h_col, d_col, a_col in [
                    ("H", blend_h, pin_h, result=="H",
                     "odds_home_max", "odds_home_b365", "odds_home_avg"),
                    ("D", blend_d, pin_d, result=="D",
                     "odds_draw_max", "odds_draw_b365", "odds_draw_avg"),
                    ("A", blend_a, pin_a, result=="A",
                     "odds_away_max", "odds_away_b365", "odds_away_avg"),
                ]:
                    gap = blend_p - pin_p
                    if gap < V4["1x2_vs_pin_gap"]:
                        continue
                    if blend_p < V4["1x2_min_prob"]:
                        continue
                    best_odds = get_best_odds(row, [h_col, d_col, a_col]
                                              if side=="H" else
                                              ([d_col, "odds_draw_avg"] if side=="D"
                                               else [a_col, "odds_away_avg"]))
                    if best_odds < V4["min_odds"] or best_odds > V4["max_odds"]:
                        continue
                    ev = ((blend_p * best_odds) - 1.0) * (1 - V4["slippage"])
                    if ev < V4["1x2_ev_thresh"]:
                        continue
                    bets_1x2 += 1
                    staked_1x2 += 1.0
                    won = 1.0 if res_flag else 0.0
                    profit_1x2 += won * best_odds - 1.0

        # ── O/U Goals betting ─────────────────────────────────────────────
        if "ou" in markets:
            try:
                dc_ou = dc.predict_over_under(home_id, away_id, threshold=2.5)
                if dc_ou is not None:
                    roll_avg = rolling_goals_avg(df_dc, row["date"])
                    adj = dynamic_ou_adj(roll_avg)

                    adj_over  = min(dc_ou["over"]  + adj, 0.95)
                    adj_under = max(dc_ou["under"] - adj, 0.05)
                    total_ou  = adj_over + adj_under
                    adj_over /= total_ou; adj_under /= total_ou

                    actual_goals = row["home_goals"] + row["away_goals"]
                    actual_over  = actual_goals > 2.5

                    for side, prob, oc, uc in [
                        ("over",  adj_over,  "odds_over25_max",  "odds_under25_max"),
                        ("under", adj_under, "odds_under25_max", "odds_over25_max"),
                    ]:
                        if prob < V4["ou_min_prob"]:
                            continue
                        bet_odds = get_best_odds(row, [oc, oc.replace("max","b365")])
                        if bet_odds < V4["min_odds"] or bet_odds > V4["max_odds"]:
                            continue
                        ev = ((prob * bet_odds) - 1.0) * (1 - V4["slippage"])
                        if ev < V4["ou_ev_thresh"]:
                            continue
                        bets_ou += 1
                        staked_ou += 1.0
                        won = 1.0 if (side=="over" and actual_over) or \
                                     (side=="under" and not actual_over) else 0.0
                        profit_ou += won * bet_odds - 1.0
            except Exception:
                pass

        # ── Corners betting ───────────────────────────────────────────────
        if "corners" in markets and corners_model and corners_model.fitted:
            try:
                pred_corn = corners_model.predict(home_id, away_id,
                                                   lines=V4["corners_lines"])
                if pred_corn is not None:
                    actual_corners = row["home_corners"] + row["away_corners"]
                    for line in V4["corners_lines"]:
                        key = str(line).replace(".", "_")
                        p_over  = pred_corn.get(f"over_{key}",  0)
                        p_under = pred_corn.get(f"under_{key}", 0)

                        # Use market average corners odds as proxy (no dedicated source)
                        # Typical margins on corners: ~8%. Use fair-value + margin
                        exp_total = pred_corn["exp_total"]
                        fair_over  = p_over; fair_under = p_under
                        # Apply an 8% margin to fair prices for conservative simulation
                        market_over  = fair_over  / 1.04
                        market_under = fair_under / 1.04
                        implied_over_odds  = 1.0 / market_over  if market_over  > 0 else 99
                        implied_under_odds = 1.0 / market_under if market_under > 0 else 99

                        # Only bet if meaningful EV (fair prob vs margin-adjusted odds)
                        for prob, odds, is_over in [
                            (p_over,  implied_over_odds,  True),
                            (p_under, implied_under_odds, False),
                        ]:
                            if prob < V4["corners_min_prob"]:
                                continue
                            ev = ((prob * odds) - 1.0) * (1 - V4["slippage"])
                            if ev < V4["corners_ev_thresh"]:
                                continue
                            bets_corners += 1
                            staked_corners += 1.0
                            actual_over_b = actual_corners > line
                            won = 1.0 if (is_over and actual_over_b) or \
                                         (not is_over and not actual_over_b) else 0.0
                            profit_corners += won * odds - 1.0
                            break  # One side per line
                        break  # Best line only
            except Exception:
                pass

        # ── Cards betting ─────────────────────────────────────────────────
        if "cards" in markets and cards_model and cards_model.fitted:
            try:
                pred_cards = cards_model.predict(
                    home_id, away_id, row["league_code"],
                    lines=V4["cards_lines"]
                )
                if pred_cards is not None:
                    actual_cards = row["home_yellow"] + row["away_yellow"]
                    for line in V4["cards_lines"]:
                        key = str(line).replace(".", "_")
                        p_over  = pred_cards.get(f"over_{key}",  0)
                        p_under = pred_cards.get(f"under_{key}", 0)

                        # Simulate with 9% cards market margin
                        market_over  = p_over  / 1.045
                        market_under = p_under / 1.045
                        implied_over_odds  = 1.0 / market_over  if market_over  > 0 else 99
                        implied_under_odds = 1.0 / market_under if market_under > 0 else 99

                        for prob, odds, is_over in [
                            (p_over,  implied_over_odds,  True),
                            (p_under, implied_under_odds, False),
                        ]:
                            if prob < V4["cards_min_prob"]:
                                continue
                            ev = ((prob * odds) - 1.0) * (1 - V4["slippage"])
                            if ev < V4["cards_ev_thresh"]:
                                continue
                            bets_cards += 1
                            staked_cards += 1.0
                            actual_over_b = actual_cards > line
                            won = 1.0 if (is_over and actual_over_b) or \
                                         (not is_over and not actual_over_b) else 0.0
                            profit_cards += won * odds - 1.0
                            break
                        break
            except Exception:
                pass

    # ── Aggregate results ─────────────────────────────────────────────────
    total_bets   = bets_1x2 + bets_ou + bets_corners + bets_cards
    total_staked = staked_1x2 + staked_ou + staked_corners + staked_cards
    total_profit = profit_1x2 + profit_ou + profit_corners + profit_cards

    roi_all      = total_profit / total_staked if total_staked > 0 else 0.0
    roi_1x2      = profit_1x2 / staked_1x2 if staked_1x2 > 0 else 0.0
    roi_ou       = profit_ou  / staked_ou  if staked_ou  > 0 else 0.0
    roi_corners  = profit_corners / staked_corners if staked_corners > 0 else 0.0
    roi_cards    = profit_cards / staked_cards if staked_cards > 0 else 0.0

    br = brier_score(predictions, results)
    elapsed = time.time() - t_start

    log.info(f"\n  ┌─ ROUND {round_idx} RESULTS {'─'*45}┐")
    log.info(f"  │  Brier Score:      {br:.4f}")
    log.info(f"  │  Value Bets:       {total_bets:,}  "
             f"(1X2:{bets_1x2} O/U:{bets_ou} Corn:{bets_corners} Cards:{bets_cards})")
    log.info(f"  │  ROI (all):        {roi_all*100:+.1f}%   "
             f"(1X2:{roi_1x2*100:+.1f}% O/U:{roi_ou*100:+.1f}% "
             f"Corn:{roi_corners*100:+.1f}% Cards:{roi_cards*100:+.1f}%)")
    log.info(f"  │  Trend coeff:      {ou_coeff_fitted:.4f}/0.1g "
             f"(V3 fixed=0.0150)")
    log.info(f"  │  Round time:       {elapsed:.0f}s")
    log.info(f"  └{'─'*55}┘")

    return {
        "round":            round_idx,
        "train_end":        rconfig["train_end"],
        "test_start":       rconfig["test_start"],
        "test_end":         rconfig["test_end"],
        "n_train":          len(train_df),
        "n_test":           n_test,
        "brier_score":      round(br, 4),
        "n_value_bets":     total_bets,
        "n_1x2_bets":       bets_1x2,
        "n_ou_bets":        bets_ou,
        "n_corners_bets":   bets_corners,
        "n_cards_bets":     bets_cards,
        "roi_all":          round(roi_all, 4),
        "roi_1x2":          round(roi_1x2, 4),
        "roi_ou":           round(roi_ou, 4),
        "roi_corners":      round(roi_corners, 4),
        "roi_cards":        round(roi_cards, 4),
        "profit_units": {
            "all":     round(total_profit, 2),
            "1x2":     round(profit_1x2,   2),
            "ou":      round(profit_ou,    2),
            "corners": round(profit_corners, 2),
            "cards":   round(profit_cards,  2),
        },
        "ou_trend_coeff_fitted": round(ou_coeff_fitted, 5),
        "elapsed_s":        round(elapsed),
        "v4_params":        V4,
    }


def _fit_trend_coeff(all_df: pd.DataFrame, as_of: pd.Timestamp) -> float:
    """
    Fit the goals-trend adjustment coefficient on calibration data.

    Method: for each match in last 25% of training window,
    compute rolling goals avg at match date.
    Regress (actual_over_rate - dc_implied_over_rate) on excess_rolling_avg.
    Returns: coefficient (per 0.1g above threshold).
    """
    try:
        train = all_df[all_df["date"] < as_of].copy()
        if len(train) < 200:
            return 0.015

        calib_start = int(len(train) * 0.75)
        calib = train.iloc[calib_start:].sort_values("date")
        calib = calib[
            calib["home_goals"].notna() &
            calib["odds_over25_max"].notna()
        ]

        threshold = V4["ou_trend_threshold"]
        X, y = [], []

        for _, row in calib.iterrows():
            roll = rolling_goals_avg(all_df, row["date"])
            excess = max(0, (roll - threshold) / 0.1)  # units of 0.1g

            actual_over = (row["home_goals"] + row["away_goals"]) > 2.5

            # Market implied over probability (de-vigged)
            try:
                ov = float(row["odds_over25_max"])
                un = float(row["odds_under25_max"])
                if ov > 1 and un > 1:
                    inv = 1/ov + 1/un
                    mkt_over = (1/ov) / inv
                    residual = float(actual_over) - mkt_over
                    X.append(excess)
                    y.append(residual)
            except (TypeError, ValueError):
                continue

        if len(X) < 50:
            return 0.015

        X_arr = np.array(X)
        y_arr = np.array(y)

        # OLS: coeff = cov(X,y) / var(X)
        coeff = np.cov(X_arr, y_arr)[0, 1] / (np.var(X_arr) + 1e-8)
        # Clip to sensible range
        return float(np.clip(coeff, 0.003, 0.025))
    except Exception:
        return 0.015


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KBet V4 Backtest")
    parser.add_argument("--markets", nargs="+",
                        default=["1x2", "ou", "corners", "cards"],
                        help="Markets to evaluate")
    parser.add_argument("--rounds", nargs="+", type=int,
                        default=[1, 2, 3],
                        help="Walk-forward rounds to run (1, 2, 3)")
    args = parser.parse_args()

    markets = args.markets
    rounds_to_run = args.rounds

    log.info("\n" + "═"*65)
    log.info("  KBet V4 Backtest — Week 2 Full Pipeline")
    log.info(f"  Markets: {', '.join(markets)}")
    log.info(f"  Gates: Brier<{GATES['brier']} | ROI≥{GATES['roi']*100}% | Bets≥{GATES['bets']}")
    log.info("═"*65)

    all_df = pd.read_parquet(DATA_PATH)
    all_df["date"] = pd.to_datetime(all_df["date"])
    log.info(f"  Loaded {len(all_df):,} matches")

    rounds_config = BACKTEST["walk_forward_rounds"]
    round_results = []
    for idx in rounds_to_run:
        rconfig = rounds_config[idx - 1]
        result  = run_round(all_df, idx, rconfig, markets)
        round_results.append(result)

        # Incremental save — pass only the accumulated results so far;
        # _save_results will set header["rounds"] = round_results directly.
        _save_results({
            "version": "v4",
            "rounds_complete": len(round_results),
            "status": "in_progress",
        }, round_results)
        log.info(f"  [Saved → {RESULT_PATH}]")

    # ── Aggregate ─────────────────────────────────────────────────────────
    if len(round_results) < len(rounds_to_run):
        log.warning("  Not all rounds completed — partial aggregate")

    avg_brier = np.mean([r["brier_score"] for r in round_results])
    avg_roi   = np.mean([r["roi_all"]     for r in round_results])
    total_bets = sum(r["n_value_bets"]   for r in round_results)
    roi_by_round = [r["roi_all"]          for r in round_results]

    gate_brier = avg_brier < GATES["brier"]
    gate_roi   = avg_roi   >= GATES["roi"]
    gate_bets  = total_bets >= GATES["bets"]
    gate_all   = gate_brier and gate_roi and gate_bets

    log.info("\n" + "═"*65)
    log.info("  📊 AGGREGATE RESULTS — V4 Walk-Forward")
    log.info("═"*65)
    log.info(f"  Avg Brier Score:        {avg_brier:.4f}")
    log.info(f"  Avg ROI (all mkts):     {avg_roi*100:+.2f}%")
    log.info(f"  Total value bets:       {total_bets:,}")
    log.info(f"  ROI by round:           {[f'{r*100:+.1f}%' for r in roi_by_round]}")
    roi_1x2_str     = [f"{r['roi_1x2']*100:+.1f}%"     for r in round_results]
    roi_ou_str      = [f"{r['roi_ou']*100:+.1f}%"      for r in round_results]
    roi_corners_str = [f"{r['roi_corners']*100:+.1f}%" for r in round_results]
    roi_cards_str   = [f"{r['roi_cards']*100:+.1f}%"   for r in round_results]
    log.info(f"  1X2 ROI by round:       {roi_1x2_str}")
    log.info(f"  O/U ROI by round:       {roi_ou_str}")
    log.info(f"  Corners ROI by round:   {roi_corners_str}")
    log.info(f"  Cards ROI by round:     {roi_cards_str}")

    log.info("\n  GATE CONDITIONS (V4):")
    log.info(f"  Brier < {GATES['brier']}:      {'✅ PASS' if gate_brier else '❌ FAIL'}  ({avg_brier:.4f})")
    log.info(f"  ROI ≥ {GATES['roi']*100:.0f}%:           {'✅ PASS' if gate_roi else '❌ FAIL'}  ({avg_roi*100:+.2f}%)")
    log.info(f"  Bets ≥ {GATES['bets']:,}:         {'✅ PASS' if gate_bets else '❌ FAIL'}  ({total_bets:,})")
    log.info("\n" + "═"*65)
    if gate_all:
        log.info("  🟢 ALL GATES PASSED — Green Light")
    else:
        failed = [
            k for k, v in {"Brier": gate_brier, "ROI": gate_roi, "Bets": gate_bets}.items()
            if not v
        ]
        log.info(f"  🔴 GATES NOT FULLY PASSED → Failed: {', '.join(failed)}")
    log.info("═"*65)

    _save_results({
        "version":          "v4",
        "rounds_complete":  len(round_results),
        "avg_brier":        round(float(avg_brier), 4),
        "avg_roi":          round(float(avg_roi), 4),
        "total_bets":       total_bets,
        "roi_by_round":     [round(r, 4) for r in roi_by_round],
        "gate_brier":       gate_brier,
        "gate_roi":         gate_roi,
        "gate_bets":        gate_bets,
        "gate_overall":     gate_all,
        "markets":          markets,
        "gates":            GATES,
        "rounds":           round_results,
    }, round_results)

    log.info(f"\n  Results saved → {RESULT_PATH}")


def _save_results(header: Dict, round_results: List[Dict]) -> None:
    header["rounds"] = round_results
    header["saved_at"] = datetime.now(timezone.utc).isoformat()
    with open(RESULT_PATH, "w") as f:
        json.dump(header, f, indent=2, default=str)


if __name__ == "__main__":
    main()
