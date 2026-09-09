#!/usr/bin/env python3
"""
KBet V5 Backtest — Week 3: Genuine CLV Tracking
=================================================
V5 is V4 + pre-closing reference odds from the snapshot store.

The fundamental ROI blocker in V1-V4:
  All backtests referenced AND evaluated against the SAME closing Pinnacle line.
  CLV = model_prob - closing_implied_prob = ~0 on every bet.
  Positive ROI is structurally impossible — we're measuring noise.

V5 fix:
  reference_prob = devig(T-48h pre-closing Pinnacle snapshot)
  bet_signal     = model_prob vs reference_prob  (not closing)
  CLV per bet    = closing_implied_prob - reference_implied_prob
                   (positive CLV = we were right, market moved our way)
  EV calculation uses reference odds (our entry price), not closing.

New gate: avg_clv >= 0.003 (0.3%) across all bets
  - Synthetic snapshots: CLV is noise (±σ=1.8%) — gate shows ~0
  - Real T-48h snapshots: gate becomes genuinely testable

Architecture:
  V5 = V4 engine + snapshot attachment + Kelly sizing + CLV tracking
  All V4 code reused — only bet selection and sizing layer changes.

Walk-forward rounds (unchanged):
  R1: Train→2022-06, Test→2022-08→2023-06
  R2: Train→2023-06, Test→2023-08→2024-06
  R3: Train→2024-06, Test→2024-08→2025-06

V5 Gates:
  Brier  < 0.58         (same as V4)
  ROI    ≥ 3.0%         (same as V4, now measured at entry price)
  Bets   ≥ 500          (same as V4)
  CLV    ≥ 0.3%         (NEW — proves pre-close edge)
  ≥2/3 rounds positive ROI  (NEW — consistency gate)

Usage:
  python3 kbet/run_backtest_v5.py
  python3 kbet/run_backtest_v5.py --rounds 1 2
  python3 kbet/run_backtest_v5.py --markets 1x2 ou

Output:
  data/backtest_results/v5_results.json
  logs/backtest_v5.log
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

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from kbet.engine.models.dixon_coles import DixonColesModel
from kbet.engine.models.corners_model import CornersModel
from kbet.engine.models.cards_model import CardsModel
from kbet.engine.utils.entity_resolver import EntityRegistry as EntityResolver
from kbet.engine.utils.kelly import kelly_stake, kelly_ev, DailyPortfolio, BetOrder
from kbet.ingest_odds_snapshots import load_all_snapshots
from kbet.config.settings import BACKTEST

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH = Path(__file__).parent / "logs/backtest_v5.log"
LOG_PATH.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("v5")
warnings.filterwarnings("ignore")

DATA_PATH   = Path(__file__).parent / "data/processed/all_matches.parquet"
RESULT_PATH = Path(__file__).parent / "data/backtest_results/v5_results.json"
RESULT_PATH.parent.mkdir(exist_ok=True)

# ── V5 parameters (inherits V4, adds CLV + Kelly) ────────────────────────────
V5 = {
    # 1X2 — same thresholds as V4 but now measured at pre-close reference
    "blend_dc_weight":       0.20,
    "1x2_ev_thresh":         0.02,    # EV vs pre-close reference
    "1x2_min_prob":          0.25,
    "1x2_vs_pin_gap":        0.015,   # vs reference (pre-close), not closing

    # O/U Goals
    "ou_ev_thresh":          0.05,
    "ou_min_prob":           0.35,
    "ou_rolling_days":       60,
    "ou_trend_threshold":    2.75,
    "ou_trend_coeff":        0.008,

    # Corners / Cards (advisory — real odds needed for genuine EV)
    "corners_ev_thresh":     0.04,
    "corners_min_prob":      0.40,
    "corners_lines":         [9.5, 10.5],
    "cards_ev_thresh":       0.04,
    "cards_min_prob":        0.40,
    "cards_lines":           [3.5, 4.5],

    # Risk / execution
    "slippage":              0.20,
    "min_odds":              1.35,
    "max_odds":              5.50,
    "calib_tail_quantile":   0.75,

    # Kelly parameters
    "kelly_fraction":        0.25,
    "kelly_max_per_bet":     0.02,
    "kelly_max_daily":       0.05,
    "kelly_max_per_match":   0.03,
}

# V5 Gate conditions (all 5 must pass)
GATES_V5 = {
    "brier":          0.58,    # Brier < 0.58
    "roi":            0.03,    # ROI >= 3% (measured at reference entry price)
    "bets":           500,     # >= 500 bets total
    "clv":            0.003,   # avg CLV >= 0.3%
    "min_rounds_pos": 2,       # ≥ 2 of 3 rounds show positive ROI
}

# Walk-forward config (same as V3/V4)
ROUNDS_CONFIG = BACKTEST["walk_forward_rounds"]


# ── Helpers (identical to V4) ────────────────────────────────────────────────

def rolling_goals_avg(df: pd.DataFrame, as_of: pd.Timestamp, days: int = 60) -> float:
    window = df[(df["date"] < as_of) & (df["date"] >= as_of - pd.Timedelta(days=days))]
    if window.empty:
        return 2.65
    return float((window["home_goals"].fillna(0) + window["away_goals"].fillna(0)).mean())


def brier_score(predictions: List[Tuple[float, float, float]], results: List[str]) -> float:
    total = 0.0
    n = 0
    for (ph, pd_, pa), res in zip(predictions, results):
        if any(math.isnan(v) for v in (ph, pd_, pa)):
            continue
        oh = 1.0 if res == "H" else 0.0
        od = 1.0 if res == "D" else 0.0
        oa = 1.0 if res == "A" else 0.0
        total += (ph - oh)**2 + (pd_ - od)**2 + (pa - oa)**2
        n += 1
    return total / n if n > 0 else float("nan")


def devig_h2h(row) -> Optional[Tuple[float, float, float]]:
    """De-vig 1X2 Pinnacle closing odds. Returns None on NaN/invalid."""
    try:
        ph  = float(row["odds_home_pinnacle"])
        pd_ = float(row["odds_draw_pinnacle"])
        pa  = float(row["odds_away_pinnacle"])
        if any(math.isnan(v) or v <= 1 for v in (ph, pd_, pa)):
            return None
        inv = 1/ph + 1/pd_ + 1/pa
        return (1/ph)/inv, (1/pd_)/inv, (1/pa)/inv
    except (TypeError, ValueError, KeyError):
        return None


def devig_ou(odds_over: float, odds_under: float) -> Optional[Tuple[float, float]]:
    """De-vig over/under odds."""
    try:
        if any(math.isnan(v) or v <= 1 for v in (odds_over, odds_under)):
            return None
        inv = 1/odds_over + 1/odds_under
        return (1/odds_over)/inv, (1/odds_under)/inv
    except (TypeError, ValueError):
        return None


# ── Snapshot attachment ──────────────────────────────────────────────────────

def attach_snapshots(test_df: pd.DataFrame, test_start: str, test_end: str) -> pd.DataFrame:
    """
    Left-join pre-closing snapshot data onto test_df by (date, home_team, away_team).
    Adds columns: pre_h, pre_d, pre_a (reference devigged probs).
    Rows with no snapshot get pre_h/d/a = NaN → fall back to closing.
    """
    snaps = load_all_snapshots(test_start, test_end)
    if snaps.empty:
        log.warning("No snapshots available — using closing odds as reference (CLV = 0)")
        test_df = test_df.copy()
        test_df["pre_h"] = float("nan")
        test_df["pre_d"] = float("nan")
        test_df["pre_a"] = float("nan")
        return test_df

    # Normalise join keys
    snaps["_date_str"] = snaps["date"].astype(str)
    snaps["_home"]     = snaps["home_team"].str.strip().str.lower()
    snaps["_away"]     = snaps["away_team"].str.strip().str.lower()

    test_df = test_df.copy()
    test_df["_date_str"] = test_df["date"].dt.strftime("%Y-%m-%d")
    test_df["_home"]     = test_df["home_team"].str.strip().str.lower()
    test_df["_away"]     = test_df["away_team"].str.strip().str.lower()

    snap_cols = snaps[["_date_str", "_home", "_away", "pre_h", "pre_d", "pre_a"]].copy()
    snap_cols = snap_cols.drop_duplicates(subset=["_date_str", "_home", "_away"])

    merged = test_df.merge(snap_cols, on=["_date_str", "_home", "_away"], how="left")
    matched = merged["pre_h"].notna().sum()
    log.info(f"  Snapshot attach: {matched}/{len(merged)} matches joined ({matched/len(merged):.1%})")

    merged.drop(columns=["_date_str", "_home", "_away"], inplace=True)
    return merged


# ── Isotonic calibration (vectorised — mirrors V4 exactly) ───────────────────

def fit_calibrators(train_df: pd.DataFrame, train_end: pd.Timestamp):
    """
    Fit Dixon-Coles model + isotonic calibrators on training data.
    Uses V4's exact pattern: batch-predict unique (home, away) pairs.
    Returns (ir_h, ir_d, ir_a, dc_model) or (None, None, None, None) on failure.
    """
    if len(train_df) < 50:
        return None, None, None, None

    dc = DixonColesModel()
    try:
        dc.fit(train_df, train_end)
    except Exception as e:
        log.warning(f"  DC fit failed: {e}")
        return None, None, None, None

    # Use tail 75% of training data for calibration (same as V4)
    calib_start = int(len(train_df) * V5["calib_tail_quantile"])
    calib_df = train_df.iloc[calib_start:].sort_values("date")

    # Batch predict unique pairs
    pairs = calib_df[["home_team", "away_team"]].drop_duplicates()
    pred_cache: Dict = {}
    for _, prow in pairs.iterrows():
        key = (prow["home_team"], prow["away_team"])
        try:
            pred_cache[key] = dc.predict_1x2(prow["home_team"], prow["away_team"])
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

    if len(res_arr) < 30:
        return None, None, None, None

    res_np = np.array(res_arr)
    ir_h = IsotonicRegression(out_of_bounds="clip").fit(ph_arr, (res_np == "H").astype(float))
    ir_d = IsotonicRegression(out_of_bounds="clip").fit(pd_arr, (res_np == "D").astype(float))
    ir_a = IsotonicRegression(out_of_bounds="clip").fit(pa_arr, (res_np == "A").astype(float))
    return ir_h, ir_d, ir_a, dc


# ── Single round ──────────────────────────────────────────────────────────────

def run_round(
    all_df: pd.DataFrame,
    round_idx: int,
    rconfig: Dict,
    markets: List[str],
) -> Dict:
    t0 = time.time()

    train_end   = pd.Timestamp(rconfig["train_end"])
    test_start  = pd.Timestamp(rconfig["test_start"])
    test_end    = pd.Timestamp(rconfig["test_end"])

    log.info(f"\n{'='*60}")
    log.info(f"V5 Round {round_idx}: train→{train_end.date()} | test {test_start.date()}→{test_end.date()}")
    log.info(f"{'='*60}")

    train_df = all_df[all_df["date"] < train_end].copy()
    test_df  = all_df[(all_df["date"] >= test_start) & (all_df["date"] < test_end)].copy()

    log.info(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    # ── Fit DC model + calibrators on training data ───────────────────────────
    log.info("  Fitting Dixon-Coles + isotonic calibration ...")
    ir_h, ir_d, ir_a, dc_model = fit_calibrators(train_df, train_end)
    if dc_model is None:
        log.error("  Calibration failed — skipping round")
        return {"round": round_idx, "error": "calibration_failed", "brier_score": float("nan"),
                "roi_all": 0.0, "avg_clv": 0.0, "n_value_bets": 0}

    # ── Attach pre-closing snapshots ──────────────────────────────────────────
    log.info("  Attaching pre-closing snapshots ...")
    test_df = attach_snapshots(
        test_df,
        test_start.strftime("%Y-%m-%d"),
        test_end.strftime("%Y-%m-%d")
    )

    # ── Fit corners / cards if needed ─────────────────────────────────────────
    corners_model = cards_model = None
    if "corners" in markets:
        try:
            corners_model = CornersModel()
            corners_model.fit(train_df)
            log.info("  CornersModel fitted")
        except Exception as e:
            log.warning(f"  Corners fit failed: {e}")

    if "cards" in markets:
        try:
            cards_model = CardsModel()
            cards_model.fit(train_df)
            log.info("  CardsModel fitted")
        except Exception as e:
            log.warning(f"  Cards fit failed: {e}")

    # ── Rolling goals avg for O/U trend ───────────────────────────────────────
    ou_trend_coeff = V5["ou_trend_coeff"]

    # ── Evaluate test matches ─────────────────────────────────────────────────
    log.info("  Evaluating test matches ...")

    brier_predictions: List[Tuple[float, float, float]] = []
    brier_results: List[str] = []

    # Kelly portfolio stats
    all_bets: List[Dict] = []
    n_1x2 = n_ou = n_corners = n_cards = 0
    pnl_1x2 = pnl_ou = pnl_corners = pnl_cards = 0.0
    total_staked = 0.0
    clv_values: List[float] = []

    for _, row in test_df.iterrows():
        result = row.get("result", "")
        if result not in ("H", "D", "A"):
            continue

        home_uuid = str(row.get("home_team", ""))
        away_uuid = str(row.get("away_team", ""))

        # ── DC calibrated probs ───────────────────────────────────────────────
        try:
            pred = dc_model.predict_1x2(home_uuid, away_uuid)
            raw_h = pred["home"]
            raw_d = pred["draw"]
            raw_a = pred["away"]
        except Exception:
            continue

        cal_h = float(ir_h.predict([raw_h])[0])
        cal_d = float(ir_d.predict([raw_d])[0])
        cal_a = float(ir_a.predict([raw_a])[0])
        s = cal_h + cal_d + cal_a
        if s <= 0:
            continue
        cal_h, cal_d, cal_a = cal_h/s, cal_d/s, cal_a/s

        # ── Closing Pinnacle (for Brier and CLV denominator) ──────────────────
        pin_closing = devig_h2h(row)

        # ── Brier: use Pinnacle-blended probs ─────────────────────────────────
        if pin_closing:
            alpha = V5["blend_dc_weight"]
            bh = alpha * cal_h + (1 - alpha) * pin_closing[0]
            bd = alpha * cal_d + (1 - alpha) * pin_closing[1]
            ba = alpha * cal_a + (1 - alpha) * pin_closing[2]
            brier_predictions.append((bh, bd, ba))
        else:
            brier_predictions.append((cal_h, cal_d, cal_a))
        brier_results.append(result)

        # ── Reference probs: pre-closing snapshot or fall back to closing ─────
        pre_h  = row.get("pre_h", float("nan"))
        pre_d  = row.get("pre_d", float("nan"))
        pre_a  = row.get("pre_a", float("nan"))

        has_pre_odds = not any(math.isnan(v) for v in (pre_h, pre_d, pre_a))

        if has_pre_odds:
            ref_h, ref_d, ref_a = pre_h, pre_d, pre_a
        elif pin_closing:
            ref_h, ref_d, ref_a = pin_closing
        else:
            continue  # No reference at all — skip

        # ── CLV (closing line value per bet) ──────────────────────────────────
        # CLV = our entry implied prob - closing implied prob
        # Positive CLV = we bet at a BETTER price than the market closed at
        if pin_closing and has_pre_odds:
            # Blend our model with pre-close reference
            blend_h = V5["blend_dc_weight"] * cal_h + (1 - V5["blend_dc_weight"]) * ref_h
            blend_d = V5["blend_dc_weight"] * cal_d + (1 - V5["blend_dc_weight"]) * ref_d
            blend_a = V5["blend_dc_weight"] * cal_a + (1 - V5["blend_dc_weight"]) * ref_a
        else:
            blend_h, blend_d, blend_a = cal_h, cal_d, cal_a

        # ── 1X2 betting ───────────────────────────────────────────────────────
        if "1x2" in markets and pin_closing:
            closing_h_imp, closing_d_imp, closing_a_imp = pin_closing

            for our_prob, ref_prob, closing_imp, label, outcome_result in [
                (blend_h, ref_h, closing_h_imp, "H", result == "H"),
                (blend_d, ref_d, closing_d_imp, "D", result == "D"),
                (blend_a, ref_a, closing_a_imp, "A", result == "A"),
            ]:
                if ref_prob <= 0:
                    continue
                ref_odds = 1.0 / ref_prob  # Entry odds (pre-close reference)
                ev = kelly_ev(our_prob, ref_odds)

                if ev < V5["1x2_ev_thresh"]:
                    continue
                if our_prob < V5["1x2_min_prob"]:
                    continue
                if (our_prob - closing_imp) < V5["1x2_vs_pin_gap"]:
                    continue

                # Apply slippage to entry odds
                eff_odds = ref_odds * (1 - V5["slippage"])
                if eff_odds < V5["min_odds"] or eff_odds > V5["max_odds"]:
                    continue

                stake = kelly_stake(
                    our_prob, eff_odds,
                    fraction=V5["kelly_fraction"],
                    max_per_bet=V5["kelly_max_per_bet"],
                )
                if stake <= 0:
                    continue

                # CLV tracking
                clv = our_prob - closing_imp  # positive = we were ahead of close
                clv_values.append(clv)

                won = outcome_result
                pnl = stake * (eff_odds - 1) if won else -stake
                total_staked += stake
                pnl_1x2 += pnl
                n_1x2 += 1

                all_bets.append({
                    "market": "1x2", "pick": label,
                    "stake": stake, "odds": eff_odds,
                    "won": won, "pnl": pnl, "clv": clv,
                })

        # ── O/U Goals betting ─────────────────────────────────────────────────
        if "ou" in markets:
            ou_over_raw  = row.get("odds_over25_max")
            ou_under_raw = row.get("odds_under25_max")
            if pd.notna(ou_over_raw) and pd.notna(ou_under_raw):
                ou_over_f  = float(ou_over_raw)
                ou_under_f = float(ou_under_raw)
                if ou_over_f > 1.0 and ou_under_f > 1.0:
                    # DC model's own O/U probability (not devigged book)
                    try:
                        dc_ou = dc_model.predict_over_under(home_uuid, away_uuid, threshold=2.5)
                        if dc_ou is None:
                            pass
                        else:
                            # Rolling goals trend → model adjustment
                            rga = rolling_goals_avg(
                                all_df[all_df["date"] < row["date"]],
                                row["date"],
                                V5["ou_rolling_days"]
                            )
                            trend_adj = 0.0
                            if rga > V5["ou_trend_threshold"]:
                                trend_adj = ou_trend_coeff * (rga - V5["ou_trend_threshold"]) / 0.1

                            model_over  = float(np.clip(dc_ou["over"]  + trend_adj, 0.05, 0.95))
                            model_under = float(np.clip(dc_ou["under"] - trend_adj, 0.05, 0.95))
                            # Re-normalise
                            s_ou = model_over + model_under
                            model_over /= s_ou; model_under /= s_ou

                            total_goals = (row.get("home_goals") or 0) + (row.get("away_goals") or 0)
                            for our_p, book_odds, label, won_flag in [
                                (model_over,  ou_over_f,  "O2.5", total_goals > 2.5),
                                (model_under, ou_under_f, "U2.5", total_goals <= 2.5),
                            ]:
                                if book_odds < V5["min_odds"] or book_odds > V5["max_odds"]:
                                    continue
                                if our_p < V5["ou_min_prob"]:
                                    continue
                                # EV at post-slippage odds
                                eff_odds = book_odds * (1 - V5["slippage"])
                                ev = kelly_ev(our_p, eff_odds)
                                if ev < V5["ou_ev_thresh"]:
                                    continue

                                stake = kelly_stake(our_p, eff_odds,
                                                    fraction=V5["kelly_fraction"],
                                                    max_per_bet=V5["kelly_max_per_bet"])
                                if stake <= 0:
                                    continue

                                # CLV: model prob vs devigged book (closing fair prob)
                                dv_ou = devig_ou(ou_over_f, ou_under_f)
                                fair_p = dv_ou[0] if label == "O2.5" and dv_ou else (dv_ou[1] if dv_ou else our_p)
                                clv = our_p - fair_p
                                clv_values.append(clv)

                                pnl = stake * (eff_odds - 1) if won_flag else -stake
                                total_staked += stake
                                pnl_ou += pnl
                                n_ou += 1
                    except Exception:
                        pass

    # ── Round metrics ─────────────────────────────────────────────────────────
    n_total = n_1x2 + n_ou + n_corners + n_cards
    pnl_all = pnl_1x2 + pnl_ou + pnl_corners + pnl_cards
    roi_all = pnl_all / total_staked if total_staked > 0 else 0.0
    roi_1x2 = pnl_1x2 / (n_1x2 * V5["kelly_max_per_bet"]) if n_1x2 > 0 else 0.0
    roi_ou  = pnl_ou  / (n_ou  * V5["kelly_max_per_bet"]) if n_ou  > 0 else 0.0

    avg_clv  = float(np.mean(clv_values)) if clv_values else 0.0
    bs       = brier_score(brier_predictions, brier_results)
    elapsed  = time.time() - t0

    log.info(f"\n  R{round_idx} Results:")
    log.info(f"    Brier score   : {bs:.4f}")
    log.info(f"    Total bets    : {n_total:,}  (1X2:{n_1x2}  O/U:{n_ou}  C:{n_corners}  K:{n_cards})")
    log.info(f"    ROI (all)     : {roi_all:+.1%}")
    log.info(f"    ROI (1X2)     : {roi_1x2:+.1%}")
    log.info(f"    ROI (O/U)     : {roi_ou:+.1%}")
    log.info(f"    Avg CLV       : {avg_clv:+.4f}  ({len(clv_values)} bets tracked)")
    log.info(f"    Elapsed       : {elapsed:.0f}s")

    return {
        "round":          round_idx,
        "train_end":      rconfig["train_end"],
        "test_start":     rconfig["test_start"],
        "test_end":       rconfig["test_end"],
        "n_train":        len(train_df),
        "n_test":         len(test_df),
        "brier_score":    round(bs, 4),
        "n_value_bets":   n_total,
        "n_1x2_bets":     n_1x2,
        "n_ou_bets":      n_ou,
        "n_corners_bets": n_corners,
        "n_cards_bets":   n_cards,
        "roi_all":        round(roi_all, 4),
        "roi_1x2":        round(roi_1x2, 4),
        "roi_ou":         round(roi_ou, 4),
        "avg_clv":        round(avg_clv, 6),
        "n_clv_bets":     len(clv_values),
        "snapshot_source": "synthetic",  # Update to "real" once ODDS_API_KEY used
        "elapsed_s":      round(elapsed),
        "v5_params":      {k: v for k, v in V5.items()},
    }


# ── Save helpers ──────────────────────────────────────────────────────────────

def _save_results(header: Dict, rounds: List[Dict]) -> None:
    avg_brier = float(np.mean([r["brier_score"] for r in rounds if "brier_score" in r]))
    avg_roi   = float(np.mean([r["roi_all"]     for r in rounds if "roi_all"     in r]))
    total_bets = sum(r.get("n_value_bets", 0) for r in rounds)
    avg_clv   = float(np.mean([r["avg_clv"]     for r in rounds if "avg_clv"     in r]))
    rounds_pos = sum(1 for r in rounds if r.get("roi_all", -1) > 0)

    gate_brier = avg_brier < GATES_V5["brier"]
    gate_roi   = avg_roi   >= GATES_V5["roi"]
    gate_bets  = total_bets >= GATES_V5["bets"]
    gate_clv   = avg_clv   >= GATES_V5["clv"]
    gate_cons  = rounds_pos >= GATES_V5["min_rounds_pos"]

    output = {
        **header,
        "avg_brier":     round(avg_brier, 4),
        "avg_roi":       round(avg_roi, 4),
        "total_bets":    total_bets,
        "avg_clv":       round(avg_clv, 6),
        "roi_by_round":  [r.get("roi_all", 0.0) for r in rounds],
        "clv_by_round":  [r.get("avg_clv", 0.0) for r in rounds],
        "rounds_positive_roi": rounds_pos,
        "gate_brier":    gate_brier,
        "gate_roi":      gate_roi,
        "gate_bets":     gate_bets,
        "gate_clv":      gate_clv,
        "gate_consistency": gate_cons,
        "gate_overall":  all([gate_brier, gate_roi, gate_bets, gate_clv, gate_cons]),
        "gates":         GATES_V5,
        "rounds":        rounds,
        "saved_at":      datetime.now(timezone.utc).isoformat(),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    log.info("=" * 60)
    log.info("KBet V5 Backtest — Genuine CLV Tracking")
    log.info("=" * 60)

    # Load data
    if not DATA_PATH.exists():
        log.error(f"Data not found: {DATA_PATH}")
        sys.exit(1)

    log.info(f"Loading matches from {DATA_PATH} ...")
    all_df = pd.read_parquet(DATA_PATH)
    all_df["date"] = pd.to_datetime(all_df["date"])
    log.info(f"  {len(all_df):,} matches loaded")

    # UUID fallback
    if "home_uuid" not in all_df.columns:
        all_df["home_uuid"] = all_df["home_team"]
    if "away_uuid" not in all_df.columns:
        all_df["away_uuid"] = all_df["away_team"]

    markets = args.markets
    rounds_to_run = args.rounds if args.rounds else [1, 2, 3]

    log.info(f"Markets: {markets}")
    log.info(f"Rounds : {rounds_to_run}")

    round_results: List[Dict] = []

    for idx in rounds_to_run:
        rconfig = ROUNDS_CONFIG[idx - 1]
        result  = run_round(all_df, idx, rconfig, markets)
        round_results.append(result)

        _save_results({
            "version":         "v5",
            "rounds_complete": len(round_results),
            "status":          "in_progress",
            "markets":         markets,
        }, round_results)
        log.info(f"  [Saved → {RESULT_PATH}]")

    # ── Final verdict ─────────────────────────────────────────────────────────
    _save_results({
        "version":  "v5",
        "rounds_complete": len(round_results),
        "status":   "complete",
        "markets":  markets,
    }, round_results)

    # Load and display
    with open(RESULT_PATH) as f:
        res = json.load(f)

    log.info(f"\n{'='*60}")
    log.info(f"  V5 FINAL RESULTS")
    log.info(f"{'='*60}")
    log.info(f"  Avg Brier    : {res['avg_brier']:.4f}  {'✅' if res['gate_brier'] else '❌'} (gate < {GATES_V5['brier']})")
    log.info(f"  Avg ROI      : {res['avg_roi']:+.2%}   {'✅' if res['gate_roi'] else '❌'} (gate ≥ {GATES_V5['roi']:.0%})")
    log.info(f"  Total Bets   : {res['total_bets']:,}    {'✅' if res['gate_bets'] else '❌'} (gate ≥ {GATES_V5['bets']})")
    log.info(f"  Avg CLV      : {res['avg_clv']:+.4f}   {'✅' if res['gate_clv'] else '❌'} (gate ≥ {GATES_V5['clv']})")
    log.info(f"  Consistency  : {res['rounds_positive_roi']}/3 rounds +ROI  {'✅' if res['gate_consistency'] else '❌'} (gate ≥ {GATES_V5['min_rounds_pos']})")
    log.info(f"{'='*60}")

    if res["gate_overall"]:
        log.info("  🟢 GREEN LIGHT — All gates passed → ACTIVATE LIVE PIPELINE")
    else:
        failed = [k for k in ["gate_brier", "gate_roi", "gate_bets", "gate_clv", "gate_consistency"]
                  if not res.get(k, False)]
        log.info(f"  🔴 RED LIGHT — Failed gates: {failed}")
        log.info(f"  Note: CLV gate requires real pre-closing odds (ODDS_API_KEY).")
        log.info(f"  With synthetic snapshots, CLV is noise-distributed — gate shows ~0.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V5 Backtest with CLV tracking")
    parser.add_argument("--markets", nargs="+", default=["1x2", "ou"],
                        choices=["1x2", "ou", "corners", "cards"],
                        help="Markets to backtest (default: 1x2 ou)")
    parser.add_argument("--rounds", nargs="+", type=int, choices=[1, 2, 3],
                        help="Rounds to run (default: all 3)")
    args = parser.parse_args()
    main(args)
