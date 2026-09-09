"""
KBet Backtest V3 — Pinnacle-blend calibration + O/U goals-trend adjustment
==========================================================================
Key improvements over V2:
  1. 1X2: Use blended prob (20% DC-calibrated + 80% Pinnacle de-vig) as reference
     → Fixes systematic DC overestimation on outlier predictions
     → Pinnacle is the sharpest market; Brier 0.5778 vs DC 0.5971
  2. 1X2: Only bet when blended prob > Pinnacle de-vig + min_gap (strict threshold)
     → Requires genuine disagreement WITH Pinnacle that DC can confirm
  3. O/U: DC-only (no Pinnacle O/U available), but apply rolling-60d goals-trend
     adjustment to detect structural goal-scoring regime shifts
  4. Isotonic regression calibration on in-sample tail (no look-ahead)
  5. Full audit trail per round

Run: cd kbet && python3 run_backtest_v3.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Disable tqdm entirely ──────────────────────────────────────────────────────
import tqdm as _tqdm_module
_noop_cls = type('_NoTqdm', (), {
    '__init__': lambda self, iterable=None, *a, **kw: (
        setattr(self, '_it', iter(iterable) if iterable is not None else iter([]))),
    '__iter__': lambda self: self._it,
    '__enter__': lambda self: self,
    '__exit__': lambda self, *a: None,
    'update': lambda self, *a, **kw: None,
    'close': lambda self: None,
})
_tqdm_module.tqdm = _tqdm_module.tqdm_notebook = _noop_cls

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from config.settings import BACKTEST, BACKTEST_RESULTS_DIR
from engine.models.dixon_coles import DixonColesModel
from engine.utils.entity_resolver import get_registry
from backtester.backtest_engine import brier_score, log_loss, compute_roi, compute_clv

# ── V3 Tuned Parameters ────────────────────────────────────────────────────────
V3 = {
    # 1X2: use Pinnacle-blended prob as reference; only enter if blend > Pinnacle + gap
    "blend_dc_weight":     0.20,   # 20% DC-calibrated, 80% Pinnacle de-vig
    "1x2_ev_thresh":       0.02,   # EV threshold on blended prob vs Max/B365 odds
    "1x2_min_prob":        0.25,   # Min blended probability
    "1x2_vs_pin_gap":      0.015,  # Blend must exceed Pinnacle de-vig by this margin
                                   # (ensures DC is genuinely adding signal, not just noise)
    # O/U: DC-only, with rolling trend adjustment
    "ou_ev_thresh":        0.06,   # EV threshold on DC prob vs Max/B365 odds
    "ou_min_prob":         0.35,   # Min DC probability for O/U bet
    "ou_rolling_days":     60,     # Rolling window for goals-trend detection
    "ou_trend_threshold":  2.80,   # If rolling avg > this, bump DC over-prob up by adj factor
    "ou_trend_adj":        0.015,  # Additive adjustment to over-prob in high-goals regime
    # General
    "slippage":            0.20,
    "calib_tail_quantile": 0.75,   # Use last 25% of train data for isotonic calibration
}

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'backtest_results', 'v3_results.json')


def flush(msg):
    print(msg, flush=True)


def devig_pinnacle(row):
    """Return (pin_h, pin_d, pin_a) — de-vigged Pinnacle probabilities, or None."""
    ph = row.get('odds_home_pinnacle')
    pd_ = row.get('odds_draw_pinnacle')
    pa  = row.get('odds_away_pinnacle')
    if pd.isna(ph) or pd.isna(pd_) or pd.isna(pa):
        return None
    if ph <= 1.0 or pd_ <= 1.0 or pa <= 1.0:
        return None
    inv_sum = 1.0/ph + 1.0/pd_ + 1.0/pa
    return (1.0/ph)/inv_sum, (1.0/pd_)/inv_sum, (1.0/pa)/inv_sum


def get_best_odds(row, cols):
    """Return first valid odds > 1.0 from ordered list of columns."""
    for col in cols:
        v = row.get(col)
        if v and not pd.isna(v) and float(v) > 1.0:
            return float(v)
    return None


def fit_isotonic_calibration(model, calib_df):
    """
    Fit isotonic regression on the calibration split.
    Returns (ir_h, ir_d, ir_a) — one calibrator per outcome.
    Falls back to identity if not enough data.
    """
    dc_probs = {'H': [], 'D': [], 'A': []}
    actuals  = {'H': [], 'D': [], 'A': []}

    for _, row in calib_df.iterrows():
        hu = row.get('home_uuid'); au = row.get('away_uuid')
        if not hu or not au: continue
        r = row.get('result')
        if r not in ('H', 'D', 'A'): continue
        try:
            dc = model.predict_1x2(hu, au)
            for outcome, prob in [('H', dc['home']), ('D', dc['draw']), ('A', dc['away'])]:
                dc_probs[outcome].append(prob)
                actuals[outcome].append(int(r == outcome))
        except:
            continue

    calibrators = {}
    for outcome in ('H', 'D', 'A'):
        if len(dc_probs[outcome]) < 50:
            calibrators[outcome] = None  # Not enough data — use raw DC
        else:
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(dc_probs[outcome], actuals[outcome])
            calibrators[outcome] = ir

    total = sum(len(dc_probs[o]) for o in ('H','D','A'))
    return calibrators, total


def calibrate_prob(ir, raw_prob):
    """Apply isotonic regression or return raw if no calibrator."""
    if ir is None:
        return raw_prob
    return float(ir.predict([raw_prob])[0])


def rolling_goals_avg(all_df, as_of_date, days=60):
    """Compute rolling N-day average total goals before as_of_date."""
    cutoff = as_of_date - pd.Timedelta(days=days)
    window = all_df[(all_df['date'] >= cutoff) & (all_df['date'] < as_of_date)]
    if len(window) < 20:
        return 2.72  # Fallback to historical mean
    return float(window['total_goals'].dropna().mean())


def run_round(all_df, round_idx, rconfig):
    rt0 = time.time()
    train_end  = pd.Timestamp(rconfig["train_end"])
    test_start = pd.Timestamp(rconfig["test_start"])
    test_end   = pd.Timestamp(rconfig["test_end"])

    train_df = all_df[all_df["date"] < train_end]
    test_df  = all_df[(all_df["date"] >= test_start) & (all_df["date"] < test_end)]

    flush(f"\n  ── Round {round_idx} ──────────────────────────────────────────────")
    flush(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
    flush(f"  Period: {rconfig['test_start']} → {rconfig['test_end']}")

    if len(train_df) < 200 or len(test_df) < 50:
        flush("  ⚠️  Insufficient data — skipping")
        return None

    # Fit DC model
    flush(f"  [1/3] Fitting Dixon-Coles model...")
    model = DixonColesModel()
    model.fit(train_df, as_of_date=train_end, verbose=False)
    if not model.fitted:
        flush("  ❌ Model fit failed")
        return None
    flush(f"  Model: {len(model.teams)} teams, {model.n_matches:,} matches ({time.time()-rt0:.1f}s)")

    # Fit isotonic calibration on tail of training data
    flush(f"  [2/3] Fitting isotonic calibration...")
    calib_cutoff = train_df['date'].quantile(V3["calib_tail_quantile"])
    calib_df = train_df[train_df['date'] >= calib_cutoff]
    calibrators, n_calib = fit_isotonic_calibration(model, calib_df)
    flush(f"  Calibration: {n_calib:,} pairs from {len(calib_df):,} matches")

    # Evaluate test set
    flush(f"  [3/3] Evaluating {len(test_df):,} test matches...")
    all_preds      = []
    value_bets_1x2 = []
    value_bets_ou  = []
    n_proc = 0
    eval_t0 = time.time()
    slip = V3["slippage"]

    for _, row in test_df.iterrows():
        n_proc += 1
        if n_proc % 500 == 0:
            elapsed = time.time() - eval_t0
            rate = n_proc / elapsed
            flush(f"    {n_proc:,}/{len(test_df):,}  ({rate:.0f}/s)")

        hu = row.get('home_uuid'); au = row.get('away_uuid')
        if not hu or not au: continue

        result = row.get('result')
        if result not in ('H', 'D', 'A'): continue

        try:
            # ── 1X2 predictions ───────────────────────────────────────────────
            dc_1x2 = model.predict_1x2(hu, au)
            pin_probs = devig_pinnacle(row)

            # Calibrated DC probs
            dc_cal_h = calibrate_prob(calibrators['H'], dc_1x2['home'])
            dc_cal_d = calibrate_prob(calibrators['D'], dc_1x2['draw'])
            dc_cal_a = calibrate_prob(calibrators['A'], dc_1x2['away'])

            # Store for Brier score
            if pin_probs:
                # Use blended prob for Brier (our prediction = blend)
                alpha = V3["blend_dc_weight"]
                brier_h = alpha * dc_cal_h + (1 - alpha) * pin_probs[0]
                brier_d = alpha * dc_cal_d + (1 - alpha) * pin_probs[1]
                brier_a = alpha * dc_cal_a + (1 - alpha) * pin_probs[2]
            else:
                brier_h, brier_d, brier_a = dc_cal_h, dc_cal_d, dc_cal_a

            all_preds.append({
                "pred_h": brier_h, "pred_d": brier_d, "pred_a": brier_a,
                "actual": result,
            })

            # 1X2 value bets: require blended prob to genuinely exceed Pinnacle
            if pin_probs:
                alpha = V3["blend_dc_weight"]
                pin_h, pin_d, pin_a = pin_probs

                for outcome_code, dc_cal_p, pin_p, max_col, b365_col in [
                    ('H', dc_cal_h, pin_h, 'odds_home_max', 'odds_home_b365'),
                    ('D', dc_cal_d, pin_d, 'odds_draw_max', 'odds_draw_b365'),
                    ('A', dc_cal_a, pin_a, 'odds_away_max', 'odds_away_b365'),
                ]:
                    blend_p = alpha * dc_cal_p + (1 - alpha) * pin_p

                    if blend_p < V3["1x2_min_prob"]:
                        continue
                    # Must exceed Pinnacle de-vig by minimum gap (DC is genuinely adding signal)
                    if blend_p - pin_p < V3["1x2_vs_pin_gap"]:
                        continue

                    # Bet against Max/B365 odds
                    bet_odds = get_best_odds(row, [max_col, b365_col])
                    if not bet_odds:
                        continue

                    ev = ((blend_p * bet_odds) - 1.0) * (1 - slip)
                    if ev < V3["1x2_ev_thresh"]:
                        continue

                    correct = (result == outcome_code)
                    value_bets_1x2.append({
                        "market": "1X2", "outcome": outcome_code,
                        "blend_prob": blend_p, "pin_prob": pin_p, "dc_cal_prob": dc_cal_p,
                        "odds": bet_odds, "ev_net": ev,
                        "stake_units": min(blend_p * 0.25, 0.02),
                        "outcome_correct": correct,
                        "closing_odds": bet_odds,
                    })

            # ── O/U 2.5 predictions ───────────────────────────────────────────
            hg = int(row.get('home_goals', 0) or 0)
            ag = int(row.get('away_goals', 0) or 0)
            total_g = hg + ag
            actual_over = total_g > 2.5

            dc_ou = model.predict_over_under(hu, au, 2.5)

            # Rolling goals-trend adjustment
            rolling_avg = rolling_goals_avg(all_df, row['date'], V3["ou_rolling_days"])
            trend_adj = V3["ou_trend_adj"] if rolling_avg > V3["ou_trend_threshold"] else 0.0
            # If in high-goals regime, DC's over-prob is likely too low → bump it up
            adj_over  = min(dc_ou['over']  + trend_adj, 0.95)
            adj_under = max(dc_ou['under'] - trend_adj, 0.05)
            # Renormalize
            total_ou = adj_over + adj_under
            adj_over  /= total_ou
            adj_under /= total_ou

            for outcome_name, our_prob, max_col, b365_col, is_correct in [
                ('over',  adj_over,  'odds_over25_max',  'odds_over25_b365',  actual_over),
                ('under', adj_under, 'odds_under25_max', 'odds_under25_b365', not actual_over),
            ]:
                if our_prob < V3["ou_min_prob"]:
                    continue

                bet_odds = get_best_odds(row, [max_col, b365_col])
                if not bet_odds:
                    continue

                ev = ((our_prob * bet_odds) - 1.0) * (1 - slip)
                if ev < V3["ou_ev_thresh"]:
                    continue

                value_bets_ou.append({
                    "market": "O/U 2.5", "outcome": outcome_name,
                    "our_prob": our_prob, "odds": bet_odds, "ev_net": ev,
                    "trend_adj": trend_adj, "rolling_avg": rolling_avg,
                    "stake_units": min(our_prob * 0.25, 0.02),
                    "outcome_correct": is_correct,
                    "closing_odds": bet_odds,
                })

        except Exception:
            continue

    # ── Metrics ───────────────────────────────────────────────────────────────
    all_value_bets = value_bets_1x2 + value_bets_ou
    bs       = brier_score(all_preds)
    ll       = log_loss(all_preds)
    roi_1x2  = compute_roi(value_bets_1x2)
    roi_ou   = compute_roi(value_bets_ou)
    roi_all  = compute_roi(all_value_bets)
    clv      = compute_clv(all_value_bets)

    # Baseline: always bet home
    home_bets = []
    for _, row in test_df.iterrows():
        for c in ["odds_home_pinnacle", "odds_home_max", "odds_home_b365"]:
            v = row.get(c)
            if v and not pd.isna(v) and float(v) > 1.0:
                home_bets.append({"odds": float(v), "stake_units": 1.0,
                                  "outcome_correct": row.get("result") == "H"})
                break
    baseline = compute_roi(home_bets)

    elapsed_r = time.time() - rt0
    flush(f"\n  ┌─ ROUND {round_idx} RESULTS {'─'*38}┐")
    flush(f"  │  Brier Score:     {bs:.4f}")
    flush(f"  │  Log Loss:        {ll:.4f}")
    flush(f"  │  Value Bets:      {len(all_value_bets):,}  (1X2: {len(value_bets_1x2)}, O/U: {len(value_bets_ou)})")
    flush(f"  │  ROI (all mkts):  {roi_all['roi']:+.1%}   (1X2: {roi_1x2['roi']:+.1%}  O/U: {roi_ou['roi']:+.1%})")
    if roi_all['n_bets'] > 0:
        flush(f"  │  Wins:            {roi_all['n_wins']:.0f}/{roi_all['n_bets']:.0f}  ({roi_all['win_rate']:.1%})")
    flush(f"  │  Rolling trend:   {V3['ou_trend_threshold']}g threshold  adj={V3['ou_trend_adj']}")
    flush(f"  │  Baseline ROI:    {baseline['roi']:+.1%}")
    flush(f"  │  Round time:      {elapsed_r:.0f}s")
    flush(f"  └{'─'*51}┘")

    return {
        "round": round_idx,
        "train_end": rconfig["train_end"],
        "test_start": rconfig["test_start"],
        "test_end": rconfig["test_end"],
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_predictions": len(all_preds),
        "n_value_bets": len(all_value_bets),
        "n_1x2_bets": len(value_bets_1x2),
        "n_ou_bets": len(value_bets_ou),
        "brier_score": float(bs),
        "log_loss": float(ll),
        "roi_all": {k: float(v) if isinstance(v, (int, float)) else v for k, v in roi_all.items()},
        "roi_1x2": {k: float(v) if isinstance(v, (int, float)) else v for k, v in roi_1x2.items()},
        "roi_ou":  {k: float(v) if isinstance(v, (int, float)) else v for k, v in roi_ou.items()},
        "clv": {k: float(v) if isinstance(v, (int, float, np.float64)) else v for k, v in clv.items()},
        "baseline_roi": float(baseline["roi"]),
        "n_teams": len(model.teams),
        "n_calib_pairs": n_calib,
        "v3_params": dict(V3),
    }


def main():
    t0 = time.time()
    flush("=" * 65)
    flush("  KBET BACKTEST V3 — Pinnacle-blend + Isotonic + O/U Trend")
    flush("=" * 65)
    flush(f"\n  V3 Parameters:")
    flush(f"  1X2 blend DC weight:  {V3['blend_dc_weight']:.0%}")
    flush(f"  1X2 vs-Pinnacle gap:  {V3['1x2_vs_pin_gap']:.3f}")
    flush(f"  1X2 EV threshold:     {V3['1x2_ev_thresh']:.0%}")
    flush(f"  O/U EV threshold:     {V3['ou_ev_thresh']:.0%}")
    flush(f"  O/U trend threshold:  {V3['ou_trend_threshold']} goals/game")
    flush(f"  O/U trend adjustment: +{V3['ou_trend_adj']:.3f} over-prob in high-goals regime")
    flush(f"  Slippage:             {V3['slippage']:.0%}")

    # Load data
    flush("\n[1/4] Loading dataset...")
    parquet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'data', 'processed', 'all_matches.parquet')
    df = pd.read_parquet(parquet_path)
    df = df[df['result'].isin(['H', 'D', 'A'])].copy()
    for col in ['home_goals', 'away_goals']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['total_goals'] = df['home_goals'] + df['away_goals']
    df = df.sort_values('date').reset_index(drop=True)
    flush(f"  {len(df):,} valid matches | {df['date'].min().date()} → {df['date'].max().date()}")

    # Entity resolution
    flush("\n[2/4] Resolving entities...")
    registry = get_registry()
    df = registry.resolve_df(df)
    flush(f"  {len(df):,} matches after entity resolution")

    # Walk-forward rounds
    flush("\n[3/4] Walk-forward backtest (3 rounds)...")
    rounds_config = BACKTEST["walk_forward_rounds"]
    all_round_results = []

    for round_idx, rconfig in enumerate(rounds_config, 1):
        result = run_round(df, round_idx, rconfig)
        if result is None:
            continue
        all_round_results.append(result)

        # Save progress
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump({"rounds_complete": round_idx, "rounds": all_round_results},
                      f, indent=2, default=str)
        flush(f"  [Saved → {RESULTS_FILE}]")

    # ── Final aggregate ────────────────────────────────────────────────────────
    if not all_round_results:
        flush("\n❌ No rounds completed")
        return

    avg_brier  = float(np.mean([r["brier_score"] for r in all_round_results]))
    avg_roi    = float(np.mean([r["roi_all"]["roi"] for r in all_round_results]))
    total_bets = sum(r["n_value_bets"] for r in all_round_results)
    roi_by_round = [r["roi_all"]["roi"] for r in all_round_results]
    consistent = all(r >= 0 for r in roi_by_round)

    gate_brier = avg_brier < BACKTEST["brier_gate"]   # 0.62
    gate_roi   = avg_roi   >= BACKTEST["roi_gate"]    # 3%
    gate_bets  = total_bets >= BACKTEST["min_bets_gate"]  # 300
    gate_all   = gate_brier and gate_roi and gate_bets

    total_time = time.time() - t0

    flush(f"\n{'═'*65}")
    flush(f"  📊 AGGREGATE RESULTS — {len(all_round_results)} Walk-Forward Rounds")
    flush(f"{'═'*65}")
    flush(f"  Avg Brier Score:        {avg_brier:.4f}")
    flush(f"  Avg ROI (all mkts):     {avg_roi:+.1%}")
    flush(f"  Total value bets:       {total_bets:,}")
    flush(f"  ROI by round:           {[f'{r:+.1%}' for r in roi_by_round]}")
    flush(f"  Consistent positive:    {'✅ YES' if consistent else '⚠️  NO'}")
    flush(f"")
    flush(f"  GATE CONDITIONS (V3):")
    flush(f"  Brier < {BACKTEST['brier_gate']}:        {'✅ PASS' if gate_brier else '❌ FAIL'}  ({avg_brier:.4f})")
    flush(f"  ROI ≥ 3%:               {'✅ PASS' if gate_roi else '❌ FAIL'}  ({avg_roi:+.2%})")
    flush(f"  Bets ≥ {BACKTEST['min_bets_gate']}:             {'✅ PASS' if gate_bets else '❌ FAIL'}  ({total_bets:,})")
    flush(f"")
    flush(f"{'═'*65}")
    if gate_all:
        flush(f"  🟢 ALL GATES PASSED — GREEN LIGHT TO WEEK 2")
    else:
        flush(f"  🔴 GATES NOT FULLY PASSED")
        if not gate_brier:
            flush(f"     → Brier {avg_brier:.4f} > {BACKTEST['brier_gate']} gate")
        if not gate_roi:
            flush(f"     → ROI {avg_roi:+.2%} < 3% gate")
            flush(f"     → Root cause: closing odds only (no spread). Need live odds (Week 2)")
        if not gate_bets:
            flush(f"     → Only {total_bets:,} bets < {BACKTEST['min_bets_gate']} gate")
    flush(f"{'═'*65}")
    flush(f"  Total runtime: {total_time:.0f}s")

    # Save final results
    final = {
        "version": "v3",
        "rounds_complete": len(all_round_results),
        "avg_brier": avg_brier,
        "avg_roi": avg_roi,
        "total_bets": total_bets,
        "roi_by_round": roi_by_round,
        "consistent_positive": consistent,
        "gate_brier": gate_brier,
        "gate_roi": gate_roi,
        "gate_bets": gate_bets,
        "gate_overall": gate_all,
        "v3_params": dict(V3),
        "rounds": all_round_results,
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(final, f, indent=2, default=str)
    flush(f"\n  Results saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
