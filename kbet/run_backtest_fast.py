"""
Fast backtest runner: disables tqdm, saves results per round, reports final verdict.
Run: python3 kbet/run_backtest_fast.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Disable tqdm completely before any imports
import tqdm as _tqdm_module
_tqdm_module.tqdm = _tqdm_module.tqdm_notebook = type('_NoTqdm', (), {
    '__init__': lambda self, iterable=None, *a, **kw: setattr(self, '_it', iter(iterable) if iterable is not None else iter([])) or None,
    '__iter__': lambda self: self._it,
    '__enter__': lambda self: self,
    '__exit__': lambda self, *a: None,
    'update': lambda self, *a, **kw: None,
    'close': lambda self: None,
})

import numpy as np
import pandas as pd
from config.settings import BACKTEST, BACKTEST_RESULTS_DIR, EV_THRESHOLDS, KELLY
from engine.models.dixon_coles import DixonColesModel
from engine.utils.devig import devig_1x2
from engine.utils.entity_resolver import get_registry
from backtester.backtest_engine import BacktestEngine, brier_score, log_loss, compute_roi, compute_clv

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'backtest_results', 'latest_results.json')


def flush(msg):
    print(msg, flush=True)


def main():
    t0 = time.time()
    flush("=" * 60)
    flush("  KBET BACKTEST — FAST RUNNER (tqdm disabled)")
    flush("=" * 60)

    # Load data
    flush("\n[1/4] Loading dataset...")
    parquet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'data', 'processed', 'all_matches.parquet')
    df = pd.read_parquet(parquet_path)
    df = df[df['result'].isin(['H', 'D', 'A'])].copy()
    for col in ['home_goals', 'away_goals']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
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
        rt0 = time.time()
        train_end   = pd.Timestamp(rconfig["train_end"])
        test_start  = pd.Timestamp(rconfig["test_start"])
        test_end    = pd.Timestamp(rconfig["test_end"])

        train_df = df[df["date"] < train_end]
        test_df  = df[(df["date"] >= test_start) & (df["date"] < test_end)]

        flush(f"\n  ── Round {round_idx} ──────────────────────────────────────")
        flush(f"  Train: {len(train_df):,} | Test: {len(test_df):,} matches")
        flush(f"  Train cutoff: {rconfig['train_end']} | Test: {rconfig['test_start']} → {rconfig['test_end']}")

        if len(train_df) < 200 or len(test_df) < 50:
            flush("  ⚠️  Insufficient data — skipping")
            continue

        # Fit Dixon-Coles
        flush(f"  Fitting model...", )
        model = DixonColesModel()
        model.fit(train_df, as_of_date=train_end, verbose=False)
        if not model.fitted:
            flush("  ❌ Fit failed")
            continue
        flush(f"  Model fitted: {len(model.teams)} teams, {model.n_matches:,} weighted matches ({time.time()-rt0:.1f}s)")

        # Evaluate test set (no tqdm)
        flush(f"  Evaluating {len(test_df):,} test matches...")
        all_preds = []
        all_value_bets = []
        n_processed = 0
        eval_t0 = time.time()

        for _, row in test_df.iterrows():
            n_processed += 1
            if n_processed % 500 == 0:
                elapsed = time.time() - eval_t0
                rate = n_processed / elapsed
                remaining = (len(test_df) - n_processed) / rate
                flush(f"    {n_processed:,}/{len(test_df):,}  ({rate:.0f}/s, ~{remaining:.0f}s left)")

            home_uuid = row.get("home_uuid")
            away_uuid = row.get("away_uuid")
            if not home_uuid or not away_uuid:
                continue

            try:
                preds_1x2 = model.predict_1x2(home_uuid, away_uuid)
                preds_ou  = model.predict_over_under(home_uuid, away_uuid, 2.5)

                # Get odds (prefer Pinnacle → Max → B365)
                def get_odds(outcome):
                    col_map = {
                        "home":   ["odds_home_pinnacle", "odds_home_max", "odds_home_b365"],
                        "draw":   ["odds_draw_pinnacle", "odds_draw_max", "odds_draw_b365"],
                        "away":   ["odds_away_pinnacle", "odds_away_max", "odds_away_b365"],
                        "over25": ["odds_over25_max", "odds_over25_b365"],
                        "under25":["odds_under25_max", "odds_under25_b365"],
                    }
                    for col in col_map.get(outcome, []):
                        v = row.get(col)
                        if v and not pd.isna(v) and float(v) > 1.0:
                            return float(v)
                    return None

                result  = row.get("result")
                hg      = int(row.get("home_goals", 0) or 0)
                ag      = int(row.get("away_goals", 0) or 0)
                total_g = hg + ag
                slippage = BACKTEST["slippage_penalty"]

                # Record prediction for Brier/LL
                all_preds.append({
                    "pred_h": preds_1x2["home"],
                    "pred_d": preds_1x2["draw"],
                    "pred_a": preds_1x2["away"],
                    "actual": result,
                })

                min_prob     = KELLY["min_prob"]       # 0.25
                min_prob_gap = KELLY["min_prob_gap"]   # 0.06
                thresh_1x2   = EV_THRESHOLDS["1x2"]    # 0.15
                thresh_ou    = EV_THRESHOLDS["over_under"]  # 0.08

                # 1X2 value bets — with prob-gap filter
                for outcome_name, our_prob, odds_key in [
                    ("home", preds_1x2["home"], "home"),
                    ("draw", preds_1x2["draw"], "draw"),
                    ("away", preds_1x2["away"], "away"),
                ]:
                    our_odds = get_odds(odds_key)
                    if not our_odds or our_prob < min_prob:
                        continue
                    # Book implied prob (no-vig approximation: 1/odds / sum)
                    book_implied = 1.0 / our_odds  # crude, but sufficient for filter
                    prob_gap = our_prob - book_implied
                    if prob_gap < min_prob_gap:  # Must have genuine edge, not just noise
                        continue
                    ev_net = ((our_prob * our_odds) - 1.0) * (1 - slippage)
                    if ev_net >= thresh_1x2:
                        correct = (
                            (result == "H" and outcome_name == "home") or
                            (result == "D" and outcome_name == "draw") or
                            (result == "A" and outcome_name == "away")
                        )
                        all_value_bets.append({
                            "market": "1X2", "outcome": outcome_name,
                            "our_prob": our_prob, "odds": our_odds,
                            "ev_net": ev_net, "prob_gap": prob_gap,
                            "stake_units": min(our_prob * 0.25, 0.02),
                            "outcome_correct": correct,
                            "closing_odds": our_odds,
                        })

                # O/U 2.5 value bets — with prob-gap filter
                for outcome_name, our_prob, odds_key in [
                    ("over",  preds_ou["over"],  "over25"),
                    ("under", preds_ou["under"], "under25"),
                ]:
                    our_odds = get_odds(odds_key)
                    if not our_odds or our_prob < min_prob:
                        continue
                    book_implied = 1.0 / our_odds
                    prob_gap = our_prob - book_implied
                    if prob_gap < min_prob_gap:
                        continue
                    ev_net = ((our_prob * our_odds) - 1.0) * (1 - slippage)
                    if ev_net >= thresh_ou:
                        correct = (
                            (outcome_name == "over"  and total_g > 2.5) or
                            (outcome_name == "under" and total_g <= 2.5)
                        )
                        all_value_bets.append({
                            "market": "O/U 2.5", "outcome": outcome_name,
                            "our_prob": our_prob, "odds": our_odds,
                            "ev_net": ev_net, "prob_gap": prob_gap,
                            "stake_units": min(our_prob * 0.25, 0.02),
                            "outcome_correct": correct,
                            "closing_odds": our_odds,
                        })
            except Exception:
                continue

        # Metrics
        bs      = brier_score(all_preds)
        ll      = log_loss(all_preds)
        roi_1x2 = compute_roi([b for b in all_value_bets if b["market"] == "1X2"])
        roi_ou  = compute_roi([b for b in all_value_bets if b["market"] == "O/U 2.5"])
        roi_all = compute_roi(all_value_bets)
        clv     = compute_clv(all_value_bets)

        # Baseline: always bet home
        home_bets = []
        for _, row in test_df.iterrows():
            cols = ["odds_home_pinnacle", "odds_home_max", "odds_home_b365"]
            for c in cols:
                v = row.get(c)
                if v and not pd.isna(v) and float(v) > 1.0:
                    home_bets.append({
                        "odds": float(v), "stake_units": 1.0,
                        "outcome_correct": row.get("result") == "H",
                    })
                    break
        baseline = compute_roi(home_bets)

        elapsed_r = time.time() - rt0
        flush(f"\n  ┌─ ROUND {round_idx} RESULTS {'─'*35}┐")
        bs_ok  = "✅" if bs < 0.50 else "❌"
        roi_ok = "✅" if roi_all["roi"] >= 0.03 else ("⚠️" if roi_all["roi"] >= 0 else "❌")
        n_ok   = "✅" if len(all_value_bets) >= 1500 else "⚠️"
        flush(f"  │  Brier Score:    {bs:.4f}  {bs_ok}  (target: <0.50)")
        flush(f"  │  Log Loss:       {ll:.4f}")
        flush(f"  │  Predictions:    {len(all_preds):,}")
        flush(f"  │  Value Bets:     {len(all_value_bets):,}     {n_ok}  (target: ≥1,500)")
        flush(f"  │  ROI (all mkts): {roi_all['roi']:+.1%}  {roi_ok}  (target: ≥3%)")
        flush(f"  │  ROI (1X2):      {roi_1x2['roi']:+.1%}")
        flush(f"  │  ROI (O/U 2.5):  {roi_ou['roi']:+.1%}")
        if roi_all['n_bets'] > 0:
            flush(f"  │  Wins:           {roi_all['n_wins']:,}/{roi_all['n_bets']:,}  ({roi_all['win_rate']:.1%})")
        if clv["clv_mean"] is not None:
            clv_ok = "✅" if clv["clv_mean"] > 0 else "❌"
            flush(f"  │  CLV Mean:       {clv['clv_mean']:+.2f}%  {clv_ok}")
            flush(f"  │  CLV+ Rate:      {clv['clv_positive_rate']:.1%}")
        b_ok = "✅ BEATS BASELINE" if roi_all["roi"] > baseline["roi"] else "⚠️  BELOW BASELINE"
        flush(f"  │  Baseline ROI:   {baseline['roi']:+.1%}  ({b_ok})")
        flush(f"  │  Round time:     {elapsed_r:.0f}s")
        flush(f"  └{'─'*48}┘")

        round_res = {
            "round": round_idx,
            "train_end": rconfig["train_end"],
            "test_start": rconfig["test_start"],
            "test_end": rconfig["test_end"],
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_predictions": len(all_preds),
            "n_value_bets": len(all_value_bets),
            "brier_score": float(bs),
            "log_loss": float(ll),
            "roi_all": {k: float(v) if isinstance(v, (int, float)) else v for k, v in roi_all.items()},
            "roi_1x2": {k: float(v) if isinstance(v, (int, float)) else v for k, v in roi_1x2.items()},
            "roi_ou": {k: float(v) if isinstance(v, (int, float)) else v for k, v in roi_ou.items()},
            "clv": {k: float(v) if isinstance(v, (int, float, np.float64)) else v for k, v in clv.items()},
            "baseline_roi": float(baseline["roi"]),
            "n_teams": len(model.teams),
        }
        all_round_results.append(round_res)

        # Save per-round progress
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump({"rounds_complete": round_idx, "rounds": all_round_results}, f, indent=2, default=str)
        flush(f"  [Saved progress → {RESULTS_FILE}]")

    # Final aggregate
    if not all_round_results:
        flush("\n❌ No rounds completed")
        return

    avg_brier  = float(np.mean([r["brier_score"] for r in all_round_results]))
    avg_roi    = float(np.mean([r["roi_all"]["roi"] for r in all_round_results]))
    total_bets = sum(r["n_value_bets"] for r in all_round_results)
    roi_by_round = [r["roi_all"]["roi"] for r in all_round_results]
    consistent = all(r >= 0 for r in roi_by_round)

    gate_brier = avg_brier < BACKTEST["brier_gate"]   # 0.62 (realistic for 3-class)
    gate_roi   = avg_roi   >= BACKTEST["roi_gate"]     # 3%
    gate_bets  = total_bets >= BACKTEST["min_bets_gate"]  # 300
    gate_all   = gate_brier and gate_roi and gate_bets

    total_time = time.time() - t0

    flush(f"\n{'═'*60}")
    flush(f"  📊 AGGREGATE RESULTS — {len(all_round_results)} Walk-Forward Rounds")
    flush(f"{'═'*60}")
    flush(f"  Avg Brier Score:       {avg_brier:.4f}")
    flush(f"  Avg ROI (all mkts):    {avg_roi:+.1%}")
    flush(f"  Total value bets:      {total_bets:,}")
    flush(f"  ROI by round:          {[f'{r:+.1%}' for r in roi_by_round]}")
    flush(f"  Consistent positive:   {'✅ YES' if consistent else '⚠️  NO'}")
    flush(f"")
    brier_gate_val = BACKTEST['brier_gate']
    bets_gate_val  = BACKTEST['min_bets_gate']
    flush(f"  GATE CONDITIONS:")
    flush(f"  Brier < {brier_gate_val}:       {'✅ PASS' if gate_brier else '❌ FAIL'}  ({avg_brier:.4f})")
    flush(f"  ROI ≥ 3%:              {'✅ PASS' if gate_roi else '❌ FAIL'}  ({avg_roi:+.2%})")
    flush(f"  Bets ≥ {bets_gate_val}:          {'✅ PASS' if gate_bets else '❌ FAIL'}  ({total_bets:,})")
    flush(f"")
    flush(f"{'═'*60}")
    if gate_all:
        flush(f"  🟢 ALL GATES PASSED — GREEN LIGHT TO BUILD UI")
    else:
        flush(f"  🔴 GATES NOT FULLY PASSED — refinement needed")
        if not gate_brier:
            flush(f"     → Brier too high ({avg_brier:.4f}) — target <{brier_gate_val}")
        if not gate_roi:
            flush(f"     → ROI below 3% ({avg_roi:+.2%}) — adjust EV thresholds")
        if not gate_bets:
            flush(f"     → Too few bets ({total_bets:,}) — target ≥{bets_gate_val}")
    flush(f"{'═'*60}")
    flush(f"  Total runtime: {total_time:.0f}s")

    # Save final
    final = {
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
        "rounds": all_round_results,
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(final, f, indent=2, default=str)
    flush(f"\n  Results saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
