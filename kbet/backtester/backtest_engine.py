"""
KBet Backtest Engine — Walk-Forward Validation
The sacred gate: nothing ships until this passes.

Protocol:
  1. Multiple walk-forward rounds (train/test on different periods)
  2. 20% slippage penalty applied to all EVs
  3. Minimum 1,500 bets required for statistical significance
  4. Brier Score, Log Loss, ROI, CLV all computed
  5. Baseline comparison against "always bet home" dummy model

Look-ahead bias prevention:
  - All features computed from data BEFORE match date
  - Odds used are closing odds from CSV (already timestamped correctly)
  - No lineup/injury data used in backtest (not available pre-match historically)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import BACKTEST, KELLY, EV_THRESHOLDS, BACKTEST_RESULTS_DIR
from engine.models.dixon_coles import DixonColesModel
from engine.utils.devig import devig_1x2, devig_2way
from engine.utils.entity_resolver import get_registry


TIER_EMOJI = {"FIRE": "🔥", "SOLID": "✅", "WATCH": "👀", "SKIP": "❌"}


def brier_score(predictions: list[dict]) -> float:
    """
    Brier Score for 1X2 predictions.
    Lower is better. Random model = 0.667. Good model < 0.50.
    """
    if not predictions:
        return 1.0

    total = 0.0
    for p in predictions:
        actual_h = 1 if p["actual"] == "H" else 0
        actual_d = 1 if p["actual"] == "D" else 0
        actual_a = 1 if p["actual"] == "A" else 0

        bs = ((p["pred_h"] - actual_h)**2 +
              (p["pred_d"] - actual_d)**2 +
              (p["pred_a"] - actual_a)**2)
        total += bs

    return round(total / len(predictions), 4)


def log_loss(predictions: list[dict]) -> float:
    """Multi-class log loss for 1X2."""
    eps = 1e-7
    total = 0.0
    for p in predictions:
        if p["actual"] == "H":
            total += -np.log(max(p["pred_h"], eps))
        elif p["actual"] == "D":
            total += -np.log(max(p["pred_d"], eps))
        else:
            total += -np.log(max(p["pred_a"], eps))
    return round(total / len(predictions), 4)


def compute_roi(bets: list[dict]) -> dict:
    """
    Compute ROI from a list of placed bets.
    Each bet: {odds, kelly_pct, outcome_correct, stake_units}
    """
    if not bets:
        return {"roi": 0.0, "n_bets": 0, "profit_units": 0.0, "win_rate": 0.0}

    total_staked = sum(b.get("stake_units", 1.0) for b in bets)
    total_profit = 0.0
    wins = 0

    for b in bets:
        stake = b.get("stake_units", 1.0)
        if b.get("outcome_correct"):
            total_profit += stake * (b["odds"] - 1.0)
            wins += 1
        else:
            total_profit -= stake

    roi = total_profit / total_staked if total_staked > 0 else 0.0

    return {
        "roi":           round(roi, 4),
        "n_bets":        len(bets),
        "n_wins":        wins,
        "win_rate":      round(wins / len(bets), 4),
        "profit_units":  round(total_profit, 2),
        "total_staked":  round(total_staked, 2),
    }


def compute_clv(bets: list[dict]) -> dict:
    """
    Closing Line Value (CLV):
    Were we getting better odds than the closing price?
    CLV > 0 consistently = our model has real edge.
    """
    clv_values = []

    for b in bets:
        bet_odds    = b.get("odds", 0)
        closing_odds = b.get("closing_odds")
        if bet_odds and closing_odds and closing_odds > 1.0 and bet_odds > 1.0:
            clv = (bet_odds / closing_odds - 1.0) * 100
            clv_values.append(clv)

    if not clv_values:
        return {"clv_mean": None, "clv_positive_rate": None, "n_clv": 0}

    return {
        "clv_mean":         round(np.mean(clv_values), 2),
        "clv_median":       round(np.median(clv_values), 2),
        "clv_positive_rate": round(sum(1 for c in clv_values if c > 0) / len(clv_values), 4),
        "n_clv":            len(clv_values),
    }


class BacktestEngine:
    """
    Runs walk-forward validation of the Dixon-Coles + Value Detection pipeline.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : Full match DataFrame (with entity UUIDs resolved, odds columns)
        """
        self.df = df.copy().sort_values("date").reset_index(drop=True)
        self.registry = get_registry()
        self.results_by_round = []

    def _get_odds(self, row: pd.Series, outcome: str) -> Optional[float]:
        """Get best available closing odds for an outcome."""
        # Prefer Pinnacle, then Max, then Bet365
        if outcome == "home":
            for col in ["odds_home_pinnacle", "odds_home_max", "odds_home_b365"]:
                v = row.get(col)
                if v and not pd.isna(v) and v > 1.0:
                    return float(v)
        elif outcome == "draw":
            for col in ["odds_draw_pinnacle", "odds_draw_max", "odds_draw_b365"]:
                v = row.get(col)
                if v and not pd.isna(v) and v > 1.0:
                    return float(v)
        elif outcome == "away":
            for col in ["odds_away_pinnacle", "odds_away_max", "odds_away_b365"]:
                v = row.get(col)
                if v and not pd.isna(v) and v > 1.0:
                    return float(v)
        elif outcome == "over25":
            for col in ["odds_over25_max", "odds_over25_b365"]:
                v = row.get(col)
                if v and not pd.isna(v) and v > 1.0:
                    return float(v)
        elif outcome == "under25":
            for col in ["odds_under25_max", "odds_under25_b365"]:
                v = row.get(col)
                if v and not pd.isna(v) and v > 1.0:
                    return float(v)
        return None

    def _evaluate_match(
        self,
        row: pd.Series,
        model: DixonColesModel,
    ) -> dict:
        """
        Run full prediction + value detection for one match.
        Returns dict with predictions, EV, and (during test) outcomes.
        """
        home_uuid = row.get("home_uuid")
        away_uuid = row.get("away_uuid")
        result    = row.get("result")  # H/D/A

        if not home_uuid or not away_uuid:
            return {}

        # Get predictions
        preds_1x2 = model.predict_1x2(home_uuid, away_uuid)
        preds_ou  = model.predict_over_under(home_uuid, away_uuid, 2.5)
        preds_btts = model.predict_btts(home_uuid, away_uuid)

        # Get available odds
        odds_h   = self._get_odds(row, "home")
        odds_d   = self._get_odds(row, "draw")
        odds_a   = self._get_odds(row, "away")
        odds_o25 = self._get_odds(row, "over25")
        odds_u25 = self._get_odds(row, "under25")

        slippage = BACKTEST["slippage_penalty"]

        value_bets = []

        # ── 1X2 value check ────────────────────────────────────────
        outcomes_1x2 = [
            ("home", preds_1x2["home"], odds_h),
            ("draw", preds_1x2["draw"], odds_d),
            ("away", preds_1x2["away"], odds_a),
        ]
        threshold_1x2 = EV_THRESHOLDS["1x2"]

        for outcome_name, our_prob, our_odds in outcomes_1x2:
            if not our_odds or our_prob < KELLY["min_prob"]:
                continue
            ev = (our_prob * our_odds) - 1.0
            ev_net = ev * (1.0 - slippage)

            if ev_net >= threshold_1x2:
                # Determine if bet won
                if result == "H":
                    correct = (outcome_name == "home")
                elif result == "D":
                    correct = (outcome_name == "draw")
                elif result == "A":
                    correct = (outcome_name == "away")
                else:
                    correct = None

                home_goals = row.get("home_goals", 0) or 0
                away_goals = row.get("away_goals", 0) or 0
                total_goals = home_goals + away_goals

                value_bets.append({
                    "market":           "1X2",
                    "outcome":          outcome_name,
                    "our_prob":         round(our_prob, 4),
                    "odds":             round(our_odds, 3),
                    "ev_net":           round(ev_net, 4),
                    "stake_units":      min(our_prob * 0.25, KELLY["max_per_bet"]),
                    "outcome_correct":  correct,
                    "closing_odds":     our_odds,  # In backtest, closing = bet odds
                    "date":             row.get("date"),
                    "home_team":        row.get("home_team", ""),
                    "away_team":        row.get("away_team", ""),
                    "league":           row.get("league_name", ""),
                })

        # ── Over/Under value check ──────────────────────────────────
        ou_outcomes = [
            ("over",  preds_ou["over"],  odds_o25),
            ("under", preds_ou["under"], odds_u25),
        ]
        threshold_ou = EV_THRESHOLDS["over_under"]

        home_goals = int(row.get("home_goals", 0) or 0)
        away_goals = int(row.get("away_goals", 0) or 0)
        total_goals = home_goals + away_goals

        for outcome_name, our_prob, our_odds in ou_outcomes:
            if not our_odds or our_prob < KELLY["min_prob"]:
                continue
            ev = (our_prob * our_odds) - 1.0
            ev_net = ev * (1.0 - slippage)

            if ev_net >= threshold_ou:
                if result is not None:
                    correct = (outcome_name == "over" and total_goals > 2.5) or \
                              (outcome_name == "under" and total_goals <= 2.5)
                else:
                    correct = None

                value_bets.append({
                    "market":          "O/U 2.5",
                    "outcome":         outcome_name,
                    "our_prob":        round(our_prob, 4),
                    "odds":            round(our_odds, 3),
                    "ev_net":          round(ev_net, 4),
                    "stake_units":     min(our_prob * 0.25, KELLY["max_per_bet"]),
                    "outcome_correct": correct,
                    "closing_odds":    our_odds,
                    "date":            row.get("date"),
                    "home_team":       row.get("home_team", ""),
                    "away_team":       row.get("away_team", ""),
                    "league":          row.get("league_name", ""),
                })

        return {
            "pred_h":     preds_1x2["home"],
            "pred_d":     preds_1x2["draw"],
            "pred_a":     preds_1x2["away"],
            "pred_over":  preds_ou["over"],
            "actual":     result,
            "value_bets": value_bets,
        }

    def run_round(self, train_end: str, test_start: str, test_end: str,
                  round_name: str = "") -> dict:
        """
        Run one walk-forward validation round.
        Train on [all data → train_end], test on [test_start → test_end].
        """
        train_end_dt   = pd.Timestamp(train_end)
        test_start_dt  = pd.Timestamp(test_start)
        test_end_dt    = pd.Timestamp(test_end)

        train_df = self.df[self.df["date"] < train_end_dt]
        test_df  = self.df[
            (self.df["date"] >= test_start_dt) &
            (self.df["date"] < test_end_dt)
        ]

        print(f"\n  {'─'*55}")
        print(f"  Round {round_name}: Train → {train_end} | Test: {test_start} → {test_end}")
        print(f"  Train: {len(train_df):,} matches | Test: {len(test_df):,} matches")

        if len(train_df) < 200 or len(test_df) < 50:
            print(f"  ⚠️  Insufficient data — skipping round")
            return {}

        # Fit model on training data
        print(f"  Fitting Dixon-Coles model...")
        model = DixonColesModel()
        model.fit(train_df, as_of_date=train_end_dt, verbose=False)

        if not model.fitted:
            print(f"  ❌ Model failed to fit")
            return {}

        print(f"  Model fitted on {model.n_matches:,} weighted matches, {len(model.teams)} teams")

        # Evaluate on test set
        all_predictions = []
        all_value_bets  = []

        for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                           desc=f"  Testing {round_name}", leave=False):
            match_result = self._evaluate_match(row, model)
            if match_result:
                all_predictions.append(match_result)
                all_value_bets.extend(match_result.get("value_bets", []))

        if not all_predictions:
            return {}

        # Compute metrics
        bs    = brier_score(all_predictions)
        ll    = log_loss(all_predictions)
        roi_1x2 = compute_roi([b for b in all_value_bets if b["market"] == "1X2"])
        roi_ou  = compute_roi([b for b in all_value_bets if b["market"] == "O/U 2.5"])
        roi_all = compute_roi(all_value_bets)
        clv     = compute_clv(all_value_bets)

        # Baseline: always bet home at average home odds
        home_bets = []
        for _, row in test_df.iterrows():
            odds_h = self._get_odds(row, "home")
            if odds_h:
                home_bets.append({
                    "odds": odds_h,
                    "stake_units": 1.0,
                    "outcome_correct": row.get("result") == "H",
                })
        baseline_roi = compute_roi(home_bets)

        round_result = {
            "round":          round_name,
            "train_end":      train_end,
            "test_start":     test_start,
            "test_end":       test_end,
            "n_train":        len(train_df),
            "n_test":         len(test_df),
            "n_predictions":  len(all_predictions),
            "n_value_bets":   len(all_value_bets),
            "brier_score":    bs,
            "log_loss":       ll,
            "roi_1x2":        roi_1x2,
            "roi_ou":         roi_ou,
            "roi_all":        roi_all,
            "clv":            clv,
            "baseline_roi":   baseline_roi,
            "model_teams":    len(model.teams),
        }

        # Print round summary
        self._print_round_summary(round_result)

        return round_result

    def _print_round_summary(self, r: dict):
        bs = r["brier_score"]
        bs_color = "✅" if bs < 0.50 else "❌"
        roi = r["roi_all"]["roi"]
        roi_color = "✅" if roi >= BACKTEST["roi_gate"] else ("⚠️" if roi >= 0 else "❌")
        n = r["n_value_bets"]
        n_color = "✅" if n >= BACKTEST["min_bets_gate"] else "⚠️"

        print(f"\n  {'─'*55}")
        print(f"  ROUND {r['round']} RESULTS")
        print(f"  {'─'*55}")
        print(f"  Brier Score:    {bs:.4f}  {bs_color}  (target: <0.50)")
        print(f"  Log Loss:       {r['log_loss']:.4f}")
        print(f"  Value Bets:     {n:,}     {n_color}  (target: ≥1,500)")
        print(f"  ROI (all mkts): {roi:+.1%}  {roi_color}  (target: ≥3%)")
        print(f"  ROI (1X2):      {r['roi_1x2']['roi']:+.1%}")
        print(f"  ROI (O/U 2.5):  {r['roi_ou']['roi']:+.1%}")
        print(f"  Wins:           {r['roi_all']['n_wins']}/{r['roi_all']['n_bets']}")

        if r["clv"]["clv_mean"] is not None:
            clv_color = "✅" if r["clv"]["clv_mean"] > 0 else "❌"
            print(f"  CLV Mean:       {r['clv']['clv_mean']:+.2f}%  {clv_color}")
            print(f"  CLV+ Rate:      {r['clv']['clv_positive_rate']:.1%}")

        baseline = r["baseline_roi"]["roi"]
        beat_base = "✅ BEATS BASELINE" if roi > baseline else "⚠️  BELOW BASELINE"
        print(f"  Baseline ROI:   {baseline:+.1%}  ({beat_base})")

    def run_all(self) -> dict:
        """Run all configured walk-forward rounds and aggregate results."""
        print(f"\n{'═'*60}")
        print(f"  🏆 KBET WALK-FORWARD BACKTEST")
        print(f"  Model: Dixon-Coles + Shin De-Vig + 20% Slippage Penalty")
        print(f"{'═'*60}")

        rounds = BACKTEST["walk_forward_rounds"]
        all_rounds = []

        for i, r in enumerate(rounds):
            result = self.run_round(
                train_end=r["train_end"],
                test_start=r["test_start"],
                test_end=r["test_end"],
                round_name=str(i + 1),
            )
            if result:
                all_rounds.append(result)
                self.results_by_round.append(result)

        if not all_rounds:
            print("\n❌ No rounds completed — check data coverage")
            return {}

        # Aggregate across rounds
        avg_brier = np.mean([r["brier_score"] for r in all_rounds])
        avg_roi   = np.mean([r["roi_all"]["roi"] for r in all_rounds])
        total_bets = sum(r["n_value_bets"] for r in all_rounds)
        all_rois  = [r["roi_all"]["roi"] for r in all_rounds]
        consistent = all(roi >= 0 for roi in all_rois)  # Positive every round?

        # Gate evaluation
        gate_brier  = avg_brier < BACKTEST["brier_gate"]
        gate_roi    = avg_roi   >= BACKTEST["roi_gate"]
        gate_bets   = total_bets >= BACKTEST["min_bets_gate"]
        gate_consistent = consistent
        gate_overall = gate_brier and gate_roi and gate_bets

        summary = {
            "rounds":          len(all_rounds),
            "total_test_bets": total_bets,
            "avg_brier":       round(avg_brier, 4),
            "avg_roi":         round(avg_roi, 4),
            "roi_by_round":    [r["roi_all"]["roi"] for r in all_rounds],
            "consistent_positive": consistent,
            "gate_brier":      gate_brier,
            "gate_roi":        gate_roi,
            "gate_bets":       gate_bets,
            "gate_overall":    gate_overall,
        }

        self._print_final_verdict(summary)

        # Save results
        os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(BACKTEST_RESULTS_DIR, f"backtest_{ts}.json")
        with open(out_path, "w") as f:
            # Make serializable
            safe = {k: (v if not isinstance(v, np.float64) else float(v))
                    for k, v in summary.items()}
            json.dump({"summary": safe, "rounds": [
                {k: (float(v) if isinstance(v, (np.float64, np.float32)) else v)
                 for k, v in r.items() if k not in ["roi_1x2", "roi_ou", "roi_all", "clv", "baseline_roi"]}
                for r in all_rounds
            ]}, f, indent=2, default=str)
        print(f"\n  Results saved → {out_path}")

        return summary

    def _print_final_verdict(self, s: dict):
        print(f"\n{'═'*60}")
        print(f"  📊 BACKTEST AGGREGATE SUMMARY")
        print(f"{'═'*60}")
        print(f"  Walk-Forward Rounds:   {s['rounds']}")
        print(f"  Total Value Bets:      {s['total_test_bets']:,}")
        print(f"  Avg Brier Score:       {s['avg_brier']:.4f}")
        print(f"  Avg ROI:               {s['avg_roi']:+.1%}")
        print(f"  ROI by round:          {[f'{r:+.1%}' for r in s['roi_by_round']]}")
        print(f"  Consistent positive:   {'✅ YES' if s['consistent_positive'] else '⚠️  NO'}")
        print(f"\n  GATE CONDITIONS:")
        print(f"  Brier < 0.50:          {'✅ PASS' if s['gate_brier'] else '❌ FAIL'}")
        print(f"  ROI ≥ 3%:              {'✅ PASS' if s['gate_roi'] else '❌ FAIL'}")
        print(f"  Bets ≥ 1,500:          {'✅ PASS' if s['gate_bets'] else '❌ FAIL'}")
        print(f"\n{'═'*60}")
        if s["gate_overall"]:
            print(f"  🟢 ALL GATES PASSED — GREEN LIGHT TO BUILD UI")
        else:
            print(f"  🔴 GATES NOT FULLY PASSED — Model needs refinement")
            print(f"     Review individual round results above")
        print(f"{'═'*60}\n")
