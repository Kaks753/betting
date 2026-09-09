"""
KBet Value Detector — The Money Layer
Computes Expected Value for each market and flags value bets.

Logic:
  EV = (our_probability × decimal_odds) - 1
  After slippage penalty: effective_EV = EV × (1 - slippage)
  If effective_EV > threshold → VALUE BET

Confidence tiers:
  🔥 FIRE:  EV > 15%
  ✅ SOLID: EV 10-15%
  👀 WATCH: EV 5-10% (1X2) or market threshold
  ❌ SKIP:  Below threshold
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import EV_THRESHOLDS, KELLY, BACKTEST
from engine.utils.devig import devig_1x2, devig_2way, blend_books


SLIPPAGE = BACKTEST["slippage_penalty"]   # 20% haircut for execution drag


def compute_ev(our_prob: float, decimal_odds: float) -> float:
    """
    Expected Value = (probability × odds) - 1
    Positive EV = value bet
    """
    if decimal_odds <= 1.0 or our_prob <= 0 or our_prob >= 1:
        return -99.0
    return (our_prob * decimal_odds) - 1.0


def apply_slippage(ev: float, slippage: float = SLIPPAGE) -> float:
    """Apply execution drag penalty to EV."""
    return ev * (1.0 - slippage)


def kelly_stake(our_prob: float, decimal_odds: float,
                fraction: float = KELLY["fraction"]) -> float:
    """
    Fractional Kelly stake as % of bankroll.
    Returns 0 if no edge.
    """
    b = decimal_odds - 1.0
    q = 1.0 - our_prob
    if b <= 0 or our_prob <= 0:
        return 0.0

    k = (our_prob * b - q) / b  # Full Kelly
    k_frac = k * fraction        # Quarter Kelly

    # Hard caps
    k_frac = max(0.0, k_frac)
    k_frac = min(k_frac, KELLY["max_per_bet"])

    return round(k_frac * 100, 2)  # Return as percentage


def confidence_tier(ev: float, market: str = "1x2") -> str:
    """Return confidence tier label based on EV."""
    if ev >= 0.15:
        return "FIRE"
    elif ev >= 0.10:
        return "SOLID"
    elif ev >= EV_THRESHOLDS.get(market, 0.05):
        return "WATCH"
    else:
        return "SKIP"


def evaluate_1x2(
    home_prob_model: float,
    draw_prob_model: float,
    away_prob_model: float,
    odds_home: Optional[float],
    odds_draw: Optional[float],
    odds_away: Optional[float],
    odds_home_pin: Optional[float] = None,
    odds_draw_pin: Optional[float] = None,
    odds_away_pin: Optional[float] = None,
) -> dict:
    """
    Evaluate value in 1X2 market.
    Compares our model probabilities vs de-vigged market probabilities.

    Returns dict with value flags for each outcome.
    """
    market = "1x2"
    threshold = EV_THRESHOLDS[market]
    min_prob = KELLY["min_prob"]

    results = {"market": "1X2", "bets": []}

    # De-vig available odds — prefer Pinnacle (3× weight) + Bet365 (1× weight)
    book_results = []
    weights = []

    if all(o and o > 1.0 for o in [odds_home_pin, odds_draw_pin, odds_away_pin]):
        r = devig_1x2(odds_home_pin, odds_draw_pin, odds_away_pin, method="shin")
        if r:
            book_results.append(r)
            weights.append(3.0)  # Pinnacle weighted 3×

    if all(o and o > 1.0 for o in [odds_home, odds_draw, odds_away]):
        r = devig_1x2(odds_home, odds_draw, odds_away, method="shin")
        if r:
            book_results.append(r)
            weights.append(1.0)

    outcomes = [
        ("home", home_prob_model, odds_home),
        ("draw", draw_prob_model, odds_draw),
        ("away", away_prob_model, odds_away),
    ]

    for outcome_name, our_prob, our_odds in outcomes:
        if our_prob < min_prob:
            continue  # Skip very unlikely outcomes (Kelly protection)
        if not our_odds or our_odds <= 1.0:
            continue

        ev = compute_ev(our_prob, our_odds)
        ev_net = apply_slippage(ev)
        stake  = kelly_stake(our_prob, our_odds) if ev_net >= threshold else 0.0
        tier   = confidence_tier(ev_net, market)

        # Market probability (de-vigged)
        if book_results:
            # Blended market prob
            total_w = sum(weights)
            mkt_prob = sum(
                r[outcome_name] * w / total_w
                for r, w in zip(book_results, weights)
            )
        else:
            mkt_prob = 1.0 / our_odds

        edge = our_prob - mkt_prob

        result = {
            "outcome":    outcome_name,
            "our_prob":   round(our_prob, 4),
            "mkt_prob":   round(mkt_prob, 4),
            "edge":       round(edge, 4),
            "odds":       our_odds,
            "ev_raw":     round(ev, 4),
            "ev_net":     round(ev_net, 4),
            "kelly_pct":  stake,
            "tier":       tier,
            "is_value":   (ev_net >= threshold),
        }
        results["bets"].append(result)

    return results


def evaluate_over_under(
    over_prob_model: float,
    odds_over: Optional[float],
    odds_under: Optional[float],
    threshold_goals: float = 2.5,
) -> dict:
    """Evaluate value in Over/Under market."""
    market = "over_under"
    ev_threshold = EV_THRESHOLDS[market]
    results = {"market": f"O/U {threshold_goals}", "bets": []}

    under_prob_model = 1.0 - over_prob_model

    outcomes = [
        ("over",  over_prob_model,  odds_over),
        ("under", under_prob_model, odds_under),
    ]

    # De-vig market odds
    mkt = devig_2way(odds_over, odds_under) if (odds_over and odds_under) else None

    for name, our_prob, our_odds in outcomes:
        if not our_odds or our_odds <= 1.0 or our_prob < KELLY["min_prob"]:
            continue

        ev = compute_ev(our_prob, our_odds)
        ev_net = apply_slippage(ev)
        stake = kelly_stake(our_prob, our_odds) if ev_net >= ev_threshold else 0.0
        tier  = confidence_tier(ev_net, market)
        mkt_prob = (mkt["yes"] if name == "over" else mkt["no"]) if mkt else 1.0 / our_odds

        results["bets"].append({
            "outcome":   name,
            "our_prob":  round(our_prob, 4),
            "mkt_prob":  round(mkt_prob, 4),
            "edge":      round(our_prob - mkt_prob, 4),
            "odds":      our_odds,
            "ev_raw":    round(ev, 4),
            "ev_net":    round(ev_net, 4),
            "kelly_pct": stake,
            "tier":      tier,
            "is_value":  (ev_net >= ev_threshold),
        })

    return results


def evaluate_btts(
    btts_yes_prob: float,
    odds_yes: Optional[float],
    odds_no: Optional[float],
) -> dict:
    """Evaluate value in Both Teams To Score market."""
    market = "btts"
    ev_threshold = EV_THRESHOLDS[market]
    results = {"market": "BTTS", "bets": []}

    mkt = devig_2way(odds_yes, odds_no) if (odds_yes and odds_no) else None

    for name, our_prob, our_odds in [
        ("yes", btts_yes_prob,          odds_yes),
        ("no",  1.0 - btts_yes_prob,    odds_no),
    ]:
        if not our_odds or our_odds <= 1.0 or our_prob < KELLY["min_prob"]:
            continue

        ev = compute_ev(our_prob, our_odds)
        ev_net = apply_slippage(ev)
        stake = kelly_stake(our_prob, our_odds) if ev_net >= ev_threshold else 0.0
        tier  = confidence_tier(ev_net, market)
        mkt_prob = (mkt["yes"] if name == "yes" else mkt["no"]) if mkt else 1.0 / our_odds

        results["bets"].append({
            "outcome":   f"BTTS {name.upper()}",
            "our_prob":  round(our_prob, 4),
            "mkt_prob":  round(mkt_prob, 4),
            "edge":      round(our_prob - mkt_prob, 4),
            "odds":      our_odds,
            "ev_raw":    round(ev, 4),
            "ev_net":    round(ev_net, 4),
            "kelly_pct": stake,
            "tier":      tier,
            "is_value":  (ev_net >= ev_threshold),
        })

    return results


def evaluate_corners(
    corners_over_prob: float,
    odds_over: Optional[float],
    odds_under: Optional[float],
    line: float = 9.5,
) -> dict:
    """Evaluate value in Corner Kick markets."""
    market = "corners"
    ev_threshold = EV_THRESHOLDS[market]
    results = {"market": f"Corners O/U {line}", "bets": []}

    mkt = devig_2way(odds_over, odds_under) if (odds_over and odds_under) else None

    for name, our_prob, our_odds in [
        ("over",  corners_over_prob,          odds_over),
        ("under", 1.0 - corners_over_prob,    odds_under),
    ]:
        if not our_odds or our_odds <= 1.0 or our_prob < KELLY["min_prob"]:
            continue

        ev = compute_ev(our_prob, our_odds)
        ev_net = apply_slippage(ev)
        stake = kelly_stake(our_prob, our_odds) if ev_net >= ev_threshold else 0.0
        tier  = confidence_tier(ev_net, market)
        mkt_prob = (mkt["yes"] if name == "over" else mkt["no"]) if mkt else 1.0 / our_odds

        results["bets"].append({
            "outcome":   f"Corners {name}",
            "our_prob":  round(our_prob, 4),
            "mkt_prob":  round(mkt_prob, 4),
            "edge":      round(our_prob - mkt_prob, 4),
            "odds":      our_odds,
            "ev_raw":    round(ev, 4),
            "ev_net":    round(ev_net, 4),
            "kelly_pct": stake,
            "tier":      tier,
            "is_value":  (ev_net >= ev_threshold),
        })

    return results


def evaluate_yellow_cards(
    cards_over_prob: float,
    odds_over: Optional[float],
    odds_under: Optional[float],
    line: float = 3.5,
) -> dict:
    """Evaluate value in Yellow Card markets."""
    market = "yellow_cards"
    ev_threshold = EV_THRESHOLDS[market]
    results = {"market": f"Yellow Cards O/U {line}", "bets": []}

    mkt = devig_2way(odds_over, odds_under) if (odds_over and odds_under) else None

    for name, our_prob, our_odds in [
        ("over",  cards_over_prob,          odds_over),
        ("under", 1.0 - cards_over_prob,    odds_under),
    ]:
        if not our_odds or our_odds <= 1.0 or our_prob < KELLY["min_prob"]:
            continue

        ev = compute_ev(our_prob, our_odds)
        ev_net = apply_slippage(ev)
        stake = kelly_stake(our_prob, our_odds) if ev_net >= ev_threshold else 0.0
        tier  = confidence_tier(ev_net, market)
        mkt_prob = (mkt["yes"] if name == "over" else mkt["no"]) if mkt else 1.0 / our_odds

        results["bets"].append({
            "outcome":   f"YCards {name}",
            "our_prob":  round(our_prob, 4),
            "mkt_prob":  round(mkt_prob, 4),
            "edge":      round(our_prob - mkt_prob, 4),
            "odds":      our_odds,
            "ev_raw":    round(ev, 4),
            "ev_net":    round(ev_net, 4),
            "kelly_pct": stake,
            "tier":      tier,
            "is_value":  (ev_net >= ev_threshold),
        })

    return results


def portfolio_kelly_cap(bets: list[dict], daily_cap: float = None) -> list[dict]:
    """
    Apply portfolio-level Kelly cap.
    If total daily stake > daily_cap, scale all stakes proportionally.
    Handles correlated bets (same day = correlated).
    """
    cap = daily_cap or KELLY["max_daily"] * 100  # as percentage

    value_bets = [b for b in bets if b.get("is_value") and b.get("kelly_pct", 0) > 0]
    total_stake = sum(b["kelly_pct"] for b in value_bets)

    if total_stake <= cap:
        return bets

    scale = cap / total_stake
    for b in value_bets:
        b["kelly_pct"] = round(b["kelly_pct"] * scale, 2)
        b["kelly_scaled"] = True

    return bets


if __name__ == "__main__":
    print("\n── Value Detector Test ─────────────────────")
    # Simulate: our model says home team wins 55%, bookie offers 2.20
    result = evaluate_1x2(
        home_prob_model=0.55,
        draw_prob_model=0.25,
        away_prob_model=0.20,
        odds_home=2.20,
        odds_draw=3.40,
        odds_away=3.80,
        odds_home_pin=2.15,
        odds_draw_pin=3.50,
        odds_away_pin=3.70,
    )

    for bet in result["bets"]:
        print(f"\n  {bet['outcome'].upper()}")
        print(f"  Our prob:  {bet['our_prob']:.1%}  |  Mkt prob: {bet['mkt_prob']:.1%}")
        print(f"  Edge:      {bet['edge']:+.1%}")
        print(f"  EV (net):  {bet['ev_net']:+.1%}")
        print(f"  Stake:     {bet['kelly_pct']}% of bankroll")
        print(f"  Tier:      {bet['tier']}")
        print(f"  Value:     {'✅ YES' if bet['is_value'] else '❌ NO'}")
