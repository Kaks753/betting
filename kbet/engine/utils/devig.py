"""
KBet De-Vigging Engine — Shin's Method + Multiplicative
Properly removes bookmaker margin from odds to extract true market probabilities.

Why NOT flat 5% reduction:
  - Bookmakers concentrate margin on longshots (favorite-longshot bias)
  - A flat cut underestimates the edge on favorites, overestimates on longshots
  - Shin's method solves this properly

Supported methods:
  1. multiplicative  — simple proportional removal (fast baseline)
  2. shin            — Shin's method (most accurate, our primary)
  3. power           — Power method (alternative to Shin)
"""

import numpy as np
from typing import Optional


def multiplicative_devig(odds: list[float]) -> list[float]:
    """
    Multiplicative method: proportionally remove the overround.
    Fastest. Used as fallback.

    P_true_i = (1/odds_i) / sum(1/odds_j)
    """
    raw_probs = [1.0 / o for o in odds if o and o > 1.0]
    if not raw_probs:
        return []
    total = sum(raw_probs)
    return [p / total for p in raw_probs]


def shin_devig(odds: list[float], iterations: int = 50) -> list[float]:
    """
    Shin's method — iterative approach to find true probabilities.
    Accounts for favorite-longshot bias properly.

    Based on: Shin (1993) "Measuring the Incidence of Insider Trading"
    Applied to betting by: Clarke et al., Graham & Stott

    Returns: list of true probabilities summing to 1.0
    """
    n = len(odds)
    if n < 2:
        return multiplicative_devig(odds)

    raw = [1.0 / o for o in odds if o and o > 1.0]
    if len(raw) != n:
        return multiplicative_devig(odds)

    overround = sum(raw)
    if overround <= 1.0:
        return raw  # No margin — already fair

    # Initial estimate via multiplicative
    p = [r / overround for r in raw]

    # Shin's iterative z (insider trading proportion)
    z = 0.0
    for _ in range(iterations):
        # Estimate z from current p
        numerator   = overround - 1.0
        denominator = sum(np.sqrt(pi * (1 - pi)) for pi in p) if n > 2 else 1.0

        if denominator == 0:
            break

        z_new = numerator / denominator
        z_new = np.clip(z_new, 0.0, 0.5)

        if abs(z_new - z) < 1e-8:
            break
        z = z_new

        # Update probabilities using Shin formula
        p_new = []
        for r_i, p_i in zip(raw, p):
            # Shin's formula: p_true = (sqrt(z^2 + 4*(1-z)*r_i^2/overround) - z) / (2*(1-z))
            if abs(1 - z) < 1e-10:
                p_new.append(r_i / overround)
            else:
                discriminant = z**2 + 4 * (1 - z) * (r_i / overround)
                if discriminant < 0:
                    discriminant = 0
                p_i_new = (np.sqrt(discriminant) - z) / (2 * (1 - z))
                p_new.append(max(0.0, p_i_new))

        total = sum(p_new)
        if total > 0:
            p = [pi / total for pi in p_new]

    return p


def power_devig(odds: list[float]) -> list[float]:
    """
    Power method — find k such that sum((1/odds)^k) = 1.
    Alternative to Shin, easier to compute.
    """
    from scipy.optimize import brentq

    raw = [1.0 / o for o in odds if o and o > 1.0]
    if not raw:
        return []

    def equation(k):
        return sum(r**k for r in raw) - 1.0

    try:
        k = brentq(equation, 0.5, 3.0, xtol=1e-8)
        probs = [r**k for r in raw]
        total = sum(probs)
        return [p / total for p in probs]
    except Exception:
        return multiplicative_devig(odds)


def devig_1x2(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    method: str = "shin",
    pinnacle_weight: float = 1.0,
) -> dict:
    """
    Remove vig from 1X2 odds and return true market probabilities.

    Parameters
    ----------
    odds_home, odds_draw, odds_away : Decimal odds
    method : "shin" | "multiplicative" | "power"
    pinnacle_weight : Weight multiplier for Pinnacle odds (use 3.0 for Pinnacle)

    Returns
    -------
    {"home": p, "draw": p, "away": p, "overround": %, "margin": %}
    """
    odds = [odds_home, odds_draw, odds_away]

    # Validate odds
    if any(o is None or o <= 1.0 for o in odds):
        return None

    raw_probs = [1.0 / o for o in odds]
    overround = sum(raw_probs)
    margin    = (overround - 1.0) / overround * 100

    if method == "shin":
        probs = shin_devig(odds)
    elif method == "power":
        probs = power_devig(odds)
    else:
        probs = multiplicative_devig(odds)

    if len(probs) != 3:
        return None

    return {
        "home":      round(probs[0], 6),
        "draw":      round(probs[1], 6),
        "away":      round(probs[2], 6),
        "overround": round(overround, 4),
        "margin_pct": round(margin, 2),
    }


def devig_2way(odds_yes: float, odds_no: float, method: str = "shin") -> dict:
    """
    Remove vig from 2-way markets (Over/Under, BTTS, Asian Handicap).
    Returns {"yes": p, "no": p, "margin_pct": m}
    """
    if odds_yes is None or odds_no is None or odds_yes <= 1.0 or odds_no <= 1.0:
        return None

    odds = [odds_yes, odds_no]
    overround = sum(1.0 / o for o in odds)
    margin = (overround - 1.0) / overround * 100

    if method == "shin":
        probs = shin_devig(odds)
    elif method == "power":
        probs = power_devig(odds)
    else:
        probs = multiplicative_devig(odds)

    if len(probs) != 2:
        return None

    return {
        "yes":        round(probs[0], 6),
        "no":         round(probs[1], 6),
        "margin_pct": round(margin, 2),
    }


def blend_books(book_results: list[dict], weights: list[float]) -> dict:
    """
    Blend multiple de-vigged book probabilities using weights.
    Use: Pinnacle weight=3.0, Bet365 weight=1.0, Max weight=2.0

    book_results: list of dicts from devig_1x2()
    weights: matching weights list
    """
    if not book_results or not weights:
        return None

    valid = [(r, w) for r, w in zip(book_results, weights) if r is not None]
    if not valid:
        return None

    total_weight = sum(w for _, w in valid)
    home = sum(r["home"] * w for r, w in valid) / total_weight
    draw = sum(r["draw"] * w for r, w in valid) / total_weight
    away = sum(r["away"] * w for r, w in valid) / total_weight

    # Renormalize
    total = home + draw + away
    return {
        "home": round(home / total, 6),
        "draw": round(draw / total, 6),
        "away": round(away / total, 6),
    }


if __name__ == "__main__":
    # Quick test
    print("\n── De-Vig Test ────────────────────────────")
    # Typical Bet365 1X2 odds with ~7.5% margin
    h, d, a = 2.10, 3.40, 3.60

    raw = [1/h, 1/d, 1/a]
    print(f"  Raw odds:     H={h}  D={d}  A={a}")
    print(f"  Raw probs:    H={raw[0]:.3f}  D={raw[1]:.3f}  A={raw[2]:.3f}  Sum={sum(raw):.3f}")

    for method in ["multiplicative", "shin", "power"]:
        result = devig_1x2(h, d, a, method=method)
        print(f"\n  [{method.upper()}]")
        print(f"  True probs:   H={result['home']:.4f}  D={result['draw']:.4f}  A={result['away']:.4f}")
        print(f"  Margin: {result['margin_pct']:.2f}%  Overround: {result['overround']:.4f}")
