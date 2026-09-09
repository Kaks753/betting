"""
Fractional Kelly Criterion Staking — Week 3
============================================
Full Kelly is theoretically optimal but practically disastrous (large drawdowns,
over-sensitivity to probability mis-estimation). We use Quarter Kelly (fraction=0.25)
with hard caps on per-bet and daily exposure.

Formula:
    full_kelly  = edge / (odds - 1)   where edge = prob * odds - 1
    stake_frac  = full_kelly * fraction
    stake_pct   = min(stake_frac, max_per_bet)

Position limits enforced by DailyPortfolio:
    - Per bet: max 2% bankroll
    - Per match: max 2 bets, max 3% combined
    - Daily total: max 5% bankroll
    - Minimum EV: 3% (don't bet on tiny edges)
    - Minimum odds: 1.35 (below this, book margin eats EV)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Default staking parameters ──────────────────────────────────────────────

DEFAULT_FRACTION    = 0.25   # Quarter Kelly
DEFAULT_MAX_PER_BET = 0.02   # 2% bankroll per bet
DEFAULT_MAX_DAILY   = 0.05   # 5% bankroll per day
DEFAULT_MAX_PER_MATCH = 0.03 # 3% per match (both bets combined)
DEFAULT_MAX_BETS_PER_MATCH = 2
DEFAULT_MIN_EV      = 0.03   # Don't bet < 3% EV
DEFAULT_MIN_ODDS    = 1.35   # Don't bet < 1.35 decimal
DEFAULT_MAX_ODDS    = 8.0    # Don't bet > 8.0 decimal (too noisy)


# ── Core Kelly function ──────────────────────────────────────────────────────

def kelly_stake(
    prob: float,
    odds: float,
    fraction: float = DEFAULT_FRACTION,
    max_per_bet: float = DEFAULT_MAX_PER_BET,
    min_ev: float = DEFAULT_MIN_EV,
    min_odds: float = DEFAULT_MIN_ODDS,
    max_odds: float = DEFAULT_MAX_ODDS,
) -> float:
    """
    Compute fractional Kelly stake as a fraction of bankroll.

    Parameters
    ----------
    prob        : model probability of the outcome occurring
    odds        : decimal odds offered by the bookmaker
    fraction    : Kelly fraction (0.25 = quarter Kelly)
    max_per_bet : hard cap on stake (fraction of bankroll)
    min_ev      : reject bets below this EV threshold
    min_odds    : reject bets below this decimal odds (book margin)
    max_odds    : reject bets above this decimal odds (too volatile)

    Returns
    -------
    float : stake as fraction of bankroll (0.0 = don't bet)
    """
    # Guard: odds validity
    if odds < min_odds or odds > max_odds:
        return 0.0

    # Guard: probability validity
    if prob <= 0.0 or prob >= 1.0:
        return 0.0

    # Compute edge
    edge = prob * odds - 1.0
    if edge < min_ev:
        return 0.0

    # Full Kelly
    full_kelly = edge / (odds - 1.0)
    if full_kelly <= 0.0:
        return 0.0

    # Fractional Kelly with hard cap
    stake = min(full_kelly * fraction, max_per_bet)
    return round(stake, 6)


def kelly_ev(prob: float, odds: float) -> float:
    """Return raw EV (edge): prob * odds - 1. Negative = no value."""
    return prob * odds - 1.0


# ── Daily Portfolio Manager ──────────────────────────────────────────────────

@dataclass
class BetOrder:
    """A single bet order with Kelly-computed stake."""
    match_id: str          # e.g. "Arsenal_Chelsea_2024-09-14"
    home: str
    away: str
    market: str            # "1x2", "ou", "corners", "cards"
    pick: str              # "H", "D", "A", "O2.5", "U2.5", etc.
    model_prob: float      # our model's probability
    book_odds: float       # decimal odds available
    ev: float              # model_prob * book_odds - 1
    stake_pct: float       # fraction of bankroll to stake
    stake_units: float     # stake_pct * 100 (units on 100-unit bank)
    confidence: str        # "FIRE" | "SOLID" | "WATCH"
    clv_ref_prob: float = 0.0    # closing line implied prob (for CLV calc)
    clv: float = 0.0             # closing_line_value = model_prob - clv_ref_prob
    result: Optional[str] = None # "W", "L", "V" (void) — filled post-match
    profit_units: Optional[float] = None  # settled P&L in units

    def __post_init__(self):
        self.stake_units = round(self.stake_pct * 100, 4)

    @property
    def implied_prob(self) -> float:
        """Book's implied probability (raw, includes margin)."""
        if self.book_odds <= 1.0:
            return 1.0
        return 1.0 / self.book_odds

    @property
    def confidence_stars(self) -> str:
        if self.ev >= 0.15:
            return "★★★★★"
        elif self.ev >= 0.10:
            return "★★★★☆"
        elif self.ev >= 0.05:
            return "★★★☆☆"
        elif self.ev >= 0.03:
            return "★★☆☆☆"
        else:
            return "★☆☆☆☆"


@dataclass
class DailyPortfolio:
    """
    Manages bet selection and sizing for a single day.
    Enforces position limits to prevent over-concentration.
    """
    bankroll: float = 1000.0   # Reference bankroll in currency units
    fraction: float = DEFAULT_FRACTION
    max_per_bet: float = DEFAULT_MAX_PER_BET
    max_daily: float = DEFAULT_MAX_DAILY
    max_per_match: float = DEFAULT_MAX_PER_MATCH
    max_bets_per_match: int = DEFAULT_MAX_BETS_PER_MATCH
    min_ev: float = DEFAULT_MIN_EV
    min_odds: float = DEFAULT_MIN_ODDS
    max_odds: float = DEFAULT_MAX_ODDS

    _bets: List[BetOrder] = field(default_factory=list, init=False)
    _match_exposure: Dict[str, float] = field(default_factory=dict, init=False)
    _match_bet_count: Dict[str, int] = field(default_factory=dict, init=False)
    _daily_exposure: float = field(default=0.0, init=False)

    def try_add(
        self,
        match_id: str,
        home: str,
        away: str,
        market: str,
        pick: str,
        model_prob: float,
        book_odds: float,
        clv_ref_prob: float = 0.0,
    ) -> Optional[BetOrder]:
        """
        Evaluate and potentially add a bet to today's portfolio.
        Returns the BetOrder if accepted, None if rejected.
        """
        ev = kelly_ev(model_prob, book_odds)

        # ── Gate 1: EV floor ─────────────────────────────────────────────
        if ev < self.min_ev:
            log.debug(f"  ✗ {home} v {away} [{market}/{pick}] ev={ev:.3f} below min_ev")
            return None

        # ── Gate 2: Compute Kelly stake ──────────────────────────────────
        stake = kelly_stake(
            prob=model_prob,
            odds=book_odds,
            fraction=self.fraction,
            max_per_bet=self.max_per_bet,
            min_ev=self.min_ev,
            min_odds=self.min_odds,
            max_odds=self.max_odds,
        )
        if stake <= 0.0:
            return None

        # ── Gate 3: Daily exposure cap ──────────────────────────────────
        if self._daily_exposure + stake > self.max_daily:
            log.debug(f"  ✗ {home} v {away} [{market}/{pick}] daily cap reached {self._daily_exposure:.3f}")
            return None

        # ── Gate 4: Per-match exposure cap ──────────────────────────────
        match_exp = self._match_exposure.get(match_id, 0.0)
        if match_exp + stake > self.max_per_match:
            log.debug(f"  ✗ {home} v {away} [{market}/{pick}] match cap reached {match_exp:.3f}")
            return None

        # ── Gate 5: Per-match bet count ─────────────────────────────────
        if self._match_bet_count.get(match_id, 0) >= self.max_bets_per_match:
            log.debug(f"  ✗ {home} v {away} [{market}/{pick}] match bet count limit")
            return None

        # ── Confidence tier ─────────────────────────────────────────────
        if ev >= 0.15:
            conf = "FIRE"
        elif ev >= 0.08:
            conf = "SOLID"
        else:
            conf = "WATCH"

        # ── CLV ─────────────────────────────────────────────────────────
        clv = model_prob - clv_ref_prob if clv_ref_prob > 0 else 0.0

        bet = BetOrder(
            match_id=match_id,
            home=home,
            away=away,
            market=market,
            pick=pick,
            model_prob=model_prob,
            book_odds=book_odds,
            ev=ev,
            stake_pct=stake,
            stake_units=stake * 100,
            confidence=conf,
            clv_ref_prob=clv_ref_prob,
            clv=clv,
        )

        # ── Accept ──────────────────────────────────────────────────────
        self._bets.append(bet)
        self._daily_exposure += stake
        self._match_exposure[match_id] = match_exp + stake
        self._match_bet_count[match_id] = self._match_bet_count.get(match_id, 0) + 1

        log.debug(
            f"  ✓ {home} v {away} [{market}/{pick}] "
            f"ev={ev:.1%} stake={stake:.2%} conf={conf}"
        )
        return bet

    @property
    def bets(self) -> List[BetOrder]:
        return list(self._bets)

    @property
    def total_exposure(self) -> float:
        return self._daily_exposure

    @property
    def total_bets(self) -> int:
        return len(self._bets)

    def summary(self) -> str:
        lines = [
            f"Portfolio: {self.total_bets} bets | "
            f"Exposure: {self._daily_exposure:.2%} | "
            f"Bankroll: {self.bankroll:.0f}"
        ]
        for b in sorted(self._bets, key=lambda x: -x.ev):
            lines.append(
                f"  {b.home} v {b.away} | {b.market}/{b.pick} | "
                f"EV={b.ev:.1%} | Odds={b.book_odds:.2f} | "
                f"Stake={b.stake_pct:.2%} ({b.stake_units:.2f}u) | "
                f"{b.confidence} {b.confidence_stars}"
            )
        return "\n".join(lines)


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("=== Kelly Staking Unit Tests ===\n")

    # Test 1: Standard value bet
    s = kelly_stake(prob=0.45, odds=2.40)
    print(f"Test 1 — Arsenal home 45% @ 2.40: stake={s:.4f} ({s*100:.2f}u on 100-unit bank)")
    assert 0 < s <= 0.02, f"Expected 0 < stake <= 0.02, got {s}"

    # Test 2: No value
    s = kelly_stake(prob=0.30, odds=2.00)
    print(f"Test 2 — No value 30% @ 2.00: stake={s:.4f} (expected 0)")
    assert s == 0.0, f"Expected 0, got {s}"

    # Test 3: Below min odds
    s = kelly_stake(prob=0.90, odds=1.10)
    print(f"Test 3 — Below min odds 90% @ 1.10: stake={s:.4f} (expected 0)")
    assert s == 0.0

    # Test 4: Huge edge — should be capped at max_per_bet
    s = kelly_stake(prob=0.80, odds=3.00, max_per_bet=0.02)
    print(f"Test 4 — Huge edge 80% @ 3.00: stake={s:.4f} (capped at 0.02)")
    assert s == 0.02

    # Test 5: Portfolio daily cap
    port = DailyPortfolio(max_daily=0.05, max_per_bet=0.02)
    bets_added = 0
    for i in range(10):
        b = port.try_add(
            match_id=f"Match{i}_TeamA_TeamB",
            home=f"Team{i*2}",
            away=f"Team{i*2+1}",
            market="1x2",
            pick="H",
            model_prob=0.48,
            book_odds=2.30,
        )
        if b:
            bets_added += 1
    print(f"Test 5 — Daily cap: accepted {bets_added}/10 bets | exposure={port.total_exposure:.2%}")
    assert port.total_exposure <= 0.05 + 1e-9

    # Test 6: Per-match bet count limit
    # max_bets_per_match=2 means max 2 bets per match regardless of exposure
    # Use small max_per_bet so each bet ~1% → plenty of headroom to hit count limit cleanly
    port2 = DailyPortfolio(max_per_match=0.10, max_per_bet=0.01, max_daily=0.50)
    b1 = port2.try_add("M1_A_B", "Arsenal", "Chelsea", "1x2", "H", 0.45, 2.40)
    b2 = port2.try_add("M1_A_B", "Arsenal", "Chelsea", "ou",  "O2.5", 0.58, 1.90)
    b3 = port2.try_add("M1_A_B", "Arsenal", "Chelsea", "corners", "O9.5", 0.55, 2.10)
    print(f"Test 6 — Per-match 2-bet count limit: b1={b1 is not None}, b2={b2 is not None}, b3={b3 is None}(count cap)")
    assert b1 is not None   # bet 1: accepted
    assert b2 is not None   # bet 2: accepted  
    assert b3 is None       # bet 3: REJECTED — count limit 2 per match

    print("\n✅ All Kelly tests passed!")
    print("\n" + port.summary())
