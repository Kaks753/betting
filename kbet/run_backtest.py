"""
KBet — Week 1 Master Runner
============================
Run this file to execute the full backtest pipeline:

  1. Download historical match data (football-data.co.uk, free CSVs)
  2. Build entity resolution registry
  3. Walk-forward validation (Dixon-Coles + Shin De-Vig + 20% slippage)
  4. Report: Brier Score, ROI, CLV, Gate conditions
  5. Verdict: GREEN LIGHT or needs work

Usage:
  python run_backtest.py              # Normal run (uses cache if exists)
  python run_backtest.py --refresh    # Force re-download all data
  python run_backtest.py --quick      # Run on Premier League only (fast test)
"""

import os
import sys
import argparse
import pandas as pd
from colorama import Fore, Style, init

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import LEAGUES, DATA_PROCESSED_DIR
from engine.scrapers.download_data import download_all, get_dataset_summary
from engine.utils.entity_resolver import get_registry
from backtester.backtest_engine import BacktestEngine

init(autoreset=True)


def banner():
    print(f"""
{Fore.GREEN}╔═══════════════════════════════════════════════════════════╗
║         🏆  K B E T  —  Quantitative Edge Engine          ║
║         Week 1: Backtest Validation Pipeline               ║
║         "We don't gamble. We exploit inefficiencies."      ║
╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")


def prepare_dataframe(df: pd.DataFrame, quick_mode: bool = False) -> pd.DataFrame:
    """
    Final data preparation before backtest:
    - Apply entity resolution (add home_uuid, away_uuid)
    - Filter to seasons with enough data
    - Handle cold start for promoted teams
    - Report coverage stats
    """
    print(f"{Fore.CYAN}[PREP] Applying entity resolution...{Style.RESET_ALL}")
    registry = get_registry()
    df = registry.resolve_df(df)

    if quick_mode:
        print(f"{Fore.YELLOW}[PREP] Quick mode: filtering to Premier League only{Style.RESET_ALL}")
        df = df[df["league_code"] == "E0"].copy()

    # Filter to rows with valid results
    df = df[df["result"].isin(["H", "D", "A"])].copy()

    # Ensure numeric goal columns
    for col in ["home_goals", "away_goals"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    print(f"{Fore.GREEN}[PREP] Ready: {len(df):,} valid matches across "
          f"{df['league_name'].nunique()} leagues{Style.RESET_ALL}")
    print(f"       Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"       Odds coverage (B365): "
          f"{df['odds_home_b365'].notna().sum():,} matches "
          f"({100*df['odds_home_b365'].notna().mean():.0f}%)")

    return df


def run_unit_tests():
    """Quick sanity checks before running full backtest."""
    print(f"\n{Fore.CYAN}[TEST] Running unit tests...{Style.RESET_ALL}")

    # Test 1: De-vig engine
    from engine.utils.devig import devig_1x2, shin_devig
    result = devig_1x2(2.10, 3.40, 3.60, method="shin")
    assert result is not None, "De-vig returned None"
    total = result["home"] + result["draw"] + result["away"]
    assert abs(total - 1.0) < 0.001, f"Probs don't sum to 1: {total}"
    assert result["margin_pct"] > 0, "Margin should be positive"
    print(f"  ✅ De-vig: H={result['home']:.3f} D={result['draw']:.3f} "
          f"A={result['away']:.3f} margin={result['margin_pct']:.1f}%")

    # Test 2: Value detector
    from engine.models.value_detector import compute_ev, kelly_stake
    ev = compute_ev(0.55, 2.20)
    assert abs(ev - 0.21) < 0.01, f"EV wrong: {ev}"
    stake = kelly_stake(0.55, 2.20)
    assert 0 < stake < 2.0, f"Stake out of range: {stake}"
    print(f"  ✅ Value: EV={ev:+.1%}, Kelly stake={stake:.2f}%")

    # Test 3: Entity resolver
    from engine.utils.entity_resolver import get_registry
    reg = get_registry()
    uid1 = reg.resolve("Man City")
    uid2 = reg.resolve("Manchester City FC")
    assert uid1 == uid2, f"Entity mismatch: {uid1} != {uid2}"
    print(f"  ✅ Entity: 'Man City' == 'Manchester City FC' → {uid1[:8]}...")

    # Test 4: Dixon-Coles basic structure
    from engine.models.dixon_coles import tau, time_weight
    assert tau(0, 0, 1.2, 1.0, -0.1) != 1.0, "Tau correction not applied"
    w = time_weight(107)
    assert abs(w - 0.5) < 0.05, f"Half-life wrong at 107 days: {w}"
    print(f"  ✅ Dixon-Coles: tau(0,0)≠1, time_weight(107days)≈{w:.2f}")

    print(f"{Fore.GREEN}  All unit tests passed ✅{Style.RESET_ALL}\n")


def main():
    parser = argparse.ArgumentParser(description="KBet Week 1 Backtest Pipeline")
    parser.add_argument("--refresh", action="store_true", help="Force re-download data")
    parser.add_argument("--quick",   action="store_true", help="Premier League only (fast)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip unit tests")
    args = parser.parse_args()

    banner()

    # ── Step 0: Unit Tests ─────────────────────────────────────────────────────
    if not args.skip_tests:
        run_unit_tests()

    # ── Step 1: Download Data ──────────────────────────────────────────────────
    print(f"{Fore.CYAN}{'─'*60}")
    print(f"STEP 1: Downloading historical data from football-data.co.uk")
    print(f"{'─'*60}{Style.RESET_ALL}")

    df = download_all(force_refresh=args.refresh)

    if df.empty:
        print(f"{Fore.RED}❌ No data downloaded. Check internet connection.{Style.RESET_ALL}")
        sys.exit(1)

    get_dataset_summary(df)

    # ── Step 2: Entity Resolution + Preparation ────────────────────────────────
    print(f"{Fore.CYAN}{'─'*60}")
    print(f"STEP 2: Entity resolution & data preparation")
    print(f"{'─'*60}{Style.RESET_ALL}")

    df = prepare_dataframe(df, quick_mode=args.quick)

    # ── Step 3: Walk-Forward Backtest ──────────────────────────────────────────
    print(f"{Fore.CYAN}{'─'*60}")
    print(f"STEP 3: Walk-Forward Validation Backtest")
    print(f"        Dixon-Coles + Shin De-Vig + 20% Slippage Penalty")
    print(f"{'─'*60}{Style.RESET_ALL}")

    engine = BacktestEngine(df)
    summary = engine.run_all()

    # ── Step 4: Final Verdict ─────────────────────────────────────────────────
    if summary.get("gate_overall"):
        print(f"""
{Fore.GREEN}╔═══════════════════════════════════════════════════════════╗
║  🟢 GREEN LIGHT — Model validated. Ready to build UI.      ║
║                                                            ║
║  Next steps:                                               ║
║  Week 2: Add ClubElo + Open-Meteo + Odds API               ║
║  Week 3: Next.js dashboard on Vercel                       ║
║  Week 4: Full UI + Soft launch                             ║
╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")
    else:
        print(f"""
{Fore.YELLOW}╔═══════════════════════════════════════════════════════════╗
║  🟡 MODEL NEEDS REFINEMENT                                 ║
║                                                            ║
║  Review results above. Common fixes:                       ║
║  • More training data (add more leagues/seasons)           ║
║  • Adjust EV thresholds per market                         ║  
║  • Tune Dixon-Coles xi decay parameter                     ║
║  • Check entity resolution (team name mismatches?)         ║
╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")

    return summary


if __name__ == "__main__":
    main()
