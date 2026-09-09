"""
KBet Data Downloader — football-data.co.uk CSV fetcher
Downloads historical match data + closing odds for all configured leagues/seasons.
No API key required. 100% free.
"""

import os
import sys
import time
import requests
import pandas as pd
from io import StringIO
from typing import Optional
from colorama import Fore, Style, init

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import (
    FOOTBALL_DATA_CO_UK_BASE, LEAGUES, SEASONS,
    FDCUK_COLUMNS, DATA_RAW_DIR, DATA_PROCESSED_DIR
)

init(autoreset=True)


def log(msg: str, level: str = "INFO"):
    colors = {"INFO": Fore.CYAN, "OK": Fore.GREEN, "WARN": Fore.YELLOW, "ERR": Fore.RED}
    print(f"{colors.get(level, '')}[{level}] {msg}{Style.RESET_ALL}")


def resolve_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return first matching column name from candidates list."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw football-data.co.uk columns to our unified internal schema.
    Handles column name variations across seasons/leagues.
    """
    out = {}

    for field, candidates in FDCUK_COLUMNS.items():
        col = resolve_column(df, candidates)
        if col:
            out[field] = df[col]

    result = pd.DataFrame(out)

    # Parse date — handle DD/MM/YY and DD/MM/YYYY
    if "date" in result.columns:
        result["date"] = pd.to_datetime(
            result["date"], dayfirst=True, errors="coerce"
        )

    # Drop rows with no date or no teams (blank rows at end of CSV)
    result = result.dropna(subset=["date", "home_team", "away_team"])
    result = result[result["home_team"].str.strip() != ""]

    return result


def download_league_season(league_code: str, season: str, league_name: str) -> Optional[pd.DataFrame]:
    """
    Download one league/season CSV from football-data.co.uk
    URL pattern: https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv
    """
    url = f"{FOOTBALL_DATA_CO_UK_BASE}/mmz4281/{season}/{league_code}.csv"

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            return None  # Season/league combo doesn't exist — normal
        resp.raise_for_status()

        # Parse CSV
        df = pd.read_csv(StringIO(resp.text), encoding="unicode_escape", on_bad_lines="skip")
        if df.empty or len(df) < 5:
            return None

        df = normalize_columns(df)
        if df.empty:
            return None

        # Add metadata columns
        df["league_code"]  = league_code
        df["league_name"]  = league_name
        df["season"]       = season

        return df

    except requests.RequestException as e:
        log(f"Network error {league_code}/{season}: {e}", "WARN")
        return None
    except Exception as e:
        log(f"Parse error {league_code}/{season}: {e}", "WARN")
        return None


def download_all(force_refresh: bool = False) -> pd.DataFrame:
    """
    Download all configured leagues × seasons.
    Saves individual CSVs to DATA_RAW_DIR.
    Returns combined DataFrame.
    """
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    combined_path = os.path.join(DATA_PROCESSED_DIR, "all_matches.parquet")

    if os.path.exists(combined_path) and not force_refresh:
        log(f"Loading cached combined dataset from {combined_path}", "OK")
        return pd.read_parquet(combined_path)

    all_frames = []
    total = len(LEAGUES) * len(SEASONS)
    done = 0

    log(f"Downloading {total} league/season combinations...", "INFO")

    for league_code, country, league_name in LEAGUES:
        league_frames = []

        for season in SEASONS:
            done += 1
            cache_path = os.path.join(DATA_RAW_DIR, f"{league_code}_{season}.csv")

            if os.path.exists(cache_path) and not force_refresh:
                df = pd.read_csv(cache_path, parse_dates=["date"])
                league_frames.append(df)
                print(f"  [{done}/{total}] {league_name} {season} — cached ({len(df)} rows)", end="\r")
            else:
                df = download_league_season(league_code, season, league_name)
                if df is not None:
                    df.to_csv(cache_path, index=False)
                    league_frames.append(df)
                    log(f"  [{done}/{total}] {league_name} {season} — {len(df)} matches", "OK")
                else:
                    log(f"  [{done}/{total}] {league_name} {season} — not found (skipped)", "WARN")

                time.sleep(0.3)  # Polite delay — respect the server

        if league_frames:
            league_df = pd.concat(league_frames, ignore_index=True)
            all_frames.append(league_df)

    if not all_frames:
        log("No data downloaded! Check internet connection.", "ERR")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    # Save combined parquet (fast loading for backtester)
    combined.to_parquet(combined_path, index=False)
    log(f"\n✅ Combined dataset: {len(combined):,} matches across {combined['league_name'].nunique()} leagues", "OK")
    log(f"   Date range: {combined['date'].min().date()} → {combined['date'].max().date()}", "OK")
    log(f"   Saved to: {combined_path}", "OK")

    return combined


def get_dataset_summary(df: pd.DataFrame):
    """Print a clean summary of the downloaded dataset."""
    print(f"\n{'─'*60}")
    print(f"  {'KBET DATASET SUMMARY':^58}")
    print(f"{'─'*60}")
    print(f"  Total matches:     {len(df):>10,}")
    print(f"  Leagues:           {df['league_name'].nunique():>10}")
    print(f"  Seasons:           {df['season'].nunique():>10}")
    print(f"  Date range:        {str(df['date'].min().date()):>10} → {str(df['date'].max().date())}")
    print(f"  Teams:             {df['home_team'].nunique():>10}")

    # Check odds coverage
    has_b365     = df["odds_home_b365"].notna().sum()
    has_pinnacle = df["odds_home_pinnacle"].notna().sum() if "odds_home_pinnacle" in df.columns else 0
    has_max      = df["odds_home_max"].notna().sum() if "odds_home_max" in df.columns else 0
    has_ou       = df["odds_over25_b365"].notna().sum() if "odds_over25_b365" in df.columns else 0

    print(f"\n  ODDS COVERAGE:")
    print(f"  Bet365 1X2:        {has_b365:>10,} ({100*has_b365/len(df):.1f}%)")
    print(f"  Pinnacle 1X2:      {has_pinnacle:>10,} ({100*has_pinnacle/len(df):.1f}%)")
    print(f"  Max odds 1X2:      {has_max:>10,} ({100*has_max/len(df):.1f}%)")
    print(f"  O/U 2.5:           {has_ou:>10,} ({100*has_ou/len(df):.1f}%)")

    print(f"\n  BY LEAGUE:")
    league_counts = df.groupby("league_name").agg(
        matches=("date", "count"),
        date_from=("date", "min"),
        date_to=("date", "max")
    ).sort_values("matches", ascending=False)
    for name, row in league_counts.iterrows():
        print(f"  {name:<25} {row['matches']:>5} matches  "
              f"({str(row['date_from'].date())} → {str(row['date_to'].date())})")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KBet Data Downloader")
    parser.add_argument("--refresh", action="store_true", help="Force re-download all data")
    args = parser.parse_args()

    df = download_all(force_refresh=args.refresh)
    if not df.empty:
        get_dataset_summary(df)
