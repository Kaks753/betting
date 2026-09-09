"""
KBet WEMA — Weighted Exponentially Moving Average
Computes form metrics per team with decay weighting.

Weights: Recent (60%) · Mid (30%) · Old (10%)
Applied to: xG for, xG against, shots, corners, yellow cards, goals
"""

import numpy as np
import pandas as pd
from typing import Optional


WEMA_WEIGHTS = [0.60, 0.30, 0.10]  # Recent, Mid, Old
WEMA_BUCKETS = [3, 4, 4]           # Last 3 | Games 4-7 | Games 8-11


def _wema_series(values: list[float]) -> float:
    """
    Compute WEMA for a list of recent values (most recent first).
    Returns weighted average. Falls back to simple mean if <3 values.
    """
    if not values:
        return np.nan
    if len(values) < 3:
        return float(np.mean(values))

    bucket_means = []
    idx = 0
    for size, weight in zip(WEMA_BUCKETS, WEMA_WEIGHTS):
        bucket = values[idx: idx + size]
        if bucket:
            bucket_means.append((float(np.mean(bucket)), weight))
        idx += size

    if not bucket_means:
        return float(np.mean(values))

    total_weight = sum(w for _, w in bucket_means)
    return sum(v * w for v, w in bucket_means) / total_weight


def compute_team_form(
    df: pd.DataFrame,
    team_uuid: str,
    as_of_date: pd.Timestamp,
    n_games: int = 11
) -> dict:
    """
    Compute WEMA form metrics for a team as of a given date.
    Only uses matches BEFORE as_of_date (no look-ahead).

    Parameters
    ----------
    df          : Full match DataFrame with home_uuid, away_uuid
    team_uuid   : Internal team UUID
    as_of_date  : Cut-off date (exclusive — backtest safety)
    n_games     : How many recent games to consider

    Returns
    -------
    dict with keys: goals_for, goals_against, xg_for, xg_against,
                    shots_for, shots_against, corners_for, corners_against,
                    yellows_for, yellows_against, n_played, win_rate, draw_rate
    """
    # Get all matches for this team before as_of_date
    is_home = df["home_uuid"] == team_uuid
    is_away = df["away_uuid"] == team_uuid
    is_before = df["date"] < as_of_date

    team_matches = df[is_before & (is_home | is_away)].copy()
    team_matches = team_matches.sort_values("date", ascending=False).head(n_games)

    if len(team_matches) == 0:
        return _empty_form()

    goals_for_list, goals_against_list = [], []
    shots_for_list, shots_against_list = [], []
    corners_for_list, corners_against_list = [], []
    yellows_for_list, yellows_against_list = [], []
    results = []

    for _, row in team_matches.iterrows():
        at_home = row["home_uuid"] == team_uuid

        if at_home:
            gf = row.get("home_goals", np.nan)
            ga = row.get("away_goals", np.nan)
            sf = row.get("home_shots", np.nan)
            sa = row.get("away_shots", np.nan)
            cf = row.get("home_corners", np.nan)
            ca = row.get("away_corners", np.nan)
            yf = row.get("home_yellow", np.nan)
            ya = row.get("away_yellow", np.nan)
            res = row.get("result", "")
            result_val = 1 if res == "H" else (0.5 if res == "D" else 0)
        else:
            gf = row.get("away_goals", np.nan)
            ga = row.get("home_goals", np.nan)
            sf = row.get("away_shots", np.nan)
            sa = row.get("home_shots", np.nan)
            cf = row.get("away_corners", np.nan)
            ca = row.get("home_corners", np.nan)
            yf = row.get("away_yellow", np.nan)
            ya = row.get("home_yellow", np.nan)
            res = row.get("result", "")
            result_val = 1 if res == "A" else (0.5 if res == "D" else 0)

        if not pd.isna(gf): goals_for_list.append(float(gf))
        if not pd.isna(ga): goals_against_list.append(float(ga))
        if not pd.isna(sf): shots_for_list.append(float(sf))
        if not pd.isna(sa): shots_against_list.append(float(sa))
        if not pd.isna(cf): corners_for_list.append(float(cf))
        if not pd.isna(ca): corners_against_list.append(float(ca))
        if not pd.isna(yf): yellows_for_list.append(float(yf))
        if not pd.isna(ya): yellows_against_list.append(float(ya))
        results.append(result_val)

    n = len(team_matches)
    wins  = sum(1 for r in results if r == 1)
    draws = sum(1 for r in results if r == 0.5)

    return {
        "goals_for":         _wema_series(goals_for_list),
        "goals_against":     _wema_series(goals_against_list),
        "shots_for":         _wema_series(shots_for_list),
        "shots_against":     _wema_series(shots_against_list),
        "corners_for":       _wema_series(corners_for_list),
        "corners_against":   _wema_series(corners_against_list),
        "yellows_for":       _wema_series(yellows_for_list),
        "yellows_against":   _wema_series(yellows_against_list),
        "n_played":          n,
        "win_rate":          wins / n if n > 0 else 0.33,
        "draw_rate":         draws / n if n > 0 else 0.25,
    }


def _empty_form() -> dict:
    """Return neutral form metrics for teams with no history (cold start)."""
    return {
        "goals_for":       1.3,   # League average approximation
        "goals_against":   1.3,
        "shots_for":       11.0,
        "shots_against":   11.0,
        "corners_for":     5.0,
        "corners_against": 5.0,
        "yellows_for":     1.8,
        "yellows_against": 1.8,
        "n_played":        0,
        "win_rate":        0.33,
        "draw_rate":       0.25,
    }


def build_form_cache(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute WEMA form for every team at every match date.
    Much faster than computing on-the-fly during backtest.

    Returns df with additional columns:
      home_gf, home_ga, home_shots_f, home_corners_f, home_yellows_f
      away_gf, away_ga, away_shots_f, away_corners_f, away_yellows_f
    """
    from tqdm import tqdm

    df = df.copy().sort_values("date").reset_index(drop=True)

    home_form_records = []
    away_form_records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building WEMA form cache"):
        date = row["date"]

        h_form = compute_team_form(df.iloc[:idx], row["home_uuid"], date)
        a_form = compute_team_form(df.iloc[:idx], row["away_uuid"], date)

        home_form_records.append({
            "home_gf":         h_form["goals_for"],
            "home_ga":         h_form["goals_against"],
            "home_shots_f":    h_form["shots_for"],
            "home_shots_a":    h_form["shots_against"],
            "home_corners_f":  h_form["corners_for"],
            "home_corners_a":  h_form["corners_against"],
            "home_yellows_f":  h_form["yellows_for"],
            "home_yellows_a":  h_form["yellows_against"],
            "home_n_played":   h_form["n_played"],
            "home_win_rate":   h_form["win_rate"],
        })
        away_form_records.append({
            "away_gf":         a_form["goals_for"],
            "away_ga":         a_form["goals_against"],
            "away_shots_f":    a_form["shots_for"],
            "away_shots_a":    a_form["shots_against"],
            "away_corners_f":  a_form["corners_for"],
            "away_corners_a":  a_form["corners_against"],
            "away_yellows_f":  a_form["yellows_for"],
            "away_yellows_a":  a_form["yellows_against"],
            "away_n_played":   a_form["n_played"],
            "away_win_rate":   a_form["win_rate"],
        })

    home_form_df = pd.DataFrame(home_form_records)
    away_form_df = pd.DataFrame(away_form_records)

    df = pd.concat([df.reset_index(drop=True), home_form_df, away_form_df], axis=1)
    return df
