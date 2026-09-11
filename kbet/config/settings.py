"""
KBet Configuration — Central settings for the entire engine
"""

# ─── Data Sources ──────────────────────────────────────────────────────────────
FOOTBALL_DATA_CO_UK_BASE = "https://football-data.co.uk"

# Leagues to download: (country_code, division_code, display_name)
LEAGUES = [
    ("E0",  "England",      "Premier League"),
    ("E1",  "England",      "Championship"),
    ("SP1", "Spain",        "La Liga"),
    ("D1",  "Germany",      "Bundesliga"),
    ("I1",  "Italy",        "Serie A"),
    ("F1",  "France",       "Ligue 1"),
    ("N1",  "Netherlands",  "Eredivisie"),
    ("P1",  "Portugal",     "Primeira Liga"),
    ("B1",  "Belgium",      "Pro League"),
    ("G1",  "Greece",       "Super League"),
]

# Seasons to download (football-data.co.uk format)
SEASONS = [
    "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"
]

# CSV column mappings from football-data.co.uk
FDCUK_COLUMNS = {
    "date":      ["Date"],
    "home_team": ["HomeTeam"],
    "away_team": ["AwayTeam"],
    "home_goals": ["FTHG", "HG"],
    "away_goals": ["FTAG", "AG"],
    "home_ht":   ["HTHG"],
    "away_ht":   ["HTAG"],
    "result":    ["FTR"],
    "ht_result": ["HTR"],
    # Match stats
    "home_shots":   ["HS"],
    "away_shots":   ["AS"],
    "home_sot":     ["HST"],
    "away_sot":     ["AST"],
    "home_corners": ["HC"],
    "away_corners": ["AC"],
    "home_fouls":   ["HF"],
    "away_fouls":   ["AF"],
    "home_yellow":  ["HY"],
    "away_yellow":  ["AY"],
    "home_red":     ["HR"],
    "away_red":     ["AR"],
    # Odds columns (best available)
    "odds_home_b365": ["B365H"],
    "odds_draw_b365": ["B365D"],
    "odds_away_b365": ["B365A"],
    "odds_home_pinnacle": ["PSH", "PH"],
    "odds_draw_pinnacle": ["PSD", "PD"],
    "odds_away_pinnacle": ["PSA", "PA"],
    "odds_home_max": ["MaxH", "BbMxH"],
    "odds_draw_max": ["MaxD", "BbMxD"],
    "odds_away_max": ["MaxA", "BbMxA"],
    "odds_home_avg": ["AvgH", "BbAvH"],
    "odds_draw_avg": ["AvgD", "BbAvD"],
    "odds_away_avg": ["AvgA", "BbAvA"],
    # Over/Under odds
    "odds_over25_b365":  ["B365>2.5"],
    "odds_under25_b365": ["B365<2.5"],
    "odds_over25_max":   ["Max>2.5"],
    "odds_under25_max":  ["Max<2.5"],
    "odds_over25_avg":   ["Avg>2.5"],
    "odds_under25_avg":  ["Avg<2.5"],
}

# ─── Model Parameters ──────────────────────────────────────────────────────────
# 18mo window + xi 0.010 live (70d half-life) — was 1095 3yr 0.0065 107d stale (Chelsea case)
DIXON_COLES = {
    "xi": 0.0065,           # Base half-life 107d; live uses 0.010 (70d) via daily_card.py live branch
    "xi_live": 0.010,       # Live recency (70d) — new coach/squad overhaul
    "min_games": 5,         # Minimum games before model trusts team ratings
    "home_advantage": 0.25, # Initial home advantage parameter (log scale)
}

WEMA_WEIGHTS = {
    "recent":  (0, 3,  0.60),   # Last 3 games: 60%
    "mid":     (3, 7,  0.30),   # Games 4-7: 30%
    "old":     (7, 11, 0.10),   # Games 8-11: 10%
}

# ─── Value Detection Thresholds (EV%) by market ────────────────────────────────
# NOTE: EV thresholds raised significantly after backtest v1 analysis:
# v1 result: 7% EV gave 2,700+ bets/season with -5% avg ROI (model not precise enough at that level)
# v2 fix: raise 1X2 to 15%, O/U to 8%, add prob-gap filter in Kelly settings
EV_THRESHOLDS = {
    "1x2":          0.15,   # 1X2 result: 15% EV minimum (was 7% — too low for DC precision)
    "over_under":   0.08,   # O/U 2.5: 8% EV minimum (was 3.5% — generated too many false positives)
    "btts":         0.10,   # Both Teams To Score: 10%
    "asian_hcap":   0.08,   # Asian Handicap: 8%
    "corners":      0.08,   # Corner markets: 8%
    "yellow_cards": 0.08,   # Yellow card markets: 8%
}

# ─── Bankroll & Staking ────────────────────────────────────────────────────────
KELLY = {
    "fraction":     0.25,   # Quarter Kelly
    "max_per_bet":  0.02,   # Max 2% bankroll per single bet
    "max_daily":    0.05,   # Max 5% bankroll per day (portfolio Kelly)
    "min_ev":       0.08,   # Never bet below 8% EV regardless of Kelly (raised from 3.5%)
    "min_prob":     0.25,   # Never bet on outcomes with <25% probability (raised from 20%)
    "min_prob_gap": 0.06,   # Minimum probability gap: our_prob - book_implied >= 6%
}

# ─── Backtest Settings ─────────────────────────────────────────────────────────
# Sharpened 0.60 (was 0.62) — Brier 0.62 barely beats random 0.667, 0.60 validates real signal
# Dixon-Coles papers 0.55-0.63, our 0.576 V3 / 0.579 V4 — 0.60 is honest gate
BACKTEST = {
    "min_bets_gate":   300,
    "slippage_penalty": 0.20,
    "walk_forward_rounds": [
        {"train_end": "2022-06-01", "test_start": "2022-08-01", "test_end": "2023-06-01"},
        {"train_end": "2023-06-01", "test_start": "2023-08-01", "test_end": "2024-06-01"},
        {"train_end": "2024-06-01", "test_start": "2024-08-01", "test_end": "2025-06-01"},
    ],
    "brier_gate": 0.60,       # Tightened 0.62→0.60 (sharpened)
    "roi_gate":   0.03,
    "clv_gate":   0.003,      # 0.3% CLV gate for live proof (G1)
}

# ─── Confidence Tiers ─────────────────────────────────────────────────────────
CONFIDENCE_TIERS = {
    "FIRE":  0.15,   # EV > 15% — highest confidence
    "SOLID": 0.10,   # EV 10-15%
    "WATCH": 0.05,   # EV 5-10%
}

# ─── Paths ─────────────────────────────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR        = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
BACKTEST_RESULTS_DIR = os.path.join(BASE_DIR, "data", "backtest_results")
LOGS_DIR            = os.path.join(BASE_DIR, "logs")
