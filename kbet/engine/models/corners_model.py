"""
Corners Model — Week 2.
Poisson regression for match total corners (home + away).

Why corners markets?
  - Softer market: bookmakers apply larger margins (~7-9%) vs goals (~5-6%)
  - Less liquid → slower line movement → more residual edge at opening
  - Data: home_corners + away_corners at 100% coverage in all_matches.parquet

Model specification:
  E[corners_home] = exp(mu + alpha_i + beta_j + rho_home)
  E[corners_away] = exp(mu + alpha_j + beta_i)
  where:
    alpha_i = home team attacking corner tendency
    beta_j  = opponent defensive corner tendency (concedes)
    rho_home = home field corner advantage

  Total expected corners = E[home] + E[away]
  P(total > line) = 1 - Poisson.CDF(floor(line), E[total])

Features used per match:
  - Rolling 6-match home corners avg (home team's last 6 games at home)
  - Rolling 6-match away corners conceded avg (away team's last 6 away)
  - League-level corner intercept (leagues differ markedly: EPL ~10.5, Bundesliga ~9.8)
  - Referee tendency (corners per game) — not yet, pending data

Walk-forward: fitted on training data, applied to test. No look-ahead.

Usage:
    model = CornersModel()
    model.fit(train_df)
    pred = model.predict(home_team, away_team, league_code, as_of_date, history_df)
    print(pred)
    # {"exp_total": 10.2, "over_9.5": 0.62, "under_9.5": 0.38,
    #  "over_10.5": 0.45, "under_10.5": 0.55}
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

logger = logging.getLogger(__name__)

# Default corner lines to compute probabilities for
DEFAULT_LINES = [8.5, 9.5, 10.5, 11.5, 12.5]

# Minimum matches to have a team-level parameter (else use league mean)
MIN_TEAM_MATCHES = 8


class CornersModel:
    """
    Poisson-based corners model with team-level attack/defense parameters.

    Parameters
    ----------
    decay_xi : float
        Time-decay parameter (same scheme as DC model). Default 0.006.
    min_matches : int
        Minimum historical matches to fit model. Default 100.
    """

    def __init__(self, decay_xi: float = 0.006, min_matches: int = 100):
        self.decay_xi = decay_xi
        self.min_matches = min_matches
        self.fitted = False

        # Parameters (set during fit)
        self.intercept: float = 0.0          # log(mu) — league/global mean
        self.home_advantage: float = 0.0     # log-scale home corner advantage
        self.attack: Dict[str, float] = {}   # team -> log attack param
        self.defense: Dict[str, float] = {}  # team -> log defense param (conceded)
        self.league_intercepts: Dict[str, float] = {}
        self.teams: List[str] = []
        self.n_matches: int = 0
        self._league_means: Dict[str, float] = {}

    # -----------------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------------

    def fit(self, matches: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        """
        Fit corners model on historical match data up to as_of_date.

        Parameters
        ----------
        matches : DataFrame with columns [date, home_team, away_team,
                  home_corners, away_corners, league_code]
        as_of_date : exclude matches on or after this date
        """
        df = matches[
            (matches["date"] < as_of_date) &
            matches["home_corners"].notna() &
            matches["away_corners"].notna()
        ].copy()

        if len(df) < self.min_matches:
            logger.warning(f"CornersModel: only {len(df)} matches — below min {self.min_matches}")
            self.fitted = False
            return

        # Time-decay weights
        df["days_ago"] = (as_of_date - df["date"]).dt.days
        df["weight"] = np.exp(-self.decay_xi * df["days_ago"])
        df["total_corners"] = df["home_corners"] + df["away_corners"]

        self.n_matches = len(df)
        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        team_idx = {t: i for i, t in enumerate(self.teams)}
        n_teams = len(self.teams)

        # League-level corner means (for intercept initialisation)
        self._league_means = df.groupby("league_code")["total_corners"].mean().to_dict()

        # ---- Optimise parameters via weighted Poisson log-likelihood --------
        # Parameter vector layout:
        #   [0] = global log-intercept
        #   [1] = home_advantage (additive log-scale)
        #   [2..n_teams+1] = attack params (home attacker)
        #   [n_teams+2..2*n_teams+1] = defense params (concedes)

        n_params = 2 + 2 * n_teams
        x0 = np.zeros(n_params)
        x0[0] = np.log(df["total_corners"].mean() / 2)  # per-team contribution

        # Pre-compute arrays for speed
        home_idx = df["home_team"].map(team_idx).values
        away_idx = df["away_team"].map(team_idx).values
        corners_h = df["home_corners"].values.astype(float)
        corners_a = df["away_corners"].values.astype(float)
        weights   = df["weight"].values

        def neg_log_likelihood(params):
            intercept = params[0]
            home_adv  = params[1]
            attack    = params[2:n_teams+2]
            defense   = params[n_teams+2:]

            # Expected corners
            log_lam_h = intercept + home_adv + attack[home_idx] + defense[away_idx]
            log_lam_a = intercept             + attack[away_idx] + defense[home_idx]
            lam_h = np.exp(log_lam_h)
            lam_a = np.exp(log_lam_a)

            # Poisson log-likelihood (vectorized)
            ll_h = weights * (corners_h * log_lam_h - lam_h)
            ll_a = weights * (corners_a * log_lam_a - lam_a)
            return -(ll_h.sum() + ll_a.sum())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                neg_log_likelihood, x0,
                method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-8},
            )

        params = result.x
        self.intercept      = params[0]
        self.home_advantage = params[1]
        for i, team in enumerate(self.teams):
            self.attack[team]  = params[2 + i]
            self.defense[team] = params[n_teams + 2 + i]

        self.fitted = True
        logger.debug(
            f"CornersModel fit: {n_teams} teams, {self.n_matches} matches, "
            f"home_adv={self.home_advantage:.3f}, "
            f"avg_exp_total={2*np.exp(self.intercept+self.home_advantage/2):.1f}"
        )

    # -----------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------

    def predict(
        self,
        home_team: str,
        away_team: str,
        lines: Optional[List[float]] = None,
    ) -> Optional[Dict]:
        """
        Predict corner probabilities for a match.

        Returns dict with expected totals and over/under probabilities
        for each line in `lines`.
        Returns None if model not fitted or team unknown.
        """
        if not self.fitted:
            return None

        if lines is None:
            lines = DEFAULT_LINES

        # Get team parameters (fall back to 0 for unknown teams)
        atk_h = self.attack.get(home_team, 0.0)
        def_h = self.defense.get(home_team, 0.0)
        atk_a = self.attack.get(away_team, 0.0)
        def_a = self.defense.get(away_team, 0.0)

        unknown_home = home_team not in self.attack
        unknown_away = away_team not in self.attack

        exp_h = np.exp(self.intercept + self.home_advantage + atk_h + def_a)
        exp_a = np.exp(self.intercept + atk_a + def_h)
        exp_total = exp_h + exp_a

        # Total corners = sum of two independent Poissons = Poisson(exp_total)
        result = {
            "exp_home":  round(float(exp_h), 2),
            "exp_away":  round(float(exp_a), 2),
            "exp_total": round(float(exp_total), 2),
            "unknown_home": unknown_home,
            "unknown_away": unknown_away,
        }

        for line in lines:
            # P(total > line) = P(X > floor(line)) where X ~ Poisson(exp_total)
            k = int(line)  # for x.5 lines, floor == x
            p_over  = 1.0 - poisson.cdf(k, exp_total)
            p_under = 1.0 - p_over
            key = str(line).replace(".", "_")
            result[f"over_{key}"]  = round(float(p_over),  4)
            result[f"under_{key}"] = round(float(p_under), 4)

        return result

    def predict_ev(
        self,
        home_team: str,
        away_team: str,
        odds_over: float,
        odds_under: float,
        line: float = 9.5,
        slippage: float = 0.20,
    ) -> Dict[str, float]:
        """
        Compute expected value for corners over/under bet.

        Returns {"ev_over": float, "ev_under": float,
                 "best_side": "over"|"under"|"none",
                 "exp_total": float}
        """
        pred = self.predict(home_team, away_team, lines=[line])
        if pred is None:
            return {"ev_over": -99, "ev_under": -99, "best_side": "none", "exp_total": 0}

        key = str(line).replace(".", "_")
        p_over  = pred[f"over_{key}"]
        p_under = pred[f"under_{key}"]
        exp_total = pred["exp_total"]

        ev_over  = ((p_over  * odds_over)  - 1.0) * (1 - slippage)
        ev_under = ((p_under * odds_under) - 1.0) * (1 - slippage)

        best = "none"
        if ev_over > 0 and ev_over >= ev_under:
            best = "over"
        elif ev_under > 0:
            best = "under"

        return {
            "ev_over":   round(ev_over,  4),
            "ev_under":  round(ev_under, 4),
            "p_over":    round(p_over,   4),
            "p_under":   round(p_under,  4),
            "best_side": best,
            "exp_total": exp_total,
        }
