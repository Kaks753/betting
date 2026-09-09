"""
Cards Model — Week 2.
Negative Binomial regression for total yellow cards in a match.

Why cards markets?
  - Very soft market: bookmakers rarely sharp on cards (less data, more variance)
  - Cards markets run at ~8-10% margin, leaving room to find genuine edge
  - Data: home_yellow + away_yellow at 100% coverage in all_matches.parquet

Why Negative Binomial, not Poisson?
  - Yellow cards are overdispersed: variance > mean
  - Poisson underestimates the probability of very low (0-1) and very high (8+)
  - NegBinom with dispersion r fits the tail correctly

Model specification:
  mu_i = exp(intercept + alpha_home + beta_away + league_effect + season_effect)
  r (dispersion) is fitted globally

  P(total > line) = 1 - NegBinom.CDF(floor(line), mu_i, r)

Features:
  - Home/away team "aggressiveness" (rolling yellow cards generated)
  - Home/away team "discipline" (rolling yellow cards conceded)
  - League effect (Serie A averages 5.1 cards/game, Premier League 3.8)
  - Season effect (end-of-season matches → more cards at top/bottom of table)

Usage:
    model = CardsModel()
    model.fit(train_df)
    pred = model.predict("Arsenal", "Chelsea", "E0")
    print(pred)
    # {"exp_total": 4.1, "over_3.5": 0.72, "under_3.5": 0.28, ...}
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import nbinom

logger = logging.getLogger(__name__)

DEFAULT_LINES = [2.5, 3.5, 4.5, 5.5]
MIN_TEAM_MATCHES = 6


class CardsModel:
    """
    Negative Binomial cards model.

    Parameters
    ----------
    decay_xi : float
        Time-decay parameter. Default 0.005 (slightly slower than DC — cards
        are more league/referee driven than team-level).
    min_matches : int
        Minimum training matches. Default 80.
    """

    def __init__(self, decay_xi: float = 0.005, min_matches: int = 80):
        self.decay_xi   = decay_xi
        self.min_matches = min_matches
        self.fitted = False

        self.intercept: float = 0.0
        self.home_advantage: float = 0.0   # cards: home usually gets fewer
        self.attack: Dict[str, float]  = {}  # "aggressiveness" — generates cards
        self.defense: Dict[str, float] = {}  # "discipline" — concedes cards
        self.league_fx: Dict[str, float] = {}
        self.dispersion: float = 5.0        # r parameter of NegBinom
        self.teams: List[str] = []
        self.n_matches: int = 0
        self._mean_cards: float = 4.0       # fallback

    # -----------------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------------

    def fit(self, matches: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        """
        Fit on historical data up to as_of_date.

        matches must have: date, home_team, away_team,
                           home_yellow, away_yellow, league_code
        """
        df = matches[
            (matches["date"] < as_of_date) &
            matches["home_yellow"].notna() &
            matches["away_yellow"].notna()
        ].copy()

        if len(df) < self.min_matches:
            logger.warning(f"CardsModel: only {len(df)} matches — using defaults")
            self._mean_cards = 4.0
            self.fitted = False
            return

        df["days_ago"]   = (as_of_date - df["date"]).dt.days
        df["weight"]     = np.exp(-self.decay_xi * df["days_ago"])
        df["total_cards"] = df["home_yellow"] + df["away_yellow"]
        self._mean_cards = float(df["total_cards"].mean())

        self.n_matches = len(df)
        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        team_idx = {t: i for i, t in enumerate(self.teams)}
        n_teams = len(self.teams)

        leagues = sorted(df["league_code"].unique())
        league_idx = {lg: i for i, lg in enumerate(leagues)}
        n_leagues = len(leagues)

        # Parameter layout:
        # [0] intercept, [1] home_adv
        # [2..n_teams+1] = attack, [n_teams+2..2*n_teams+1] = defense
        # [2*n_teams+2..2*n_teams+n_leagues+1] = league effects
        # [-1] = log(dispersion r)

        n_params = 2 + 2*n_teams + n_leagues + 1
        x0 = np.zeros(n_params)
        x0[0] = np.log(self._mean_cards / 2)
        x0[-1] = np.log(5.0)  # dispersion

        home_ti = df["home_team"].map(team_idx).values
        away_ti = df["away_team"].map(team_idx).values
        league_li = df["league_code"].map(league_idx).values
        cards_h = df["home_yellow"].values.astype(float)
        cards_a = df["away_yellow"].values.astype(float)
        wts     = df["weight"].values

        def neg_log_likelihood(params):
            intercept = params[0]
            home_adv  = params[1]
            attack    = params[2:n_teams+2]
            defense   = params[n_teams+2:2*n_teams+2]
            league_e  = params[2*n_teams+2:2*n_teams+2+n_leagues]
            log_r     = params[-1]
            r = np.exp(log_r)  # dispersion, must be positive

            lam_h = np.exp(intercept + home_adv + attack[home_ti] + defense[away_ti] + league_e[league_li])
            lam_a = np.exp(intercept            + attack[away_ti] + defense[home_ti] + league_e[league_li])

            # Negative Binomial log-likelihood
            # NegBinom(k; mu, r): logP = log C(k+r-1,k) + k*log(mu/(mu+r)) + r*log(r/(mu+r))
            def nb_ll(k, mu):
                p = mu / (mu + r)
                return (gammaln(k + r) - gammaln(r) - gammaln(k + 1)
                        + r * np.log(r / (mu + r))
                        + k * np.log(np.clip(p, 1e-10, 1)))

            ll = wts * (nb_ll(cards_h, lam_h) + nb_ll(cards_a, lam_a))
            return -ll.sum()

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
        self.dispersion     = float(np.exp(params[-1]))

        for i, team in enumerate(self.teams):
            self.attack[team]  = params[2 + i]
            self.defense[team] = params[n_teams + 2 + i]

        for i, lg in enumerate(leagues):
            self.league_fx[lg] = params[2*n_teams + 2 + i]

        self.fitted = True
        logger.debug(
            f"CardsModel fit: {n_teams} teams, {self.n_matches} matches, "
            f"dispersion r={self.dispersion:.2f}, mean={self._mean_cards:.2f}"
        )

    # -----------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------

    def predict(
        self,
        home_team: str,
        away_team: str,
        league_code: str,
        lines: Optional[List[float]] = None,
    ) -> Optional[Dict]:
        """
        Predict card probabilities for a match.

        Returns dict with expected total and over/under for each line.
        """
        if not self.fitted:
            return None

        if lines is None:
            lines = DEFAULT_LINES

        atk_h = self.attack.get(home_team, 0.0)
        def_h = self.defense.get(home_team, 0.0)
        atk_a = self.attack.get(away_team, 0.0)
        def_a = self.defense.get(away_team, 0.0)
        lg_fx = self.league_fx.get(league_code, 0.0)

        unknown_home = home_team not in self.attack
        unknown_away = away_team not in self.attack

        exp_h = np.exp(self.intercept + self.home_advantage + atk_h + def_a + lg_fx)
        exp_a = np.exp(self.intercept + atk_a + def_h + lg_fx)
        exp_total = exp_h + exp_a

        result = {
            "exp_home":     round(float(exp_h), 2),
            "exp_away":     round(float(exp_a), 2),
            "exp_total":    round(float(exp_total), 2),
            "dispersion_r": round(self.dispersion, 3),
            "unknown_home": unknown_home,
            "unknown_away": unknown_away,
        }

        # NegBinom parameters: mu = exp_total, r = dispersion
        # scipy nbinom uses (n=r, p=r/(r+mu))
        r = self.dispersion
        p = r / (r + exp_total)

        for line in lines:
            k = int(line)   # for x.5 lines
            p_under = float(nbinom.cdf(k, r, p))
            p_over  = 1.0 - p_under
            key = str(line).replace(".", "_")
            result[f"over_{key}"]  = round(p_over,  4)
            result[f"under_{key}"] = round(p_under, 4)

        return result

    def predict_ev(
        self,
        home_team: str,
        away_team: str,
        league_code: str,
        odds_over: float,
        odds_under: float,
        line: float = 3.5,
        slippage: float = 0.20,
    ) -> Dict[str, float]:
        """Compute EV for cards over/under bet."""
        pred = self.predict(home_team, away_team, league_code, lines=[line])
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
