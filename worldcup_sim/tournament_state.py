"""Bayesian tournament-state engine (Phase 1).

Every team carries a rating as a distribution, not a point. The prior mean is
the pre-tournament Elo; the prior std encodes how unsure we are about it.
After each played match we run a Normal-Normal conjugate update: the match
gives a noisy performance estimate, and the posterior pulls the mean toward
that estimate while shrinking the variance. The Monte Carlo engine samples
from the posterior, so early in the tournament a team's outcomes are spread
wide and they tighten as real evidence comes in.

This is intentionally not Elo. Elo nudges a point estimate by a fixed K and
never tells you how confident it is. Here the confidence is the whole point:
it is what lets a 0-0 against Cabo Verde widen Spain's cone of outcomes
instead of just shaving a few points off a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from . import form as form_mod


# --- update constants ---------------------------------------------------------

# Prior spread on the Elo rating before any match is played.
PRIOR_STD = 72.0
# A single match is a noisy read on true strength. This is the measurement
# noise of one performance estimate, in Elo points. Larger -> matches move
# the rating less and the posterior shrinks more slowly.
OBS_STD = 132.0
# Each goal of (clipped) margin is worth this many Elo points of performance.
GOALS_TO_ELO = 80.0
# A blowout should not read as an unbounded performance; cap the margin.
MARGIN_CAP = 3

# Per-match fatigue bump and between-matchday recovery.
FATIGUE_PER_MATCH = 0.34
FATIGUE_RECOVERY = 0.22
MORALE_DECAY = 0.6


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class TeamState:
    """Mutable per-team state that evolves across the tournament."""

    team: str
    rating_mean: float
    rating_std: float = PRIOR_STD
    form: float = 0.0
    fatigue: float = 0.0
    morale: float = 0.0          # exponential average of result sign, [-1, 1]
    yellow_cards: int = 0
    injuries: int = 0
    played: int = 0
    # Goal-difference performance running totals, handy for diagnostics.
    gf: int = 0
    ga: int = 0

    def posterior_strength_distribution(self):
        """Return (mean, std) of the current rating posterior."""
        return self.rating_mean, self.rating_std

    def sample_strength(self, rng: np.random.Generator) -> float:
        """Draw one rating from the posterior, plus the current form bonus.

        Fatigue trims a little off the top so a tired team plays slightly
        below its rating. The form bonus is applied here, not folded into the
        posterior mean, so it stays a transient effect that decays on its own.
        """
        base = rng.normal(self.rating_mean, self.rating_std)
        base += form_mod.form_rating_bonus(self.form)
        base -= 34.0 * self.fatigue
        return base

    @property
    def suspended_next(self) -> bool:
        """Yellow-card accumulation suspension proxy (two yellows = miss one)."""
        return self.yellow_cards >= 2 and self.yellow_cards % 2 == 0


def _performance_estimate(opp_rating: float, gd: int) -> float:
    """Noisy single-match read of a team's strength, in Elo points."""
    margin = _clip(gd, -MARGIN_CAP, MARGIN_CAP)
    return opp_rating + GOALS_TO_ELO * margin


class TournamentState:
    """Holds every team's TeamState and applies match evidence."""

    def __init__(self, priors: Dict[str, float]):
        self.teams: Dict[str, TeamState] = {
            t: TeamState(team=t, rating_mean=r) for t, r in priors.items()
        }

    def get(self, team: str) -> TeamState:
        return self.teams[team]

    # -- the Bayesian update ---------------------------------------------------
    def update_after_match(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        *,
        expected_gd_home: Optional[float] = None,
        quality_home: float = 0.0,
        yellows_home: int = 0,
        yellows_away: int = 0,
        injuries_home: int = 0,
        injuries_away: int = 0,
    ) -> None:
        """Fold one played match into both teams' posteriors and form."""
        hs, as_ = self.teams[home], self.teams[away]

        # Opponent uncertainty is folded into the measurement noise: a result
        # against a team we are unsure about tells us less.
        obs_var_h = OBS_STD ** 2 + as_.rating_std ** 2
        obs_var_a = OBS_STD ** 2 + hs.rating_std ** 2

        y_h = _performance_estimate(as_.rating_mean, home_goals - away_goals)
        y_a = _performance_estimate(hs.rating_mean, away_goals - home_goals)

        self._conjugate_update(hs, y_h, obs_var_h)
        self._conjugate_update(as_, y_a, obs_var_a)

        # Form: how far the actual margin beat the model's expectation.
        if expected_gd_home is None:
            expected_gd_home = (hs.rating_mean - as_.rating_mean) / 250.0
        sig_h = form_mod.performance_signal(
            expected_gd_home, home_goals - away_goals, quality=quality_home)
        sig_a = form_mod.performance_signal(
            -expected_gd_home, away_goals - home_goals, quality=-quality_home)
        hs.form = form_mod.update_form(hs.form, sig_h)
        as_.form = form_mod.update_form(as_.form, sig_a)

        # Morale, fatigue, cards, injuries, running record.
        self._bookkeep(hs, home_goals, away_goals, yellows_home, injuries_home)
        self._bookkeep(as_, away_goals, home_goals, yellows_away, injuries_away)

    @staticmethod
    def _conjugate_update(st: TeamState, y: float, obs_var: float) -> None:
        prior_prec = 1.0 / (st.rating_std ** 2)
        obs_prec = 1.0 / obs_var
        post_prec = prior_prec + obs_prec
        post_var = 1.0 / post_prec
        st.rating_mean = post_var * (st.rating_mean * prior_prec + y * obs_prec)
        st.rating_std = float(np.sqrt(post_var))

    @staticmethod
    def _bookkeep(st: TeamState, gf: int, ga: int, yellows: int, inj: int) -> None:
        result = 1 if gf > ga else (-1 if gf < ga else 0)
        st.morale = MORALE_DECAY * st.morale + (1 - MORALE_DECAY) * result
        st.fatigue = _clip(st.fatigue + FATIGUE_PER_MATCH, 0.0, 1.0)
        st.yellow_cards += yellows
        st.injuries += inj
        st.played += 1
        st.gf += gf
        st.ga += ga

    def recover_between_matchdays(self) -> None:
        """Bleed off some fatigue in the rest days between rounds."""
        for st in self.teams.values():
            st.fatigue = _clip(st.fatigue - FATIGUE_RECOVERY, 0.0, 1.0)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Plain dict of the current state, for reports and diagnostics."""
        return {
            t: {
                "rating_mean": round(s.rating_mean, 1),
                "rating_std": round(s.rating_std, 1),
                "form": round(s.form, 3),
                "fatigue": round(s.fatigue, 3),
                "morale": round(s.morale, 3),
                "played": s.played,
            }
            for t, s in self.teams.items()
        }
