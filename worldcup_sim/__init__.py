"""Dynamic tournament-state layer for the 2026 World Cup simulator.

The original engine (worldcup_2026_sim.py) treats team strength as a fixed
pre-tournament number. This package turns it into something that moves as
results arrive: a Bayesian rating that updates after every played match,
latent form, fatigue and travel load, style matchups, and matchday-aware
tactical intent. live_forecast.py wires it all together.
"""

from .priors import elo_priors, group_draw, DATA_DIR

__all__ = ["elo_priors", "group_draw", "DATA_DIR"]
