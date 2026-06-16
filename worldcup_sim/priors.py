"""Pre-tournament priors and the bracket, pulled from the base engine.

We deliberately reuse the numbers and the draw that already live in
worldcup_2026_sim.py rather than keeping a second copy that can drift. Elo
is the prior mean for every team's rating; the group draw and the official
2026 bracket slot tables come straight from the Tournament resolver.
"""

from __future__ import annotations

import os
from typing import Dict, List

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA_DIR = os.path.normpath(DATA_DIR)


def _base_engine():
    # Imported lazily so the lightweight modules (styles, venues, state) stay
    # importable even if the big engine's optional deps are not all present.
    import worldcup_2026_sim as base
    return base


def elo_priors() -> Dict[str, float]:
    """Team -> pre-tournament Elo rating for the 48-team 2026 field."""
    base = _base_engine()
    return {team: feats[0] for team, feats in base._FEATURES_2026.items()}


def fifa_points() -> Dict[str, float]:
    """Team -> FIFA ranking points (reporting only)."""
    base = _base_engine()
    return {team: feats[2] for team, feats in base._FEATURES_2026.items()}


def group_draw() -> Dict[str, List[str]]:
    """Group letter -> the four teams drawn into it."""
    base = _base_engine()
    return {g: list(teams) for g, teams in base._REAL_GROUPS_2026.items()}


def team_to_group() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for g, teams in group_draw().items():
        for t in teams:
            out[t] = g
    return out


def host_nations() -> Dict[str, str]:
    """The three co-hosts, used for the home-soil edge."""
    return {"Mexico": "Mexico", "Canada": "Canada", "United States": "USA"}


def bracket():
    """The Tournament class, used only for its static slot tables."""
    return _base_engine().Tournament
