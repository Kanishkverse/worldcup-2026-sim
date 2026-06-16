"""Player-level strength, aggregated up to the team (player -> team).

The team-level engine treats a side as one Elo number. That cannot say what
France look like without Mbappe, or Portugal without their first-choice spine.
This module builds the real 26-player squads, picks the available starting XI
through the base engine's formation logic, and aggregates the players who are
actually on the pitch into a team rating.

The aggregation is deliberately star-sensitive. A plain average of eleven
ratings barely moves when you remove one man, because a deep squad just
promotes the next name. Real football is not like that: the best player is
worth more than his slot in an average. So each line blends its mean with its
best man, and the single best attacker enters again as an explicit term. The
result, mapped to Elo, drops by a meaningful amount when a star sits out and
barely at all when a squad player does.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Real player metrics and offline features, set before the base engine loads.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("WC_PLAYERS_CSV", os.path.join(_ROOT, "data", "players_2026_real.csv"))
os.environ.setdefault("WC_LIVE_FEATURES", "0")

import worldcup_2026_sim as base  # noqa: E402
from worldcup_2026_sim import Position, Player, Team  # noqa: E402

from .priors import elo_priors  # noqa: E402

# Rank-decay for the XI: the best players carry more weight, so the gap down
# to a replacement when a star sits out actually shows up. Plus a talisman
# term on the single best man in the squad. Both are tuned so that losing a
# star costs more on a thin squad than on a deep one.
_RANK_DECAY = 0.80
_W_CORE, _W_TALISMAN = 0.86, 0.14
_MAX_W_DEFENCE = 0.20


def _line(xi: List[Player], positions) -> List[float]:
    return [p.effective_rating() for p in xi if p.position in positions]


def _rank_weighted(ratings: List[float], decay: float = _RANK_DECAY) -> float:
    if not ratings:
        return 0.30
    rs = np.array(sorted(ratings, reverse=True))
    w = decay ** np.arange(len(rs))
    return float(np.dot(w / w.sum(), rs))


def team_overall(team: Team) -> float:
    """Aggregate the available starting XI into one strength scalar in ~[0.3,1].

    The rank-decay core leans on the best players, and the talisman term is the
    squad's single best available man. Remove a star and the core slips (a
    replacement enters at a low weight) and, if he was the best in the squad,
    the talisman term slips too.
    """
    xi = team.starting_xi()
    core = _rank_weighted([p.effective_rating() for p in xi])
    avail = [p.effective_rating() for p in team.squad if p.available]
    talisman = max(avail) if avail else core
    return _W_CORE * core + _W_TALISMAN * talisman


def attack_defence(team: Team) -> Tuple[float, float]:
    xi = team.starting_xi()
    atk = _rank_weighted(_line(xi, (Position.MID, Position.FWD)))
    dfn = (1.0 - _MAX_W_DEFENCE) * float(np.mean(_line(xi, (Position.GK, Position.DEF)) or [0.3])) \
        + _MAX_W_DEFENCE * max(_line(xi, (Position.GK, Position.DEF)) or [0.3])
    return atk, dfn


# ---------------------------------------------------------------------------
# Load the real player-level teams once, and calibrate overall -> Elo.
# ---------------------------------------------------------------------------

_INV = None
_ELO_FIT: Optional[Tuple[float, float]] = None
_FULL_OVERALL: Dict[str, float] = {}


def inventory():
    """The 48 real player-level Team objects (cached)."""
    global _INV
    if _INV is None:
        _INV = base.build_world_cup_2026(rng_seed=2026)
    return _INV


def _fit_elo() -> Tuple[float, float]:
    """Least-squares map from full-squad overall strength to the Elo priors."""
    global _ELO_FIT, _FULL_OVERALL
    if _ELO_FIT is None:
        inv = inventory()
        priors = elo_priors()
        teams = [t for t in inv.teams if t in priors]
        x = np.array([team_overall(inv.teams[t]) for t in teams])
        y = np.array([priors[t] for t in teams])
        a, b = np.polyfit(x, y, 1)
        _ELO_FIT = (float(a), float(b))
        _FULL_OVERALL = {t: float(ov) for t, ov in zip(teams, x)}
    return _ELO_FIT


def overall_to_elo(overall: float) -> float:
    a, b = _fit_elo()
    return a * overall + b


def player_elo(team_name: str) -> float:
    """Current Elo-equivalent of a team from its available players."""
    inv = inventory()
    return overall_to_elo(team_overall(inv.teams[team_name]))


def player_elo_priors() -> Dict[str, float]:
    """Every team's player-aggregated Elo, at full available strength."""
    _fit_elo()
    inv = inventory()
    return {t: player_elo(t) for t in inv.teams}


def elo_per_overall() -> float:
    """Elo points per unit of aggregate strength (the star-effect slope)."""
    return _fit_elo()[0]


# ---------------------------------------------------------------------------
# Availability: who is on the teamsheet before kickoff
# ---------------------------------------------------------------------------

def rule_out(team: Team, name_substrings: Iterable[str]) -> List[str]:
    """Mark matching players unavailable; return the names actually ruled out."""
    out = []
    subs = [s.lower() for s in name_substrings]
    for p in team.squad:
        if any(s in p.name.lower() for s in subs):
            p.available = False
            out.append(p.name)
    return out


def reset_availability(team: Team) -> None:
    for p in team.squad:
        p.available = True


def star_impact(team_name: str, name_substrings: Iterable[str]) -> Dict[str, float]:
    """Elo cost of losing the named player(s), via the aggregation. Restores
    availability afterward so the inventory is left untouched."""
    inv = inventory()
    team = inv.teams[team_name]
    full = player_elo(team_name)
    out = rule_out(team, name_substrings)
    without = player_elo(team_name)
    reset_availability(team)
    return {"ruled_out": out, "elo_full": round(full, 1),
            "elo_without": round(without, 1), "elo_drop": round(full - without, 1)}
