"""Where each team stands right now, for the per-game simulator to use.

The per-game simulator works off the pre-tournament player ratings. This
folds in what has happened since: it replays the played results onto the
player-aggregated ratings through the Bayesian state engine and reports each
team's current effective rating (posterior mean plus its form bonus) and its
form. The per-game simulator turns that into a small shift on the squad, so a
fixture is played by the teams as they are now, not as they were at the draw.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, Tuple

from . import form as form_mod
from . import squads
from .tournament_state import TournamentState

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(_ROOT, "data", "results_2026.csv")


def _load_rows(path: str):
    rows = []
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: (int(r.get("matchday", 1)), r.get("group", "")))
    return rows


def current_state(results_path: str = DEFAULT_RESULTS) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (effective_rating, form) per team after replaying real results.

    effective_rating is the posterior mean plus the form bonus, on the same
    player-aggregated Elo scale per_game starts from, so the difference between
    the two is exactly how far the tournament has moved a team.
    """
    priors = squads.player_elo_priors()
    state = TournamentState(priors)
    last_md = None
    for r in _load_rows(results_path):
        md = int(r.get("matchday", 1))
        if last_md is not None and md != last_md:
            state.recover_between_matchdays()
        state.update_after_match(r["home"].strip(), r["away"].strip(),
                                 int(r["home_goals"]), int(r["away_goals"]))
        last_md = md

    effective, form = {}, {}
    for t, s in state.teams.items():
        effective[t] = s.rating_mean + form_mod.form_rating_bonus(s.form)
        form[t] = s.form
    return effective, form


def movement(results_path: str = DEFAULT_RESULTS) -> Dict[str, float]:
    """How far each team has moved from its pre-tournament rating (Elo points)."""
    priors = squads.player_elo_priors()
    eff, _ = current_state(results_path)
    return {t: eff[t] - priors[t] for t in priors}
