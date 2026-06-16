"""Per-game simulator over the real match physics.

Where the tournament engine fires a calibrated Poisson per match, this runs the
base engine's full minute-by-minute simulation for a single fixture, many
times, and returns the whole outcome distribution. That means the real model:
formation-based XI selection, the player cohesion graph, accumulating fatigue,
sendings-off, threshold substitutions, the manager's half-time tactical
adjustment, and extra time plus penalties in a knockout.

Because it works on the real squads, you set the teamsheet before kickoff.
Rule a player out and the formation reshapes around the absence, the cohesion
graph is recomputed, and the outcome distribution shifts accordingly. That is
the player-to-team idea made physical: France without Mbappe is a different
team here, not a hand-edited number.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import worldcup_2026_sim as base
from . import squads


@dataclass
class PerGameResult:
    team_a: str
    team_b: str
    runs: int
    a_win: float
    draw: float
    b_win: float
    mean_goals_a: float
    mean_goals_b: float
    over_2_5: float
    modal_score: Tuple[int, int]
    scorelines: Dict[Tuple[int, int], float] = field(default_factory=dict)
    pens_rate: float = 0.0

    def summary(self) -> str:
        a, b = self.team_a, self.team_b
        top = sorted(self.scorelines.items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines = [
            f"{a} vs {b}  ({self.runs} simulations)",
            f"  {a} win {self.a_win*100:5.1f}%   draw {self.draw*100:5.1f}%   "
            f"{b} win {self.b_win*100:5.1f}%",
            f"  expected goals {self.mean_goals_a:.2f} - {self.mean_goals_b:.2f}   "
            f"over 2.5: {self.over_2_5*100:.0f}%",
            f"  most likely {self.modal_score[0]}-{self.modal_score[1]}   "
            f"top scorelines: " + ", ".join(f"{s[0]}-{s[1]} {p*100:.0f}%" for s, p in top),
        ]
        return "\n".join(lines)


@contextmanager
def _teamsheet(team, out_players: Iterable[str], fatigue_bump: float):
    """Apply availability/fatigue for one fixture, then restore the squad."""
    snap = [(p, p.available, p.fatigue) for p in team.squad]
    if out_players:
        squads.rule_out(team, out_players)
    if fatigue_bump:
        for p in team.squad:
            p.fatigue = float(np.clip(p.fatigue + fatigue_bump, 0.0, 1.0))
    try:
        yield team
    finally:
        for p, avail, fat in snap:
            p.available, p.fatigue = avail, fat


class PerGameSimulator:
    def __init__(self, use_manager: bool = True, seed: int = 0,
                 config: Optional["base.EngineConfig"] = None):
        self.rng = np.random.default_rng(seed)
        self.sim = base.MatchSimulator(self.rng, config=config)
        self.agent = (base.TacticalManagerAgent(base.build_default_news_store(),
                                                use_llm=False)
                      if use_manager else None)

    def fixture(
        self,
        team_a: str,
        team_b: str,
        runs: int = 2000,
        knockout: bool = False,
        out_a: Optional[Iterable[str]] = None,
        out_b: Optional[Iterable[str]] = None,
        fatigue_a: float = 0.0,
        fatigue_b: float = 0.0,
        common_seed: Optional[int] = None,
    ) -> PerGameResult:
        # Common random numbers: pass the same common_seed to two fixtures and
        # the random stream is identical, so the only thing that differs is the
        # teamsheet. That is what makes a one-player change visible against the
        # Monte Carlo noise instead of drowning in it.
        if common_seed is not None:
            self.rng = np.random.default_rng(common_seed)
            self.sim.rng = self.rng
        inv = squads.inventory()
        ta, tb = inv.teams[team_a], inv.teams[team_b]

        aw = dr = bw = 0
        ga_tot = gb_tot = over = pens = 0
        scores: Counter = Counter()
        with _teamsheet(ta, out_a or [], fatigue_a), \
                _teamsheet(tb, out_b or [], fatigue_b):
            for _ in range(runs):
                res = self.sim.simulate(ta, tb, knockout=knockout,
                                        tactical_agent=self.agent)
                ga, gb = res.home_goals, res.away_goals
                ga_tot += ga; gb_tot += gb
                over += int(ga + gb > 2)
                scores[(ga, gb)] += 1
                if res.went_to_pens:
                    pens += 1
                    winner = res.winner
                    aw += int(winner == team_a); bw += int(winner == team_b)
                elif ga > gb:
                    aw += 1
                elif gb > ga:
                    bw += 1
                else:
                    dr += 1

        modal = max(scores, key=scores.get)
        return PerGameResult(
            team_a=team_a, team_b=team_b, runs=runs,
            a_win=aw / runs, draw=dr / runs, b_win=bw / runs,
            mean_goals_a=ga_tot / runs, mean_goals_b=gb_tot / runs,
            over_2_5=over / runs, modal_score=modal,
            scorelines={s: c / runs for s, c in scores.items()},
            pens_rate=pens / runs,
        )
