"""Group standings and the 2026 tiebreakers.

FIFA ranks a group on points, then overall goal difference, then goals for,
and only then drops into the head-to-head record between the teams still
level, fair play (fewest disciplinary points), and finally drawing of lots.
We reproduce that order exactly, with the drawing of lots made deterministic
from the team name so a run is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Standing:
    team: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    gf: int = 0
    ga: int = 0
    cards: int = 0   # disciplinary points (yellow=1, red=3), for fair play

    @property
    def points(self) -> int:
        return 3 * self.won + self.drawn

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def apply(self, scored: int, conceded: int) -> None:
        self.played += 1
        self.gf += scored
        self.ga += conceded
        if scored > conceded:
            self.won += 1
        elif scored == conceded:
            self.drawn += 1
        else:
            self.lost += 1


@dataclass
class Match:
    home: str
    away: str
    home_goals: int
    away_goals: int
    cards_home: int = 0
    cards_away: int = 0


def _records(matches: List[Match], teams: List[str]) -> Dict[str, Standing]:
    recs = {t: Standing(team=t) for t in teams}
    for m in matches:
        if m.home not in recs or m.away not in recs:
            continue
        recs[m.home].apply(m.home_goals, m.away_goals)
        recs[m.away].apply(m.away_goals, m.home_goals)
        recs[m.home].cards += m.cards_home
        recs[m.away].cards += m.cards_away
    return recs


def _head_to_head(cluster: List[str], matches: List[Match]) -> Dict[str, tuple]:
    """Mini-table (points, gd, gf) among only the tied teams."""
    sub = [m for m in matches if m.home in cluster and m.away in cluster]
    mini = {t: Standing(team=t) for t in cluster}
    for m in sub:
        mini[m.home].apply(m.home_goals, m.away_goals)
        mini[m.away].apply(m.away_goals, m.home_goals)
    return {t: (mini[t].points, mini[t].gd, mini[t].gf) for t in cluster}


def rank_group(matches: List[Match], teams: List[str]) -> List[Standing]:
    """Return the four teams ordered by the full 2026 tiebreaker chain."""
    recs = _records(matches, teams)

    # First pass: points, overall GD, overall GF.
    ordered = sorted(recs.values(),
                     key=lambda s: (s.points, s.gd, s.gf), reverse=True)

    # Resolve any cluster still level on all three.
    out: List[Standing] = []
    i = 0
    while i < len(ordered):
        j = i + 1
        key = (ordered[i].points, ordered[i].gd, ordered[i].gf)
        while j < len(ordered) and (ordered[j].points, ordered[j].gd, ordered[j].gf) == key:
            j += 1
        cluster = ordered[i:j]
        if len(cluster) == 1:
            out.append(cluster[0])
        else:
            h2h = _head_to_head([s.team for s in cluster], matches)
            cluster.sort(key=lambda s: (h2h[s.team], -s.cards, s.team),
                         reverse=True)
            out.extend(cluster)
        i = j
    return out
