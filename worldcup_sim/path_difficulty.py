"""Bracket path difficulty (Phase 6).

Reaching a semifinal off the back of three beatable opponents is not the same
achievement as grinding through a group of death and two heavyweights, but a
plain win-probability table cannot tell them apart. This tracker records, for
every Monte Carlo run, the strength of each opponent a team faced on its
knockout path, then reports who tends to get the soft draws and who earns
their deep runs against real opposition.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

# Stages that count as a "late round" for the favourable-path question.
_LATE_STAGES = {"SF", "FINAL", "CHAMPION"}


class PathTracker:
    def __init__(self, strengths: Dict[str, float]):
        self.raw_strength = dict(strengths)
        self._field_mean = float(np.mean(list(strengths.values())))
        self.runs = 0
        # All knockout opponent strengths a team has faced, across every run.
        self._all_opp: Dict[str, List[float]] = defaultdict(list)
        # Per-run mean opponent strength, only for runs where the team went deep.
        self._deep_means: Dict[str, List[float]] = defaultdict(list)
        self._deep_count: Dict[str, int] = defaultdict(int)
        # Scratch space for the run in progress.
        self._cur: Dict[str, List[float]] = defaultdict(list)

    def start_run(self) -> None:
        self._cur = defaultdict(list)
        self.runs += 1

    def record_ko_match(self, team_a: str, team_b: str) -> None:
        """Log one knockout tie: each side faced the other's strength."""
        self._cur[team_a].append(self.raw_strength.get(team_b, self._field_mean))
        self._cur[team_b].append(self.raw_strength.get(team_a, self._field_mean))

    def end_run(self, reached: Dict[str, str]) -> None:
        for team, opps in self._cur.items():
            self._all_opp[team].extend(opps)
            if reached.get(team) in _LATE_STAGES and opps:
                self._deep_means[team].append(float(np.mean(opps)))
                self._deep_count[team] += 1

    def summary(self) -> List[Dict[str, float]]:
        """One row per team that played at least one knockout match."""
        rows: List[Dict[str, float]] = []
        for team, opps in self._all_opp.items():
            if not opps:
                continue
            deep = self._deep_means.get(team, [])
            deep_mean = float(np.mean(deep)) if deep else float("nan")
            # Positive luck = late runs came against below-average opposition.
            path_luck = (self._field_mean - deep_mean) if deep else float("nan")
            rows.append({
                "team": team,
                "raw_strength": round(self.raw_strength.get(team, float("nan")), 1),
                "average_path_strength": round(float(np.mean(opps)), 1),
                "path_variance": round(float(np.var(opps)), 1),
                "deep_run_rate": round(self._deep_count[team] / self.runs, 4),
                "late_path_strength": round(deep_mean, 1) if deep else float("nan"),
                "path_luck": round(path_luck, 1) if deep else float("nan"),
            })
        rows.sort(key=lambda r: r["deep_run_rate"], reverse=True)
        self._label(rows)
        return rows

    def _label(self, rows: List[Dict[str, float]]) -> None:
        """Tag deep teams as earning it on strength or riding a soft path."""
        if not rows:
            return
        strong = np.percentile([r["raw_strength"] for r in rows], 75)
        for r in rows:
            if r["deep_run_rate"] < 0.05 or r["path_luck"] != r["path_luck"]:
                r["path_note"] = ""
            elif r["raw_strength"] >= strong:
                r["path_note"] = "raw strength"
            elif r["path_luck"] > 0:
                r["path_note"] = "favourable path"
            else:
                r["path_note"] = "earned the hard way"
