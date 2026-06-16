"""Group results vs predictions, turned into a knockout correction.

The idea the user asked for, end to end:

  1. Build every team from its players, so the rating is an aggregate of who is
     available, not a hand-set number.
  2. For each group game that has actually been played, run the per-game
     physics simulator to get what the model expected, and compare it to what
     happened. The gap is the team's residual: did it beat its own model or
     fall short of it?
  3. Turn those residuals into a per-team Elo correction.
  4. Carry the correction into the elimination rounds and see how the knockout
     and title odds move once the model has been told where it was wrong.

The group stage is, in effect, a calibration fold for the knockouts.

    python3 scripts/group_to_ko_correction.py --per-game-runs 300 --ko-runs 3000
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldcup_sim import engine, squads
from worldcup_sim.per_game import PerGameSimulator
from worldcup_sim.priors import group_draw
from worldcup_sim.standings import Match
from worldcup_sim.tournament_state import TournamentState

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "data", "results_2026.csv")
REPORT = os.path.join(ROOT, "reports", "group_to_ko_correction.md")

# Goals-above-expectation per game -> Elo correction, then a clamp. Few games
# of evidence, so the gain is modest and bounded.
K_RESIDUAL = 45.0
CORRECTION_CLAMP = 110.0


def load_played(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def learn_corrections(rows, sim, runs, log):
    """Per-game predicted GD vs actual for each played match -> Elo correction."""
    residuals: Dict[str, List[float]] = defaultdict(list)
    log.append("## Predicted vs actual, game by game\n")
    log.append("| Match | Predicted GD | Actual | Residual (home) |")
    log.append("|---|---|---|---|")
    for r in rows:
        h, a = r["home"].strip(), r["away"].strip()
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        res = sim.fixture(h, a, runs=runs)
        exp_gd = res.mean_goals_a - res.mean_goals_b
        actual_gd = hg - ag
        resid = actual_gd - exp_gd
        residuals[h].append(resid)
        residuals[a].append(-resid)
        log.append(f"| {h} {hg}-{ag} {a} | {exp_gd:+.2f} | {actual_gd:+d} | "
                   f"{resid:+.2f} |")
    log.append("")

    corrections: Dict[str, float] = {}
    for team, rs in residuals.items():
        mean_res = float(np.mean(rs))
        corr = float(np.clip(K_RESIDUAL * mean_res, -CORRECTION_CLAMP, CORRECTION_CLAMP))
        corrections[team] = corr
    return corrections, residuals


def pinned_by_group(rows):
    groups = group_draw()
    team_grp = {t: g for g, ts in groups.items() for t in ts}
    pinned = defaultdict(dict)
    for r in rows:
        h, a = r["home"].strip(), r["away"].strip()
        pinned[team_grp[h]][frozenset((h, a))] = Match(
            home=h, away=a, home_goals=int(r["home_goals"]),
            away_goals=int(r["away_goals"]))
    return pinned


def ko_forecast(priors: Dict[str, float], pinned, runs, seed):
    state = TournamentState(priors)
    return engine.forecast(state, pinned, runs=runs, seed=seed)["probs"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-game-runs", type=int, default=300)
    ap.add_argument("--ko-runs", type=int, default=3000)
    ap.add_argument("--results", default=RESULTS)
    args = ap.parse_args()

    rows = load_played(args.results)
    base_elo = squads.player_elo_priors()
    sim = PerGameSimulator(use_manager=True, seed=11)

    log: List[str] = ["# Group-to-knockout correction\n",
                      f"Player-aggregated ratings, {len(rows)} played group "
                      f"games, per-game physics at {args.per_game_runs} sims "
                      "each. Residuals become Elo corrections carried into the "
                      "elimination rounds.\n"]

    corrections, residuals = learn_corrections(rows, sim, args.per_game_runs, log)
    corrected = {t: base_elo[t] + corrections.get(t, 0.0) for t in base_elo}

    log.append("## Learned corrections (teams with games played)\n")
    log.append("| Team | Games | Mean residual | Correction (Elo) | "
               "Player Elo | Corrected |")
    log.append("|---|---|---|---|---|---|")
    for team in sorted(residuals, key=lambda t: corrections[t], reverse=True):
        rs = residuals[team]
        log.append(f"| {team} | {len(rs)} | {np.mean(rs):+.2f} | "
                   f"{corrections[team]:+.0f} | {base_elo[team]:.0f} | "
                   f"{corrected[team]:.0f} |")
    log.append("")

    pinned = pinned_by_group(rows)
    before = ko_forecast(base_elo, pinned, args.ko_runs, seed=2026)
    after = ko_forecast(corrected, pinned, args.ko_runs, seed=2026)

    log.append("## Effect on the elimination rounds\n")
    log.append("Title and deep-round odds before and after the correction, for "
               "the teams that moved most.\n")
    log.append("| Team | Champ before | Champ after | SF before | SF after |")
    log.append("|---|---|---|---|---|")
    moved = sorted(before, key=lambda t: abs(after[t]["CHAMPION"] - before[t]["CHAMPION"]),
                   reverse=True)[:12]
    for t in moved:
        log.append(f"| {t} | {before[t]['CHAMPION']*100:.1f}% | "
                   f"{after[t]['CHAMPION']*100:.1f}% | {before[t]['SF']*100:.1f}% | "
                   f"{after[t]['SF']*100:.1f}% |")
    log.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log))
    print("\n".join(log))
    print(f"\nWrote {os.path.relpath(REPORT, ROOT)}")


if __name__ == "__main__":
    main()
