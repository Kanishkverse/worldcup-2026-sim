"""Probabilistic scoring of the match model (metrics for Phase 8).

A forecast that says everyone has a chance is useless; one that is cocksure
and wrong is worse. The honest test is whether the probabilities are both
sharp and calibrated. This script backtests the new match model on the
32-team World Cups of 2010-2022, where we know who actually won and who
reached the semifinals, and reports:

  * Brier score    - squared error of the whole champion probability vector
  * log loss       - how surprised the model was by the real champion
  * a calibration curve for "reach the semifinal" probabilities

It writes reports/model_diagnostics.md.

    python3 scripts/model_diagnostics.py --runs 3000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import worldcup_2026_sim as base
from worldcup_sim import engine, match_context as mc
from worldcup_sim.standings import Match, rank_group

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "reports", "model_diagnostics.md")

# 32-team bracket, reused from the base engine's historical resolver.
T32 = base.Tournament32


def _fields(year: int):
    groups: Dict[str, List[str]] = {}
    elo: Dict[str, float] = {}
    for g, members in base._HISTORICAL_FIELDS[year].items():
        groups[g] = [t for (t, _e, _m, _r) in members]
        for (t, e, _m, _r) in members:
            elo[t] = e
    return groups, elo


def _sim_group(teams, elo, rng) -> List:
    matches = []
    n = len(teams)
    for i in range(n):
        for j in range(i + 1, n):
            ga, gb, _ = engine.simulate_match(
                teams[i], teams[j], elo[teams[i]], elo[teams[j]],
                mc.knockout_modifier(), mc.knockout_modifier(), rng,
                venue=None, apply_host_edge=False)
            matches.append(Match(home=teams[i], away=teams[j],
                                 home_goals=ga, away_goals=gb))
    return rank_group(matches, teams)


def _ko(name_a, name_b, elo, rng) -> str:
    _, _, w = engine.simulate_match(
        name_a, name_b, elo[name_a], elo[name_b],
        mc.knockout_modifier(), mc.knockout_modifier(), rng,
        venue=None, knockout=True, apply_host_edge=False)
    return w if w is not None else name_a


def simulate_year(year: int, runs: int, seed: int) -> Tuple[Counter, Counter, List[str]]:
    groups, elo = _fields(year)
    rng = np.random.default_rng(seed)
    champ, sf = Counter(), Counter()
    teams_all = list(elo.keys())
    for _ in range(runs):
        standings = {g: _sim_group(t, elo, rng) for g, t in groups.items()}
        slot = {}
        for g, ranked in standings.items():
            slot[f"1{g}"] = ranked[0].team
            slot[f"2{g}"] = ranked[1].team
        win = {}
        for mid, sa, sb in T32.R16_SLOTS_32:
            win[mid] = _ko(slot[sa], slot[sb], elo, rng)
        for mid, fa, fb in T32.QF_SLOTS_32:
            win[mid] = _ko(win[fa], win[fb], elo, rng)
        for mid, fa, fb in T32.SF_SLOTS_32:
            # both QF winners feeding a semifinal reached the SF
            sf[win[fa]] += 1
            sf[win[fb]] += 1
            win[mid] = _ko(win[fa], win[fb], elo, rng)
        _, fa, fb = T32.FINAL_SLOT_32
        champ[_ko(win[fa], win[fb], elo, rng)] += 1
    return champ, sf, teams_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2500)
    args = ap.parse_args()

    years = [2010, 2014, 2018, 2022]
    brier_list, logloss_list = [], []
    calib_points: List[Tuple[float, int]] = []
    champ_rank_lines = []

    for yi, year in enumerate(years):
        truth = base._HISTORICAL_RESULTS[year]
        actual_champ = truth["champion"]
        actual_sf = set(truth["semifinalists"])
        champ, sf, teams = simulate_year(year, args.runs, seed=100 + yi)

        p_champ = {t: champ[t] / args.runs for t in teams}
        # Multiclass Brier over the champion vector.
        brier = sum((p_champ[t] - (1.0 if t == actual_champ else 0.0)) ** 2
                    for t in teams)
        ll = -math.log(max(p_champ[actual_champ], 1e-6))
        brier_list.append(brier)
        logloss_list.append(ll)

        ranked = sorted(p_champ.items(), key=lambda kv: kv[1], reverse=True)
        rank = [t for t, _ in ranked].index(actual_champ) + 1
        champ_rank_lines.append(
            f"| {year} | {actual_champ} | {rank} | "
            f"{p_champ[actual_champ]*100:.1f}% | {brier:.4f} | {ll:.3f} |")

        for t in teams:
            calib_points.append((sf[t] / args.runs, 1 if t in actual_sf else 0))

    # Calibration curve for "reach the semifinal".
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 1.01)]
    calib_rows = []
    for lo, hi in bins:
        pts = [(p, y) for (p, y) in calib_points if lo <= p < hi]
        if not pts:
            continue
        mean_pred = np.mean([p for p, _ in pts])
        emp = np.mean([y for _, y in pts])
        calib_rows.append((lo, hi, len(pts), mean_pred, emp))

    print(f"Mean Brier  : {np.mean(brier_list):.4f}")
    print(f"Mean logloss: {np.mean(logloss_list):.3f}")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    L = []
    L.append("# Model diagnostics\n")
    L.append(f"Backtest of the new match model on the 32-team World Cups, "
             f"{args.runs} simulations per tournament. No in-tournament "
             "updates here: this isolates the match engine and bracket, scored "
             "against the real champion and semifinalists.\n")
    L.append("## Champion scoring\n")
    L.append("| Year | Champion | Model rank | Champion prob | Brier | Log loss |")
    L.append("|---|---|---|---|---|---|")
    L.extend(champ_rank_lines)
    L.append("")
    L.append(f"- Mean Brier score: **{np.mean(brier_list):.4f}** "
             "(lower is better; a uniform 1-of-32 guess scores about 0.97).")
    L.append(f"- Mean log loss: **{np.mean(logloss_list):.3f}** "
             "(uniform guess scores about 3.47).\n")
    L.append("## Calibration: reach the semifinal\n")
    L.append("All teams across all four tournaments, bucketed by predicted "
             "probability of reaching the semifinal, against how often it "
             "actually happened.\n")
    L.append("| Predicted bin | Teams | Mean predicted | Actual rate |")
    L.append("|---|---|---|---|")
    for lo, hi, n, mp, emp in calib_rows:
        L.append(f"| {lo:.2f}-{hi:.2f} | {n} | {mp*100:.1f}% | {emp*100:.1f}% |")
    L.append("")
    L.append("A well-calibrated model tracks the diagonal: the predicted and "
             "actual columns should be close in every bin. The known soft spot "
             "is form blindness. The model rates on pre-tournament Elo, so a "
             "side that catches fire (2018 France, 2022 Argentina from their "
             "respective ranks) is scored as a mid-pack contender. That is "
             "exactly the gap the live engine's Bayesian updates and form "
             "layer are built to close once real results arrive.\n")
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"Wrote {os.path.relpath(REPORT, ROOT)}")


if __name__ == "__main__":
    main()
