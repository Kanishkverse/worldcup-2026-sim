"""Live 2026 forecast: condition the simulation on results as they arrive.

Run order:
  1. load the pre-tournament Elo priors,
  2. load the real played results,
  3. replay them through the Bayesian state engine (ratings, form, fatigue),
  4. estimate each team's qualification outlook for the tactical states,
  5. simulate the rest of the tournament many times and aggregate.

Point it at a different results file with --results, or set WC_RESULTS_CSV.
As more games are played you only edit the CSV; nothing else changes.

    python3 live_forecast.py --runs 4000
    python3 live_forecast.py --results data/results_2026.csv --out-prefix results/live
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List

from worldcup_sim.priors import elo_priors, group_draw
from worldcup_sim.standings import Match
from worldcup_sim.tournament_state import TournamentState
from worldcup_sim import engine

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(ROOT, "data", "results_2026.csv")


def load_results(path: str) -> List[dict]:
    rows: List[dict] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    # Replay in tournament order so fatigue and form accumulate correctly.
    rows.sort(key=lambda r: (int(r.get("matchday", 1)), r.get("group", "")))
    return rows


def build_state(priors, result_rows):
    """Replay real results into a fresh Bayesian state, by matchday."""
    state = TournamentState(priors)
    pinned_by_group: Dict[str, Dict[frozenset, Match]] = defaultdict(dict)
    last_md = None
    for r in result_rows:
        md = int(r.get("matchday", 1))
        if last_md is not None and md != last_md:
            state.recover_between_matchdays()
        home, away = r["home"].strip(), r["away"].strip()
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        state.update_after_match(home, away, hg, ag)
        grp = r.get("group", "").strip()
        pinned_by_group[grp][frozenset((home, away))] = Match(
            home=home, away=away, home_goals=hg, away_goals=ag)
        last_md = md
    return state, pinned_by_group


def _write_probs_csv(path: str, probs: Dict[str, Dict[str, float]]) -> None:
    cols = ["R32", "R16", "QF", "SF", "FINAL", "CHAMPION"]
    rows = sorted(probs.items(), key=lambda kv: kv[1]["CHAMPION"], reverse=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["team", "advance_R16", "quarterfinal", "semifinal",
                    "final", "champion"])
        for team, p in rows:
            w.writerow([team, f"{p['R16']:.4f}", f"{p['QF']:.4f}",
                        f"{p['SF']:.4f}", f"{p['FINAL']:.4f}",
                        f"{p['CHAMPION']:.4f}"])


def _write_path_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    cols = ["team", "raw_strength", "average_path_strength", "path_variance",
            "deep_run_rate", "late_path_strength", "path_luck", "path_note"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main() -> None:
    ap = argparse.ArgumentParser(description="Live 2026 World Cup forecast.")
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--results", default=os.getenv("WC_RESULTS_CSV", DEFAULT_RESULTS))
    ap.add_argument("--out-prefix", default=os.path.join(ROOT, "results", "live_forecast"))
    args = ap.parse_args()

    priors = elo_priors()
    result_rows = load_results(args.results)
    state, pinned_by_group = build_state(priors, result_rows)

    print(f"Loaded {len(result_rows)} real result(s) from "
          f"{os.path.relpath(args.results, ROOT)}.")
    print(f"Simulating the remaining tournament {args.runs} times.\n")

    out = engine.forecast(state, pinned_by_group, runs=args.runs, seed=args.seed)
    probs = out["probs"]

    ranked = sorted(probs.items(), key=lambda kv: kv[1]["CHAMPION"], reverse=True)
    print(f"{'Team':24} {'Champ':>7} {'Final':>7} {'SF':>7} {'R16':>7}")
    print("-" * 56)
    for team, p in ranked[:16]:
        print(f"{team:24} {p['CHAMPION']*100:6.1f}% {p['FINAL']*100:6.1f}% "
              f"{p['SF']*100:6.1f}% {p['R16']*100:6.1f}%")

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    _write_probs_csv(args.out_prefix + "_probs.csv", probs)
    _write_path_csv(args.out_prefix + "_path.csv", out["path"])
    with open(args.out_prefix + ".json", "w", encoding="utf-8") as fh:
        json.dump({"runs": out["runs"], "probs": probs,
                   "champions": out["champions"], "qual_prob": out["qual_prob"],
                   "state": out["state"]}, fh, indent=2)
    print(f"\nWrote {os.path.relpath(args.out_prefix, ROOT)}_probs.csv, "
          f"_path.csv, .json")


if __name__ == "__main__":
    main()
