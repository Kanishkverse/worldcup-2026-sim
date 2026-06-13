"""Pure prediction engine for the 2026 World Cup — who actually wins, in %.

No LLM anywhere. One job: the sharpest win probabilities we can get from
public signals.

    match model   Elo win expectancy -> calibrated Poisson goal means
                  (draws fall out naturally; knockouts get ET + penalties)
    bracket       the validated Tournament class from ../fifa (official 2026
                  groups, third-place matrix, R32 seeding) with all
                  already-played real results pinned
    validation    backtest on the 2010/2014/2018/2022 World Cups before
                  trusting a single 2026 number
    output        elo_2026_probs.csv  (+ backtest_report.csv)

Usage:
    python3 elo_predict.py --backtest          # validate the match model
    python3 elo_predict.py --runs 20000        # 2026 prediction
"""

import argparse
import os
import sys
import time
from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("WC_PLAYERS_CSV",
                      os.path.join(ROOT, "data", "players_2026_real.csv"))
os.environ.setdefault("WC_LIVE_FEATURES", "0")   # snapshot Elo, reproducible

import worldcup_2026_sim as wc

HOSTS = {"United States", "Mexico", "Canada"}
HOST_ELO_BONUS = 80.0      # standard home-advantage bump, all matches in NA
TOTAL_GOALS = 2.64         # real World Cup average 2010-2022
MAX_G = 12                 # Poisson truncation for the calibration solve


# ==============================================================================
# Elo -> Poisson calibration
# ==============================================================================

def _outcome_probs(lam_a: float, lam_b: float):
    """P(A win), P(draw), P(B win) for independent Poisson scores."""
    ga = np.exp(-lam_a) * np.power(lam_a, np.arange(MAX_G + 1)) \
        / [np.math.factorial(k) for k in range(MAX_G + 1)]
    gb = np.exp(-lam_b) * np.power(lam_b, np.arange(MAX_G + 1)) \
        / [np.math.factorial(k) for k in range(MAX_G + 1)]
    joint = np.outer(ga, gb)
    p_a = np.tril(joint, -1).sum()
    p_d = np.trace(joint)
    return p_a, p_d, 1.0 - p_a - p_d


def build_lambda_map(total: float = TOTAL_GOALS):
    """For each Elo diff, the goal-mean split whose Poisson outcome matches
    the Elo expectancy W_e = P(win) + 0.5 P(draw).  Returns an interpolator."""
    drs = np.arange(0.0, 1001.0, 20.0)
    diffs = []
    for dr in drs:
        we = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        lo, hi = 0.0, total - 0.02
        for _ in range(40):                       # bisection on the goal split
            mid = 0.5 * (lo + hi)
            pa, pd_, _ = _outcome_probs((total + mid) / 2, (total - mid) / 2)
            if pa + 0.5 * pd_ < we:
                lo = mid
            else:
                hi = mid
        diffs.append(0.5 * (lo + hi))
    return drs, np.array(diffs)


_DRS, _DIFFS = build_lambda_map()


def lambdas_for(dr: float):
    """Goal means (lam_for_stronger_side applied with sign of dr)."""
    sign = 1.0 if dr >= 0 else -1.0
    d = float(np.interp(abs(dr), _DRS, _DIFFS))
    return (TOTAL_GOALS + sign * d) / 2.0, (TOTAL_GOALS - sign * d) / 2.0


# ==============================================================================
# Match simulator (drop-in for Tournament.sim)
# ==============================================================================

class EloMatchSimulator:
    """Calibrated Elo->Poisson scorelines; ET + penalties in knockouts."""

    def __init__(self, rng: np.random.Generator, host_bonus: bool = True):
        self.rng = rng
        self.host_bonus = host_bonus

    def _elo(self, team) -> float:
        e = float(team.features.elo)
        if self.host_bonus and team.name in HOSTS:
            e += HOST_ELO_BONUS
        return e

    def simulate(self, team_a, team_b, knockout=False, tactical_agent=None,
                 verbose=False):
        dr = self._elo(team_a) - self._elo(team_b)
        lam_a, lam_b = lambdas_for(dr)
        hg = int(self.rng.poisson(lam_a))
        ag = int(self.rng.poisson(lam_b))
        res = wc.MatchResult(home=team_a.name, away=team_b.name,
                             home_goals=hg, away_goals=ag)
        if knockout and hg == ag:
            # extra time: 30 minutes at a slightly lower scoring rate
            res.home_goals += int(self.rng.poisson(lam_a * 30 / 90 * 0.85))
            res.away_goals += int(self.rng.poisson(lam_b * 30 / 90 * 0.85))
            if res.home_goals == res.away_goals:
                res.went_to_pens = True
                we = 1.0 / (1.0 + 10 ** (-dr / 400.0))
                p_a = float(np.clip(0.5 + 0.5 * (we - 0.5), 0.35, 0.65))
                if self.rng.random() < p_a:
                    res.home_pens, res.away_pens = 5, 4
                else:
                    res.home_pens, res.away_pens = 4, 5
        return res


# ==============================================================================
# Monte Carlo
# ==============================================================================

def run_2026(runs: int, seed: int, no_pin: bool = False) -> pd.DataFrame:
    inventory = wc.build_world_cup_2026(rng_seed=seed)
    reaches, champs, t0 = [], Counter(), time.time()
    for i in range(runs):
        rng = np.random.default_rng(seed + i)
        t = wc.Tournament(inventory, rng=rng)
        t.sim = EloMatchSimulator(rng)
        if no_pin:
            t.PIN_REAL_RESULTS = False
        reach = t.run()
        reaches.append(reach)
        champs[next(k for k, v in reach.items() if v == "CHAMPION")] += 1
        if (i + 1) % max(runs // 10, 1) == 0:
            print(f"  {i+1}/{runs}  ({time.time()-t0:.0f}s)  "
                  f"leader: {champs.most_common(1)[0]}")
    return pd.DataFrame(aggregate(reaches))


def aggregate(reaches):
    order = ["GROUP", "R32", "R16", "QF", "SF", "FINAL", "CHAMPION"]
    rank = {s: i for i, s in enumerate(order)}
    teams = sorted({t for r in reaches for t in r})
    rows = []
    for team in teams:
        counts = Counter(r.get(team, "GROUP") for r in reaches)
        def at_least(stage):
            return 100.0 * sum(n for s, n in counts.items()
                               if rank[s] >= rank[stage]) / len(reaches)
        rows.append({"Country": team,
                     "Reach R16 %": round(at_least("R16"), 2),
                     "Reach QF %": round(at_least("QF"), 2),
                     "Reach SF %": round(at_least("SF"), 2),
                     "Reach Final %": round(at_least("FINAL"), 2),
                     "Win Trophy %": round(at_least("CHAMPION"), 2)})
    rows.sort(key=lambda r: (-r["Win Trophy %"], -r["Reach Final %"], r["Country"]))
    return rows


# ==============================================================================
# Backtest — would this model have called 2010-2022 sensibly?
# ==============================================================================

ACTUAL = {2010: "Spain", 2014: "Germany", 2018: "France", 2022: "Argentina"}


def backtest(runs: int, seed: int) -> pd.DataFrame:
    rows = []
    for year, winner in ACTUAL.items():
        inventory = wc.build_historical_world_cup(year, rng_seed=seed)
        champs = Counter()
        finals = Counter()
        for i in range(runs):
            rng = np.random.default_rng(seed + i)
            t = wc.Tournament32(inventory, rng=rng)
            t.sim = EloMatchSimulator(rng, host_bonus=False)
            reach = t.run()
            for k, v in reach.items():
                if v == "CHAMPION":
                    champs[k] += 1
                if v in ("CHAMPION", "FINAL"):
                    finals[k] += 1
        ranked = [t for t, _ in champs.most_common()]
        rows.append({
            "year": year, "actual": winner,
            "p_actual_%": round(100 * champs.get(winner, 0) / runs, 1),
            "model_rank_of_actual": ranked.index(winner) + 1 if winner in ranked else None,
            "model_top3": ", ".join(f"{t} {100*c/runs:.0f}%"
                                    for t, c in champs.most_common(3)),
            "p_actual_final_%": round(100 * finals.get(winner, 0) / runs, 1),
        })
        print(rows[-1])
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--backtest-runs", type=int, default=2000)
    ap.add_argument("--no-pin", action="store_true",
                    help="ignore already-played real 2026 results")
    args = ap.parse_args()

    if args.backtest:
        df = backtest(args.backtest_runs, args.seed)
        df.to_csv("backtest_report.csv", index=False)
        print("\n", df.to_string(index=False))
        print("saved -> backtest_report.csv")
        return

    df = run_2026(args.runs, args.seed, no_pin=args.no_pin)
    df.to_csv("elo_2026_probs.csv", index=False)
    print(df.head(15).to_string(index=False))
    print(f"saved -> elo_2026_probs.csv ({args.runs} tournaments)")


if __name__ == "__main__":
    main()
