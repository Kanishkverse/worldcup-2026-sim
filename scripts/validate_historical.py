"""Validate the live engine's machinery against real tournaments.

Unit tests show each piece works in isolation. This goes further: it drives
the four mechanisms that actually matter for a live forecast with real
historical scorelines and checks them against what those tournaments did.

  1. Results ingestion - does the CSV -> standings -> qualifiers path
     reproduce the real group outcomes, tiebreakers and all?
  2. Calibration       - do the simulated frequencies match the 2010-2022
     champions and semifinalists (Brier, log loss, beat-uniform)?
  3. Bayesian updates  - after feeding a group's real results in, is the
     posterior ordering closer to the real final table than the prior Elo was?
  4. Form dynamics     - does form track over/under-performance, stay bounded,
     and regress instead of running away?

The fixtures in data/historical_results.csv are deliberately the famous ones:
Argentina's loss to Saudi Arabia, Germany's two group-stage exits, Croatia's
2018 run. They are exactly the cases a static model gets wrong.

    python3 scripts/validate_historical.py
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import worldcup_2026_sim as base
from worldcup_sim import engine, form as form_mod
from worldcup_sim.standings import Match, rank_group
from worldcup_sim.tournament_state import TournamentState
from scripts.model_diagnostics import simulate_year  # reuse the backtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "data", "historical_results.csv")
REPORT = os.path.join(ROOT, "reports", "historical_validation.md")

# Real final group order (after official tiebreakers) for each fixture.
ACTUAL_ORDER = {
    (2022, "C"): ["Argentina", "Poland", "Mexico", "Saudi Arabia"],
    (2018, "F"): ["Sweden", "Mexico", "South Korea", "Germany"],
    (2018, "D"): ["Croatia", "Argentina", "Nigeria", "Iceland"],
    (2022, "E"): ["Japan", "Spain", "Germany", "Costa Rica"],
}


def load_fixtures() -> Dict[tuple, List[dict]]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    with open(FIXTURES, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            groups[(int(r["year"]), r["group"])].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: int(r["matchday"]))
    return groups


def _priors_for(year: int, teams: List[str]) -> Dict[str, float]:
    feats = base.features_historical(year)
    return {t: feats[t].elo for t in teams}


def _matches(rows: List[dict]) -> List[Match]:
    return [Match(home=r["home"], away=r["away"],
                  home_goals=int(r["home_goals"]), away_goals=int(r["away_goals"]))
            for r in rows]


def _line(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# 1. Results ingestion
# ---------------------------------------------------------------------------

def validate_ingestion(fixtures, log) -> bool:
    log.append("## 1. Results ingestion\n")
    log.append("Ingest the real scorelines, rebuild the table, and compare the "
               "computed order to the real final standings. This exercises the "
               "tiebreakers too: three of these four groups were settled on "
               "goal difference.\n")
    log.append("| Tournament | Group | Computed order | Matches reality |")
    log.append("|---|---|---|---|")
    all_ok = True
    for (year, grp), rows in sorted(fixtures.items()):
        teams = ACTUAL_ORDER[(year, grp)]
        ranked = rank_group(_matches(rows), list(teams))
        computed = [s.team for s in ranked]
        ok = computed == ACTUAL_ORDER[(year, grp)]
        all_ok &= ok
        log.append(f"| {year} | {grp} | {', '.join(computed)} | {_line(ok)} |")
    log.append("")
    return all_ok


# ---------------------------------------------------------------------------
# 2. Calibration
# ---------------------------------------------------------------------------

def validate_calibration(log, runs: int) -> bool:
    log.append("## 2. Calibration\n")
    log.append("Backtest the match model on the four 32-team World Cups and "
               "score the champion probabilities. Uniform 1-of-32 guessing "
               "scores Brier 0.97 / log loss 3.47; the model must beat both.\n")
    log.append("| Year | Champion | Model rank | Champion prob | Brier | Log loss |")
    log.append("|---|---|---|---|---|---|")
    import math
    briers, lls, ranks = [], [], []
    for yi, year in enumerate([2010, 2014, 2018, 2022]):
        truth = base._HISTORICAL_RESULTS[year]
        champ, _sf, teams = simulate_year(year, runs, seed=200 + yi)
        p = {t: champ[t] / runs for t in teams}
        ac = truth["champion"]
        brier = sum((p[t] - (1.0 if t == ac else 0.0)) ** 2 for t in teams)
        ll = -math.log(max(p[ac], 1e-6))
        rank = sorted(p, key=p.get, reverse=True).index(ac) + 1
        briers.append(brier); lls.append(ll); ranks.append(rank)
        log.append(f"| {year} | {ac} | {rank} | {p[ac]*100:.1f}% | "
                   f"{brier:.3f} | {ll:.3f} |")
    mb, ml, mr = np.mean(briers), np.mean(lls), np.mean(ranks)
    ok = mb < 0.97 and ml < 3.47
    log.append("")
    log.append(f"- Mean Brier **{mb:.3f}** (uniform 0.97), "
               f"mean log loss **{ml:.3f}** (uniform 3.47), "
               f"champion mean rank **{mr:.2f}** of 32. {_line(ok)}\n")
    return ok


# ---------------------------------------------------------------------------
# 3. Bayesian updates
# ---------------------------------------------------------------------------

def _order_rank(teams_by_strength: List[str], actual: List[str]) -> float:
    """Kendall tau between a predicted ordering and the real final order."""
    pos_pred = {t: i for i, t in enumerate(teams_by_strength)}
    pos_true = {t: i for i, t in enumerate(actual)}
    teams = actual
    tau, _ = kendalltau([pos_pred[t] for t in teams],
                        [pos_true[t] for t in teams])
    return tau


def validate_bayes(fixtures, log) -> bool:
    log.append("## 3. Bayesian updates\n")
    log.append("Feed each group's real results through the state engine and ask "
               "whether the posterior ordering ends up closer to the real final "
               "table than the pre-tournament Elo was. Closeness is Kendall tau "
               "against the actual order (1.0 = identical). The posterior should "
               "also be more certain than the prior (smaller rating std).\n")
    log.append("| Tournament | Group | Prior tau | Posterior tau | Std shrinks |")
    log.append("|---|---|---|---|---|")
    prior_taus, post_taus = [], []
    std_ok = True
    for (year, grp), rows in sorted(fixtures.items()):
        actual = ACTUAL_ORDER[(year, grp)]
        priors = _priors_for(year, actual)
        st = TournamentState(priors)
        prior_std = np.mean([st.get(t).rating_std for t in actual])

        prior_order = sorted(actual, key=lambda t: priors[t], reverse=True)
        for r in rows:
            st.update_after_match(r["home"], r["away"],
                                  int(r["home_goals"]), int(r["away_goals"]))
        post_order = sorted(actual, key=lambda t: st.get(t).rating_mean,
                            reverse=True)
        post_std = np.mean([st.get(t).rating_std for t in actual])

        pt = _order_rank(prior_order, actual)
        qt = _order_rank(post_order, actual)
        prior_taus.append(pt); post_taus.append(qt)
        shrank = post_std < prior_std
        std_ok &= shrank
        log.append(f"| {year} | {grp} | {pt:+.2f} | {qt:+.2f} | {_line(shrank)} |")

    mp, mq = np.mean(prior_taus), np.mean(post_taus)
    ok = mq > mp and std_ok
    log.append("")
    log.append(f"- Mean ordering agreement with reality: prior **{mp:+.2f}** -> "
               f"posterior **{mq:+.2f}**. Conditioning on results moves the "
               f"ranking toward the truth. {_line(ok)}\n")
    return ok


# ---------------------------------------------------------------------------
# 4. Form dynamics
# ---------------------------------------------------------------------------

def _form_trajectory(year, grp, rows):
    """Return {team: [form after each of its matches]} for one group."""
    actual = ACTUAL_ORDER[(year, grp)]
    st = TournamentState(_priors_for(year, actual))
    traj: Dict[str, List[float]] = defaultdict(list)
    pairs = []  # (performance vs expectation, resulting form change)
    peak = 0.0
    for r in rows:
        h, a = r["home"], r["away"]
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        exp_gd = (st.get(h).rating_mean - st.get(a).rating_mean) / 250.0
        fb_h, fb_a = st.get(h).form, st.get(a).form
        st.update_after_match(h, a, hg, ag)
        pairs.append(((hg - ag) - exp_gd, st.get(h).form - fb_h))
        pairs.append(((ag - hg) + exp_gd, st.get(a).form - fb_a))
        traj[h].append(st.get(h).form)
        traj[a].append(st.get(a).form)
        peak = max(peak, abs(st.get(h).form), abs(st.get(a).form))
    return traj, pairs, peak


def validate_form(fixtures, log) -> bool:
    log.append("## 4. Form dynamics\n")
    log.append("Track latent form through each real sequence. It must stay "
               "bounded, move with over/under-performance, and the famous "
               "trajectories must look right: a shock spikes and then regresses, "
               "a collapse trends down, a recovery turns back up.\n")

    max_abs = 0.0
    all_pairs = []
    traj: Dict[tuple, Dict[str, List[float]]] = {}
    for key, rows in sorted(fixtures.items()):
        t, p, pk = _form_trajectory(key[0], key[1], rows)
        traj[key] = t
        all_pairs.extend(p)
        max_abs = max(max_abs, pk)

    bounded = max_abs <= form_mod.FORM_CLIP + 1e-9
    xs = np.array([p[0] for p in all_pairs])
    ys = np.array([p[1] for p in all_pairs])
    corr = float(np.corrcoef(xs, ys)[0, 1])
    directional = corr > 0.5

    # Saudi Arabia 2022: beating Argentina spikes form, then two losses regress
    # it. No runaway, clear mean reversion.
    saudi = traj[(2022, "C")]["Saudi Arabia"]
    saudi_spike = saudi[0] > 0.5
    saudi_regress = abs(saudi[-1]) < saudi[0]

    # Argentina 2022: the Saudi loss dips form, two wins pull it back positive.
    arg = traj[(2022, "C")]["Argentina"]
    arg_dip = arg[0] < 0
    arg_recover = arg[-1] > 0

    # Germany 2018: lost the opener, lost the decider, crashed out - trends down.
    ger = traj[(2018, "F")]["Germany"]
    ger_down = ger[-1] < 0

    # Japan 2022: beat Germany and Spain - ends clearly positive.
    jpn = traj[(2022, "E")]["Japan"]
    jpn_up = jpn[-1] > 0.3

    log.append(f"- Boundedness: max |form| observed **{max_abs:.2f}** "
               f"<= clip {form_mod.FORM_CLIP}. {_line(bounded)}")
    log.append(f"- Directionality: corr(beating expectation, form change) "
               f"**{corr:+.2f}** (> 0.5). {_line(directional)}")
    log.append(f"- Saudi Arabia 2022 spikes after beating Argentina "
               f"({saudi[0]:+.2f}) then regresses ({saudi[-1]:+.2f}). "
               f"{_line(saudi_spike and saudi_regress)}")
    log.append(f"- Argentina 2022 dips after the Saudi loss ({arg[0]:+.2f}) "
               f"then recovers ({arg[-1]:+.2f}). {_line(arg_dip and arg_recover)}")
    log.append(f"- Germany 2018 trends down to elimination ({ger[-1]:+.2f}). "
               f"{_line(ger_down)}")
    log.append(f"- Japan 2022 ends on a high ({jpn[-1]:+.2f}). {_line(jpn_up)}")
    log.append("")
    return (bounded and directional and saudi_spike and saudi_regress
            and arg_dip and arg_recover and ger_down and jpn_up)


def _expected_gd(elo_h: float, elo_a: float) -> float:
    lam_h, lam_a = engine.lambdas_for((elo_h - elo_a) * engine.GAP_SCALE)
    return lam_h - lam_a


def validate_correction(fixtures, log) -> bool:
    """The group-to-knockout correction mechanism, checked against reality.

    For each famous group, predict every game's expected goal difference, take
    the residual against the real scoreline, and turn it into an Elo
    correction. Two checks: the corrected ratings rank the real final table
    better than the raw Elo did, and the headline over/under-performers get the
    right sign (Saudi Arabia 2022 up, Germany 2018 down)."""
    log.append("## 5. Group-to-knockout correction\n")
    log.append("The same residual-to-correction step the 2026 pipeline runs, "
               "here on real groups. Expected goal difference comes from the "
               "rating gap; the residual against the real result becomes an Elo "
               "correction. The corrected order should sit closer to the real "
               "final table than the prior, and the famous shocks should be "
               "signed correctly.\n")
    log.append("| Tournament | Group | Prior tau | Corrected tau |")
    log.append("|---|---|---|---|")
    K = 45.0
    prior_taus, corr_taus = [], []
    corrections: Dict[tuple, Dict[str, float]] = {}
    for (year, grp), rows in sorted(fixtures.items()):
        actual = ACTUAL_ORDER[(year, grp)]
        elo = _priors_for(year, actual)
        resid: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            h, a = r["home"], r["away"]
            gd = int(r["home_goals"]) - int(r["away_goals"])
            e = _expected_gd(elo[h], elo[a])
            resid[h].append(gd - e)
            resid[a].append(-(gd - e))
        corr = {t: float(np.clip(K * np.mean(resid[t]), -110, 110)) for t in actual}
        corrections[(year, grp)] = corr
        corrected_elo = {t: elo[t] + corr[t] for t in actual}

        prior_order = sorted(actual, key=lambda t: elo[t], reverse=True)
        corr_order = sorted(actual, key=lambda t: corrected_elo[t], reverse=True)
        pt = _order_rank(prior_order, actual)
        ct = _order_rank(corr_order, actual)
        prior_taus.append(pt); corr_taus.append(ct)
        log.append(f"| {year} | {grp} | {pt:+.2f} | {ct:+.2f} |")

    mp, mc = np.mean(prior_taus), np.mean(corr_taus)
    saudi = corrections[(2022, "C")]["Saudi Arabia"] > 0
    germany = corrections[(2018, "F")]["Germany"] < 0
    ok = mc > mp and saudi and germany
    log.append("")
    log.append(f"- Ordering vs reality: prior **{mp:+.2f}** -> corrected "
               f"**{mc:+.2f}**. {_line(mc > mp)}")
    log.append(f"- Saudi Arabia 2022 corrected up "
               f"({corrections[(2022,'C')]['Saudi Arabia']:+.0f} Elo), "
               f"Germany 2018 corrected down "
               f"({corrections[(2018,'F')]['Germany']:+.0f} Elo). "
               f"{_line(saudi and germany)}\n")
    return ok


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2000)
    args = ap.parse_args()

    fixtures = load_fixtures()
    log: List[str] = ["# Historical validation\n",
                      "Real scorelines from four famous group stages, plus the "
                      "2010-2022 champion backtest. Each section ends in a "
                      "pass/fail gate; the same gates are asserted in "
                      "tests/test_historical_validation.py.\n"]

    r1 = validate_ingestion(fixtures, log)
    r2 = validate_calibration(log, args.runs)
    r3 = validate_bayes(fixtures, log)
    r4 = validate_form(fixtures, log)
    r5 = validate_correction(fixtures, log)

    overall = all([r1, r2, r3, r4, r5])
    log.insert(2, f"**Overall: {_line(overall)}** "
                  f"(ingestion {_line(r1)}, calibration {_line(r2)}, "
                  f"bayes {_line(r3)}, form {_line(r4)}, "
                  f"correction {_line(r5)})\n")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log))

    print("\n".join(log))
    print(f"\nWrote {os.path.relpath(REPORT, ROOT)}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
