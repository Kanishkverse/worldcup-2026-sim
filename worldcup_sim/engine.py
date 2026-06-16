"""Dynamic Monte Carlo engine (Phase 8 core).

This is where the layers meet. Each simulated match starts from two sampled
posterior strengths, then the expected goals are bent by style matchup, by the
venue (altitude, heat, travel), and by what each team needs from the game on
that matchday. The tournament walk reuses the base engine's official 2026
bracket slot tables so the structure is identical to the static simulator;
only the per-match model is richer.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import match_context as mc
from . import styles as styles_mod
from . import venues as venues_mod
from .priors import bracket, group_draw, host_nations, team_to_group
from .path_difficulty import PathTracker
from .standings import Match, Standing, rank_group
from .tournament_state import TournamentState

TOTAL_GOALS = 2.64
HOST_EDGE_GROUP = 55.0    # Elo points of home comfort for a co-host in the group
HOST_EDGE_KO = 25.0       # smaller in the knockouts, fewer guaranteed home venues

# Rating gaps are scaled by this before they hit the goal model. Below 1.0 it
# compresses favourites toward the field and lifts the upset rate.
# scripts/calibrate_upsets.py shows the bare match model is favourite-heavy and
# would want about 0.5 to match history on its own. The full engine, though,
# adds three more sources of upset variance (posterior rating sampling, style
# matchups, venue load), so 0.5 would double-count and flatten the forecast.
# 0.85 is a deliberate middle: it takes the edge off the favourite bias and
# lets the stochastic layers supply the rest.
GAP_SCALE = 0.85

# Standard four-team group schedule by draw index, with matchday tags. The real
# matchday-1 results line up with the (0,1)/(2,3) pairings of the draw order.
_SCHEDULE = [
    (1, 0, 1), (1, 2, 3),
    (2, 0, 2), (2, 1, 3),
    (3, 0, 3), (3, 1, 2),
]

_NEUTRAL_VENUES = [
    "MetLife", "SoFi", "ATT", "NRG", "Mercedes-Benz", "Hard Rock",
    "Lincoln", "Levi's", "Arrowhead", "Lumen", "Gillette",
]


# ---------------------------------------------------------------------------
# Elo -> Poisson calibration: find the goal-mean split that makes the Poisson
# win/draw/loss probabilities reproduce the Elo win expectancy, holding the
# total at 2.64 goals (the 2010-2022 World Cup average).
# ---------------------------------------------------------------------------

def _poisson_we(lam_s: float, lam_w: float, kmax: int = 12) -> float:
    ks = np.arange(0, kmax + 1)
    ps = np.exp(-lam_s) * lam_s ** ks / np.array([math.factorial(k) for k in ks])
    pw = np.exp(-lam_w) * lam_w ** ks / np.array([math.factorial(k) for k in ks])
    win = draw = 0.0
    for i in ks:
        for j in ks:
            p = ps[i] * pw[j]
            if i > j:
                win += p
            elif i == j:
                draw += p
    return win + 0.5 * draw


def _delta_for_we(we_target: float) -> float:
    lo, hi = 0.0, TOTAL_GOALS
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        we = _poisson_we((TOTAL_GOALS + mid) / 2, (TOTAL_GOALS - mid) / 2)
        if we < we_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _build_lambda_map():
    drs = np.arange(0, 1001, 20.0)
    diffs = np.array([_delta_for_we(1.0 / (1.0 + 10 ** (-dr / 400.0))) for dr in drs])
    return drs, diffs


_DRS, _DIFFS = _build_lambda_map()


def lambdas_for(dr: float) -> Tuple[float, float]:
    """Goal means (lam_a, lam_b) for a rating gap dr = rating_a - rating_b."""
    sign = 1.0 if dr >= 0 else -1.0
    d = float(np.interp(abs(dr), _DRS, _DIFFS))
    return (TOTAL_GOALS + sign * d) / 2.0, (TOTAL_GOALS - sign * d) / 2.0


def _we_for(dr: float) -> float:
    return 1.0 / (1.0 + 10 ** (-dr / 400.0))


# ---------------------------------------------------------------------------
# Match resolution
# ---------------------------------------------------------------------------

class MatchEnv:
    """Per-iteration scratch state: where each team last played (for travel)."""

    def __init__(self):
        self.last_venue: Dict[str, str] = {}
        self.load: Dict[str, float] = defaultdict(float)

    def travel(self, team: str, venue: str) -> float:
        prev = self.last_venue.get(team)
        km = venues_mod.distance_km(prev, venue) if prev else 0.0
        self.last_venue[team] = venue
        return km


def _host_edge(team: str, knockout: bool) -> float:
    if team in host_nations():
        return HOST_EDGE_KO if knockout else HOST_EDGE_GROUP
    return 0.0


def simulate_match(
    name_a: str, name_b: str,
    strength_a: float, strength_b: float,
    ctx_a: mc.ContextModifier, ctx_b: mc.ContextModifier,
    rng: np.random.Generator,
    *,
    venue: Optional[str] = None,
    env: Optional[MatchEnv] = None,
    knockout: bool = False,
    apply_host_edge: bool = True,
) -> Tuple[int, int, Optional[str]]:
    """Resolve one match, returning (goals_a, goals_b, winner_or_None)."""
    he_a = _host_edge(name_a, knockout) if apply_host_edge else 0.0
    he_b = _host_edge(name_b, knockout) if apply_host_edge else 0.0
    eff_a = strength_a + he_a
    eff_b = strength_b + he_b

    lam_a, lam_b = lambdas_for((eff_a - eff_b) * GAP_SCALE)

    # Style matchup (+/-15%).
    sa, sb = styles_mod.matchup_modifier(name_a, name_b)
    lam_a *= sa
    lam_b *= sb

    # Tactical intent: my attack and the opponent's defensive scaling.
    lam_a *= ctx_a.attack_mult * ctx_b.defense_mult
    lam_b *= ctx_b.attack_mult * ctx_a.defense_mult

    # Venue: altitude / heat / travel, accumulated into per-iteration load.
    if venue is not None:
        km_a = env.travel(name_a, venue) if env else 0.0
        km_b = env.travel(name_b, venue) if env else 0.0
        ve_a = venues_mod.venue_modifier(venue, name_a, km_a)
        ve_b = venues_mod.venue_modifier(venue, name_b, km_b)
        if env:
            env.load[name_a] = min(1.0, env.load[name_a] + ve_a.fatigue_add)
            env.load[name_b] = min(1.0, env.load[name_b] + ve_b.fatigue_add)
            lam_a *= 1.0 - 0.10 * env.load[name_a]
            lam_b *= 1.0 - 0.10 * env.load[name_b]
        lam_a *= ve_a.xg_mult
        lam_b *= ve_b.xg_mult

    lam_a = max(0.05, lam_a)
    lam_b = max(0.05, lam_b)
    ga, gb = int(rng.poisson(lam_a)), int(rng.poisson(lam_b))

    if not knockout:
        return ga, gb, (name_a if ga > gb else name_b if gb > ga else None)

    if ga == gb:
        # Extra time at a reduced rate, then penalties tilted slightly by Elo.
        et_a = int(rng.poisson(lam_a * (30.0 / 90.0) * 0.85))
        et_b = int(rng.poisson(lam_b * (30.0 / 90.0) * 0.85))
        ga, gb = ga + et_a, gb + et_b
        if ga == gb:
            p_a = float(np.clip(0.5 + 0.5 * (_we_for(eff_a - eff_b) - 0.5), 0.35, 0.65))
            return ga, gb, (name_a if rng.random() < p_a else name_b)
    return ga, gb, (name_a if ga > gb else name_b)


# ---------------------------------------------------------------------------
# Group stage
# ---------------------------------------------------------------------------

def _pick_venue(name_a: str, name_b: str, salt: int) -> str:
    if "Mexico" in (name_a, name_b):
        return "Azteca"
    if "Canada" in (name_a, name_b):
        return "BMO"
    if "United States" in (name_a, name_b):
        return _NEUTRAL_VENUES[salt % len(_NEUTRAL_VENUES)]
    return _NEUTRAL_VENUES[salt % len(_NEUTRAL_VENUES)]


def _sample_cards(rng: np.random.Generator) -> Tuple[int, int]:
    return int(rng.poisson(1.8)), int(rng.poisson(1.8))


def simulate_group(
    teams: List[str],
    pinned: Dict[frozenset, Match],
    strengths: Dict[str, float],
    qual_prob: Dict[str, float],
    rng: np.random.Generator,
    env: MatchEnv,
    *,
    use_context: bool = True,
) -> List[Match]:
    """Play the six group games (pinned where real), return all match rows."""
    matches: List[Match] = []
    # Live tally so matchday-2/3 context can read current points and GD.
    tally = {t: Standing(team=t) for t in teams}

    for salt, (md, i, j) in enumerate(_SCHEDULE):
        ta, tb = teams[i], teams[j]
        key = frozenset((ta, tb))
        if key in pinned:
            m = pinned[key]
        else:
            if use_context:
                ca = mc.context_modifier(tally[ta].points, tally[ta].gd,
                                         qual_prob.get(ta, 0.5), md)
                cb = mc.context_modifier(tally[tb].points, tally[tb].gd,
                                         qual_prob.get(tb, 0.5), md)
            else:
                ca = cb = mc.knockout_modifier()
            venue = _pick_venue(ta, tb, salt)
            ga, gb, _ = simulate_match(ta, tb, strengths[ta], strengths[tb],
                                       ca, cb, rng, venue=venue, env=env)
            ka, kb = _sample_cards(rng)
            m = Match(home=ta, away=tb, home_goals=ga, away_goals=gb,
                      cards_home=ka, cards_away=kb)
        matches.append(m)
        tally[m.home].apply(m.home_goals, m.away_goals)
        tally[m.away].apply(m.away_goals, m.home_goals)
    return matches


# ---------------------------------------------------------------------------
# One full tournament
# ---------------------------------------------------------------------------

_STAGE_ORDER = ["GROUP", "R32", "R16", "QF", "SF", "FINAL", "CHAMPION"]


def run_tournament(
    state: TournamentState,
    pinned_by_group: Dict[str, Dict[frozenset, Match]],
    qual_prob: Dict[str, float],
    rng: np.random.Generator,
    tracker: Optional[PathTracker] = None,
) -> Dict[str, str]:
    """Simulate one tournament from the current state; return team -> stage."""
    T = bracket()
    groups = group_draw()
    env = MatchEnv()

    # Fix each team's strength for this iteration (samples the posterior).
    strengths = {t: state.get(t).sample_strength(rng) for t in state.teams}

    reached: Dict[str, str] = {}
    standings: Dict[str, List[Standing]] = {}
    for g, teams in groups.items():
        ms = simulate_group(teams, pinned_by_group.get(g, {}), strengths,
                            qual_prob, rng, env)
        ranked = rank_group(ms, teams)
        standings[g] = ranked
        for s in ranked:
            reached[s.team] = "GROUP"

    # Slot table: winners and runners-up, then the eight best third places.
    slot_team: Dict[str, str] = {}
    for g, ranked in standings.items():
        slot_team[f"1{g}"] = ranked[0].team
        slot_team[f"2{g}"] = ranked[1].team
    thirds = sorted((ranked[2] for ranked in standings.values()),
                    key=lambda s: (s.points, s.gd, s.gf), reverse=True)[:8]
    third_by_group = {team_to_group()[s.team]: s.team for s in thirds}
    for slot_token, group_letter in T._match_thirds_to_slots(
            list(third_by_group.keys())).items():
        slot_team[slot_token] = third_by_group[group_letter]

    if tracker is not None:
        tracker.start_run()

    winners: Dict[int, str] = {}

    def play(name_a: str, name_b: str, stage: str) -> str:
        venue = _NEUTRAL_VENUES[(hash((name_a, name_b)) % len(_NEUTRAL_VENUES))]
        _, _, w = simulate_match(
            name_a, name_b, strengths[name_a], strengths[name_b],
            mc.knockout_modifier(), mc.knockout_modifier(), rng,
            venue=venue, env=env, knockout=True)
        reached[name_a] = stage
        reached[name_b] = stage
        if tracker is not None:
            tracker.record_ko_match(name_a, name_b)
        return w if w is not None else name_a

    for mid, sa, sb in T.R32_SLOTS:
        winners[mid] = play(slot_team[sa], slot_team[sb], "R32")
    for mid, _, _ in T.R32_SLOTS:
        reached[winners[mid]] = "R16"

    def play_round(slots, stage_in, stage_next):
        for mid, fa, fb in slots:
            winners[mid] = play(winners[fa], winners[fb], stage_in)
        for mid, _, _ in slots:
            reached[winners[mid]] = stage_next

    play_round(T.R16_SLOTS, "R16", "QF")
    play_round(T.QF_SLOTS, "QF", "SF")
    play_round(T.SF_SLOTS, "SF", "FINAL")
    _, fa, fb = T.FINAL_SLOT
    champion = play(winners[fa], winners[fb], "FINAL")
    reached[champion] = "CHAMPION"

    if tracker is not None:
        tracker.end_run(reached)
    return reached


# ---------------------------------------------------------------------------
# Qualification pre-estimate (drives the matchday tactical states)
# ---------------------------------------------------------------------------

def estimate_qualification(
    state: TournamentState,
    pinned_by_group: Dict[str, Dict[frozenset, Match]],
    runs: int = 400,
    seed: int = 99,
) -> Dict[str, float]:
    """Cheap pre-sim of the group stage to get a qualification outlook.

    Runs with neutral context (no tactical states yet, since those depend on
    this estimate) and counts top-two finishes plus a partial credit for the
    third places that usually advance.
    """
    rng = np.random.default_rng(seed)
    groups = group_draw()
    credit: Dict[str, float] = defaultdict(float)
    flat_qual = {t: 0.5 for t in state.teams}
    for _ in range(runs):
        strengths = {t: state.get(t).sample_strength(rng) for t in state.teams}
        env = MatchEnv()
        for g, teams in groups.items():
            ms = simulate_group(teams, pinned_by_group.get(g, {}), strengths,
                                flat_qual, rng, env, use_context=False)
            ranked = rank_group(ms, teams)
            credit[ranked[0].team] += 1.0
            credit[ranked[1].team] += 1.0
            credit[ranked[2].team] += 0.67
    return {t: min(0.99, credit[t] / runs) for t in state.teams}


# ---------------------------------------------------------------------------
# Top-level forecast
# ---------------------------------------------------------------------------

def forecast(
    state: TournamentState,
    pinned_by_group: Dict[str, Dict[frozenset, Match]],
    runs: int = 2000,
    seed: int = 2026,
    qual_prob: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Run the full Monte Carlo and aggregate per-team stage probabilities."""
    if qual_prob is None:
        qual_prob = estimate_qualification(state, pinned_by_group, seed=seed + 1)

    rng = np.random.default_rng(seed)
    tracker = PathTracker({t: state.get(t).rating_mean for t in state.teams})
    reach_at_least = {t: Counter() for t in state.teams}
    champions: Counter = Counter()

    for _ in range(runs):
        reached = run_tournament(state, pinned_by_group, qual_prob, rng, tracker)
        for team, stage in reached.items():
            r = _STAGE_ORDER.index(stage)
            for s in _STAGE_ORDER[1:r + 1]:
                reach_at_least[team][s] += 1
        champ = next(t for t, s in reached.items() if s == "CHAMPION")
        champions[champ] += 1

    probs: Dict[str, Dict[str, float]] = {}
    for t in state.teams:
        c = reach_at_least[t]
        probs[t] = {s: c[s] / runs for s in _STAGE_ORDER[1:]}
        probs[t]["CHAMPION"] = champions[t] / runs

    return {
        "runs": runs,
        "probs": probs,
        "champions": dict(champions),
        "qual_prob": qual_prob,
        "path": tracker.summary(),
        "state": state.snapshot(),
    }
