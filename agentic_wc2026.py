"""
Agentic 2026 World Cup simulator — RunPod 4090 / Ollama (qwen) edition.

v2: built ON TOP of the original, backtest-validated match engine instead of a
parallel one. `AgenticMatchSimulator` subclasses `worldcup_2026_sim.MatchSimulator`
and keeps its calibrated minute-by-minute Poisson goal process, player-graph
cohesion, fatigue model, red cards, threshold substitutions, extra time and
penalty model untouched. The LLM agents steer it at three bounded points:

  pre-match   both managers (LLM) pick mentality / pressing from a scouting
              report; mapped to the engine's TacticalModifiers (clamped ±15%)
  per goal    every goal the calibrated process produces is contested by
              agents: the attacker (LLM) chooses shoot / pass / dribble and
              the goalkeeper (LLM) chooses hold / rush / narrow — good
              decisions keep the goal, bad ones get it DENIED. The base xG is
              compensated for the expected denial rate, so total scoring stays
              at the engine's validated level (~2.6-2.8 goals/match) while
              decisions move outcomes at the margin.
  half time   both managers (LLM) adjust the modifiers and make up to three
              named substitutions from the real bench; the engine re-runs the
              cohesion graph over the new XI.

Real FIFA squads come from players_2026_real.csv; the official 48-team format,
R32 bracket and real-result pinning come from worldcup_2026_sim.Tournament.

LLM failures are never silent: a parse failure falls back to a sensible
default and is counted (reported at the end); 5 consecutive transport
failures abort the run rather than degrade into a non-agentic sim.

Run (RunPod, Ollama already serving — see README_RUNPOD.md):
    python3 agentic_wc2026.py --demo "Brazil,France"        # 1 match, verbose
    python3 agentic_wc2026.py --runs 35 --seed 2026         # full tournaments
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Data-file defaults (read by worldcup_2026_sim at run time, override via CLI).
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("WC_PLAYERS_CSV", os.path.join(_HERE, "players_2026_real.csv"))
os.environ.setdefault("WC_ELO_URL", os.path.join(_HERE, "eloratings.csv"))
os.environ.setdefault("WC_MV_CSV", os.path.join(_HERE, "mv.csv"))
os.environ.setdefault("WC_RESULTS_CSV", os.path.join(_HERE, "results_2026.csv"))

import numpy as np

wc = None  # worldcup_2026_sim, imported in main() after env/CLI is settled

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("agentic2026")

_STAGES = ["GROUP", "R32", "R16", "QF", "SF", "FINAL", "CHAMPION"]


# ==============================================================================
# LLM client — OpenAI-protocol against local Ollama; never fails silently
# ==============================================================================


class LLMClient:
    """Chat wrapper with call/fallback accounting and a hard circuit breaker."""

    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self, base_url: str, model: str, timeout: float = 45.0,
                 offline: bool = False):
        self.base_url, self.model, self.timeout = base_url, model, timeout
        self.offline = offline
        self.calls = 0
        self.transport_failures = 0
        self.parse_fallbacks = 0
        self._consecutive = 0
        self._client = None

    def _get(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url,
                                  api_key=os.getenv("WC_LLM_API_KEY", "EMPTY").strip(),
                                  timeout=self.timeout, max_retries=1)
        return self._client

    def chat(self, system: str, user: str, max_tokens: int = 160) -> str:
        if self.offline:
            return ""
        self.calls += 1
        try:
            r = self._get().chat.completions.create(
                model=self.model, max_tokens=max_tokens, temperature=0.7,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            self._consecutive = 0
            text = r.choices[0].message.content or ""
            # qwen3-style reasoning blocks are not part of the decision
            return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        except Exception as exc:
            self.transport_failures += 1
            self._consecutive += 1
            LOG.warning("LLM call failed (%d consecutive): %s", self._consecutive, exc)
            if self._consecutive >= self.MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{self.MAX_CONSECUTIVE_FAILURES} consecutive LLM failures — "
                    f"is Ollama serving {self.model} at {self.base_url}? Aborting "
                    "instead of silently degrading to a non-agentic sim."
                ) from exc
            return ""

    def decide(self, system: str, user: str, default: dict,
               max_tokens: int = 160) -> dict:
        """Chat + extract the first JSON object; count fallbacks to default."""
        text = self.chat(system, user, max_tokens=max_tokens)
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                got = json.loads(m.group(0))
                if isinstance(got, dict):
                    return {**default, **got}
            except (json.JSONDecodeError, TypeError):
                pass
        if not self.offline:
            self.parse_fallbacks += 1
        return dict(default)

    def report(self) -> str:
        return (f"{self.calls} LLM calls, {self.transport_failures} transport "
                f"failures, {self.parse_fallbacks} parse fallbacks "
                f"({100.0 * self.parse_fallbacks / max(self.calls, 1):.1f}%)")


class RandomAgentClient(LLMClient):
    """Ablation arm: same decision points and action space, zero intelligence.

    Every agent decision is drawn uniformly at random. If results match the
    LLM runs, the champion spread is sampling noise, not emergent tactics.
    `rng` is reset to the per-run generator so each run stays reproducible.
    """

    def __init__(self):
        super().__init__("", "random-agents", offline=True)
        self.random_mode = True
        self.rng = np.random.default_rng(0)   # replaced per run


# ==============================================================================
# Manager agents — personas from the real squad data
# ==============================================================================


MANAGER_SYSTEM = (
    "You are the national-team head coach in a 2026 World Cup match. "
    "Answer with ONE JSON object only, no prose."
)

PLAYER_SYSTEM = (
    "You are a professional footballer in a 2026 World Cup match. "
    "Answer with ONE JSON object only, no prose."
)

_DEF_PLAN = {"mentality": "balanced", "pressing": 0.5}
_MENTALITY = {"attacking": 1.0, "balanced": 0.0, "defensive": -1.0}


def _squad_brief(team, xi) -> str:
    rows = [f"  {p.position.name:<3} {p.name:<28} rating {p.effective_rating():.2f}"
            for p in xi]
    return f"{team.name} XI:\n" + "\n".join(rows)


def plan_to_mods(plan: dict):
    """Map a manager plan onto the engine's clamped TacticalModifiers.

    Tighter than the engine's own 0.70-1.40 clamp: the agentic layer may tilt
    the validated lambda model by at most ~15% per channel.
    """
    ment = _MENTALITY.get(str(plan.get("mentality", "balanced")).lower(), 0.0)
    press = float(np.clip(plan.get("pressing", 0.5), 0.0, 1.0))
    atk = (1.0 + 0.10 * ment) * (1.0 + 0.06 * (press - 0.5))
    dfd = (1.0 - 0.07 * ment) * (1.0 - 0.04 * (press - 0.5))
    mods = wc.TacticalModifiers(
        attack_mult=float(np.clip(atk, 0.85, 1.15)),
        defense_mult=float(np.clip(dfd, 0.85, 1.15)),
        rationale=str(plan.get("key_instruction") or plan.get("talk") or "agent plan"),
    )
    return mods


class ManagerAgent:
    """One LLM persona per team: pre-match plan + half-time reaction."""

    def __init__(self, team, llm: LLMClient):
        self.team, self.llm = team, llm

    def plan(self, opponent, knockout: bool) -> dict:
        if getattr(self.llm, "random_mode", False):
            r = self.llm.rng
            self.llm.calls += 1
            return {"mentality": str(r.choice(["attacking", "balanced", "defensive"])),
                    "pressing": float(r.random()), "key_instruction": "random"}
        user = (
            f"You manage {self.team.name} against {opponent.name} "
            f"({'knockout — no replays' if knockout else 'group stage'}).\n"
            f"{_squad_brief(self.team, self.team.starting_xi())}\n"
            f"Opponent attack {opponent.attack_base():.2f} / defense {opponent.defense_base():.2f}; "
            f"your attack {self.team.attack_base():.2f} / defense {self.team.defense_base():.2f}.\n"
            'JSON: {"mentality": "attacking|balanced|defensive", '
            '"pressing": 0.0-1.0, "key_instruction": "<8 words"}'
        )
        return self.llm.decide(MANAGER_SYSTEM, user, _DEF_PLAN)

    def halftime(self, opponent, my_goals: int, opp_goals: int, state) -> dict:
        if getattr(self.llm, "random_mode", False):
            r = self.llm.rng
            self.llm.calls += 1
            offs = [p for p in state.xi if p.position is not wc.Position.GK]
            bench = list(state.bench)
            subs = []
            for _ in range(int(r.integers(0, 4))):
                if not offs or not bench:
                    break
                off = offs.pop(int(r.integers(len(offs))))
                on = bench.pop(int(r.integers(len(bench))))
                subs.append({"off": off.name, "on": on.name})
            return {"mentality": str(r.choice(["attacking", "balanced", "defensive"])),
                    "pressing": float(r.random()), "substitutions": subs,
                    "talk": "random"}
        bench_txt = "; ".join(f"{p.name} ({p.position.name} {p.effective_rating():.2f})"
                              for p in state.bench[:10]) or "none"
        tired = sorted(state.xi, key=lambda p: -state.match_fatigue[id(p)])[:5]
        tired_txt = "; ".join(f"{p.name} (fatigue {state.match_fatigue[id(p)]:.2f})"
                              for p in tired)
        user = (
            f"Half-time. {self.team.name} {my_goals}-{opp_goals} {opponent.name}.\n"
            f"Most tired starters: {tired_txt}\nBench: {bench_txt}\n"
            "Up to 3 substitutions (exact names off->on, same position preferred).\n"
            'JSON: {"mentality": "attacking|balanced|defensive", "pressing": 0.0-1.0, '
            '"substitutions": [{"off": "<name>", "on": "<name>"}], "talk": "<8 words"}'
        )
        return self.llm.decide(MANAGER_SYSTEM, user,
                               {**_DEF_PLAN, "substitutions": []}, max_tokens=220)


# ==============================================================================
# Agentic match simulator — the validated engine with agent steering
# ==============================================================================


def make_simulator_cls():
    """Class factory (worldcup_2026_sim is imported after CLI parsing)."""

    class AgenticMatchSimulator(wc.MatchSimulator):
        """Calibrated core, agentic steering.

        The goal PROCESS (how many scoring moments arise per minute) is the
        parent's validated model. Each scoring moment is then contested by the
        player agents and can be DENIED; base_xg is divided by the expected
        keep-rate so net scoring matches the parent's calibration. Decisions
        therefore shift WHO scores and WHICH moments survive — not the
        physics of the match.
        """

        DENY_BASE = 0.10          # expected denial rate the xG is compensated for
        # weight of each on-pitch position in the "who is on the end of it" draw
        _SCORER_W = {"FWD": 1.00, "MID": 0.55, "DEF": 0.12, "GK": 0.0}

        def __init__(self, llm: LLMClient, rng: np.random.Generator,
                     keeper_agents: bool = True, verbose: bool = False):
            cfg = wc.EngineConfig(base_xg=1.40 / (1.0 - 0.10))
            super().__init__(rng, config=cfg)
            self.llm = llm
            self.keeper_agents = keeper_agents
            self.verbose = verbose

        # ---- agent decisions -------------------------------------------------
        def _attacker_decision(self, p, mate, team_name: str, opp_name: str,
                               minute: int, score: Tuple[int, int]) -> str:
            if getattr(self.llm, "random_mode", False):
                self.llm.calls += 1
                return str(self.rng.choice(["shoot", "pass", "dribble"]))
            user = (
                f"Minute {minute}. You are {p.name} ({p.position.name}, rating "
                f"{p.effective_rating():.2f}) of {team_name}, clean chance vs "
                f"{opp_name} (score {score[0]}-{score[1]}). Best-placed teammate: "
                f"{mate.name} (rating {mate.effective_rating():.2f}).\n"
                'JSON: {"action": "shoot|pass|dribble", "reason": "<6 words"}'
            )
            d = self.llm.decide(PLAYER_SYSTEM, user, {"action": "shoot"}, max_tokens=80)
            return d.get("action") if d.get("action") in ("shoot", "pass", "dribble") \
                else "shoot"

        def _keeper_decision(self, gk, striker, minute: int) -> str:
            if not self.keeper_agents:
                return "hold_line"
            if getattr(self.llm, "random_mode", False):
                self.llm.calls += 1
                return str(self.rng.choice(["hold_line", "rush_out", "narrow_angle"]))
            user = (
                f"Minute {minute}. You are goalkeeper {gk.name} (rating "
                f"{gk.effective_rating():.2f}). {striker.name} is through on goal.\n"
                'JSON: {"action": "hold_line|rush_out|narrow_angle"}'
            )
            d = self.llm.decide(PLAYER_SYSTEM, user, {"action": "hold_line"},
                                max_tokens=60)
            return d.get("action") if d.get("action") in ("hold_line", "rush_out",
                                                          "narrow_angle") \
                else "hold_line"

        # ---- per-goal duel -----------------------------------------------------
        def _pick_scorer(self, state):
            pool = [p for p in state.xi if p.position is not wc.Position.GK]
            if not pool:
                return state.xi[0]
            w = np.array([self._SCORER_W[p.position.name] *
                          max(p.effective_rating(), 0.05) for p in pool])
            if w.sum() <= 0:
                return pool[0]
            return pool[int(self.rng.choice(len(pool), p=w / w.sum()))]

        def _agentic_goal(self, minute: int, st_atk, st_def, result,
                          score: Tuple[int, int]) -> bool:
            """Contest one engine-generated scoring moment. True = goal stands."""
            shooter = self._pick_scorer(st_atk)
            mates = [m for m in st_atk.xi if m is not shooter
                     and m.position in (wc.Position.FWD, wc.Position.MID)]
            mate = max(mates, key=lambda m: m.effective_rating()) if mates else shooter
            action = self._attacker_decision(shooter, mate, st_atk.team.name,
                                             st_def.team.name, minute, score)
            # Denial probability around DENY_BASE; good decisions lower it.
            if action == "pass" and mate is not shooter:
                deny = 0.07 if mate.effective_rating() > shooter.effective_rating() \
                    else 0.16
                shooter = mate
            elif action == "dribble":                       # high variance
                deny = 0.04 if self.rng.random() < 0.5 else 0.22
            else:
                deny = self.DENY_BASE

            gk = next((p for p in st_def.xi if p.position is wc.Position.GK), None)
            gk_act = "hold_line"
            if gk is not None:
                gk_act = self._keeper_decision(gk, shooter, minute)
                deny += {"hold_line": 0.0, "narrow_angle": 0.02,
                         "rush_out": float(self.rng.choice([-0.06, 0.10]))}[gk_act]
                deny += 0.10 * (gk.effective_rating() - 0.55)   # keeper quality
            deny = float(np.clip(deny, 0.02, 0.35))

            if self.rng.random() < deny:
                result.events.append((minute, st_def.team.name,
                                      f"DENIED {shooter.name}'s {action} "
                                      f"(GK {gk_act})"))
                if self.verbose:
                    LOG.info("  %d' DENIED %s (%s) by GK %s [p=%.2f]", minute,
                             shooter.name, action, gk_act, deny)
                return False
            result.events.append((minute, st_atk.team.name,
                                  f"GOAL {shooter.name} ({action}, GK {gk_act})"))
            if self.verbose:
                LOG.info("  %d' GOAL %s — %s (%s; GK %s)", minute,
                         st_atk.team.name, shooter.name, action, gk_act)
            return True

        # ---- engine loop with contested goals ----------------------------------
        def _simulate_period(self, minutes, state_a, state_b, mods_a, mods_b,
                             result):
            """Parent's calibrated loop, with every goal contested by agents."""
            ha = hb = 0
            p_red = self.config.red_card_p_per_min
            for t in minutes:
                if self.rng.random() < p_red:
                    state_a.apply_red_card(t, self.rng, result)
                if self.rng.random() < p_red:
                    state_b.apply_red_card(t, self.rng, result)

                lam_a = self._minute_lambda(state_a.attack_cohesion,
                                            state_b.defense_cohesion,
                                            state_a.team_fatigue(), mods_a)
                lam_b = self._minute_lambda(state_b.attack_cohesion,
                                            state_a.defense_cohesion,
                                            state_b.team_fatigue(), mods_b)
                for _ in range(int(self.rng.poisson(lam_a))):
                    if self._agentic_goal(t, state_a, state_b, result, (ha, hb)):
                        ha += 1
                        result.timeline.append((t, result.home))
                for _ in range(int(self.rng.poisson(lam_b))):
                    if self._agentic_goal(t, state_b, state_a, result, (hb, ha)):
                        hb += 1
                        result.timeline.append((t, result.away))

                state_a.tick_fatigue()
                state_b.tick_fatigue()
                if t % self.SUB_CHECK_EVERY == 0:
                    state_a.check_substitutions(t, result)
                    state_b.check_substitutions(t, result)
            return ha, hb

        # ---- manager substitutions through the engine's own bookkeeping ---------
        def _apply_manager_subs(self, state, subs: list, result) -> None:
            changed = False
            for sub in (subs or [])[:3]:
                if state.subs_used >= self.config.max_subs or not state.bench:
                    break
                if not isinstance(sub, dict):
                    continue
                off = next((p for p in state.xi
                            if p.name == str(sub.get("off", "")).strip()), None)
                on = next((p for p in state.bench
                           if p.name == str(sub.get("on", "")).strip()), None)
                if off is None or on is None:
                    continue
                if off.position is wc.Position.GK and on.position is not wc.Position.GK:
                    continue  # managers never trade the keeper for an outfielder
                state.bench.remove(on)
                state.xi[state.xi.index(off)] = on
                state.match_fatigue[id(on)] = on.fatigue        # fresh legs
                state._lambda_fatigue = max(on.fatigue,
                                            state._lambda_fatigue * (10.0 / 11.0))
                state.subs_used += 1
                changed = True
                result.events.append((45, state.team.name,
                                      f"SUB {on.name} for {off.name} (manager)"))
            if changed:
                state._recompute_cohesion()   # re-run the graph over the new XI

        # ---- full match ---------------------------------------------------------
        def simulate(self, team_a, team_b, knockout=False, tactical_agent=None,
                     verbose=False):
            cfg = self.config
            state_a = wc.MatchTeamState(team_a, cfg)
            state_b = wc.MatchTeamState(team_b, cfg)
            result = wc.MatchResult(home=team_a.name, away=team_b.name,
                                    home_goals=0, away_goals=0)

            mgr_a, mgr_b = ManagerAgent(team_a, self.llm), ManagerAgent(team_b, self.llm)
            plan_a, plan_b = mgr_a.plan(team_b, knockout), mgr_b.plan(team_a, knockout)
            mods_a, mods_b = plan_to_mods(plan_a), plan_to_mods(plan_b)
            if self.verbose:
                LOG.info("%s plan: %s -> %s", team_a.name, plan_a, mods_a)
                LOG.info("%s plan: %s -> %s", team_b.name, plan_b, mods_b)

            h1a, h1b = self._simulate_period(range(1, 46), state_a, state_b,
                                             mods_a, mods_b, result)

            # -- half time: manager agents adjust modifiers + make real subs ----
            ht_a = mgr_a.halftime(team_b, h1a, h1b, state_a)
            ht_b = mgr_b.halftime(team_a, h1b, h1a, state_b)
            mods_a, mods_b = plan_to_mods(ht_a), plan_to_mods(ht_b)
            self._apply_manager_subs(state_a, ht_a.get("substitutions"), result)
            self._apply_manager_subs(state_b, ht_b.get("substitutions"), result)
            if self.verbose:
                LOG.info("HT %s: %s | HT %s: %s", team_a.name,
                         ht_a.get("talk", ""), team_b.name, ht_b.get("talk", ""))

            h2a, h2b = self._simulate_period(range(46, 91), state_a, state_b,
                                             mods_a, mods_b, result)
            result.home_goals, result.away_goals = h1a + h2a, h1b + h2b

            if knockout and result.home_goals == result.away_goals:
                eta, etb = self._simulate_period(range(91, 121), state_a, state_b,
                                                 mods_a, mods_b, result)
                result.home_goals += eta
                result.away_goals += etb
                if result.home_goals == result.away_goals:
                    result.went_to_pens = True
                    result.home_pens, result.away_pens = self._shootout(team_a, team_b)

            result.events.sort()
            return result

    return AgenticMatchSimulator


# ==============================================================================
# Tournament wiring + aggregation
# ==============================================================================


def make_tournament(inventory, engine, rng):
    """Official 2026 bracket + real-result pinning, agentic match resolution."""

    class AgenticTournament(wc.Tournament):
        def _fixed_or_simulate(self, ta, tb, knockout):
            pinned = (wc._fixed_results().get(frozenset((ta.name, tb.name)))
                      if self.PIN_REAL_RESULTS else None)
            if pinned is not None:
                return super()._fixed_or_simulate(ta, tb, knockout)
            return self.engine.simulate(ta, tb, knockout=knockout)

    t = AgenticTournament(inventory, rng=rng, record=True)
    t.engine = engine
    return t


def aggregate(reaches: List[Dict[str, str]]) -> List[dict]:
    rank = {s: i for i, s in enumerate(_STAGES)}
    teams = sorted({t for r in reaches for t in r})
    rows = []
    for team in teams:
        counts = Counter(r.get(team, "GROUP") for r in reaches)
        def at_least(stage):
            return 100.0 * sum(n for s, n in counts.items()
                               if rank[s] >= rank[stage]) / len(reaches)
        rows.append({"Country": team,
                     "Reach R32 %": round(at_least("R32"), 1),
                     "Reach R16 %": round(at_least("R16"), 1),
                     "Reach QF %": round(at_least("QF"), 1),
                     "Reach SF %": round(at_least("SF"), 1),
                     "Reach Final %": round(at_least("FINAL"), 1),
                     "Win Trophy %": round(at_least("CHAMPION"), 1)})
    rows.sort(key=lambda r: (-r["Win Trophy %"], -r["Reach Final %"], r["Country"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic 2026 WC sim (Ollama/qwen)")
    ap.add_argument("--base-url", default=os.getenv("WC_LLM_BASE_URL",
                                                    "http://127.0.0.1:11434/v1"))
    ap.add_argument("--model", default=os.getenv("WC_LLM_MODEL",
                                                 "qwen2.5:14b-instruct"))
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--runs", type=int, default=1, help="full tournaments to play")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--run-offset", type=int, default=0,
                    help="continue a series: numbers runs and seeds from offset+1 "
                         "so a follow-up batch never overwrites earlier run files")
    ap.add_argument("--players", default=os.environ["WC_PLAYERS_CSV"],
                    help="per-player CSV (default: real FIFA squads)")
    ap.add_argument("--demo", metavar="TEAM_A,TEAM_B",
                    help="play ONE verbose match and exit (smoke test)")
    ap.add_argument("--no-keeper-agents", action="store_true",
                    help="skip GK agent calls (≈40%% fewer LLM calls)")
    ap.add_argument("--no-pin", action="store_true",
                    help="simulate even already-played real matches")
    ap.add_argument("--offline", action="store_true",
                    help="no LLM at all — default decisions (pipeline check only)")
    ap.add_argument("--random-agents", action="store_true",
                    help="ablation: every agent decision drawn uniformly at random "
                         "from the same action space (no LLM)")
    ap.add_argument("--out-prefix", default="agentic_2026")
    args = ap.parse_args()

    os.environ["WC_PLAYERS_CSV"] = args.players
    if not os.path.exists(args.players):
        sys.exit(f"players CSV not found: {args.players} "
                 "(upload players_2026_real.csv or pass --players)")

    global wc
    import worldcup_2026_sim as _wc
    wc = _wc

    if args.random_agents:
        llm = RandomAgentClient()
    else:
        llm = LLMClient(args.base_url, args.model, args.timeout, offline=args.offline)
    if not args.offline and not args.random_agents:
        probe = llm.chat("Reply with the single word: ok", "ready?", max_tokens=10)
        if not probe:
            sys.exit(f"Cannot reach {args.model} at {args.base_url} — "
                     "start Ollama first (see README_RUNPOD.md).")
        LOG.info("LLM online: %s @ %s", args.model, args.base_url)

    inventory = wc.build_world_cup_2026(rng_seed=args.seed)
    sim_cls = make_simulator_cls()

    # ---- single-match demo ----------------------------------------------------
    if args.demo:
        names = [s.strip() for s in args.demo.split(",")]
        if len(names) != 2 or any(n not in inventory.teams for n in names):
            sys.exit(f"--demo wants two of: {', '.join(sorted(inventory.teams))}")
        eng = sim_cls(llm, np.random.default_rng(args.seed),
                      keeper_agents=not args.no_keeper_agents, verbose=True)
        res = eng.simulate(inventory.teams[names[0]], inventory.teams[names[1]],
                           knockout=True)
        score = f"{res.home} {res.home_goals}-{res.away_goals} {res.away}"
        if res.went_to_pens:
            score += f" (pens {res.home_pens}-{res.away_pens})"
        LOG.info("FT %s | winner %s | %s", score, res.winner, llm.report())
        with open(f"{args.out_prefix}_demo.json", "w") as fh:
            json.dump({"score": score, "winner": res.winner,
                       "events": res.events, "llm": llm.report()}, fh, indent=2)
        LOG.info("saved -> %s_demo.json", args.out_prefix)
        return

    # ---- full tournaments -----------------------------------------------------
    reaches, t0 = [], time.time()
    for run_ix in range(args.runs):
        run_no = args.run_offset + run_ix + 1
        rng = np.random.default_rng(args.seed + args.run_offset + run_ix)
        if args.random_agents:
            llm.rng = rng        # keep the whole run on one reproducible stream
        eng = sim_cls(llm, rng, keeper_agents=not args.no_keeper_agents)
        tour = make_tournament(inventory, eng, rng)
        if args.no_pin:
            tour.PIN_REAL_RESULTS = False
        reach = tour.run()
        reaches.append(reach)
        champ = next(t for t, s in reach.items() if s == "CHAMPION")
        LOG.info("run %d/%d champion: %s (%.0fs elapsed, %s)",
                 run_no, args.run_offset + args.runs, champ,
                 time.time() - t0, llm.report())
        with open(f"{args.out_prefix}_run{run_no}_matches.json", "w") as fh:
            json.dump(tour.match_log, fh, indent=2, default=str)

    import pandas as pd
    df = pd.DataFrame(aggregate(reaches))
    df.to_csv(f"{args.out_prefix}_probs.csv", index=False)
    with open(f"{args.out_prefix}_meta.json", "w") as fh:
        backend = ("random-ablation" if args.random_agents else
                   "offline-defaults" if args.offline else "ollama-agentic-v2")
        json.dump({"runs": args.runs, "seed": args.seed, "model": args.model,
                   "base_url": args.base_url, "backend": backend,
                   "engine": "MatchSimulator subclass (validated core)",
                   "players_csv": args.players, "offline": args.offline,
                   "keeper_agents": not args.no_keeper_agents,
                   "llm_report": llm.report(),
                   "elapsed_s": round(time.time() - t0, 1)}, fh, indent=2)
    print(df.head(20).to_string(index=False))
    print(f"saved -> {args.out_prefix}_probs.csv, per-run match JSONs, meta")
    print(f"LLM accounting: {llm.report()}")


if __name__ == "__main__":
    main()
