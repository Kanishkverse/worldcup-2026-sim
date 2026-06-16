"""Matchday tactical intelligence (Phase 3).

A team plays the third group game differently from the first. A side already
through rotates and coasts; a side that must win throws bodies forward and
leaves space behind; a side that only needs to avoid defeat sits in. This
module reads the group situation (points, goal difference, how likely they
are to qualify, which matchday it is) and returns a tactical state plus the
attack/defence/risk modifiers that go with it.

The matchday gate matters: nobody panics in game one, so the desperate states
only switch on from matchday two onward.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TacticalState(str, Enum):
    NEED_WIN = "NEED_WIN"
    DRAW_ACCEPTABLE = "DRAW_ACCEPTABLE"
    MUST_AVOID_LOSS = "MUST_AVOID_LOSS"
    GOAL_DIFFERENCE_HUNT = "GOAL_DIFFERENCE_HUNT"
    ALREADY_QUALIFIED = "ALREADY_QUALIFIED"


@dataclass(frozen=True)
class ContextModifier:
    """Multipliers handed to the match engine for one team.

    attack_mult scales the team's own expected goals; defense_mult scales the
    opponent's expected goals (below 1.0 means they defend better). The other
    two are intent signals the engine uses for late-game pushes.
    """

    state: TacticalState
    attack_mult: float
    defense_mult: float
    sub_aggression: float   # 0..1, how readily attacking subs come on
    risk: float             # 0..1, late-game risk taking when chasing


_MODIFIERS = {
    TacticalState.DRAW_ACCEPTABLE: (1.00, 1.00, 0.50, 0.50),
    TacticalState.NEED_WIN: (1.12, 0.92, 0.85, 0.90),
    TacticalState.MUST_AVOID_LOSS: (0.93, 1.10, 0.40, 0.25),
    TacticalState.GOAL_DIFFERENCE_HUNT: (1.08, 0.98, 0.70, 0.60),
    TacticalState.ALREADY_QUALIFIED: (0.92, 0.96, 0.90, 0.40),
}


def classify(points: int, goal_difference: int, qual_prob: float,
             matchday: int) -> TacticalState:
    """Pick the tactical state from the group situation."""
    if matchday >= 2 and qual_prob >= 0.97:
        return TacticalState.ALREADY_QUALIFIED
    if matchday >= 2 and qual_prob >= 0.80:
        return TacticalState.GOAL_DIFFERENCE_HUNT
    if matchday >= 2 and qual_prob <= 0.35:
        return TacticalState.NEED_WIN
    if matchday >= 3 and qual_prob < 0.60 and points <= 3:
        return TacticalState.NEED_WIN
    if matchday >= 2 and qual_prob <= 0.55 and points <= 1:
        return TacticalState.MUST_AVOID_LOSS
    return TacticalState.DRAW_ACCEPTABLE


def context_modifier(points: int, goal_difference: int, qual_prob: float,
                     matchday: int) -> ContextModifier:
    state = classify(points, goal_difference, qual_prob, matchday)
    atk, dfn, sub, risk = _MODIFIERS[state]
    return ContextModifier(state=state, attack_mult=atk, defense_mult=dfn,
                           sub_aggression=sub, risk=risk)


def knockout_modifier() -> ContextModifier:
    """Knockouts have no group maths; everyone is in DRAW_ACCEPTABLE balance
    until extra time, where the engine applies its own late-game push."""
    atk, dfn, sub, risk = _MODIFIERS[TacticalState.DRAW_ACCEPTABLE]
    return ContextModifier(state=TacticalState.DRAW_ACCEPTABLE, attack_mult=atk,
                           defense_mult=dfn, sub_aggression=sub, risk=risk)
