"""Matchday tactical states: the right state for the situation, and modifiers
that point the right way."""

from worldcup_sim.match_context import (
    TacticalState, classify, context_modifier, knockout_modifier,
)


def test_matchday_one_is_calm():
    # Nobody panics in game one, whatever their outlook.
    assert classify(0, 0, 0.20, 1) == TacticalState.DRAW_ACCEPTABLE
    assert classify(0, 0, 0.85, 1) == TacticalState.DRAW_ACCEPTABLE


def test_secured_team_coasts_late():
    assert classify(7, 5, 0.99, 3) == TacticalState.ALREADY_QUALIFIED


def test_strong_but_not_safe_hunts_goals():
    assert classify(4, 2, 0.85, 2) == TacticalState.GOAL_DIFFERENCE_HUNT


def test_desperate_team_must_win():
    assert classify(1, -2, 0.20, 3) == TacticalState.NEED_WIN


def test_modifiers_point_the_right_way():
    need = context_modifier(1, -2, 0.20, 3)
    assert need.state == TacticalState.NEED_WIN
    assert need.attack_mult > 1.0
    assert need.defense_mult < 1.0  # they leave themselves open

    avoid = context_modifier(1, 0, 0.45, 2)
    assert avoid.state == TacticalState.MUST_AVOID_LOSS
    assert avoid.attack_mult < 1.0
    assert avoid.defense_mult > 1.0  # they sit in


def test_knockout_is_balanced():
    k = knockout_modifier()
    assert k.attack_mult == 1.0 and k.defense_mult == 1.0
