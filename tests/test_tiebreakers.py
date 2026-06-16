"""2026 group tiebreakers: points, then overall goal difference, then goals
for, then head-to-head among the teams still level."""

from worldcup_sim.standings import Match, rank_group


def _order(matches, teams):
    return [s.team for s in rank_group(matches, teams)]


def test_points_come_first():
    teams = ["A", "B", "C"]
    matches = [
        Match("A", "B", 1, 0),
        Match("A", "C", 1, 0),
        Match("B", "C", 1, 0),
    ]
    assert _order(matches, teams)[0] == "A"  # A has six points


def test_goal_difference_breaks_equal_points():
    teams = ["X", "Y", "Z"]
    matches = [
        Match("X", "Z", 5, 0),   # X: +5
        Match("Y", "Z", 1, 0),   # Y: +1
        Match("X", "Y", 0, 0),   # both draw, equal on points
    ]
    order = _order(matches, teams)
    assert order[0] == "X" and order[1] == "Y"


def test_goals_for_breaks_equal_points_and_gd():
    teams = ["A", "B", "C"]
    matches = [
        Match("A", "C", 4, 2),   # A: +2, gf 4
        Match("B", "C", 2, 0),   # B: +2, gf 2
        Match("A", "B", 1, 1),   # draw: equal points and gd, gf still splits
    ]
    order = _order(matches, teams)
    assert order[0] == "A" and order[1] == "B"


def test_head_to_head_breaks_a_full_tie():
    # A and B finish level on points, goal difference and goals for. A beat B,
    # so the head-to-head puts A above B.
    teams = ["A", "B", "C", "D"]
    matches = [
        Match("A", "B", 1, 0),
        Match("A", "C", 0, 1),
        Match("A", "D", 1, 0),
        Match("B", "C", 1, 0),
        Match("B", "D", 1, 0),
        Match("C", "D", 0, 2),
    ]
    order = _order(matches, teams)
    assert order[0] == "A" and order[1] == "B"
