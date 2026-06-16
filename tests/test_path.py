"""Path difficulty: opponent strengths are recorded, deep runs are counted,
and an easy road to a deep run shows up as positive path luck."""

from worldcup_sim.path_difficulty import PathTracker


def test_records_opponent_strength_and_deep_run():
    strengths = {"A": 2000, "B": 1900, "C": 1500, "D": 1500}
    pt = PathTracker(strengths)
    pt.start_run()
    pt.record_ko_match("A", "C")
    pt.record_ko_match("A", "D")
    pt.end_run({"A": "CHAMPION", "C": "R32", "D": "R16", "B": "GROUP"})

    rows = {r["team"]: r for r in pt.summary()}
    assert "A" in rows
    assert rows["A"]["average_path_strength"] == 1500.0
    assert rows["A"]["deep_run_rate"] > 0.0
    assert rows["A"]["path_luck"] > 0  # faced below-average opposition


def test_team_not_in_a_knockout_is_absent():
    strengths = {"A": 2000, "B": 1900, "C": 1500, "D": 1500}
    pt = PathTracker(strengths)
    pt.start_run()
    pt.record_ko_match("A", "C")
    pt.end_run({"A": "R16", "C": "R32", "B": "GROUP", "D": "GROUP"})
    teams = {r["team"] for r in pt.summary()}
    assert "B" not in teams and "D" not in teams


def test_soft_path_is_labelled_favourable():
    strengths = {"S1": 2100, "S2": 2050, "M": 1700, "w1": 1400, "w2": 1400}
    pt = PathTracker(strengths)
    pt.start_run()
    pt.record_ko_match("M", "w1")
    pt.record_ko_match("M", "w2")
    pt.record_ko_match("S1", "S2")
    pt.end_run({"M": "SF", "w1": "R32", "w2": "R16", "S1": "QF", "S2": "R32"})

    rows = {r["team"]: r for r in pt.summary()}
    assert rows["M"]["path_note"] == "favourable path"
