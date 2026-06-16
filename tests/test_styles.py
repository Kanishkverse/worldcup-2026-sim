"""Style matchups: bounded, asymmetric, and footballing-sensible."""

from worldcup_sim import styles


def test_modifiers_are_bounded():
    for a in ("Spain", "Morocco", "USA", "Haiti"):
        for b in ("Iran", "France", "Cabo Verde", "Brazil"):
            ma, mb = styles.matchup_modifier(a, b)
            assert 0.85 <= ma <= 1.15
            assert 0.85 <= mb <= 1.15


def test_possession_team_is_blunted_by_a_deep_block():
    # Spain are the archetypal possession side; Iran sit in a deep block.
    spain_mult, _ = styles.matchup_modifier("Spain", "Iran")
    # Against an open, high-line side Spain should be less restricted.
    spain_open, _ = styles.matchup_modifier("Spain", "Germany")
    assert spain_mult < 1.0
    assert spain_mult < spain_open


def test_matchup_is_symmetric_under_swap():
    a, b = styles.matchup_modifier("Spain", "Morocco")
    b2, a2 = styles.matchup_modifier("Morocco", "Spain")
    assert abs(a - a2) < 1e-9
    assert abs(b - b2) < 1e-9


def test_unknown_team_falls_back_to_neutral():
    # No row for "Atlantis" -> neutral default, no crash, still bounded.
    ma, mb = styles.matchup_modifier("Atlantis", "Spain")
    assert 0.85 <= ma <= 1.15
    assert 0.85 <= mb <= 1.15


def test_alias_resolves_to_canonical_row():
    assert styles.style_of("USA") == styles.style_of("United States")
