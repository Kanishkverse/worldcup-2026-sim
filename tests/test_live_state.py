"""Live-state sync: results move teams, and per_game responds in the right
direction."""

from worldcup_sim import live_state, squads
from worldcup_sim.per_game import PerGameSimulator, FORM_PER_ELO


def test_current_state_covers_the_field():
    eff, form = live_state.current_state()
    assert len(eff) == 48 and len(form) == 48


def test_dropping_points_to_a_minnow_lowers_a_favourite():
    mv = live_state.movement()
    # Spain were held 0-0 by Cabo Verde; their rating must fall, the minnow's
    # must rise.
    assert mv["Spain"] < 0
    assert mv["Cabo Verde"] > 0


def test_form_shift_sign_follows_the_live_rating():
    base = squads.player_elo_priors()["Germany"]
    assert PerGameSimulator._form_shift("Germany", base + 100) > 0
    assert PerGameSimulator._form_shift("Germany", base - 100) < 0
    assert PerGameSimulator._form_shift("Germany", None) == 0.0


def test_form_shift_is_clamped():
    base = squads.player_elo_priors()["Spain"]
    big = PerGameSimulator._form_shift("Spain", base - 5000)
    assert big >= -0.5 - 1e-9
