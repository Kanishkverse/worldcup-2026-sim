"""Bayesian state engine: updates move the mean the right way and shrink the
variance, and the shrinkage compounds as more matches come in."""

import numpy as np

from worldcup_sim.tournament_state import TournamentState, PRIOR_STD


def _state():
    return TournamentState({"Strong": 2100.0, "Weak": 1600.0})


def test_prior_is_elo_with_default_spread():
    st = _state()
    mean, std = st.get("Strong").posterior_strength_distribution()
    assert mean == 2100.0
    assert std == PRIOR_STD


def test_upset_pulls_means_together():
    st = _state()
    before_strong = st.get("Strong").rating_mean
    before_weak = st.get("Weak").rating_mean
    # The underdog wins.
    st.update_after_match("Weak", "Strong", 2, 0)
    assert st.get("Strong").rating_mean < before_strong
    assert st.get("Weak").rating_mean > before_weak


def test_variance_shrinks_after_a_match():
    st = _state()
    assert st.get("Strong").rating_std == PRIOR_STD
    st.update_after_match("Strong", "Weak", 1, 1)
    assert st.get("Strong").rating_std < PRIOR_STD


def test_variance_keeps_shrinking_with_more_evidence():
    st = _state()
    stds = []
    for _ in range(3):
        st.update_after_match("Strong", "Weak", 2, 1)
        stds.append(st.get("Strong").rating_std)
    assert stds[0] > stds[1] > stds[2]


def test_form_responds_to_overperformance():
    st = _state()
    # Weak side beats a much stronger side: strong positive form for the weak.
    st.update_after_match("Weak", "Strong", 3, 0)
    assert st.get("Weak").form > 0
    assert st.get("Strong").form < 0


def test_sample_is_near_the_mean_on_average():
    st = _state()
    rng = np.random.default_rng(0)
    draws = [st.get("Strong").sample_strength(rng) for _ in range(4000)]
    # No fatigue or form yet, so the sample mean should sit near the rating.
    assert abs(np.mean(draws) - 2100.0) < 15.0


def test_fatigue_accumulates_and_recovers():
    st = _state()
    st.update_after_match("Strong", "Weak", 1, 0)
    loaded = st.get("Strong").fatigue
    assert loaded > 0
    st.recover_between_matchdays()
    assert st.get("Strong").fatigue < loaded
