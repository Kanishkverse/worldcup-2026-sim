"""End-to-end smoke test: the forecast runs, returns a proper distribution for
the whole field, and respects pinned real results."""

from worldcup_sim import engine
from worldcup_sim.priors import elo_priors
from worldcup_sim.standings import Match
from worldcup_sim.tournament_state import TournamentState


def test_forecast_returns_a_valid_distribution():
    state = TournamentState(elo_priors())
    out = engine.forecast(state, pinned_by_group={}, runs=30, seed=1)
    probs = out["probs"]
    assert len(probs) == 48
    champ_total = sum(p["CHAMPION"] for p in probs.values())
    assert abs(champ_total - 1.0) < 1e-6
    for p in probs.values():
        # Monotone funnel: reaching a later stage is never more likely.
        assert p["R16"] >= p["QF"] >= p["SF"] >= p["FINAL"] >= p["CHAMPION"] - 1e-9


def test_pinned_result_conditions_the_state():
    priors = elo_priors()
    base = TournamentState(priors)
    spain_prior = base.get("Spain").rating_mean
    base.update_after_match("Spain", "Cabo Verde", 0, 0)
    # A goalless draw with a weak side drags the posterior mean down.
    assert base.get("Spain").rating_mean < spain_prior
    assert base.get("Spain").rating_std < base.get("Argentina").rating_std
