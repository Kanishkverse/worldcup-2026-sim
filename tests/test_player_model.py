"""Player-level aggregation and the per-game simulator: strength comes from
available players, ruling a player out weakens the team, and the per-game
distribution is well formed."""

from worldcup_sim import squads
from worldcup_sim.per_game import PerGameSimulator


def test_player_elo_priors_rank_top_teams_sensibly():
    pe = squads.player_elo_priors()
    assert len(pe) == 48
    top = sorted(pe, key=pe.get, reverse=True)[:6]
    # The usual suspects should populate the top of a player-aggregated table.
    assert {"Spain", "France", "Argentina", "England"} <= set(top)


def test_ruling_out_a_player_never_raises_rating():
    impact = squads.star_impact("France", ["Mbappe"])
    assert impact["ruled_out"] == ["Kylian Mbappe"]
    assert impact["elo_without"] <= impact["elo_full"]


def test_availability_is_restored_after_star_impact():
    squads.star_impact("France", ["Mbappe"])
    inv = squads.inventory()
    assert all(p.available for p in inv.teams["France"].squad)


def test_per_game_distribution_is_valid():
    sim = PerGameSimulator(use_manager=True, seed=3)
    r = sim.fixture("France", "Mexico", runs=200)
    assert abs(r.a_win + r.draw + r.b_win - 1.0) < 1e-9
    assert r.mean_goals_a > 0 and r.mean_goals_b > 0
    # France are favourites against Mexico.
    assert r.a_win > r.b_win


def _top_xi_names(team):
    return [p.name for p in sorted(team.squad, key=lambda p: p.effective_rating(),
                                   reverse=True)[:11]]


def test_removing_the_first_xi_drops_the_aggregate():
    # Deterministic, noise-free: take out the best eleven and both the
    # player-Elo and the attack base must fall clearly.
    inv = squads.inventory()
    fra = inv.teams["France"]
    full_elo = squads.player_elo("France")
    full_atk = fra.attack_base()
    out = _top_xi_names(fra)
    squads.rule_out(fra, out)
    try:
        assert squads.player_elo("France") < full_elo - 30
        assert fra.attack_base() < full_atk - 0.03
    finally:
        squads.reset_availability(fra)


def test_removing_the_first_xi_lowers_win_probability():
    # One star is within Monte Carlo noise for a squad this deep, so the
    # per-game check uses the whole first XI (a ~5 point true effect) with
    # common random numbers.
    sim = PerGameSimulator(use_manager=True, seed=5)
    out = _top_xi_names(squads.inventory().teams["France"])
    full = sim.fixture("France", "Mexico", runs=1500, common_seed=99)
    gutted = sim.fixture("France", "Mexico", runs=1500, out_a=out, common_seed=99)
    assert gutted.a_win < full.a_win - 0.02
