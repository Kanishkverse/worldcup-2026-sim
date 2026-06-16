"""Venue and travel effects: altitude only hurts the unacclimatised, heat
adds load, travel adds load, and everything stays bounded."""

from worldcup_sim import venues


def test_altitude_hurts_a_sea_level_side():
    eff = venues.venue_modifier("Azteca", "France", travel_km=0.0)
    assert eff.fatigue_add > 0.0
    assert eff.xg_mult < 1.0
    assert eff.stamina_mult < 1.0


def test_altitude_does_not_hurt_an_acclimatised_host():
    eff = venues.venue_modifier("Azteca", "Mexico", travel_km=0.0)
    assert eff.fatigue_add == 0.0
    assert eff.xg_mult == 1.0


def test_heat_adds_fatigue():
    hot = venues.venue_modifier("ATT", "England", travel_km=0.0)      # Dallas
    mild = venues.venue_modifier("Lumen", "England", travel_km=0.0)   # Seattle
    assert hot.fatigue_add > mild.fatigue_add


def test_travel_adds_fatigue_and_is_bounded():
    near = venues.venue_modifier("MetLife", "Brazil", travel_km=0.0)
    far = venues.venue_modifier("MetLife", "Brazil", travel_km=4000.0)
    assert far.fatigue_add > near.fatigue_add
    assert far.fatigue_add <= 0.40


def test_distance_is_symmetric_and_zero_on_self():
    assert venues.distance_km("BC Place", "BC Place") == 0.0
    d1 = venues.distance_km("BC Place", "Hard Rock")
    d2 = venues.distance_km("Hard Rock", "BC Place")
    assert abs(d1 - d2) < 1e-6
    assert d1 > 3000.0  # Vancouver to Miami is a long way


def test_unknown_venue_is_neutral():
    eff = venues.venue_modifier("Nowhere", "Spain", travel_km=1000.0)
    assert eff.fatigue_add == 0.0 and eff.xg_mult == 1.0
