"""Per-match market math: de-vig, derive, blend."""

import math

from worldcup_sim import market


def test_devig_sums_to_one_and_strips_margin():
    p = market.devig(2.0, 3.5, 4.0)
    assert abs(sum(p) - 1.0) < 1e-9
    # The raw prices imply more than 100%; the overround is positive.
    assert market.overround(2.0, 3.5, 4.0) > 0


def test_american_and_decimal_agree():
    dec = market.devig(2.0, 3.4, 4.2, fmt="decimal")
    amer = market.devig(100, 240, 320, fmt="american")  # +100 == 2.0 decimal
    assert abs(dec[0] - amer[0]) < 1e-9


def test_implied_from_ratings_is_ordered_and_normalised():
    fav = market.implied_from_ratings(2100, 1700)
    assert abs(sum(fav) - 1.0) < 1e-9
    assert fav[0] > fav[2]            # the stronger side is favoured
    even = market.implied_from_ratings(1900, 1900)
    assert abs(even[0] - even[2]) < 1e-3   # equal ratings -> symmetric


def test_blend_endpoints_and_normalisation():
    model = (0.5, 0.3, 0.2)
    mkt = (0.4, 0.25, 0.35)
    b = market.blend(model, mkt, w=0.5)
    assert abs(sum(b) - 1.0) < 1e-9
    all_model = market.blend(model, mkt, w=1.0)
    assert all(abs(all_model[i] - model[i]) < 1e-9 for i in range(3))
    assert abs(market.blend(model, mkt, w=0.0)[0] - mkt[0]) < 1e-9
