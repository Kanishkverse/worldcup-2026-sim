"""Lock the historical validation gates into the suite: ingestion reproduces
real group tables, the backtest beats uniform, Bayesian conditioning improves
the ordering, and form behaves on real sequences."""

from scripts import validate_historical as vh


def test_ingestion_reproduces_real_group_tables():
    assert vh.validate_ingestion(vh.load_fixtures(), [])


def test_calibration_beats_uniform():
    # Smaller run count keeps the test quick; still well under the uniform bar.
    assert vh.validate_calibration([], runs=500)


def test_bayesian_updates_improve_the_ordering():
    assert vh.validate_bayes(vh.load_fixtures(), [])


def test_form_dynamics_track_reality():
    assert vh.validate_form(vh.load_fixtures(), [])


def test_group_to_ko_correction_improves_ordering():
    assert vh.validate_correction(vh.load_fixtures(), [])
