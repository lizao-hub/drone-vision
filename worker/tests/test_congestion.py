import pytest

from algorithm.domain.congestion import (
    CongestionLevel,
    CongestionThresholds,
    classify_congestion,
)

THRESHOLDS = CongestionThresholds()


def test_no_measurable_speed_is_clear_not_a_density_guess():
    # A jammed-looking deck without a speed measurement must not be labelled.
    assert classify_congestion(0.9, None, THRESHOLDS) is CongestionLevel.CLEAR


def test_a_sparse_deck_is_never_congested_however_slow():
    assert classify_congestion(0.01, 1.0, THRESHOLDS) is CongestionLevel.CLEAR


def test_flowing_traffic_is_clear():
    assert classify_congestion(0.5, 40.0 / 3.6, THRESHOLDS) is CongestionLevel.CLEAR


def test_collapsed_speed_on_a_crowded_deck_is_severe():
    assert classify_congestion(0.5, 8.0 / 3.6, THRESHOLDS) is CongestionLevel.SEVERE


def test_between_the_two_speeds_is_moderate():
    assert classify_congestion(0.5, 20.0 / 3.6, THRESHOLDS) is CongestionLevel.MODERATE


def test_thresholds_are_validated_at_construction():
    with pytest.raises(ValueError):
        CongestionThresholds(clear_density=-0.1)
    with pytest.raises(ValueError):
        CongestionThresholds(clear_speed_kmh=0.0)
    with pytest.raises(ValueError):
        CongestionThresholds(severe_speed_kmh=0.0)
    with pytest.raises(ValueError):
        CongestionThresholds(clear_speed_kmh=12.0, severe_speed_kmh=12.0)
