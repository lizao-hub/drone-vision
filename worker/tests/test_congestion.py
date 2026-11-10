import pytest

from algorithm.domain.congestion import CongestionLevel, CongestionStabilizer

DWELL = 700
DENSE = 0.20  # well above clear_density


def _stabilizer(**kwargs) -> CongestionStabilizer:
    params = {"dwell_ms": DWELL}
    params.update(kwargs)
    return CongestionStabilizer(**params)


def _kmh(value: float) -> float:
    return value / 3.6


def test_classify_is_the_raw_threshold_matrix():
    stabilizer = _stabilizer()

    assert stabilizer.classify(DENSE, None) is CongestionLevel.CLEAR
    assert stabilizer.classify(0.02, _kmh(0.0)) is CongestionLevel.CLEAR
    assert stabilizer.classify(DENSE, _kmh(40.0)) is CongestionLevel.CLEAR
    assert stabilizer.classify(DENSE, _kmh(20.0)) is CongestionLevel.MODERATE
    assert stabilizer.classify(DENSE, _kmh(10.0)) is CongestionLevel.SEVERE


def test_classify_boundaries_belong_to_the_upper_branch():
    stabilizer = _stabilizer()

    # density == clear_density is not "below", so it can still be congested
    assert stabilizer.classify(0.08, _kmh(20.0)) is CongestionLevel.MODERATE
    # speed == clear_speed still flows, speed == severe_speed is already severe
    assert stabilizer.classify(DENSE, _kmh(30.0)) is CongestionLevel.CLEAR
    assert stabilizer.classify(DENSE, _kmh(12.0)) is CongestionLevel.SEVERE


def test_cold_start_reports_the_raw_label_without_dwell():
    stabilizer = _stabilizer()

    assert stabilizer.update("deck-1", 0, DENSE, _kmh(20.0)) is CongestionLevel.MODERATE
    assert stabilizer.update("deck-2", 0, DENSE, _kmh(10.0)) is CongestionLevel.SEVERE


def test_a_different_label_must_persist_for_the_dwell():
    stabilizer = _stabilizer()

    assert stabilizer.update("deck", 0, DENSE, _kmh(10.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 100, DENSE, _kmh(20.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 700, DENSE, _kmh(20.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 800, DENSE, _kmh(20.0)) is CongestionLevel.MODERATE


def test_the_candidate_timer_resets_when_the_raw_label_changes():
    stabilizer = _stabilizer()

    assert stabilizer.update("deck", 0, DENSE, _kmh(10.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 100, DENSE, _kmh(20.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 200, DENSE, _kmh(10.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 300, DENSE, _kmh(20.0)) is CongestionLevel.SEVERE
    # Without the reset the 100 ms candidate would have won at t=800.
    assert stabilizer.update("deck", 900, DENSE, _kmh(20.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 1000, DENSE, _kmh(20.0)) is CongestionLevel.MODERATE


def test_a_frame_by_frame_blip_never_changes_the_label():
    stabilizer = _stabilizer()

    levels = [
        stabilizer.update("deck", t, DENSE, _kmh(10.0 if (t // 100) % 2 == 0 else 20.0))
        for t in range(0, 2001, 100)
    ]

    assert set(levels) == {CongestionLevel.SEVERE}


def test_missing_speed_holds_the_previous_label():
    stabilizer = _stabilizer(max_hold_ms=3000)

    assert stabilizer.update("deck", 0, DENSE, _kmh(10.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 100, DENSE, None) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 1000, DENSE, None) is CongestionLevel.SEVERE


def test_missing_speed_gives_up_after_the_hold_window():
    stabilizer = _stabilizer(max_hold_ms=1000)

    assert stabilizer.update("deck", 0, DENSE, _kmh(10.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 100, DENSE, None) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 1100, DENSE, None) is CongestionLevel.CLEAR


def test_missing_speed_on_a_cold_deck_is_clear():
    stabilizer = _stabilizer()

    assert stabilizer.update("deck", 0, DENSE, None) is CongestionLevel.CLEAR


def test_missing_speed_drops_a_pending_candidate():
    stabilizer = _stabilizer()

    assert stabilizer.update("deck", 0, DENSE, _kmh(10.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 100, DENSE, _kmh(20.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 200, DENSE, None) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 300, DENSE, _kmh(20.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 900, DENSE, _kmh(20.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 1000, DENSE, _kmh(20.0)) is CongestionLevel.MODERATE


def test_a_repeated_timestamp_holds_everything():
    stabilizer = _stabilizer()

    assert stabilizer.update("deck", 0, DENSE, _kmh(10.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 100, DENSE, _kmh(20.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 100, DENSE, _kmh(20.0)) is CongestionLevel.SEVERE
    assert stabilizer.update("deck", 800, DENSE, _kmh(20.0)) is CongestionLevel.MODERATE


def test_a_deck_unseen_for_the_ttl_starts_cold():
    stabilizer = _stabilizer(state_ttl_ms=1000)

    assert stabilizer.update("deck", 0, DENSE, _kmh(10.0)) is CongestionLevel.SEVERE
    # Within the TTL the label survives and the dwell still applies.
    assert stabilizer.update("deck", 900, 0.02, _kmh(10.0)) is CongestionLevel.SEVERE
    # Past the TTL the id starts cold: the raw label is reported immediately.
    assert stabilizer.update("deck", 2000, 0.02, _kmh(10.0)) is CongestionLevel.CLEAR


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        CongestionStabilizer(clear_density=-0.1)
    with pytest.raises(ValueError):
        CongestionStabilizer(clear_speed_kmh=0.0)
    with pytest.raises(ValueError):
        CongestionStabilizer(severe_speed_kmh=0.0)
    with pytest.raises(ValueError):
        CongestionStabilizer(clear_speed_kmh=12.0, severe_speed_kmh=12.0)
    with pytest.raises(ValueError):
        CongestionStabilizer(dwell_ms=0)
    with pytest.raises(ValueError):
        CongestionStabilizer(state_ttl_ms=0)
    with pytest.raises(ValueError):
        CongestionStabilizer(max_hold_ms=0)
