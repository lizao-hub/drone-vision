import numpy as np
import pytest

from algorithm.domain.geometry import Box, Mask
from algorithm.domain.observations import (
    DetectedObject,
    Diagnostics,
    Observation,
    TrackedObject,
)
from algorithm.domain.vehicle_speed_runtime import VehicleSpeedRuntime, VehicleSpeedUpdate
from algorithm.domain.viaduct_runtime import (
    CongestionLevel,
    ViaductDeck,
    ViaductRuntime,
    ViaductUpdate,
)

FRAME = 10  # deck masks are 10 x 10 px, so normalised and pixel coords stay easy
FRAME_W, FRAME_H = 640, 368
STEP_MS = 125
CAR_WIDTH, CAR_HEIGHT = 0.2, 0.2
M_PER_PX = 1.95 / (CAR_WIDTH * FRAME_W)  # car width 128 px -> metres per pixel
GROUND_LINES = ((0.5, 4.0, "line-bt-000001"),)  # y, dy px/frame, track id
GROUND_PX_PER_FRAME = GROUND_LINES[0][1]


def _dy_for(kmh: float) -> float:
    """Car ``dy`` per frame that measures ``kmh`` against the ground flow."""
    relative_px_per_s = (kmh / 3.6) / M_PER_PX
    return GROUND_PX_PER_FRAME + relative_px_per_s * STEP_MS / 1000.0


def _deck(columns: slice, track_id: str) -> TrackedObject:
    mask = np.zeros((FRAME, FRAME), dtype=np.bool_)
    mask[:, columns] = True
    rows_px = np.flatnonzero(mask.any(axis=1))
    columns_px = np.flatnonzero(mask.any(axis=0))
    box = Box(
        columns_px[0] / FRAME,
        rows_px[0] / FRAME,
        (columns_px[-1] + 1) / FRAME,
        (rows_px[-1] + 1) / FRAME,
    )
    return TrackedObject(
        track_id=track_id, label="viaduct", confidence=0.9, box=box, mask=Mask(mask)
    )


def _car(x: float, y: float) -> DetectedObject:
    return DetectedObject(
        label="car",
        confidence=0.9,
        box=Box(
            x - CAR_WIDTH / 2, y - CAR_HEIGHT / 2, x + CAR_WIDTH / 2, y + CAR_HEIGHT / 2
        ),
    )


def _vehicle(x: float, y: float, track_id: str) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        label="car",
        confidence=0.9,
        box=Box(
            x - CAR_WIDTH / 2, y - CAR_HEIGHT / 2, x + CAR_WIDTH / 2, y + CAR_HEIGHT / 2
        ),
    )


def _line(y: float, track_id: str) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        label="line",
        confidence=0.6,
        box=Box(0.49, y - 0.01, 0.51, y + 0.01),
    )


def _traffic(
    index: int,
    *,
    cars: tuple[tuple[float, float, float, str | None], ...],
    lines: tuple[tuple[float, float, str], ...] = GROUND_LINES,
):
    """One frame's detections/tracks; cars are ``(x, y0, dy px, track id or None)``."""
    detections = []
    tracks = []
    for x, y0, dy_px, track_id in cars:
        y = y0 + index * dy_px / FRAME_H
        detections.append(_car(x, y))
        if track_id is not None:
            tracks.append(_vehicle(x, y, track_id))
    line_tracks = tuple(
        _line(y0 + index * dy_px / FRAME_H, track_id) for y0, dy_px, track_id in lines
    )
    return tuple(detections), tuple(tracks), line_tracks


def _observation(
    timestamp_ms: int,
    *,
    decks: tuple[TrackedObject, ...] = (),
    cars: tuple[DetectedObject, ...] = (),
    vehicles: tuple[TrackedObject, ...] = (),
    lines: tuple[TrackedObject, ...] = (),
    valid: bool = True,
    reasons: tuple[str, ...] = (),
) -> Observation:
    return Observation(
        timestamp_ms=timestamp_ms,
        frame_sequence=timestamp_ms // STEP_MS,
        generation=0,
        frame_valid=valid,
        viaduct=tuple(
            DetectedObject(
                label=deck.label,
                confidence=deck.confidence,
                box=deck.box,
                mask=deck.mask,
            )
            for deck in decks
        ),
        vehicle=cars,
        vehicle_tracks=vehicles,
        viaduct_tracks=decks,
        line_tracks=lines,
        unknown_reasons=reasons,
        diagnostics=Diagnostics(1.0, 1.0, 0.0, 1.0),
    )


_CONGESTION_KWARGS = ("clear_density", "clear_speed_kmh", "severe_speed_kmh", "dwell_ms")


def _runtime(**kwargs) -> ViaductRuntime:
    """Build the runtime; congestion thresholds go to it, the rest to the speed runtime."""
    congestion = {key: kwargs.pop(key) for key in _CONGESTION_KWARGS if key in kwargs}
    return ViaductRuntime(
        speed=VehicleSpeedRuntime(
            frame_width=FRAME_W,
            frame_height=FRAME_H,
            car_width_m=1.95,
            min_scale_samples=1,
            min_track_age=1,
            **kwargs,
        ),
        **congestion,
    )


def _step(
    runtime: ViaductRuntime,
    index: int,
    *,
    decks: tuple[TrackedObject, ...],
    cars: tuple[tuple[float, float, float, str | None], ...],
    lines: tuple[tuple[float, float, str], ...] = GROUND_LINES,
) -> ViaductUpdate:
    detections, tracks, line_tracks = _traffic(index, cars=cars, lines=lines)
    return runtime.update(
        _observation(
            index * STEP_MS,
            decks=decks,
            cars=detections,
            vehicles=tracks,
            lines=line_tracks,
        )
    )


def test_each_deck_reports_its_own_density_and_mean_speed():
    runtime = _runtime()
    decks = (_deck(slice(0, 6), "deck-000001"), _deck(slice(6, 10), "deck-000002"))
    cars = ((0.25, 0.5, -4.0, "car-bt-000001"), (0.75, 0.5, 4.0, "car-bt-000002"))
    update = _step(runtime, 0, decks=decks, cars=cars)
    update = _step(runtime, 1, decks=decks, cars=cars)

    assert update.valid
    left, right = update.decks
    assert (left.deck_id, right.deck_id) == ("deck-000001", "deck-000002")
    assert left.density == pytest.approx(4 / 60)
    assert right.density == pytest.approx(4 / 40)
    assert (left.vehicles, right.vehicles) == (1, 1)
    # 4 px/frame against a 4 px/frame ground flow -> 64 px/s; the trailing car
    # moves with the ground -> a measured zero.
    assert left.mean_speed_mps == pytest.approx(64.0 * M_PER_PX)
    assert right.mean_speed_mps == pytest.approx(0.0, abs=1e-9)


def test_the_mean_is_the_arithmetic_mean_of_that_deck_s_speeds():
    runtime = _runtime()
    decks = (_deck(slice(0, 6), "deck-000001"),)
    cars = (
        (0.2, 0.5, -4.0, "car-bt-000001"),
        (0.35, 0.55, 4.0, "car-bt-000002"),
    )
    update = _step(runtime, 0, decks=decks, cars=cars)
    update = _step(runtime, 1, decks=decks, cars=cars)

    [left] = update.decks
    reported = {item.track_id: item.speed_mps for item in update.speed.speeds}
    expected = sum(reported.values()) / len(reported)
    assert left.mean_speed_mps == pytest.approx(expected)
    assert left.mean_speed_mps == pytest.approx(32.0 * M_PER_PX)
    assert left.vehicles == 2


def test_a_box_belongs_to_the_deck_holding_its_centre():
    # Centre x = 6 -> right deck; the box straddles the boundary at x = 6.
    runtime = _runtime()
    decks = (_deck(slice(0, 6), "deck-000001"), _deck(slice(6, 10), "deck-000002"))
    cars = ((0.6, 0.5, -4.0, "car-bt-000001"),)
    _step(runtime, 0, decks=decks, cars=cars)
    update = _step(runtime, 1, decks=decks, cars=cars)

    left, right = update.decks
    # Coverage only counts the boxes the deck owns, so the straddling box adds
    # nothing to the left deck even though half of it lies there.
    assert (left.vehicles, left.mean_speed_mps) == (0, None)
    assert left.density == 0.0
    assert right.vehicles == 1
    assert right.density == pytest.approx(2 / 40)
    assert right.mean_speed_mps == pytest.approx(64.0 * M_PER_PX)


def test_a_deck_without_a_mature_track_reports_no_speed_but_keeps_its_density():
    # The left car is detected but not tracked yet (ByteTrack's first frame).
    runtime = _runtime()
    decks = (_deck(slice(0, 6), "deck-000001"), _deck(slice(6, 10), "deck-000002"))
    cars = ((0.25, 0.5, -4.0, None), (0.75, 0.5, 4.0, "car-bt-000001"))
    _step(runtime, 0, decks=decks, cars=cars)
    update = _step(runtime, 1, decks=decks, cars=cars)

    left, right = update.decks
    assert update.valid
    assert left.vehicles == 1
    assert left.density == pytest.approx(4 / 60)
    assert left.mean_speed_mps is None
    assert right.mean_speed_mps == pytest.approx(0.0, abs=1e-9)


def test_a_missing_scale_voids_the_whole_update():
    runtime = _runtime()
    decks = (_deck(slice(0, 6), "deck-000001"),)
    update = _step(runtime, 0, decks=decks, cars=())

    assert not update.valid
    assert update.decks == ()
    assert update.reasons == ("scale_unavailable",)


def test_a_missing_ground_flow_voids_the_whole_update():
    runtime = _runtime()
    decks = (_deck(slice(0, 6), "deck-000001"),)
    cars = ((0.25, 0.5, -4.0, "car-bt-000001"),)
    _step(runtime, 0, decks=decks, cars=cars, lines=())
    update = _step(runtime, 1, decks=decks, cars=cars, lines=())

    assert not update.valid
    assert update.decks == ()
    assert update.reasons == ("flow_unavailable",)


def test_a_deck_with_no_car_reports_zero_density_and_no_speed():
    runtime = _runtime()
    decks = (_deck(slice(0, 6), "deck-000001"), _deck(slice(6, 10), "deck-000002"))
    cars = ((0.25, 0.5, -4.0, "car-bt-000001"),)
    _step(runtime, 0, decks=decks, cars=cars)
    update = _step(runtime, 1, decks=decks, cars=cars)

    right = update.decks[1]
    assert update.valid
    assert right.vehicles == 0
    assert right.density == 0.0
    assert right.mean_speed_mps is None


def test_a_parked_car_is_a_measured_zero_not_a_missing_value():
    runtime = _runtime()
    decks = (_deck(slice(0, 6), "deck-000001"),)
    cars = ((0.25, 0.5, 4.0, "car-bt-000001"),)  # moves with the ground flow
    _step(runtime, 0, decks=decks, cars=cars)
    update = _step(runtime, 1, decks=decks, cars=cars)

    [left] = update.decks
    assert left.mean_speed_mps == pytest.approx(0.0, abs=1e-9)
    assert left.mean_speed_kmh == pytest.approx(0.0, abs=1e-9)


def test_no_deck_means_no_measurement():
    runtime = _runtime()
    update = _step(runtime, 0, decks=(), cars=())

    assert not update.valid
    assert update.decks == ()
    assert update.reasons == ("viaduct_unavailable",)


def test_an_invalid_frame_is_reported_and_freezes_the_decks():
    runtime = _runtime()
    decks = (_deck(slice(0, 6), "deck-000001"),)
    cars = ((0.25, 0.5, -4.0, "car-bt-000001"),)
    _step(runtime, 0, decks=decks, cars=cars)
    frozen = runtime.update(
        _observation(STEP_MS, valid=False, reasons=("frame_invalid",))
    )
    after = _step(runtime, 2, decks=decks, cars=cars)

    assert not frozen.valid
    assert frozen.decks == ()
    assert frozen.reasons == ("frame_invalid",)
    assert after.valid
    assert [item.deck_id for item in after.decks] == ["deck-000001"]
    assert [item.track_id for item in after.speed.speeds] == ["car-bt-000001"]


def test_decks_come_out_ordered_by_id_not_by_instance_order():
    runtime = _runtime()
    decks = (_deck(slice(6, 10), "deck-000002"), _deck(slice(0, 6), "deck-000001"))
    cars = ((0.25, 0.5, -4.0, "car-bt-000001"), (0.75, 0.5, 4.0, "car-bt-000002"))
    _step(runtime, 0, decks=decks, cars=cars)
    update = _step(runtime, 1, decks=decks, cars=cars)

    assert [item.deck_id for item in update.decks] == ["deck-000001", "deck-000002"]


def test_overlapping_masks_attribute_a_car_to_the_first_deck_only():
    runtime = _runtime()
    decks = (_deck(slice(0, 10), "deck-000002"), _deck(slice(0, 10), "deck-000001"))
    cars = ((0.25, 0.5, -4.0, "car-bt-000001"),)
    _step(runtime, 0, decks=decks, cars=cars)
    update = _step(runtime, 1, decks=decks, cars=cars)

    first, second = update.decks
    assert first.deck_id == "deck-000001"
    assert first.mean_speed_mps == pytest.approx(64.0 * M_PER_PX)
    assert second.mean_speed_mps is None
    # Both decks measure the same pixels, so the density is reported twice.
    assert first.density == second.density == pytest.approx(4 / 100)


def test_the_update_carries_the_per_vehicle_speeds_it_was_built_from():
    runtime = _runtime()
    decks = (_deck(slice(0, 6), "deck-000001"),)
    cars = ((0.25, 0.5, -4.0, "car-bt-000001"),)
    _step(runtime, 0, decks=decks, cars=cars)
    update = _step(runtime, 1, decks=decks, cars=cars)

    assert update.speed.timestamp_ms == STEP_MS
    assert [item.track_id for item in update.speed.speeds] == ["car-bt-000001"]
    assert update.speed.scale_px == pytest.approx(CAR_WIDTH * FRAME_W)


def _two_frame_update(cars, *, decks=None, **runtime_kwargs) -> ViaductUpdate:
    runtime = _runtime(**runtime_kwargs)
    if decks is None:
        decks = (_deck(slice(0, 6), "deck-000001"),)
    _step(runtime, 0, decks=decks, cars=cars)
    return _step(runtime, 1, decks=decks, cars=cars)


def test_a_slow_deck_with_too_few_cars_is_not_congested():
    # One crawling car: 4 / 60 px of coverage stays below clear_density.
    update = _two_frame_update(((0.25, 0.5, _dy_for(2.0), "car-bt-000001"),))

    [left] = update.decks
    assert left.density < 0.08
    assert left.congestion is CongestionLevel.CLEAR


def test_a_crowded_deck_that_still_flows_is_not_congested():
    cars = (
        (0.2, 0.5, _dy_for(35.0), "car-bt-000001"),
        (0.35, 0.5, _dy_for(35.0), "car-bt-000002"),
    )
    update = _two_frame_update(cars)

    [left] = update.decks
    assert left.density >= 0.08
    assert left.mean_speed_kmh >= 30.0
    assert left.congestion is CongestionLevel.CLEAR


def test_a_crowded_slow_deck_is_moderately_congested():
    cars = (
        (0.2, 0.5, _dy_for(20.0), "car-bt-000001"),
        (0.35, 0.5, _dy_for(20.0), "car-bt-000002"),
    )
    update = _two_frame_update(cars)

    [left] = update.decks
    assert 12.0 < left.mean_speed_kmh < 30.0
    assert left.congestion is CongestionLevel.MODERATE


def test_a_crowded_crawling_deck_is_severely_congested():
    cars = (
        (0.2, 0.5, _dy_for(10.0), "car-bt-000001"),
        (0.35, 0.5, _dy_for(10.0), "car-bt-000002"),
    )
    update = _two_frame_update(cars)

    [left] = update.decks
    assert left.mean_speed_kmh <= 12.0
    assert left.congestion is CongestionLevel.SEVERE


def test_a_deck_without_a_measurable_speed_is_clear():
    # Left: detected but not tracked yet, so no speed.  Right: a mature track
    # keeps the frame's speed measurement valid.
    decks = (_deck(slice(0, 6), "deck-000001"), _deck(slice(6, 10), "deck-000002"))
    cars = (
        (0.25, 0.5, _dy_for(20.0), None),
        (0.75, 0.5, _dy_for(20.0), "car-bt-000001"),
    )
    update = _two_frame_update(cars, decks=decks)

    left, _ = update.decks
    assert left.vehicles == 1
    assert left.density > 0.0
    assert left.mean_speed_mps is None
    assert left.congestion is CongestionLevel.CLEAR


def test_a_frame_without_a_measurable_speed_keeps_the_previous_label():
    runtime = _runtime(dwell_ms=500)
    decks = (_deck(slice(0, 6), "deck-000001"),)
    slow = (
        (0.2, 0.5, _dy_for(10.0), "car-bt-000001"),
        (0.35, 0.5, _dy_for(10.0), "car-bt-000002"),
    )
    _step(runtime, 0, decks=decks, cars=slow)  # warm-up: no valid speed yet
    update = _step(runtime, 1, decks=decks, cars=slow)
    assert update.decks[0].congestion is CongestionLevel.SEVERE

    # Frame 2: the cars are detected but no longer tracked, so the deck has no
    # measurable speed; the label must hold instead of falling back to clear.
    untracked = tuple((x, y, dy, None) for x, y, dy, _ in slow)
    update = _step(runtime, 2, decks=decks, cars=untracked)
    assert update.decks[0].mean_speed_mps is None
    assert update.decks[0].congestion is CongestionLevel.SEVERE


def test_congestion_thresholds_are_configurable():
    cars = (
        (0.2, 0.5, _dy_for(20.0), "car-bt-000001"),
        (0.35, 0.5, _dy_for(20.0), "car-bt-000002"),
    )

    strict = _two_frame_update(cars, severe_speed_kmh=25.0)
    relaxed = _two_frame_update(cars, clear_speed_kmh=15.0)

    assert strict.decks[0].congestion is CongestionLevel.SEVERE
    assert relaxed.decks[0].congestion is CongestionLevel.CLEAR


def test_invalid_congestion_settings_are_rejected():
    speed = VehicleSpeedRuntime(frame_width=FRAME_W, frame_height=FRAME_H)
    with pytest.raises(ValueError):
        ViaductRuntime(speed=speed, clear_density=-0.1)
    with pytest.raises(ValueError):
        ViaductRuntime(speed=speed, clear_speed_kmh=0.0)
    with pytest.raises(ValueError):
        ViaductRuntime(speed=speed, severe_speed_kmh=0.0)
    with pytest.raises(ValueError):
        ViaductRuntime(speed=speed, clear_speed_kmh=12.0, severe_speed_kmh=12.0)


def test_a_deck_rejects_inconsistent_values():
    with pytest.raises(ValueError):
        ViaductDeck("", 0.0, None, 0, CongestionLevel.CLEAR)
    with pytest.raises(ValueError):
        ViaductDeck("deck-000001", 1.5, None, 0, CongestionLevel.CLEAR)
    with pytest.raises(ValueError):
        ViaductDeck("deck-000001", 0.0, -1.0, 0, CongestionLevel.CLEAR)
    with pytest.raises(ValueError):
        ViaductDeck("deck-000001", 0.0, None, -1, CongestionLevel.CLEAR)
    with pytest.raises(TypeError):
        ViaductDeck("deck-000001", 0.0, None, 0, "clear")


def test_an_update_rejects_inconsistent_frames():
    invalid_speed = VehicleSpeedUpdate(0, False, (), ("frame_invalid",))
    deck = ViaductDeck("deck-000001", 0.0, None, 0, CongestionLevel.CLEAR)

    with pytest.raises(TypeError):
        ViaductUpdate(0, False, (), "not a speed", ("frame_invalid",))
    with pytest.raises(ValueError):
        ViaductUpdate(125, False, (), invalid_speed, ("frame_invalid",))
    with pytest.raises(ValueError):
        ViaductUpdate(0, False, (), invalid_speed)
    with pytest.raises(ValueError):
        ViaductUpdate(0, False, (deck,), invalid_speed, ("frame_invalid",))
    with pytest.raises(ValueError):
        ViaductUpdate(0, True, (deck,), invalid_speed)


def test_the_runtime_requires_a_vehicle_speed_runtime():
    with pytest.raises(TypeError):
        ViaductRuntime(speed=object())
