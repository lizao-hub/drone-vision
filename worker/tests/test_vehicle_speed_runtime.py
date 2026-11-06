import pytest

from algorithm.domain.geometry import Box
from algorithm.domain.observations import (
    DetectedObject,
    Diagnostics,
    Observation,
    TrackedObject,
)
from algorithm.domain.vehicle_speed_runtime import VehicleSpeedRuntime

FRAME_W, FRAME_H = 640, 368
STEP_MS = 125
CAR_WIDTH_M = 1.95
STEP_PX = 4.0
STEP_Y = STEP_PX / FRAME_H
FLOW_PX_PER_S = STEP_PX / (STEP_MS / 1000.0)


def _car(center_y: float, width_px: float = 20.0) -> DetectedObject:
    half = width_px / FRAME_W / 2
    return DetectedObject(
        label="car",
        confidence=0.9,
        box=Box(0.5 - half, center_y - 0.02, 0.5 + half, center_y + 0.02),
    )


def _vehicle(center_y: float, track_id: str = "car-bt-000001") -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        label="car",
        confidence=0.9,
        box=Box(0.45, center_y - 0.03, 0.55, center_y + 0.03),
    )


def _line(center_y: float, track_id: str = "line-bt-000001") -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        label="line",
        confidence=0.6,
        box=Box(0.4, center_y - 0.06, 0.41, center_y + 0.06),
    )


def _observation(
    timestamp_ms: int,
    *,
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
        viaduct=(),
        vehicle=cars,
        vehicle_tracks=vehicles,
        viaduct_tracks=(),
        line_tracks=lines,
        unknown_reasons=reasons,
        diagnostics=Diagnostics(1.0, 1.0, 0.0, 1.0),
    )


def _runtime(**kwargs) -> VehicleSpeedRuntime:
    return VehicleSpeedRuntime(
        frame_width=FRAME_W, frame_height=FRAME_H, car_width_m=CAR_WIDTH_M, **kwargs
    )


def _run(count: int, *, car_step_px: float = 2 * STEP_PX, **kwargs):
    runtime = kwargs.pop("runtime", None) or _runtime(**kwargs)
    cars = (_car(0.5),) * 3
    result = None
    for index in range(count):
        result = runtime.update(
            _observation(
                index * STEP_MS,
                cars=cars,
                vehicles=(_vehicle(0.5 + index * car_step_px / FRAME_H),),
                lines=(_line(0.5 + index * STEP_Y),),
            )
        )
    return runtime, result


def test_speed_is_the_relative_flow_times_the_car_scale():
    _, report = _run(5)

    assert report.valid
    assert [item.track_id for item in report.speeds] == ["car-bt-000001"]
    expected = (FLOW_PX_PER_S * 1.95 / 20.0)
    assert report.speeds[0].speed_mps == pytest.approx(expected)
    assert report.speeds[0].speed_kmh == pytest.approx(expected * 3.6)
    assert report.scale_px == pytest.approx(20.0)


def test_a_parked_car_moves_with_the_ground_and_reports_zero():
    _, report = _run(5, car_step_px=STEP_PX)

    assert report.valid
    assert report.speeds[0].speed_mps == pytest.approx(0.0, abs=1e-9)


def test_opposite_direction_is_measured_from_the_ground_flow():
    _, report = _run(5, car_step_px=-STEP_PX)

    expected = (2 * FLOW_PX_PER_S) * 1.95 / 20.0
    assert report.speeds[0].speed_mps == pytest.approx(expected)


def test_every_vehicle_track_gets_its_own_speed():
    runtime = _runtime()
    cars = (_car(0.5),) * 3
    result = None
    for index in range(5):
        result = runtime.update(
            _observation(
                index * STEP_MS,
                cars=cars,
                vehicles=(
                    _vehicle(0.5 + index * 2 * STEP_Y, "car-bt-000001"),
                    _vehicle(0.2 + index * STEP_Y, "car-bt-000002"),
                ),
                lines=(_line(0.7 + index * STEP_Y),),
            )
        )

    assert [(item.track_id, round(item.speed_kmh, 1)) for item in result.speeds] == [
        ("car-bt-000001", round(FLOW_PX_PER_S * 1.95 / 20.0 * 3.6, 1)),
        ("car-bt-000002", 0.0),
    ]


def test_first_frames_are_invalid_until_the_ground_flow_matures():
    runtime = _runtime(min_scale_samples=3)
    cars = (_car(0.5),) * 3
    for index in range(3):
        report = runtime.update(
            _observation(
                index * STEP_MS,
                cars=cars,
                vehicles=(_vehicle(0.5 + index * STEP_Y),),
                lines=(_line(0.5 + index * STEP_Y),),
            )
        )
        assert not report.valid
        assert report.reasons == ("flow_unavailable",)
        assert report.speeds == ()


def test_no_cars_means_no_scale():
    report = _runtime().update(
        _observation(0, lines=(_line(0.5),), vehicles=(_vehicle(0.5),))
    )

    assert not report.valid
    assert report.reasons == ("scale_unavailable",)
    assert report.speeds == ()


def test_invalid_frame_is_reported_and_freezes_the_state():
    runtime = _runtime()
    cars = (_car(0.5),) * 3
    runtime.update(
        _observation(
            0, cars=cars, vehicles=(_vehicle(0.5),), lines=(_line(0.5),)
        )
    )
    frozen = runtime.update(
        _observation(STEP_MS, valid=False, reasons=("frame_invalid",))
    )

    assert not frozen.valid
    assert frozen.reasons == ("frame_invalid",)
    assert frozen.speeds == ()

    # The gap is 250 ms, so the next raw velocity still integrates to 4 px/125 ms.
    for index in (2, 3, 4):
        result = runtime.update(
            _observation(
                index * STEP_MS,
                cars=cars,
                vehicles=(_vehicle(0.5 + index * 2 * STEP_Y),),
                lines=(_line(0.5 + index * STEP_Y),),
            )
        )

    assert result.valid
    assert result.speeds[0].speed_mps == pytest.approx(FLOW_PX_PER_S * 1.95 / 20.0)


def test_degraded_frame_reports_the_reason_without_touching_the_reference():
    runtime = _runtime()
    cars = (_car(0.5),) * 3
    runtime.update(
        _observation(0, cars=cars, vehicles=(_vehicle(0.5),), lines=(_line(0.5),))
    )
    degraded = runtime.update(
        _observation(
            STEP_MS,
            cars=cars,
            vehicles=(_vehicle(0.5 + 6 * STEP_Y),),
            lines=(_line(0.5 + 6 * STEP_Y),),
            reasons=("viaduct_mask_held",),
        )
    )

    assert not degraded.valid
    assert degraded.reasons == ("viaduct_mask_held",)
    assert degraded.speeds == ()

    for index in (2, 3, 4):
        result = runtime.update(
            _observation(
                index * STEP_MS,
                cars=cars,
                vehicles=(_vehicle(0.5 + index * 2 * STEP_Y),),
                lines=(_line(0.5 + index * STEP_Y),),
            )
        )

    assert result.valid
    assert result.speeds[0].speed_mps == pytest.approx(FLOW_PX_PER_S * 1.95 / 20.0)


def test_scale_uses_the_median_width():
    runtime = _runtime()
    cars = (_car(0.5, 18.0), _car(0.5, 19.0), _car(0.5, 60.0))
    result = None
    for index in range(5):
        result = runtime.update(
            _observation(
                index * STEP_MS,
                cars=cars,
                vehicles=(_vehicle(0.5 + index * 2 * STEP_Y),),
                lines=(_line(0.5 + index * STEP_Y),),
            )
        )

    assert result.scale_px == pytest.approx(19.0)
    assert result.speeds[0].speed_mps == pytest.approx(FLOW_PX_PER_S * 1.95 / 19.0)


def test_a_new_vehicle_needs_its_own_warm_up():
    runtime = _runtime()
    cars = (_car(0.5),) * 3
    for index in range(3):
        runtime.update(
            _observation(
                index * STEP_MS,
                cars=cars,
                vehicles=(_vehicle(0.5 + index * STEP_Y),),
                lines=(_line(0.5 + index * STEP_Y),),
            )
        )
    report = runtime.update(
        _observation(
            3 * STEP_MS,
            cars=cars,
            vehicles=(
                _vehicle(0.5 + 3 * STEP_Y),
                _vehicle(0.2 + 3 * STEP_Y, "car-bt-000002"),
            ),
            lines=(_line(0.5 + 3 * STEP_Y),),
        )
    )

    assert [item.track_id for item in report.speeds] == ["car-bt-000001"]


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        VehicleSpeedRuntime(frame_width=0, frame_height=10)
    with pytest.raises(ValueError):
        VehicleSpeedRuntime(frame_width=10, frame_height=10, car_width_m=0.0)
    with pytest.raises(ValueError):
        VehicleSpeedRuntime(frame_width=10, frame_height=10, min_track_age=0)
