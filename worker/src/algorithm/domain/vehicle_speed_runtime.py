"""Per-vehicle ground speed: the car's image velocity minus the deck's own flow.

A runtime-layer state machine consuming :class:`Observation` values.  It is the
single speed source for the deck statistics: the vehicle flow and the ground
flow live in one place so the subtraction happens on the same time base.

Model::

    v_car_img = EMA of one vehicle track's box-centre velocity   (px/s)
    v_ground  = mean of the lane-track velocities this frame      (px/s)
    speed     = |v_car_img - v_ground| * car_width_m / median(car widths)

The lane markings are static, so ``v_ground`` is the camera's own motion; a
parked car therefore reports 0 regardless of the pixel scale.

Policies:
    * nadir view: one pixel scale for the whole frame, no banding;
    * a track needs ``min_track_age`` velocity samples before it counts, which
      also sets how many frames a sequence needs before it reports;
    * no scale or no ground flow -> ``valid=False`` with ``speeds=()``; a
      missing speed is never reported as zero (a stopped car is a *measured*
      zero, which is a different thing);
    * invalid and degraded frames freeze the state (no widths, no flow, no
      references updated), like the other runtimes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot, isfinite
from statistics import median

from algorithm.domain.observations import Observation, TrackedObject


@dataclass(frozen=True, slots=True)
class VehicleSpeed:
    """Ground speed of one tracked vehicle."""

    timestamp_ms: int
    track_id: str
    label: str
    speed_mps: float

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, str) or not self.track_id:
            raise ValueError("track_id must be a non-empty string")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string")
        if not isfinite(self.speed_mps) or self.speed_mps < 0.0:
            raise ValueError("speed_mps must be finite and non-negative")

    @property
    def speed_kmh(self) -> float:
        return self.speed_mps * 3.6


@dataclass(frozen=True, slots=True)
class VehicleSpeedUpdate:
    """One frame's per-vehicle speeds plus the frame-level measurement state."""

    timestamp_ms: int
    valid: bool
    speeds: tuple[VehicleSpeed, ...]
    reasons: tuple[str, ...] = ()
    scale_px: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_ms, bool) or not isinstance(self.timestamp_ms, int):
            raise TypeError("timestamp_ms must be an integer")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a boolean")
        if not isinstance(self.speeds, tuple):
            raise TypeError("speeds must be an immutable tuple")
        if not isinstance(self.reasons, tuple):
            raise TypeError("reasons must be an immutable tuple")
        if self.scale_px is not None and (
            not isfinite(self.scale_px) or self.scale_px <= 0.0
        ):
            raise ValueError("scale_px must be finite and positive")
        if not self.valid:
            if not self.reasons:
                raise ValueError("an invalid update must record at least one reason")
            if self.speeds:
                raise ValueError("an invalid update cannot carry speeds")


@dataclass(slots=True)
class _TrackState:
    center_x: float
    center_y: float
    seen_ms: int
    vx: float = 0.0
    vy: float = 0.0
    age: int = 0


class VehicleSpeedRuntime:
    def __init__(
        self,
        *,
        frame_width: int,
        frame_height: int,
        car_width_m: float = 1.95,
        scale_window_ms: int = 2000, # 车宽样本的滚动时间窗：只统计最近 2s 内 Observation.vehicle 的框宽
        min_scale_samples: int = 10, # 窗口内至少多少条车宽样本才允许定尺度
        flow_alpha: float = 0.2,
        min_track_age: int = 3, # 一条轨迹至少累积多少个速度样本才允许上报（车、线分别算）
        track_timeout_ms: int = 750,
    ) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame geometry must be positive")
        if not isfinite(car_width_m) or car_width_m <= 0.0:
            raise ValueError("car_width_m must be finite and positive")
        if scale_window_ms <= 0:
            raise ValueError("scale_window_ms must be positive")
        if min_scale_samples <= 0:
            raise ValueError("min_scale_samples must be positive")
        if not 0.0 < flow_alpha <= 1.0:
            raise ValueError("flow_alpha must be within (0, 1]")
        if min_track_age <= 0:
            raise ValueError("min_track_age must be positive")
        if track_timeout_ms <= 0:
            raise ValueError("track_timeout_ms must be positive")
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.car_width_m = car_width_m
        self.scale_window_ms = scale_window_ms
        self.min_scale_samples = min_scale_samples
        self.flow_alpha = flow_alpha
        self.min_track_age = min_track_age
        self.track_timeout_ms = track_timeout_ms
        self._widths: deque[tuple[int, float]] = deque()
        self._line_states: dict[str, _TrackState] = {}
        self._vehicle_states: dict[str, _TrackState] = {}

    def update(self, observation: Observation) -> VehicleSpeedUpdate:
        timestamp_ms = observation.timestamp_ms
        self._expire(timestamp_ms)
        if not observation.frame_valid:
            return VehicleSpeedUpdate(timestamp_ms, False, (), ("frame_invalid",))
        if observation.unknown_reasons:
            return VehicleSpeedUpdate(
                timestamp_ms, False, (), observation.unknown_reasons
            )

        # Both flows advance every valid frame: the lane tracks must not wait for
        # the scale (or the vehicles for the ground flow), otherwise each missing
        # ingredient restarts the other one's warm-up.
        states = {
            track.track_id: self._advance(self._vehicle_states, track, timestamp_ms)
            for track in observation.vehicle_tracks
        }
        ground = self._ground_flow(timestamp_ms, observation.line_tracks)

        # Scale: without cars there is nothing to measure against.
        self._remember_widths(timestamp_ms, observation)
        scale = self._scale()
        if scale is None:
            return VehicleSpeedUpdate(timestamp_ms, False, (), ("scale_unavailable",))
        metres_per_pixel, median_px = scale

        if ground is None:
            return VehicleSpeedUpdate(
                timestamp_ms, False, (), ("flow_unavailable",), median_px
            )

        speeds: list[VehicleSpeed] = []
        for track in observation.vehicle_tracks:
            state = states[track.track_id]
            if state.age < self.min_track_age:
                continue
            speeds.append(
                VehicleSpeed(
                    timestamp_ms=timestamp_ms,
                    track_id=track.track_id,
                    label=track.label,
                    speed_mps=hypot(state.vx - ground[0], state.vy - ground[1])
                    * metres_per_pixel,
                )
            )
        speeds.sort(key=lambda item: item.track_id)
        return VehicleSpeedUpdate(
            timestamp_ms=timestamp_ms,
            valid=True,
            speeds=tuple(speeds),
            scale_px=median_px,
        )

    # -- internals ---------------------------------------------------------
    def _remember_widths(self, timestamp_ms: int, observation: Observation) -> None:
        for item in observation.vehicle:
            width_px = (item.box.x2 - item.box.x1) * self.frame_width
            if width_px <= 0.0:
                continue
            self._widths.append((timestamp_ms, width_px))
        horizon = timestamp_ms - self.scale_window_ms
        while self._widths and self._widths[0][0] < horizon:
            self._widths.popleft()

    def _scale(self) -> tuple[float, float] | None:
        widths = [width for _, width in self._widths]
        if len(widths) < self.min_scale_samples:
            return None
        median_px = median(widths)
        return self.car_width_m / median_px, median_px

    def _ground_flow(
        self, timestamp_ms: int, tracks: tuple[TrackedObject, ...]
    ) -> tuple[float, float] | None:
        velocities = []
        for track in tracks:
            state = self._advance(self._line_states, track, timestamp_ms)
            if state.age >= self.min_track_age:
                velocities.append((state.vx, state.vy))
        if not velocities:
            return None
        return (
            sum(item[0] for item in velocities) / len(velocities),
            sum(item[1] for item in velocities) / len(velocities),
        )

    def _advance(
        self, states: dict[str, _TrackState], track: TrackedObject, timestamp_ms: int
    ) -> _TrackState:
        center_x = (track.box.x1 + track.box.x2) / 2 * self.frame_width
        center_y = (track.box.y1 + track.box.y2) / 2 * self.frame_height
        state = states.get(track.track_id)
        if state is None or timestamp_ms <= state.seen_ms:
            state = _TrackState(center_x, center_y, timestamp_ms)
            states[track.track_id] = state
            return state
        dt_s = (timestamp_ms - state.seen_ms) / 1000.0
        vx = (center_x - state.center_x) / dt_s
        vy = (center_y - state.center_y) / dt_s
        if state.age == 0:
            state.vx, state.vy = vx, vy
        else:
            state.vx = self.flow_alpha * vx + (1.0 - self.flow_alpha) * state.vx
            state.vy = self.flow_alpha * vy + (1.0 - self.flow_alpha) * state.vy
        state.center_x = center_x
        state.center_y = center_y
        state.seen_ms = timestamp_ms
        state.age += 1
        return state

    def _expire(self, timestamp_ms: int) -> None:
        for states in (self._line_states, self._vehicle_states):
            expired = [
                track_id
                for track_id, state in states.items()
                if timestamp_ms - state.seen_ms > self.track_timeout_ms
            ]
            for track_id in expired:
                del states[track_id]
