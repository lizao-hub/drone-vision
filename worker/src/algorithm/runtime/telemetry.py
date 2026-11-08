"""Run statistics: per-frame stdout and the end-of-run summary.

Extracted from the batch loop so both entry points (serial ``batch`` and the
split ``pipeline``) print the same lines from the same counters: the split run is
only comparable to the serial one if its stdout is.  Nothing here touches the
domain; it counts observations and prints.
"""

from __future__ import annotations

from collections import Counter

from algorithm.domain.observations import Observation
from algorithm.domain.viaduct_runtime import ViaductUpdate


class FrameTelemetry:
    """Accumulate one run's counters, printing as it goes."""

    def __init__(
        self,
        *,
        log_every: int = 50,
        total_frames: int | None = None,
        stride: int = 1,
    ) -> None:
        self.log_every = log_every
        self.total_frames = total_frames
        self.stride = stride
        self.processed = 0
        self.valid = 0
        self.total_ms = 0.0
        self.track_ids: set[str] = set()
        self.line_track_ids: set[str] = set()
        self.labels: Counter[str] = Counter()
        self.reasons: Counter[str] = Counter()
        self.car_speed_samples = 0
        self.car_speed_total = 0.0
        self.car_speed_min: float | None = None
        self.car_speed_max: float | None = None
        self.car_speed_frames = 0
        self.viaduct_frames = 0
        self.viaduct_reasons: Counter[str] = Counter()
        self.deck_densities: dict[str, list[float]] = {}
        self.deck_speeds: dict[str, list[float]] = {}
        self.deck_congestion: dict[str, Counter[str]] = {}

    def observe(self, observation: Observation, viaduct: ViaductUpdate) -> None:
        """Count one finished frame and print its per-frame lines."""
        self.processed += 1
        self.total_ms += observation.diagnostics.processing_ms
        self.valid += int(observation.frame_valid)
        for track in observation.vehicle_tracks:
            self.track_ids.add(track.track_id)
            self.labels[track.label] += 1
        self.line_track_ids.update(track.track_id for track in observation.line_tracks)
        for reason in observation.unknown_reasons:
            self.reasons[reason] += 1
        car_speeds = viaduct.speed
        if car_speeds.valid and car_speeds.speeds:
            self.car_speed_frames += 1
            values = [item.speed_kmh for item in car_speeds.speeds]
            self.car_speed_samples += len(values)
            self.car_speed_total += sum(values)
            low, high = min(values), max(values)
            self.car_speed_min = (
                low if self.car_speed_min is None else min(self.car_speed_min, low)
            )
            self.car_speed_max = (
                high if self.car_speed_max is None else max(self.car_speed_max, high)
            )
            print(
                f"car_speed t={observation.timestamp_ms}ms: {len(values)} cars"
                f" (min {low:.1f} max {high:.1f} km/h)",
                flush=True,
            )
        if viaduct.valid:
            self.viaduct_frames += 1
            for item in viaduct.decks:
                self.deck_densities.setdefault(item.deck_id, []).append(item.density)
                self.deck_congestion.setdefault(item.deck_id, Counter())[
                    item.congestion.value
                ] += 1
                if item.mean_speed_kmh is not None:
                    self.deck_speeds.setdefault(item.deck_id, []).append(
                        item.mean_speed_kmh
                    )
            detail = ", ".join(
                f"{item.deck_id} {item.density:.1%} n{item.vehicles}"
                + (
                    ""
                    if item.mean_speed_kmh is None
                    else f" {item.mean_speed_kmh:.1f}km/h"
                )
                + f" {item.congestion.value}"
                for item in viaduct.decks
            )
            print(f"viaduct t={observation.timestamp_ms}ms: {detail}", flush=True)
        else:
            self.viaduct_reasons.update(viaduct.reasons)
        if self.log_every and self.processed % self.log_every == 0:
            print(
                f"[{self.processed}/{self.total_frames or '?'}] valid={self.valid} "
                f"tracks={len(observation.vehicle_tracks)} "
                f"proc={observation.diagnostics.processing_ms:.0f}ms",
                flush=True,
            )

    def summary(self, *, elapsed_s: float) -> None:
        """Print the end-of-run block: totals, tracks, speed and deck stats."""
        average_ms = self.total_ms / self.processed if self.processed else 0.0
        throughput = self.processed / elapsed_s if elapsed_s else 0.0
        print(
            f"frames={self.processed} (1/{self.stride}) valid={self.valid} "
            f"avg_proc={average_ms:.1f}ms wall={elapsed_s:.1f}s fps={throughput:.1f}"
        )
        if self.track_ids:
            print(
                f"unique_tracks={len(self.track_ids)} "
                + ", ".join(
                    f"{label}:{count}" for label, count in sorted(self.labels.items())
                )
            )
        if self.line_track_ids:
            print(f"unique_line_tracks={len(self.line_track_ids)}")
        if self.reasons:
            print(
                "reasons: "
                + ", ".join(f"{key}:{value}" for key, value in sorted(self.reasons.items()))
            )
        if self.car_speed_samples:
            mean_car = self.car_speed_total / self.car_speed_samples
            print(
                f"car_speed: samples={self.car_speed_samples} mean={mean_car:.1f} km/h"
                f" min={self.car_speed_min:.1f} max={self.car_speed_max:.1f}"
                f" frames={self.car_speed_frames}/{self.processed}"
            )
        if self.viaduct_frames:
            print(
                f"viaduct: frames={self.viaduct_frames}/{self.processed}"
                f" decks={len(self.deck_densities)}"
            )
            for deck_id in sorted(self.deck_densities):
                densities = self.deck_densities[deck_id]
                line = (
                    f"  {deck_id}: density mean={_mean(densities):.1%}"
                    f" min={min(densities):.1%} max={max(densities):.1%}"
                )
                speeds = self.deck_speeds.get(deck_id, [])
                if speeds:
                    line += f" | speed mean={_mean(speeds):.1f} km/h (frames={len(speeds)})"
                congestion = self.deck_congestion.get(deck_id)
                if congestion:
                    line += " | congestion " + ", ".join(
                        f"{key}:{value}" for key, value in sorted(congestion.items())
                    )
                print(line)
        if self.viaduct_reasons:
            print(
                "viaduct_unavailable: "
                + ", ".join(
                    f"{key}:{value}" for key, value in sorted(self.viaduct_reasons.items())
                )
            )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
