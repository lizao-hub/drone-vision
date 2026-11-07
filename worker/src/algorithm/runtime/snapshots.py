"""JSONL-ready snapshots: domain values in, plain JSON types out."""

from __future__ import annotations

from typing import Any

from algorithm.domain.geometry import Box
from algorithm.domain.observations import DetectedObject, Observation, TrackedObject
from algorithm.domain.viaduct_runtime import ViaductUpdate


def record(
    observation: Observation,
    viaduct: ViaductUpdate,
    *,
    sink_ms: float | None = None,
) -> dict[str, Any]:
    """One frame's observation plus the deck statistics, ready for ``json.dumps``.

    ``sink_ms`` is the writer's own time for this frame; it is measured after the
    pipeline returns, so the caller passes it in rather than the observation
    carrying it.
    """
    car_speeds = viaduct.speed
    return {
        "timestamp_ms": observation.timestamp_ms,
        "frame_sequence": observation.frame_sequence,
        "frame_valid": observation.frame_valid,
        "viaduct": [_detection(item) for item in observation.viaduct],
        "viaduct_tracks": [
            _tracked(track) for track in observation.viaduct_tracks
        ],
        "vehicle": [_detection(item) for item in observation.vehicle],
        "vehicle_tracks": [_tracked(track) for track in observation.vehicle_tracks],
        "line_tracks": [_tracked(track) for track in observation.line_tracks],
        "unknown_reasons": list(observation.unknown_reasons),
        "vehicle_speeds": {
            "valid": car_speeds.valid,
            "reasons": list(car_speeds.reasons),
            "scale_px": None
            if car_speeds.scale_px is None
            else round(car_speeds.scale_px, 3),
            "vehicles": [
                {
                    "track_id": item.track_id,
                    "label": item.label,
                    "speed_mps": round(item.speed_mps, 4),
                    "speed_kmh": round(item.speed_kmh, 3),
                }
                for item in car_speeds.speeds
            ],
        },
        "viaduct_stats": {
            "valid": viaduct.valid,
            "reasons": list(viaduct.reasons),
            "decks": [
                {
                    "deck_id": item.deck_id,
                    "density": round(item.density, 5),
                    "mean_speed_mps": None
                    if item.mean_speed_mps is None
                    else round(item.mean_speed_mps, 4),
                    "mean_speed_kmh": None
                    if item.mean_speed_kmh is None
                    else round(item.mean_speed_kmh, 3),
                    "mean_speed_state": item.speed_state.value,
                    "vehicles": item.vehicles,
                    "congestion": item.congestion.value,
                }
                for item in viaduct.decks
            ],
        },
        "diagnostics": {
            "segment_ms": round(observation.diagnostics.segment_ms, 2),
            "detect_ms": round(observation.diagnostics.detect_ms, 2),
            "line_ms": round(observation.diagnostics.line_ms, 2),
            "processing_ms": round(observation.diagnostics.processing_ms, 2),
            "decode_ms": round(observation.diagnostics.decode_ms, 2),
            "queue_wait_ms": round(observation.diagnostics.queue_wait_ms, 2),
            "sink_ms": round(
                0.0 if sink_ms is None else sink_ms, 2
            ),
        },
    }


def _tracked(track: TrackedObject) -> dict[str, Any]:
    return {
        "track_id": track.track_id,
        "label": track.label,
        "confidence": round(track.confidence, 4),
        "box": _box(track.box),
    }


def _detection(item: DetectedObject) -> dict[str, Any]:
    return {
        "label": item.label,
        "confidence": round(item.confidence, 4),
        "box": _box(item.box),
    }


def _box(box: Box) -> dict[str, float]:
    return {
        "x": round(box.x1, 5),
        "y": round(box.y1, 5),
        "width": round(box.x2 - box.x1, 5),
        "height": round(box.y2 - box.y1, 5),
    }
