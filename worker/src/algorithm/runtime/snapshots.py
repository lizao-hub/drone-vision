"""JSONL-ready snapshots: domain values in, plain JSON types out.

The shape is an explicit projection, not a generic dump: masks are dropped, a
box becomes ``x/y/width/height``, and floats are rounded so one JSONL line
stays readable.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from algorithm.domain.geometry import Box
from algorithm.domain.observations import (
    DetectedObject,
    Diagnostics,
    Observation,
    TrackedObject,
)
from algorithm.domain.viaduct_runtime import ViaductUpdate


def record(observation: Observation, viaduct: ViaductUpdate) -> dict[str, Any]:
    """One frame's observation plus the deck statistics, ready for ``json.dumps``."""
    speeds = viaduct.speed
    return {
        "timestamp_ms": observation.timestamp_ms,
        "frame_sequence": observation.frame_sequence,
        "frame_valid": observation.frame_valid,
        "viaduct": [_detection(item) for item in observation.viaduct],
        "viaduct_tracks": [_tracked(track) for track in observation.viaduct_tracks],
        "vehicle": [_detection(item) for item in observation.vehicle],
        "vehicle_tracks": [_tracked(track) for track in observation.vehicle_tracks],
        "line_tracks": [_tracked(track) for track in observation.line_tracks],
        "unknown_reasons": list(observation.unknown_reasons),
        "vehicle_speeds": {
            "valid": speeds.valid,
            "reasons": list(speeds.reasons),
            "scale_px": _rounded(speeds.scale_px, 3),
            "vehicles": [
                {
                    "track_id": item.track_id,
                    "label": item.label,
                    "speed_mps": round(item.speed_mps, 4),
                    "speed_kmh": round(item.speed_kmh, 3),
                }
                for item in speeds.speeds
            ],
        },
        "viaduct_stats": {
            "valid": viaduct.valid,
            "reasons": list(viaduct.reasons),
            "decks": [
                {
                    "deck_id": item.deck_id,
                    "density": round(item.density, 5),
                    "mean_speed_mps": _rounded(item.mean_speed_mps, 4),
                    "mean_speed_kmh": _rounded(item.mean_speed_kmh, 3),
                    "vehicles": item.vehicles,
                    "congestion": item.congestion.value,
                }
                for item in viaduct.decks
            ],
        },
        "diagnostics": _diagnostics(observation.diagnostics),
    }


def _rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _diagnostics(diagnostics: Diagnostics) -> dict[str, float]:
    return {name: round(value, 2) for name, value in asdict(diagnostics).items()}


def _detection(item: DetectedObject) -> dict[str, Any]:
    return {
        "label": item.label,
        "confidence": round(item.confidence, 4),
        "box": _box(item.box),
    }


def _tracked(track: TrackedObject) -> dict[str, Any]:
    return {"track_id": track.track_id, **_detection(track)}


def _box(box: Box) -> dict[str, float]:
    return {
        "x": round(box.x1, 5),
        "y": round(box.y1, 5),
        "width": round(box.x2 - box.x1, 5),
        "height": round(box.y2 - box.y1, 5),
    }
