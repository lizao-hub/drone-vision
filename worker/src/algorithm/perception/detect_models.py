"""Detection adapters (vehicle detector, lane-line detector): boxes only."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from algorithm.domain.geometry import Box
from algorithm.domain.observations import DetectedObject
from algorithm.perception.raw import clamp_unit, sdk_to_list, verify_weight_file


class DetectionAdapter:
    """Single-threshold YOLO detector run on the blacked-out frame.

    One class covers both models: the vehicle weights and the lane-line weights
    differ only in what they were trained on, so the same call shape is used for
    each.  One confidence threshold covers every class the model reports; nothing
    is filtered by label unless ``expected_labels`` is given.  Which labels receive
    identities is the tracker's decision, not this adapter's.
    """

    def __init__(
        self,
        weight_path: Path,
        *,
        device: int | str = 0,
        imgsz: int = 640,
        confidence: float = 0.30,
        expected_labels: tuple[str, ...] | None = None,
        expected_sha256: str | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if expected_labels is not None:
            if not expected_labels or len(set(expected_labels)) != len(expected_labels):
                raise ValueError("expected labels must be a non-empty unique sequence")
        if not 0.0 < confidence <= 1.0:
            raise ValueError("detect confidence must be within (0, 1]")
        if imgsz <= 0:
            raise ValueError("detect imgsz must be positive")

        verified = verify_weight_file(weight_path, expected_sha256)
        if model_factory is None:
            from ultralytics import YOLO  # type: ignore[import-not-found]

            model_factory = YOLO
        self._model = model_factory(str(verified))
        if getattr(self._model, "task", None) != "detect":
            raise ValueError(
                f"detection model must be a detector, got task {self._model.task!r}"
            )
        names = {int(index): str(name) for index, name in self._model.names.items()}
        if expected_labels is not None:
            unknown = [label for label in expected_labels if label not in set(names.values())]
            if unknown:
                raise ValueError(f"expected labels absent from the model: {unknown}")
        self._names = names
        self._labels = None if expected_labels is None else frozenset(expected_labels)
        self._confidence = confidence
        self._device = device
        self._imgsz = imgsz

    def predict(self, frame: np.ndarray) -> tuple[DetectedObject, ...]:
        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            device=self._device,
            imgsz=self._imgsz,
            # Minimal-rect letterboxing: a 1080p 16:9 frame becomes 640x384
            # instead of 640x640, dropping ~40% of the padded pixels.  Boxes stay
            # normalized to the source frame, so coordinates are unaffected.
            rect=True,
            verbose=False,
        )
        if not results:
            return ()
        result = results[0]
        if result.boxes is None:
            return ()
        coordinates = sdk_to_list(result.boxes.xyxyn)
        confidences = sdk_to_list(result.boxes.conf)
        class_ids = sdk_to_list(result.boxes.cls)
        detections: list[DetectedObject] = []
        for coords, confidence, class_id_value in zip(
            coordinates, confidences, class_ids, strict=True
        ):
            label = self._names.get(int(class_id_value))
            if label is None:
                continue
            if self._labels is not None and label not in self._labels:
                continue
            if float(confidence) < self._confidence:
                continue
            x1, y1, x2, y2 = (clamp_unit(float(value)) for value in coords)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                DetectedObject(
                    label=label,
                    confidence=float(confidence),
                    box=Box(x1, y1, x2, y2),
                )
            )
        return tuple(detections)
