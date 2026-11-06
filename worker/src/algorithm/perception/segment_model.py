"""Viaduct segmentation adapter: ultralytics ``Results`` in, ``DetectedObject`` out."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from algorithm.domain.geometry import Box, Mask
from algorithm.domain.observations import DetectedObject
from algorithm.perception.raw import clamp_unit, sdk_to_list, verify_weight_file


class ViaductSegmentAdapter:
    """Single-class viaduct segmenter.

    Masks are requested with ``retina_masks=True`` so the SDK returns them at
    source-frame resolution; callers never rasterize polygons themselves.
    """

    def __init__(
        self,
        weight_path: Path,
        *,
        device: int | str = 0,
        imgsz: int = 960,
        confidence: float = 0.25,
        expected_label: str = "viaduct",
        expected_sha256: str | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not 0.0 < confidence <= 1.0:
            raise ValueError("segment confidence must be within (0, 1]")
        if imgsz <= 0:
            raise ValueError("segment imgsz must be positive")
        verified = verify_weight_file(weight_path, expected_sha256)
        if model_factory is None:
            from ultralytics import YOLO  # type: ignore[import-not-found]

            model_factory = YOLO
        self._model = model_factory(str(verified))
        if getattr(self._model, "task", None) != "segment":
            raise ValueError(
                f"viaduct model must be a segmenter, got task {self._model.task!r}"
            )
        names = {int(index): str(name) for index, name in self._model.names.items()}
        if names != {0: expected_label}:
            raise ValueError(
                f"viaduct class map mismatch: expected {{0: {expected_label!r}}}, got {names}"
            )
        self._device = device
        self._imgsz = imgsz
        self._confidence = confidence
        self._label = expected_label

    def predict(self, frame: np.ndarray) -> tuple[DetectedObject, ...]:
        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            device=self._device,
            imgsz=self._imgsz,
            retina_masks=True,
            # FP16 is hardcoded for the segmenter: it is the slowest stage and
            # the only model with enough work to feed the tensor cores (the two
            # 640 detectors are latency-bound).  Ultralytics forces FP32 on CPU.
            quantize=16,
            # Minimal-rect letterboxing: a 1080p 16:9 frame becomes 960x544
            # instead of 960x960, dropping ~40% of the padded pixels.  Boxes and
            # retina masks are mapped back to the source frame by the SDK.
            rect=True,
            verbose=False,
        )
        if not results:
            return ()
        result = results[0]
        if result.masks is None or result.boxes is None:
            return ()
        height, width = frame.shape[:2]
        masks = result.masks.data
        confidences = sdk_to_list(result.boxes.conf)
        coordinates = sdk_to_list(result.boxes.xyxyn)
        count = min(len(masks), len(confidences), len(coordinates))
        segmentations: list[DetectedObject] = []
        for index in range(count):
            x1, y1, x2, y2 = (clamp_unit(float(value)) for value in coordinates[index])
            if x2 <= x1 or y2 <= y1:
                continue
            mask = np.asarray(masks[index].cpu().numpy() > 0.5, dtype=np.bool_)
            if mask.shape != (height, width):
                raise ValueError(
                    "segmentation mask does not match the source frame: "
                    f"{mask.shape} != {(height, width)}"
                )
            segmentations.append(
                DetectedObject(
                    label=self._label,
                    confidence=float(confidences[index]),
                    box=Box(x1, y1, x2, y2),
                    mask=Mask(mask),
                )
            )
        return tuple(segmentations)
