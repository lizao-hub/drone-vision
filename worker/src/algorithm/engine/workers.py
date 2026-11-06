"""Worker processes of the multi-process pipeline: one model, one GPU, one queue.

Nothing here imports torch at module import time.  A worker sets
``CUDA_VISIBLE_DEVICES`` to its own card *before* the first ``import torch``
(:func:`_pin_gpu`), so its ``cuda:0`` is exactly the card the parent asked for
and it cannot allocate on a neighbour's GPU -- model loading included.  That is
also why the parent must not have created a CUDA context before forking; the
pipeline falls back to ``spawn`` when it has (see
``algorithm.engine.pipeline._start_method``).

Each worker is a loop over its queue::

    segment:  (seq, slot) -> predict -> write masked + masks into the slot
                          -> SegmentDone to the parent, (seq, slot) to both
                             detectors (only when the deck is non-empty)
    vehicle:  (seq, slot) -> predict on the slot's masked frame -> DetectionDone
    line:     the same with the lane-line model

Model failures are reported in the message instead of killing the worker: the
frame becomes invalid in the parent, with the worker's traceback as the cause.  A
failure to *load* the model is fatal and reports a :class:`WorkerFailed` with the
traceback.
"""

from __future__ import annotations

import os
import queue as _queue
import traceback
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from algorithm.config import PipelineSettings
from algorithm.engine.shared import (
    LINE,
    SEGMENT,
    VEHICLE,
    DetectionDone,
    SegmentDone,
    WorkerFailed,
    WorkerReady,
    frame_view,
    mask_view,
)
from algorithm.perception.masking import deck_frame

# The one device a worker is allowed to see, after pinning.
_PINNED_DEVICE = "cuda:0"
_QUEUE_TIMEOUT_S = 0.2


@dataclass(frozen=True)
class WorkerSpec:
    """Everything one worker process needs, pickled once when it is spawned.

    One type for all three roles.  The segment worker additionally receives the
    two detector queues and the per-frame instance capacity, which is why those
    three fields have defaults; ``run_worker`` dispatches on ``role``.
    """

    role: str
    gpu: int
    name: str
    settings: PipelineSettings
    names: tuple[tuple[str, str, str], ...]
    height: int
    width: int
    q_in: Any
    q_out: Any
    stop: Any
    q_vehicle: Any = None
    q_line: Any = None
    capacity: int = 0


def _pin_gpu(gpu: int) -> None:
    """Restrict this process to physical GPU ``gpu`` before CUDA is initialised."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch  # noqa: PLC0415 - imported after the env var on purpose

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA was initialised before the worker pinned its device "
            f"(wanted GPU {gpu}); start the pipeline from a CUDA-free parent"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(f"worker cannot see GPU {gpu} after pinning")


def _warm_up(adapter: object, height: int, width: int) -> float:
    """Run one inference so cuDNN/cuBLAS autotuning is not charged to frame 0."""
    started = perf_counter()
    adapter.predict(np.zeros((height, width, 3), dtype=np.uint8))  # type: ignore[attr-defined]
    return (perf_counter() - started) * 1000.0


def _attach(
    names: tuple[tuple[str, str, str], ...],
) -> tuple[list[object], list[object], list[object]]:
    """Attach every slot's three buffers: ``(frames, maskeds, masks)`` per slot."""
    from multiprocessing.shared_memory import SharedMemory  # noqa: PLC0415

    return (
        [SharedMemory(name=slot[0]) for slot in names],
        [SharedMemory(name=slot[1]) for slot in names],
        [SharedMemory(name=slot[2]) for slot in names],
    )


def _log_failure(role: str, seq: int, exc: BaseException, seen: set[str]) -> None:
    """Print each distinct adapter failure once, not once per frame."""
    message = repr(exc)
    if message in seen:
        return
    seen.add(message)
    print(f"[mp:{role}] frame {seq} failed: {message}", flush=True)


def _segment_loop(spec: WorkerSpec) -> None:
    from algorithm.perception.segment_model import ViaductSegmentAdapter  # noqa: PLC0415

    # The loop below is the whole worker; unbundle the spec once at the top.
    gpu, settings, names = spec.gpu, spec.settings, spec.names
    height, width = spec.height, spec.width
    q_in, q_out = spec.q_in, spec.q_out
    q_vehicle, q_line, stop = spec.q_vehicle, spec.q_line, spec.stop
    capacity = spec.capacity

    _pin_gpu(gpu)
    adapter = ViaductSegmentAdapter(
        settings.viaduct_weight,
        device=_PINNED_DEVICE,
        imgsz=settings.segment_imgsz,
        confidence=settings.viaduct_confidence,
        expected_label=settings.viaduct_label,
    )
    warmup_ms = _warm_up(adapter, height, width)
    print(f"[mp:segment] GPU {gpu} ready (warmup {warmup_ms:.0f} ms)", flush=True)
    q_out.put(WorkerReady(SEGMENT))

    frames, maskeds, masks = _attach(names)
    out: np.ndarray | None = None
    failures: set[str] = set()
    try:
        while not stop.is_set():  # type: ignore[attr-defined]
            try:
                item = q_in.get(timeout=_QUEUE_TIMEOUT_S)
            except _queue.Empty:
                continue
            if item is None:
                break
            seq, slot = item
            frame = frame_view(frames[slot].buf, height, width)
            started = perf_counter()
            try:
                instances = adapter.predict(frame)
                segment_ms = (perf_counter() - started) * 1000.0
                kept = tuple(
                    instance
                    for instance in instances
                    if instance.confidence >= settings.mask_min_confidence
                )
                if len(kept) > capacity:
                    raise ValueError(
                        f"{len(kept)} viaduct instances exceed the {capacity}-mask slot"
                    )
                if out is None:
                    out = np.empty((height, width, 3), dtype=np.uint8)
                masked, empty = deck_frame(frame, kept, out=out)
                out = masked
                frame_view(maskeds[slot].buf, height, width)[:] = masked
                for position, instance in enumerate(kept):
                    mask_view(masks[slot].buf, position, height, width)[:] = (
                        instance.mask.data
                    )
                q_out.put(
                    SegmentDone(
                        seq=seq,
                        slot=slot,
                        count=len(kept),
                        boxes=tuple(
                            (
                                instance.box.x1,
                                instance.box.y1,
                                instance.box.x2,
                                instance.box.y2,
                            )
                            for instance in kept
                        ),
                        confidences=tuple(instance.confidence for instance in kept),
                        segment_ms=segment_ms,
                        empty=empty,
                        error=None,
                    )
                )
                if not empty:
                    q_vehicle.put((seq, slot))
                    q_line.put((seq, slot))
            except Exception as exc:  # noqa: BLE001 - reported, frame is invalid
                _log_failure(SEGMENT, seq, exc, failures)
                q_out.put(
                    SegmentDone(
                        seq=seq,
                        slot=slot,
                        count=0,
                        boxes=(),
                        confidences=(),
                        segment_ms=(perf_counter() - started) * 1000.0,
                        empty=True,
                        error=repr(exc),
                    )
                )
    finally:
        for segment in frames + maskeds + masks:
            segment.close()


def _detector_loop(spec: WorkerSpec) -> None:
    from algorithm.perception.detect_models import DetectionAdapter  # noqa: PLC0415

    # The loop below is the whole worker; unbundle the spec once at the top.
    role, gpu, settings, names = spec.role, spec.gpu, spec.settings, spec.names
    height, width = spec.height, spec.width
    q_in, q_out, stop = spec.q_in, spec.q_out, spec.stop

    _pin_gpu(gpu)
    if role == VEHICLE:
        adapter = DetectionAdapter(
            settings.vehicle_weight,
            device=_PINNED_DEVICE,
            imgsz=settings.detect_imgsz,
            confidence=settings.vehicle_confidence,
        )
    elif role == LINE:
        adapter = DetectionAdapter(
            settings.line_weight,
            device=_PINNED_DEVICE,
            imgsz=settings.line_imgsz,
            confidence=settings.line_confidence,
            # A swapped checkpoint must fail at wiring time, not silently track junk.
            expected_labels=(settings.line_label,),
        )
    else:  # pragma: no cover - the parent only starts the two known roles
        raise ValueError(f"unknown detector role {role!r}")

    warmup_ms = _warm_up(adapter, height, width)
    print(f"[mp:{role}] GPU {gpu} ready (warmup {warmup_ms:.0f} ms)", flush=True)
    q_out.put(WorkerReady(role))

    _, maskeds, _ = _attach(names)
    failures: set[str] = set()
    try:
        while not stop.is_set():  # type: ignore[attr-defined]
            try:
                item = q_in.get(timeout=_QUEUE_TIMEOUT_S)
            except _queue.Empty:
                continue
            if item is None:
                break
            seq, slot = item
            masked = frame_view(maskeds[slot].buf, height, width)
            started = perf_counter()
            try:
                detections = adapter.predict(masked)
                q_out.put(
                    DetectionDone(
                        role=role,
                        seq=seq,
                        slot=slot,
                        detections=detections,
                        elapsed_ms=(perf_counter() - started) * 1000.0,
                        error=None,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - reported, frame is invalid
                _log_failure(role, seq, exc, failures)
                q_out.put(
                    DetectionDone(
                        role=role,
                        seq=seq,
                        slot=slot,
                        detections=(),
                        elapsed_ms=(perf_counter() - started) * 1000.0,
                        error=repr(exc),
                    )
                )
    finally:
        for segment in maskeds:
            segment.close()


def _run(role: str, body: object, q_out: object) -> None:
    """Start ``body``; report a fatal failure to the parent and exit non-zero."""
    try:
        body()  # type: ignore[operator]
    except BaseException as exc:  # noqa: BLE001 - the parent needs the traceback
        q_out.put(WorkerFailed(role, traceback.format_exc()))
        raise SystemExit(1) from exc


def run_worker(spec: WorkerSpec) -> None:
    """Process entry point: build this worker's model, then loop until stopped."""
    if spec.role == SEGMENT:
        _run(SEGMENT, lambda: _segment_loop(spec), spec.q_out)
    else:
        _run(spec.role, lambda: _detector_loop(spec), spec.q_out)
