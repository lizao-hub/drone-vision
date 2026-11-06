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
from time import perf_counter

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


def _segment_loop(
    gpu: int,
    settings: PipelineSettings,
    names: tuple[tuple[str, str, str], ...],
    height: int,
    width: int,
    q_in: object,
    q_vehicle: object,
    q_line: object,
    q_out: object,
    stop: object,
    capacity: int,
) -> None:
    from algorithm.perception.segment_model import ViaductSegmentAdapter  # noqa: PLC0415

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


def _detector_loop(
    role: str,
    gpu: int,
    settings: PipelineSettings,
    names: tuple[tuple[str, str, str], ...],
    height: int,
    width: int,
    q_in: object,
    q_out: object,
    stop: object,
) -> None:
    from algorithm.perception.detect_models import DetectionAdapter  # noqa: PLC0415

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


def segment_worker(
    gpu: int,
    settings: PipelineSettings,
    names: tuple[tuple[str, str, str], ...],
    height: int,
    width: int,
    q_in: object,
    q_vehicle: object,
    q_line: object,
    q_out: object,
    stop: object,
    capacity: int,
) -> None:
    """Process entry point: one segmentation model on GPU ``gpu``."""
    _run(
        SEGMENT,
        lambda: _segment_loop(
            gpu,
            settings,
            names,
            height,
            width,
            q_in,
            q_vehicle,
            q_line,
            q_out,
            stop,
            capacity,
        ),
        q_out,
    )


def vehicle_worker(
    gpu: int,
    settings: PipelineSettings,
    names: tuple[tuple[str, str, str], ...],
    height: int,
    width: int,
    q_in: object,
    q_out: object,
    stop: object,
) -> None:
    """Process entry point: the vehicle detector on GPU ``gpu``."""
    _run(
        VEHICLE,
        lambda: _detector_loop(
            VEHICLE, gpu, settings, names, height, width, q_in, q_out, stop
        ),
        q_out,
    )


def line_worker(
    gpu: int,
    settings: PipelineSettings,
    names: tuple[tuple[str, str, str], ...],
    height: int,
    width: int,
    q_in: object,
    q_out: object,
    stop: object,
) -> None:
    """Process entry point: the lane-line detector on GPU ``gpu``."""
    _run(
        LINE,
        lambda: _detector_loop(
            LINE, gpu, settings, names, height, width, q_in, q_out, stop
        ),
        q_out,
    )
