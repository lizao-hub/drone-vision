"""One worker process: attach to the rings, run one stage per job, reply.

Both stages run this same loop; the handler decides what a job means.  A *stage*
failure does not kill the worker -- it is returned as the frame's error reason,
exactly as the serial pipeline records it -- so a dead worker always means an
infrastructure failure (killed, OOM, CUDA) and the coordinator may treat it as a
hard error.

Models are built here, not passed in: a torch model cannot be pickled, and each
worker should pay for, and occupy memory with, only the models it runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from queue import Empty
from time import perf_counter
from typing import Any

import numpy as np

from algorithm.config import PipelineSettings
from algorithm.perception.masking import blackout_with_union
from algorithm.perception.stages import DetectStage, SegmentStage
from algorithm.runtime.factory import build_detect_stage, build_segment_stage
from algorithm.runtime.shm import (
    DetectResult,
    FrameJob,
    FrameRing,
    FrameRingSpec,
    MaskRing,
    MaskRingSpec,
    SegmentResult,
)

# How long a worker waits for work before re-checking the stop flag; it is the
# worst-case delay when the coordinator shuts down.
_JOB_POLL_S = 0.2


def serve_segment(
    settings: PipelineSettings,
    device: int | str,
    frame_spec: FrameRingSpec,
    mask_spec: MaskRingSpec,
    jobs: Any,
    results: Any,
    stop: Any,
) -> None:
    """Segmentation worker: frame slot in, union + instance masks written out."""
    frames = FrameRing.attach(frame_spec)
    masks = MaskRing.attach(mask_spec)
    try:
        stage = build_segment_stage(settings, device)
        _serve(jobs, results, stop, lambda job: _segment(stage, frames, masks, job))
    finally:
        frames.close()
        masks.close()


def serve_detect(
    settings: PipelineSettings,
    device: int | str,
    frame_spec: FrameRingSpec,
    mask_spec: MaskRingSpec,
    jobs: Any,
    results: Any,
    stop: Any,
) -> None:
    """Detection worker: masked frame in, vehicle and lane-line boxes out."""
    frames = FrameRing.attach(frame_spec)
    masks = MaskRing.attach(mask_spec)
    try:
        stage = build_detect_stage(settings, device)
        _serve(jobs, results, stop, lambda job: _detect(stage, frames, masks, job))
    finally:
        frames.close()
        masks.close()


def _serve(jobs: Any, results: Any, stop: Any, handle: Callable[[FrameJob], Any]) -> None:
    while not stop.is_set():
        try:
            job = jobs.get(timeout=_JOB_POLL_S)
        except Empty:
            continue
        if job is None:
            return
        results.put(handle(job))


def _segment(
    stage: SegmentStage,
    frames: FrameRing,
    masks: MaskRing,
    job: FrameJob,
) -> SegmentResult:
    # The detector reads the same frame slot later, so this stage works on its
    # own copy: a backend that writes into its input must not corrupt that hop.
    frame = frames.view(job.frame_slot).copy()
    started = perf_counter()
    try:
        instances = stage.predict(frame)
    except Exception:  # noqa: BLE001 - third-party runtimes raise their own types
        return SegmentResult(
            sequence=job.sequence,
            frame_slot=job.frame_slot,
            mask_slot=job.mask_slot,
            segment_ms=(perf_counter() - started) * 1000.0,
            error="viaduct_adapter_error",
        )
    segment_ms = (perf_counter() - started) * 1000.0

    # Same union the serial blackout builds: every instance, nothing else.
    union = np.zeros(frame.shape[:2], dtype=np.bool_)
    for item in instances:
        union |= item.mask.data
    masks.write(job.mask_slot, union, [item.mask.data for item in instances])
    return SegmentResult(
        sequence=job.sequence,
        frame_slot=job.frame_slot,
        mask_slot=job.mask_slot,
        segment_ms=segment_ms,
        has_deck=bool(union.any()),
        # Instance i is plane i + 1 of the mask slot, so the boxes travel bare.
        instances=tuple(replace(item, mask=None) for item in instances),
    )


def _detect(
    stage: DetectStage,
    frames: FrameRing,
    masks: MaskRing,
    job: FrameJob,
) -> DetectResult:
    masked = blackout_with_union(
        frames.view(job.frame_slot), masks.union(job.mask_slot)
    )
    if masked is None:
        # The coordinator only dispatches a frame whose deck was found, so an
        # empty mask here is a broken invariant, not a frame to classify.
        raise RuntimeError("detect worker was handed a frame with no deck")

    started = perf_counter()
    try:
        vehicle = stage.vehicles(masked)
    except Exception:  # noqa: BLE001
        return DetectResult(
            sequence=job.sequence,
            frame_slot=job.frame_slot,
            mask_slot=job.mask_slot,
            detect_ms=(perf_counter() - started) * 1000.0,
            vehicle_error="vehicle_adapter_error",
        )
    detect_ms = (perf_counter() - started) * 1000.0

    line_started = perf_counter()
    try:
        line = stage.lines(masked)
    except Exception:  # noqa: BLE001
        # The vehicles still travel: the serial order runs the vehicle tracker
        # before it ever looks at the line model, and the coordinator must be
        # able to reproduce that precedence.
        return DetectResult(
            sequence=job.sequence,
            frame_slot=job.frame_slot,
            mask_slot=job.mask_slot,
            detect_ms=detect_ms,
            line_ms=(perf_counter() - line_started) * 1000.0,
            vehicle=vehicle,
            line_error="line_adapter_error",
        )
    return DetectResult(
        sequence=job.sequence,
        frame_slot=job.frame_slot,
        mask_slot=job.mask_slot,
        detect_ms=detect_ms,
        line_ms=(perf_counter() - line_started) * 1000.0,
        vehicle=vehicle,
        line=line,
    )
