"""The split pipeline's coordinator: decode, dispatch, track, write.

It owns everything the workers do not: the source, the two rings, the reference
counters, the three trackers, ``ViaductRuntime``, the JSONL and the run
statistics.  The workers only run models.

Frame order is the one thing that cannot be parallelised: ``ByteTracker`` rejects
a timestamp that does not advance.  Results are therefore completed strictly in
``frame_sequence`` order, which the topology gives for free -- the detection
worker handles one frame at a time, so its results arrive in order, and a frame
is always segmented before detection is dispatched for it.

One frame's round trip::

    [coordinator] --seg job--> [seg, GPU 0] --masks--> [coordinator]
    [coordinator] --detect job--> [detect, GPU 1] --boxes--> [coordinator]
                                                             trackers, JSONL

The coordinator never waits on a stage it could have overlapped: while the
detection worker is busy, the segmentation worker fills the next slot.

Failure policy: a *stage* exception is reported by the worker as the frame's
error reason and the run continues (identical to the serial pipeline); a *worker
process* that dies stops the run with :class:`WorkerDied`.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from time import perf_counter
from typing import Any, Self

from algorithm.config import PipelineSettings
from algorithm.domain.geometry import Mask
from algorithm.domain.observations import (
    DetectedObject,
    Diagnostics,
    Observation,
    TrackedObject,
)
from algorithm.domain.vehicle_speed_runtime import VehicleSpeedRuntime
from algorithm.domain.viaduct_runtime import ViaductRuntime
from algorithm.perception.pipeline import invalid_observation
from algorithm.runtime import parallel_worker
from algorithm.runtime.factory import build_track_stage, stage_devices
from algorithm.runtime.frame_packet import FramePacket
from algorithm.runtime.shm import (
    DetectResult,
    FrameJob,
    FrameRing,
    MaskRing,
    SegmentResult,
    SlotPool,
)
from algorithm.runtime.snapshots import record
from algorithm.runtime.sources import FrameSource
from algorithm.runtime.telemetry import FrameTelemetry

# How long the coordinator waits for a result before checking whether a worker
# died; it bounds how long a crashed worker can hang the run.
_RESULT_POLL_S = 0.2
# How long a worker may take to finish the frame it already started.
_SHUTDOWN_S = 10.0
# Instance masks per ring slot.  The segmenter emits one deck per carriageway; a
# frame with more instances than this cannot be transported, so the run stops
# with a traceback instead of silently dropping decks.
MASK_INSTANCES = 4


class WorkerDied(RuntimeError):
    """A worker process stopped without reporting a result."""


@dataclass(slots=True)
class _Pending:
    """One dispatched frame, from the segmentation job to its observation."""

    sequence: int
    timestamp_ms: int
    frame_slot: int
    mask_slot: int
    started: float
    segment_ms: float = 0.0
    # Filled by the segmentation result; the masks themselves stay in the ring.
    viaduct: tuple[DetectedObject, ...] = ()
    viaduct_tracks: tuple[TrackedObject, ...] = ()
    # Set once the frame's verdict exists; retirement waits for its turn.
    observation: Observation | None = None


class Coordinator:
    """Drive one split-pipeline run over an open :class:`FrameSource`."""

    def __init__(
        self,
        settings: PipelineSettings,
        source: FrameSource,
        json_path: Path,
        *,
        log_every: int = 50,
        stride: int = 1,
    ) -> None:
        self.settings = settings
        self.source = source
        self.json_path = json_path
        self.log_every = log_every
        self.stride = stride
        self.width = 0
        self.height = 0
        self.generation = 0
        self.frames: FrameRing | None = None
        self.masks: MaskRing | None = None
        self.frame_pool: SlotPool | None = None
        self.mask_pool: SlotPool | None = None
        self.track = None
        self.viaduct: ViaductRuntime | None = None
        self.telemetry: FrameTelemetry | None = None
        self.json_stream = None
        self.jobs_segment: Any = None
        self.jobs_detect: Any = None
        self.results: Any = None
        self.stop: Any = None
        self.workers: dict[str, Any] = {}
        self.pending: dict[int, _Pending] = {}
        # The next frame, decoded but not yet dispatched to the ring.
        self.prefetched: FramePacket | None = None
        # Frames whose deck ids are ready and that are waiting for the detector,
        # oldest first.  It is a queue, not a slot: the segmenter must be able to
        # run a frame ahead of the detector, otherwise the detector starves.
        self.ready: deque[_Pending] = deque()
        self.segment_busy = False
        self.detect_busy = False
        self.source_done = False

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> Self:
        self.width = self.source.width
        self.height = self.source.height
        slots = self.settings.shm_slots
        self.frames = FrameRing.allocate(
            slots=slots, height=self.height, width=self.width
        )
        self.masks = MaskRing.allocate(
            slots=slots,
            height=self.height,
            width=self.width,
            instances=MASK_INSTANCES,
        )
        self.frame_pool = SlotPool(slots)
        self.mask_pool = SlotPool(slots)
        self.track = build_track_stage(self.settings)
        self.viaduct = ViaductRuntime(
            speed=VehicleSpeedRuntime(
                frame_width=self.width,
                frame_height=self.height,
                car_width_m=self.settings.car_width_m,
            )
        )
        self.telemetry = FrameTelemetry(
            log_every=self.log_every,
            total_frames=self.source.frame_count or None,
            stride=self.stride,
        )
        self.json_stream = self.json_path.open("w", encoding="utf-8")
        self._start_workers()
        return self

    def __exit__(self, *exception: object) -> None:
        self._stop_workers()
        for ring in (self.masks, self.frames):
            if ring is not None:
                ring.unlink()
        if self.json_stream is not None:
            self.json_stream.close()

    def _start_workers(self) -> None:
        # `fork` when it exists: a worker then inherits this process's imports
        # instead of paying for them again (importing ultralytics alone is 1.5 s
        # here).  That is only safe because the coordinator never imports torch
        # or touches CUDA -- no context can be inherited mid-initialisation.
        context = mp.get_context(
            "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        )
        self.jobs_segment = context.Queue()
        self.jobs_detect = context.Queue()
        self.results = context.Queue()
        self.stop = context.Event()
        segment_card, detect_card = stage_devices(self.settings)
        assert self.frames is not None and self.masks is not None
        specs = (
            ("segment", parallel_worker.serve_segment, segment_card),
            ("detect", parallel_worker.serve_detect, detect_card),
        )
        jobs = {"segment": self.jobs_segment, "detect": self.jobs_detect}
        for name, target, card in specs:
            process = context.Process(
                target=target,
                args=(
                    self.settings,
                    card,
                    self.frames.spec,
                    self.masks.spec,
                    jobs[name],
                    self.results,
                    self.stop,
                ),
                name=f"{name}-worker",
                daemon=True,
            )
            process.start()
            self.workers[name] = process

    def _stop_workers(self) -> None:
        if self.stop is not None:
            self.stop.set()
        for process in self.workers.values():
            process.join(_SHUTDOWN_S)
        for name, process in self.workers.items():
            if process.is_alive():
                process.terminate()
                process.join(_SHUTDOWN_S)
                print(f"warning: {name} worker did not stop and was terminated")

    # -- the loop ----------------------------------------------------------
    def consume(self) -> None:
        started_at = perf_counter()
        while True:
            self._pump_detect()
            # Read one frame ahead of the segmenter.  Decoding takes ~7 ms and
            # the handoff needs none of it: prefetching here means the decode
            # overlaps the frame already in the segmenter instead of delaying
            # the next handoff by the whole decode time.
            if self.prefetched is None and not self.source_done:
                self.prefetched = self._read_eligible()
                if self.prefetched is None:
                    self.source_done = True
            if (
                not self.segment_busy
                and self.prefetched is not None
                and self.frame_pool.has_free()
            ):
                packet = self.prefetched
                self.prefetched = None
                self._dispatch_segment(packet)
            if self.source_done and not self.pending:
                break
            self._handle(self._wait_result())
        assert self.telemetry is not None
        self.telemetry.summary(elapsed_s=perf_counter() - started_at)

    def _read_eligible(self) -> FramePacket | None:
        """The next frame worth running inference on, or ``None`` at the end."""
        while True:
            packet = self.source.read()
            if packet is None:
                return None
            if packet.inference_eligible:
                return packet

    def _dispatch_segment(self, packet: FramePacket) -> None:
        assert self.frames and self.frame_pool and self.mask_pool
        frame_slot = self.frame_pool.take()
        mask_slot = self.mask_pool.take()
        self.frames.write(frame_slot, packet.image)
        # One extra reference: the segmentation worker holds the frame until its
        # result arrives.  The mask slot needs none -- the segmentation worker
        # only writes it, and the coordinator's own reference covers that.
        self.frame_pool.retain(frame_slot)
        pending = _Pending(
            sequence=packet.frame_index,
            timestamp_ms=packet.timestamp_ms,
            frame_slot=frame_slot,
            mask_slot=mask_slot,
            started=perf_counter(),
        )
        self.pending[pending.sequence] = pending
        self.segment_busy = True
        self.jobs_segment.put(
            FrameJob(
                sequence=pending.sequence,
                timestamp_ms=pending.timestamp_ms,
                generation=self.generation,
                frame_slot=frame_slot,
                mask_slot=mask_slot,
            )
        )

    def _pump_detect(self) -> None:
        """Hand the oldest frame waiting on deck ids to the detection worker."""
        if self.detect_busy or not self.ready:
            return
        assert self.mask_pool
        pending = self.ready.popleft()
        self.detect_busy = True
        self.mask_pool.retain(pending.mask_slot)
        self.jobs_detect.put(
            FrameJob(
                sequence=pending.sequence,
                timestamp_ms=pending.timestamp_ms,
                generation=self.generation,
                frame_slot=pending.frame_slot,
                mask_slot=pending.mask_slot,
            )
        )

    def _wait_result(self) -> SegmentResult | DetectResult:
        while True:
            try:
                return self.results.get(timeout=_RESULT_POLL_S)
            except Empty:
                for name, process in self.workers.items():
                    if not process.is_alive():
                        raise WorkerDied(
                            f"{name} worker died (exit code {process.exitcode})"
                        )

    def _handle(self, result: SegmentResult | DetectResult) -> None:
        pending = self.pending.get(result.sequence)
        if pending is None:
            raise RuntimeError(f"result for an unknown frame: {result.sequence}")
        if isinstance(result, SegmentResult):
            self._handle_segment(pending, result)
        else:
            self._handle_detect(pending, result)

    def _handle_segment(self, pending: _Pending, result: SegmentResult) -> None:
        assert self.frame_pool and self.track
        self.segment_busy = False
        # The segmenter is done with the pixels; the detector still needs them.
        self.frame_pool.release(pending.frame_slot)
        pending.segment_ms = result.segment_ms
        if result.error is not None:
            self._finish_invalid(pending, (result.error,))
            return

        # Instance i is plane i + 1 of the mask slot; `Mask` copies, so nothing
        # in the observation aliases a slot the ring may reuse.
        pending.viaduct = tuple(
            DetectedObject(
                label=item.label,
                confidence=item.confidence,
                box=item.box,
                mask=Mask(self.masks.plane(pending.mask_slot, index + 1)),  # type: ignore[union-attr]
            )
            for index, item in enumerate(result.instances)
        )
        # Deck ids are decided before detection is dispatched, which is the
        # serial order: a deck-tracker failure must win over a detector one.
        try:
            pending.viaduct_tracks = self.track.viaduct(
                pending.timestamp_ms, pending.viaduct
            )
        except Exception:  # noqa: BLE001
            self._finish_invalid(pending, ("viaduct_tracker_error",))
            return
        if not result.has_deck:
            self._complete(pending, self._observation(pending, not_detected=True))
            return
        self.ready.append(pending)

    def _handle_detect(self, pending: _Pending, result: DetectResult) -> None:
        assert self.mask_pool and self.track
        self.detect_busy = False
        self.mask_pool.release(pending.mask_slot)
        # Keep the detection card busy while this frame is tracked and written:
        # the next frame's masks are already waiting for it.
        self._pump_detect()
        if result.vehicle_error is not None:
            self._finish_invalid(
                pending, (result.vehicle_error,), detect_ms=result.detect_ms
            )
            return
        try:
            vehicle_tracks = self.track.vehicles(
                pending.timestamp_ms,
                result.vehicle,
                width=self.width,
                height=self.height,
                generation=self.generation,
            )
        except Exception:  # noqa: BLE001
            self._finish_invalid(
                pending, ("tracker_error",), detect_ms=result.detect_ms
            )
            return
        if result.line_error is not None:
            self._finish_invalid(
                pending,
                (result.line_error,),
                detect_ms=result.detect_ms,
                line_ms=result.line_ms,
            )
            return
        try:
            line_tracks = self.track.lines(
                pending.timestamp_ms,
                result.line,
                width=self.width,
                height=self.height,
                generation=self.generation,
            )
        except Exception:  # noqa: BLE001
            self._finish_invalid(
                pending,
                ("line_tracker_error",),
                detect_ms=result.detect_ms,
                line_ms=result.line_ms,
            )
            return
        self._complete(
            pending,
            self._observation(
                pending,
                vehicle=result.vehicle,
                vehicle_tracks=vehicle_tracks,
                line_tracks=line_tracks,
                detect_ms=result.detect_ms,
                line_ms=result.line_ms,
            ),
        )

    # -- observation assembly ----------------------------------------------
    def _observation(
        self,
        pending: _Pending,
        *,
        vehicle: tuple[DetectedObject, ...] = (),
        vehicle_tracks: tuple[TrackedObject, ...] = (),
        line_tracks: tuple[TrackedObject, ...] = (),
        not_detected: bool = False,
        detect_ms: float = 0.0,
        line_ms: float = 0.0,
    ) -> Observation:
        """The valid-frame shape, mirroring ``ObservationPipeline.process``."""
        return Observation(
            timestamp_ms=pending.timestamp_ms,
            frame_sequence=pending.sequence,
            generation=self.generation,
            frame_valid=True,
            viaduct=pending.viaduct,
            viaduct_tracks=pending.viaduct_tracks,
            vehicle=vehicle,
            vehicle_tracks=vehicle_tracks,
            line_tracks=line_tracks,
            unknown_reasons=("viaduct_not_detected",) if not_detected else (),
            diagnostics=Diagnostics(
                segment_ms=pending.segment_ms,
                detect_ms=detect_ms,
                line_ms=line_ms,
                processing_ms=(perf_counter() - pending.started) * 1000.0,
            ),
        )

    def _finish_invalid(
        self,
        pending: _Pending,
        unknown_reasons: tuple[str, ...],
        *,
        detect_ms: float = 0.0,
        line_ms: float = 0.0,
    ) -> None:
        self._complete(
            pending,
            invalid_observation(
                pending.timestamp_ms,
                pending.sequence,
                self.generation,
                unknown_reasons,
                processing_ms=(perf_counter() - pending.started) * 1000.0,
                segment_ms=pending.segment_ms,
                detect_ms=detect_ms,
                line_ms=line_ms,
            ),
        )

    def _complete(self, pending: _Pending, observation: Observation) -> None:
        """Record a verdict, then retire every frame that is now in order.

        Retiring out of order would hand the trackers and the speed runtime a
        timestamp that does not advance, so a finished frame waits its turn.
        """
        pending.observation = observation
        while self.pending:
            earliest = self.pending[min(self.pending)]
            if earliest.observation is None:
                return
            self._retire(earliest)

    def _retire(self, pending: _Pending) -> None:
        """Free the frame's slots, then measure it and write it out."""
        assert self.frame_pool and self.mask_pool and self.viaduct and self.telemetry
        assert pending.observation is not None
        del self.pending[pending.sequence]
        self.frame_pool.release(pending.frame_slot)
        self.mask_pool.release(pending.mask_slot)
        viaduct = self.viaduct.update(pending.observation)
        self.telemetry.observe(pending.observation, viaduct)
        self.json_stream.write(
            json.dumps(record(pending.observation, viaduct), ensure_ascii=False) + "\n"
        )


def run_parallel(
    settings: PipelineSettings,
    source: FrameSource,
    json_path: Path,
    *,
    log_every: int = 50,
    stride: int = 1,
) -> None:
    """Run the split pipeline to the end of ``source``; raises on worker death."""
    # A worker imports ultralytics itself; with a reachable-looking network that
    # import blocks for ~20 s on its online check, which would dominate startup.
    # The models are local files, so the offline path is the correct one here;
    # an explicit YOLO_OFFLINE in the environment still wins.
    os.environ.setdefault("YOLO_OFFLINE", "True")
    source.open()
    try:
        with Coordinator(
            settings, source, json_path, log_every=log_every, stride=stride
        ) as coordinator:
            coordinator.consume()
    finally:
        source.close()
