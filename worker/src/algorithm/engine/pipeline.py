"""The parent side of the engines: one frame in, one Observation out.

Three worker processes own the models (:mod:`algorithm.engine.workers`); this
object owns everything around them -- the ring of shared-memory slots, the
queues, the order frames are assembled in, and the failure paths.

``submit`` is non-blocking until ``window`` frames are in flight (that bound is
the backpressure), ``result`` returns the next observation in submission order,
and ``run`` does both while keeping the ring full, so throughput is set by the
slowest stage instead of by the sum of all three.

The parent loads no model and never touches CUDA, which is what lets it fork.
"""

from __future__ import annotations

import multiprocessing as _mp
import queue as _queue
import sys
import threading
from collections import deque
from dataclasses import dataclass
from time import monotonic, perf_counter
from typing import Any, Iterable, Iterator

import numpy as np

from algorithm.config import MpPipelineSettings, PipelineSettings
from algorithm.domain.geometry import Box, Mask
from algorithm.domain.observations import DetectedObject, Observation
from algorithm.engine import workers
from algorithm.engine.packet import FrameGeometryChanged, FramePacket
from algorithm.engine.shared import (
    LINE,
    SEGMENT,
    VEHICLE,
    DetectionDone,
    SegmentDone,
    Slot,
    WorkerFailed,
    WorkerReady,
    frame_view,
    mask_view,
)
from algorithm.perception.assembly import ObservationAssembler, StageResult

_READY_POLL_S = 0.2
_RESULT_POLL_S = 0.2


# A detector that never ran (failed or empty segment) contributes nothing:
# the stand-in keeps _assemble free of `if parts.vehicle else None` chains.
_NO_DETECTION = DetectionDone(
    role="", seq=-1, slot=-1, detections=(), elapsed_ms=0.0, error=None
)


@dataclass
class _Parts:
    """What has arrived for one in-flight frame."""

    timestamp_ms: int
    generation: int
    seg: SegmentDone | None = None
    vehicle: DetectionDone | None = None
    line: DetectionDone | None = None

    @property
    def vehicle_done(self) -> DetectionDone:
        """The vehicle detector's message, or an empty stand-in if it never ran."""
        return _NO_DETECTION if self.vehicle is None else self.vehicle

    @property
    def line_done(self) -> DetectionDone:
        """The line detector's message, or an empty stand-in if it never ran."""
        return _NO_DETECTION if self.line is None else self.line

    def ready(self) -> bool:
        if self.seg is None:
            return False
        # A failed frame and an empty deck never reach the detectors.
        if self.seg.error is not None or self.seg.empty:
            return True
        return self.vehicle is not None and self.line is not None


def _as_error(message: str | None) -> BaseException | None:
    """A worker sends the traceback text, not the exception object.

    ``StageResult`` carries the cause so the assembler can report it; over a
    process boundary there is no object to carry, so the text is re-wrapped.
    """
    return None if message is None else RuntimeError(message)


def _frame_error(frame: object) -> str | None:
    """What ``frame`` is missing to be a usable BGR image, or ``None`` if it is."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.size == 0:
        return "an H x W x 3 frame"
    if frame.shape[2] != 3:
        return "3 channels (BGR)"
    if frame.dtype != np.uint8:
        return "uint8 pixels"
    return None


def _start_method() -> str:
    """``fork`` when the parent is CUDA-free (fast start), ``spawn`` otherwise.

    A forked child would inherit a live CUDA context and could not re-pin its
    device, so spawn is mandatory once the parent has touched CUDA -- and on
    Windows, which has no fork.
    """
    if sys.platform == "win32" or "fork" not in _mp.get_all_start_methods():
        return "spawn"
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        return "spawn"
    return "fork"


def _warm_tracker_stack() -> None:
    """Import the tracking stack (ultralytics -> torch) before the first frame.

    The parent needs ``ultralytics.trackers.byte_tracker`` to build ByteTrack,
    and that import is the single most expensive thing the parent does (~20 s on
    a cold page cache; it pulls in torch).  It cannot happen before the workers
    are forked -- the children must set ``CUDA_VISIBLE_DEVICES`` before *their*
    first torch import -- so it is warmed in a background thread that overlaps
    the workers' model load.  Importing torch alone does not create a CUDA
    context, so a later fork is still safe.
    """
    try:
        import ultralytics.engine.results  # noqa: F401, PLC0415
        import ultralytics.trackers.byte_tracker  # noqa: F401, PLC0415
    except ImportError:
        # Let the real tracker construction report the missing dependency with
        # its own traceback instead of hiding it behind the warm-up.
        pass


class MultiProcessPipeline:
    """Three model processes, one GPU each, assembled by the calling process.

    ``submit`` is non-blocking until ``window`` frames are in flight (that bound
    is the backpressure), ``result`` returns the next observation in submission
    order, and :meth:`run` does both: it keeps the ring full, so throughput is
    set by the slowest stage rather than by the sum of all three.

    The trackers arrive pre-built as a :class:`ObservationAssembler`: the parent never
    loads a model, so it never touches CUDA, and the caller owns the one object
    that holds all cross-frame state.
    """

    def __init__(
        self,
        settings: PipelineSettings,
        mp_settings: MpPipelineSettings,
        *,
        fps: float,
        assembler: ObservationAssembler,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")

        self.settings = settings
        self.mp = mp_settings
        self.fps = fps
        self.assembler = assembler

        self._height = 0
        self._width = 0
        self._slots: list[Slot] = []
        self._slot_names: tuple[tuple[str, str, str], ...] = ()
        self._free_slots: _queue.Queue[int] = _queue.Queue()
        self._parts: dict[int, _Parts] = {}
        self._order: deque[int] = deque()
        self._order_lock = threading.Lock()
        self._results: _queue.Queue[Observation] = _queue.Queue()
        self._processes: list[_mp.process.BaseProcess] = []
        self._post_thread: threading.Thread | None = None
        self._stop: Any = None
        # The queues exist only between start() and close(), but close() must
        # also work on a pipeline that never started, so they are declared
        # here and start() fills them in.
        self._q_segment: Any = None
        self._q_vehicle: Any = None
        self._q_line: Any = None
        self._q_out: Any = None
        self._failure: BaseException | None = None
        self._closed = False

    # -- lifecycle ---------------------------------------------------------
    @property
    def window(self) -> int:
        return self.mp.window

    @property
    def started(self) -> bool:
        return bool(self._processes)

    def start(self, height: int, width: int) -> None:
        """Allocate the ring and start the workers; called by the first submit."""
        if self.started:
            raise RuntimeError("pipeline already started")
        if self._closed:
            raise RuntimeError("pipeline is closed")
        if height <= 0 or width <= 0:
            raise ValueError("frame geometry must be positive")
        self._height, self._width = int(height), int(width)
        self._allocate_ring()
        self._spawn_workers()

        loading = monotonic()
        warmer = threading.Thread(
            target=_warm_tracker_stack, name="mp-warmup", daemon=True
        )
        warmer.start()
        try:
            self._wait_ready()
            warmer.join()
        except BaseException:
            # A worker that cannot load its model leaves processes and shared
            # memory behind otherwise; surface the error with a clean slate.
            self.close()
            raise
        print(
            f"[mp] {len(self._processes)} workers ready in {monotonic() - loading:.1f}s"
            f" ({self._width}x{self._height}, window {self.window})",
            flush=True,
        )
        self._post_thread = threading.Thread(
            target=self._post_loop, name="mp-assembler", daemon=True
        )
        self._post_thread.start()

    def _allocate_ring(self) -> None:
        """The shared-memory slots and the queues that connect the processes."""
        self._slots = [
            Slot(height=self._height, width=self._width, mask_capacity=self.mp.mask_capacity)
            for _ in range(self.window)
        ]
        for index in range(self.window):
            self._free_slots.put(index)
        self._slot_names = tuple(slot.names for slot in self._slots)

        context = _mp.get_context(_start_method())
        self._q_segment = context.Queue(maxsize=self.window)
        self._q_vehicle = context.Queue(maxsize=self.window)
        self._q_line = context.Queue(maxsize=self.window)
        self._q_out = context.Queue(maxsize=self.window * 4)
        self._stop = context.Event()

    def _spawn_workers(self) -> None:
        """Start one process per spec: the segment roles, then both detectors."""
        specs = [
            self._worker_spec(
                SEGMENT, gpu, f"mp-segment-{index}-gpu{gpu}", self._q_segment
            )
            for index, gpu in enumerate(self.mp.segment_gpus)
        ]
        specs += [
            self._worker_spec(role, gpu, f"mp-{role}-gpu{gpu}", queue)
            for role, gpu, queue in (
                (VEHICLE, self.mp.vehicle_gpu, self._q_vehicle),
                (LINE, self.mp.line_gpu, self._q_line),
            )
        ]
        context = _mp.get_context(_start_method())
        for worker in specs:
            process = context.Process(
                target=workers.run_worker, args=(worker,), name=worker.name, daemon=True
            )
            process.start()
            self._processes.append(process)

    def _worker_spec(
        self, role: str, gpu: int, name: str, q_in: Any
    ) -> workers.WorkerSpec:
        """One worker's inputs; the segment role also feeds both detectors."""
        segment = role == SEGMENT
        return workers.WorkerSpec(
            role=role,
            gpu=gpu,
            name=name,
            settings=self.settings,
            names=self._slot_names,
            height=self._height,
            width=self._width,
            q_in=q_in,
            q_out=self._q_out,
            stop=self._stop,
            q_vehicle=self._q_vehicle if segment else None,
            q_line=self._q_line if segment else None,
            capacity=self.mp.mask_capacity if segment else 0,
        )

    def close(self) -> None:
        """Stop the workers, drop in-flight frames and free the shared memory."""
        if self._closed:
            return
        self._closed = True
        if self._stop is not None:
            self._stop.set()
        for queue in (self._q_segment, self._q_vehicle, self._q_line):
            if queue is None:  # never started, or already closed
                continue
            try:
                queue.put_nowait(None)
            except Exception:  # noqa: BLE001 - a full queue still stops on the event
                pass
        for process in self._processes:
            process.join(timeout=15.0)
        for process in self._processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        if self._post_thread is not None:
            self._post_thread.join(timeout=5.0)
            self._post_thread = None
        for queue in (self._q_segment, self._q_vehicle, self._q_line, self._q_out):
            if queue is None:
                continue
            try:
                queue.close()
                queue.join_thread()
            except Exception:  # noqa: BLE001
                pass
        self._q_segment = self._q_vehicle = self._q_line = self._q_out = None
        for slot in self._slots:
            slot.close_unlink()
        self._slots = []
        self._processes = []
        with self._order_lock:
            self._parts.clear()
            self._order.clear()

    # -- frame API ---------------------------------------------------------
    def submit(
        self,
        frame: np.ndarray,
        *,
        timestamp_ms: int,
        frame_sequence: int,
        generation: int = 0,
    ) -> None:
        """Queue one frame; blocks while ``window`` frames are already in flight."""
        # One validation pass serves both the start-on-first-submit path and
        # the geometry check below.
        problem = _frame_error(frame)
        if not self.started:
            if problem is not None:
                raise ValueError(f"multi-process pipeline needs {problem}")
            self.start(int(frame.shape[0]), int(frame.shape[1]))
        if problem is not None or (frame.shape[0], frame.shape[1]) != (
            self._height,
            self._width,
        ):
            detail = (
                f"{frame.shape} {frame.dtype}"
                if isinstance(frame, np.ndarray)
                else f"a {type(frame).__name__}"
            )
            raise FrameGeometryChanged(
                f"frame {detail} does not match the started geometry "
                f"{(self._height, self._width)} uint8"
            )
        # Acquire the slot before publishing the order: a blocked submit must not
        # leave a half-registered frame behind.
        slot = self._free_slots.get()
        frame_view(self._slots[slot].frame.buf, self._height, self._width)[:] = frame
        with self._order_lock:
            if frame_sequence in self._parts:  # pragma: no cover - defensive
                self._free_slots.put(slot)
                raise ValueError("frame_sequence must be unique among in-flight frames")
            self._parts[frame_sequence] = _Parts(
                timestamp_ms=timestamp_ms, generation=generation
            )
            self._order.append(frame_sequence)
        self._q_segment.put((frame_sequence, slot))

    def result(self) -> Observation:
        """Next observation in submission order; blocks until it is assembled."""
        if not self.started and not self._closed:
            raise RuntimeError("pipeline has not been started; submit a frame first")
        while True:
            try:
                return self._results.get(timeout=_RESULT_POLL_S)
            except _queue.Empty:
                self._raise_if_failed()
                if self._closed:
                    raise RuntimeError("pipeline is closed")

    def run(
        self, frames: Iterable[FramePacket]
    ) -> Iterator[tuple[np.ndarray, Observation]]:
        """Submit ahead by ``window`` frames and yield ``(frame, observation)`` pairs.

        Frames are consumed in the order given; the producer side keeps the ring
        full, which is what turns three sequential stages into a pipeline.  The
        packet is kept until its observation arrives, because ``(frame,
        observation)`` has to be yielded in submission order.
        """
        iterator = iter(frames)
        queued: deque[FramePacket] = deque()
        while True:
            self._fill(iterator, queued)
            if not queued:
                return
            observation = self.result()
            yield queued.popleft().frame, observation

    # -- internals ---------------------------------------------------------
    def _fill(
        self, iterator: Iterator[FramePacket], queued: deque[FramePacket]
    ) -> None:
        """Submit packets until ``queued`` holds ``window`` frames or none are left."""
        while len(queued) < self.window:
            packet = next(iterator, None)
            if packet is None:
                return
            self.submit(
                packet.frame,
                timestamp_ms=packet.timestamp_ms,
                frame_sequence=packet.frame_index,
            )
            queued.append(packet)

    def _wait_ready(self) -> None:
        # Roles repeat when several segment workers share the role, so readiness
        # is counted (one message per process), not de-duplicated by name.
        ready: list[str] = []
        expected = len(self._processes)
        deadline = monotonic() + self.mp.start_timeout_s
        while len(ready) < expected:
            try:
                message = self._q_out.get(timeout=_READY_POLL_S)
            except _queue.Empty:
                dead = [p.name for p in self._processes if p.exitcode is not None]
                if dead:
                    raise RuntimeError(f"worker(s) exited during startup: {dead}")
                if monotonic() > deadline:
                    raise TimeoutError(
                        f"{len(ready)}/{expected} workers ready within "
                        f"{self.mp.start_timeout_s:.0f}s: {ready}"
                    )
                continue
            if isinstance(message, WorkerReady):
                ready.append(message.role)
            elif isinstance(message, WorkerFailed):
                raise RuntimeError(
                    f"worker {message.role!r} failed to start:\n{message.traceback}"
                )
            else:
                raise RuntimeError(f"unexpected startup message {message!r}")

    def _post_loop(self) -> None:
        """Collect worker messages and publish assembled observations in order."""
        while not self._stop.is_set():
            try:
                message = self._q_out.get(timeout=0.1)
            except _queue.Empty:
                continue
            if isinstance(message, WorkerFailed):
                self._fail(
                    RuntimeError(
                        f"worker {message.role!r} crashed:\n{message.traceback}"
                    )
                )
                return
            if isinstance(message, SegmentDone):
                with self._order_lock:
                    parts = self._parts.get(message.seq)
                    if parts is None:
                        continue
                    parts.seg = message
            elif isinstance(message, DetectionDone):
                with self._order_lock:
                    parts = self._parts.get(message.seq)
                    if parts is None:
                        continue
                    if message.role == VEHICLE:
                        parts.vehicle = message
                    elif message.role == LINE:
                        parts.line = message
            else:  # pragma: no cover - no other message type is sent
                continue
            try:
                self._drain_in_order()
            except BaseException as exc:  # noqa: BLE001 - surfaced through result()
                self._fail(exc)
                return

    def _fail(self, failure: BaseException) -> None:
        """Record why the pipeline stopped; ``result()`` re-raises it to the caller."""
        self._failure = failure
        self._stop.set()

    def _drain_in_order(self) -> None:
        """Assemble every leading frame whose parts are complete; raises on failure."""
        while True:
            with self._order_lock:
                if not self._order:
                    return
                seq = self._order[0]
                parts = self._parts[seq]
                if not parts.ready():
                    return
            observation = self._assemble(seq, parts)
            with self._order_lock:
                self._order.popleft()
                del self._parts[seq]
            assert parts.seg is not None
            self._free_slots.put(parts.seg.slot)
            self._results.put(observation)

    def _assemble(self, seq: int, parts: _Parts) -> Observation:
        assert parts.seg is not None
        started = perf_counter()
        seg = parts.seg
        vehicle = parts.vehicle_done
        line = parts.line_done
        stages = StageResult(
            viaduct=self._viaduct(seg),
            vehicle=vehicle.detections,  # type: ignore[arg-type]
            line=line.detections,  # type: ignore[arg-type]
            segment_ms=seg.segment_ms,
            detect_ms=vehicle.elapsed_ms,
            line_ms=line.elapsed_ms,
            viaduct_error=_as_error(seg.error),
            vehicle_error=_as_error(vehicle.error),
            line_error=_as_error(line.error),
            deck_empty=seg.empty,
        )
        # "processing_ms" is this frame's critical-path work, not its queueing:
        # the same quantity a per-frame (non-pipelined) reader would report.
        processing_ms = (
            seg.segment_ms
            + max(vehicle.elapsed_ms, line.elapsed_ms)
            + (perf_counter() - started) * 1000.0
        )
        return self.assembler.assemble(
            parts.timestamp_ms,
            seq,
            parts.generation,
            height=self._height,
            width=self._width,
            stages=stages,
            processing_ms=processing_ms,
        )

    def _viaduct(self, seg: SegmentDone) -> tuple[DetectedObject, ...]:
        """Rebuild the instances, copying their masks out of the shared slot."""
        if not seg.count:
            return ()
        slot = self._slots[seg.slot]
        label = self.settings.viaduct_label
        return tuple(
            DetectedObject(
                label=label,
                confidence=seg.confidences[index],
                box=Box(*seg.boxes[index]),
                # Mask() copies (and freezes) the bytes, so the slot can be
                # reused by a later frame while this observation is alive.
                mask=Mask(
                    mask_view(slot.masks.buf, index, self._height, self._width)
                ),
            )
            for index in range(seg.count)
        )

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            failure, self._failure = self._failure, None
            raise RuntimeError(f"multi-process pipeline failed: {failure}") from failure
        dead = [
            f"{process.name} (exit {process.exitcode})"
            for process in self._processes
            if process.exitcode not in (None, 0)
        ]
        if dead:
            raise RuntimeError(f"worker process died: {', '.join(dead)}")
