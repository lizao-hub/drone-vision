"""Live wrapper: pull a :class:`FrameSource` on a reader thread.

The sources own their lifecycle (``open``/``read``/``close``, reconnection, file
looping) and never spawn a thread.  This module adds the one policy the live path
needs: keep only the newest frame.  A pipeline that falls behind therefore drops
frames instead of accumulating latency, which is what keeps ``relay`` real time.
"""

from __future__ import annotations

import threading
from time import monotonic

from algorithm.runtime.frame_packet import FramePacket
from algorithm.runtime.sources import FrameSource

REOPEN_DELAY_S = 1.0
OPEN_ERROR_LOG_S = 10.0
# How long to wait before asking the source again while it reports "nothing yet"
# (an RTSP reconnect in flight).  The source owns the real backoff, so this only
# has to keep the loop from spinning.
IDLE_POLL_S = 0.2


class LatestFrameQueue:
    """Single-slot queue: ``put_latest`` overwrites whatever is still pending."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._packet: FramePacket | None = None
        self._closed = False
        self._dropped = 0

    def put_latest(self, packet: FramePacket) -> None:
        with self._condition:
            if self._closed:
                return
            if self._packet is not None:
                self._dropped += 1
            self._packet = packet
            self._condition.notify()

    def get(self, timeout: float | None = None) -> FramePacket | None:
        """Newest packet, or ``None`` on timeout / after :meth:`close`."""
        with self._condition:
            if self._packet is None and not self._closed:
                self._condition.wait(timeout)
            packet, self._packet = self._packet, None
            return packet

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._packet = None
            self._condition.notify_all()

    @property
    def dropped(self) -> int:
        with self._condition:
            return self._dropped


class ThreadedSource:
    """A :class:`FrameSource` pulled on a thread; the consumer sees the newest frame."""

    def __init__(
        self,
        source: FrameSource,
        *,
        reopen_delay_s: float = REOPEN_DELAY_S,
        idle_poll_s: float = IDLE_POLL_S,
    ) -> None:
        if reopen_delay_s <= 0.0:
            raise ValueError("reopen_delay_s must be positive")
        if idle_poll_s <= 0.0:
            raise ValueError("idle_poll_s must be positive")
        self.source = source
        self.source_id = source.source_id
        self.reopen_delay_s = reopen_delay_s
        self.idle_poll_s = idle_poll_s
        self.frames = LatestFrameQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_open_error_log = 0.0

    def start(self) -> "ThreadedSource":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._read_loop, name="threaded-source", daemon=True
        )
        self._thread.start()
        return self

    def get(self, timeout: float | None = None) -> FramePacket | None:
        return self.frames.get(timeout)

    def close(self) -> None:
        self._stop.set()
        self.frames.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def dropped(self) -> int:
        return self.frames.dropped

    @property
    def fps(self) -> float:
        """The wrapped source's own frame rate (0.0 when it cannot report one)."""
        return getattr(self.source, "fps", 0.0)

    # -- internals ---------------------------------------------------------
    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.source.open()
            except Exception:  # noqa: BLE001 - the source is not reachable yet
                self._log_open_error()
                self._stop.wait(self.reopen_delay_s)
                continue
            try:
                self._pump()
            finally:
                self.source.close()
            if not self._stop.is_set():
                self._stop.wait(self.reopen_delay_s)

    def _pump(self) -> None:
        while not self._stop.is_set():
            started = monotonic()
            packet = self.source.read()
            if packet is None:
                # ``None`` means "nothing for me right now".  A finite source (a
                # file that ran out) sets ``exhausted`` and the loop reopens it;
                # a live source backing off never sets it and just needs a pause
                # before the next attempt.
                if getattr(self.source, "exhausted", False):
                    return
                self._stop.wait(self.idle_poll_s)
                continue
            self.frames.put_latest(packet)
            # Pace the reader to the source's own frame rate so a decoder that
            # runs ahead of real time does not spin; the drop-oldest queue then
            # absorbs whatever the pipeline cannot keep up with.  A live stream
            # whose ``read()`` already blocks at the stream's pace collapses
            # this wait to zero.
            fps = getattr(self.source, "fps", 0.0)
            if fps > 0.0:
                self._stop.wait(max(0.0, (1.0 / fps) - (monotonic() - started)))

    def _log_open_error(self) -> None:
        now = monotonic()
        if now - self._last_open_error_log < OPEN_ERROR_LOG_S:
            return
        self._last_open_error_log = now
        print(f"live-source: cannot open {self.source_id!r}; retrying", flush=True)
