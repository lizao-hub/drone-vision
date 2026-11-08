"""Live frame intake: a reader thread feeding a drop-oldest queue.

A live source can deliver frames faster than the pipeline consumes them.  Blocking
on the newest frame instead of queueing all of them keeps the stream in real time:
the reader always overwrites the single slot, and the consumer sees the latest
frame plus a count of the frames it skipped.

The source accepts an RTSP URL or a local video file.  Files are replayed at their
own frame rate and loop when they end, so a local clip behaves like a live stream.
"""

from __future__ import annotations

import os
import threading
from time import monotonic, monotonic_ns, sleep
from uuid import uuid4

import cv2
import numpy as np

from algorithm.runtime.frame_packet import FramePacket

DEFAULT_FPS = 25.0
REOPEN_DELAY_S = 1.0
OPEN_ERROR_LOG_S = 10.0


class LatestFrameQueue:
    """Single-slot queue: ``put_latest`` overwrites whatever is still pending."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._packet: FramePacket | None = None
        self._closed = False
        self._dropped = 0
        self._received = 0

    def put_latest(self, packet: FramePacket) -> None:
        with self._condition:
            if self._closed:
                return
            if self._packet is not None:
                self._dropped += 1
            self._packet = packet
            self._received += 1
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

    @property
    def received(self) -> int:
        with self._condition:
            return self._received

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed


class LiveSource:
    """Reader thread over an RTSP URL or a video file.

    Frames are published into :class:`LatestFrameQueue`; timestamps are
    milliseconds since this source was created and are made strictly increasing,
    because the trackers reject non-advancing time.  A dropped connection (or the
    end of a file) reopens the capture after ``reopen_delay_s``.
    """

    def __init__(
        self,
        source: str,
        *,
        rtsp_transport: str = "tcp",
        reopen_delay_s: float = REOPEN_DELAY_S,
    ) -> None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty URL or file path")
        if rtsp_transport not in ("tcp", "udp"):
            raise ValueError("rtsp_transport must be 'tcp' or 'udp'")
        if reopen_delay_s <= 0.0:
            raise ValueError("reopen_delay_s must be positive")
        self.source = source.strip()
        self.rtsp_transport = rtsp_transport
        self.reopen_delay_s = reopen_delay_s
        self.frames = LatestFrameQueue()
        self._is_file = "://" not in self.source
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_ns = monotonic_ns()
        self._session_id = uuid4().hex
        self._last_timestamp_ms = -1
        self._frame_index = 0
        self._fps = DEFAULT_FPS
        self._last_open_error_log = 0.0

    def start(self) -> "LiveSource":
        if self._thread is not None:
            return self
        if not self._is_file:
            # OpenCV reads this when the capture is created, not per frame.
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                f"rtsp_transport;{self.rtsp_transport}",
            )
        self._thread = threading.Thread(
            target=self._read_loop, name="live-source", daemon=True
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
    def fps(self) -> float:
        return self._fps

    @property
    def dropped(self) -> int:
        return self.frames.dropped

    @property
    def received(self) -> int:
        return self.frames.received

    # -- internals ---------------------------------------------------------
    def _read_loop(self) -> None:
        while not self._stop.is_set():
            capture = cv2.VideoCapture(self.source)
            if not capture.isOpened():
                capture.release()
                self._log_open_error()
                sleep(self.reopen_delay_s)
                continue
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if 0.0 < fps < 240.0:
                self._fps = fps
            period = 1.0 / self._fps if self._is_file else 0.0
            while not self._stop.is_set():
                started = monotonic()
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                self._publish(frame)
                if period > 0.0:
                    sleep(max(0.0, period - (monotonic() - started)))
            capture.release()
            if not self._stop.is_set():
                sleep(self.reopen_delay_s)

    def _publish(self, frame: np.ndarray) -> None:
        elapsed_ms = int((monotonic_ns() - self._started_ns) // 1_000_000)
        timestamp_ms = max(self._last_timestamp_ms + 1, elapsed_ms)
        self._last_timestamp_ms = timestamp_ms
        height, width = frame.shape[:2]
        self.frames.put_latest(
            FramePacket(
                source_id=self.source,
                session_id=self._session_id,
                frame_index=self._frame_index,
                timestamp_ms=timestamp_ms,
                received_at_monotonic_ms=elapsed_ms,
                image=frame,
                original_width=int(width),
                original_height=int(height),
            )
        )
        self._frame_index += 1

    def _log_open_error(self) -> None:
        now = monotonic()
        if now - self._last_open_error_log < OPEN_ERROR_LOG_S:
            return
        self._last_open_error_log = now
        print(f"live-source: cannot open {self.source!r}; retrying", flush=True)
