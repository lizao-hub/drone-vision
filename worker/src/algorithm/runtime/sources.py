"""Frame sources: OpenCV capture in, :class:`FramePacket` out.

There are exactly two: ``FileSequenceSource`` for one local video and
``RtspSource`` for a live feed.  Both implement the same small lifecycle --
``open()`` once, ``read()`` from exactly one reader until it stops, ``close()``
after that reader exits -- and neither owns a thread.  A controller therefore
decides how to pull: synchronously (``batch`` / ``pipeline``) or through a reader
thread that keeps only the newest frame
(:class:`algorithm.runtime.live.ThreadedSource`).

``FileSequenceSource`` reads one file and marks sampled frames as
inference-eligible instead of dropping them, so the timeline stays dense while
inference runs on a subset.  ``RtspSource`` reconnects with exponential backoff
and reports its connection state; a failed read is not an exception but an
explicit ``None`` while the backoff is pending.

``open_capture`` and ``make_packet`` are the two pieces both sources share, so the
transport policy, the buffer size and the strictly-increasing timestamp exist in
one place.  ``capture_factory`` stays injectable so tests can drive the sources
with a fake capture and no video files.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from threading import Event
from time import monotonic, monotonic_ns
from typing import Protocol, cast
from uuid import uuid4

import cv2
import numpy as np

from algorithm.runtime.frame_packet import FramePacket

DEFAULT_FPS = 25.0


class Capture(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def get(self, property_id: int) -> float: ...
    def set(self, property_id: int, value: float) -> bool: ...
    def release(self) -> None: ...


class FrameSource(Protocol):
    """Controller opens once; its reader owns read/close until it exits.

    Sources may expose ``request_stop()`` to interrupt waits without closing a
    native resource.  Standalone callers must likewise serialize read and close.
    """

    source_id: str

    def open(self) -> None: ...
    def read(self) -> FramePacket | None: ...
    def close(self) -> None: ...


CaptureFactory = Callable[[str], Capture]


def _default_capture_factory(source: str) -> Capture:
    return cast(Capture, cv2.VideoCapture(source))


def _strict_timestamp(previous_ms: int | None, candidate_ms: int) -> int:
    """Trackers reject non-advancing time, so a source never emits it."""
    if previous_ms is None:
        return candidate_ms
    return max(candidate_ms, previous_ms + 1)


def open_capture(
    source: str,
    *,
    rtsp_transport: str | None = None,
    capture_factory: CaptureFactory = _default_capture_factory,
) -> Capture:
    """Open a capture with the demo's transport and buffer policy.

    The buffer is pinned to one frame: a backlog in the decoder is latency, not
    content, and every consumer here wants the newest frame.
    """
    if rtsp_transport is not None:
        # OpenCV reads this when the capture is created, not per frame.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;{rtsp_transport}",
        )
    capture = capture_factory(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def make_packet(
    *,
    source_id: str,
    session_id: str,
    frame_index: int,
    timestamp_ms: int,
    image: np.ndarray,
    inference_eligible: bool = True,
) -> FramePacket:
    """The one place a :class:`FramePacket` is built from a decoded frame."""
    height, width = image.shape[:2]
    return FramePacket(
        source_id=source_id,
        session_id=session_id,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        image=image,
        original_width=int(width),
        original_height=int(height),
        inference_eligible=inference_eligible,
    )


class FileSequenceSource:
    """Read one local video as a strictly-increasing frame stream."""

    source_id = "file-sequence"

    def __init__(
        self,
        path: Path,
        *,
        capture_factory: CaptureFactory = _default_capture_factory,
        session_id: str | None = None,
        frame_stride: int = 1,
    ) -> None:
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        self.path = Path(path)
        self.capture_factory = capture_factory
        self.session_id = session_id or f"file-{uuid4().hex}"
        self.frame_stride = frame_stride
        self._capture: Capture | None = None
        self._frame_index = 0
        self._last_timestamp_ms: int | None = None
        self._stop_requested = Event()
        self._opened = False
        self._fps = DEFAULT_FPS
        self._width = 0
        self._height = 0
        self._frame_count = 0
        self.exhausted = False

    def open(self) -> None:
        self.close()
        self._stop_requested.clear()
        self._frame_index = 0
        self._last_timestamp_ms = None
        self.exhausted = False
        self._opened = True
        try:
            self._open_capture()
        except Exception:
            self._opened = False
            raise

    def read(self) -> FramePacket | None:
        if self._stop_requested.is_set():
            return None
        if not self._opened:
            raise RuntimeError("file source is not open")
        capture = self._capture
        if capture is None:
            self.exhausted = True
            return None
        ok, image = capture.read()
        if self._stop_requested.is_set():
            return None
        if not ok or image is None:
            self.exhausted = True
            return None
        timestamp_ms = _strict_timestamp(
            self._last_timestamp_ms,
            max(0, round(capture.get(cv2.CAP_PROP_POS_MSEC))),
        )
        self._last_timestamp_ms = timestamp_ms
        frame_index = self._frame_index
        self._frame_index += 1
        return make_packet(
            source_id=self.source_id,
            session_id=self.session_id,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            image=image,
            inference_eligible=(frame_index % self.frame_stride == 0),
        )

    def request_stop(self) -> None:
        """Cooperative cancellation only; never touch the native capture here."""
        self._stop_requested.set()

    def close(self) -> None:
        self.request_stop()
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()
        self._opened = False

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _open_capture(self) -> None:
        capture = self.capture_factory(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open video source: {self.path}")
        self._capture = capture
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps > 0.0:
            self._fps = fps
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            self._frame_count = frame_count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > 0 and height > 0:
            self._width = width
            self._height = height


class RtspSource:
    """Pull a live RTSP stream, reconnecting with exponential backoff."""

    source_id = "rtsp"

    def __init__(
        self,
        url: str,
        *,
        capture_factory: CaptureFactory = _default_capture_factory,
        session_id: str | None = None,
        reconnect_initial_s: float = 0.5,
        reconnect_max_s: float = 8.0,
        rtsp_transport: str | None = "tcp",
    ) -> None:
        if not url:
            raise ValueError("RTSP URL is required")
        if reconnect_initial_s < 0.0:
            raise ValueError("reconnect_initial_s must be non-negative")
        if reconnect_max_s < reconnect_initial_s:
            raise ValueError(
                "reconnect_max_s cannot be shorter than reconnect_initial_s",
            )
        if rtsp_transport not in (None, "tcp", "udp"):
            raise ValueError("rtsp_transport must be 'tcp' or 'udp'")
        self._url = url
        self.capture_factory = capture_factory
        self.session_id = session_id or f"rtsp-{uuid4().hex}"
        self.reconnect_initial_s = reconnect_initial_s
        self.reconnect_max_s = reconnect_max_s
        self.rtsp_transport = rtsp_transport
        self._capture: Capture | None = None
        self._frame_index = 0
        self._last_timestamp_ms: int | None = None
        self._next_reconnect_at = 0.0
        self._reconnect_delay_s = reconnect_initial_s
        self._closed = True
        self.stream_generation = 0
        self.connection_state = "closed"
        self.last_error: str | None = None
        self.exhausted = False
        self._fps = DEFAULT_FPS

    def open(self) -> None:
        self.close()
        self._closed = False
        self.exhausted = False
        self._frame_index = 0
        self._last_timestamp_ms = None
        self._reconnect_delay_s = self.reconnect_initial_s
        self._next_reconnect_at = 0.0
        if not self._connect():
            self._closed = True
            raise RuntimeError("cannot open RTSP source")

    def read(self) -> FramePacket | None:
        capture = self._capture
        if capture is None:
            if self._closed:
                raise RuntimeError("RTSP source is not open")
            if monotonic() < self._next_reconnect_at:
                return None
            if not self._connect():
                return None
            capture = self._capture
            assert capture is not None
        ok, image = capture.read()
        if not ok or image is None:
            self._mark_disconnected("rtsp_read_failed")
            return None
        timestamp_ms = _strict_timestamp(
            self._last_timestamp_ms, monotonic_ns() // 1_000_000
        )
        self._last_timestamp_ms = timestamp_ms
        packet = make_packet(
            source_id=self.source_id,
            session_id=self.session_id,
            frame_index=self._frame_index,
            timestamp_ms=timestamp_ms,
            image=image,
        )
        self._frame_index += 1
        return packet

    @property
    def fps(self) -> float:
        """The stream's own frame rate, or the fallback until the first connect."""
        return self._fps

    def close(self) -> None:
        self._closed = True
        self.connection_state = "closed"
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()

    def _connect(self) -> bool:
        capture = open_capture(
            self._url,
            rtsp_transport=self.rtsp_transport,
            capture_factory=self.capture_factory,
        )
        if not capture.isOpened():
            capture.release()
            self._schedule_reconnect("rtsp_open_failed")
            return False
        self._capture = capture
        # The relay derives its output rate from this, so a bogus value (0, or
        # something absurd) must not replace the fallback.
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if 0.0 < fps < 240.0:
            self._fps = fps
        self.stream_generation += 1
        self.connection_state = "connected"
        self.last_error = None
        self._reconnect_delay_s = self.reconnect_initial_s
        self._next_reconnect_at = 0.0
        return True

    def _mark_disconnected(self, reason: str) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()
        self._schedule_reconnect(reason)

    def _schedule_reconnect(self, reason: str) -> None:
        self.connection_state = "reconnecting"
        self.last_error = reason
        self._next_reconnect_at = monotonic() + self._reconnect_delay_s
        self._reconnect_delay_s = min(
            self.reconnect_max_s,
            max(
                self.reconnect_initial_s,
                self._reconnect_delay_s * 2.0,
            ),
        )
