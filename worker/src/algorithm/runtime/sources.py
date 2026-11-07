"""Frame sources: OpenCV capture in, :class:`FramePacket` out.

Every source implements the same small lifecycle: ``open()`` once, ``read()`` from
exactly one reader until it stops, ``close()`` after that reader exits.  Sources
never own a thread, so a controller can decide whether to pull synchronously
(batch) or wrap the source in a reader thread (see :mod:`algorithm.runtime.live`).

``FileSequenceSource`` replays one or more local files as one continuous timeline:
timestamps keep advancing across file boundaries, realtime pacing prevents a fast
decode from racing ahead of the pipeline, and ``frame_stride`` marks frames as
inference-eligible instead of dropping them.  ``RtspSource`` reconnects with
exponential backoff and reports its connection state; a failed read is not an
exception but an explicit ``None`` while the backoff is pending.

``capture_factory`` is injectable so tests can drive both sources with a fake
capture and no video files.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
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


class FileSequenceSource:
    """Replay local videos as one strictly-increasing timeline."""

    source_id = "file-sequence"
    pace_to_processing_fps = False

    def __init__(
        self,
        paths: Sequence[Path],
        *,
        capture_factory: CaptureFactory = _default_capture_factory,
        session_id: str | None = None,
        frame_stride: int = 1,
        realtime: bool = True,
        maximum_lag_ms: int = 120,
        loop: bool = False,
    ) -> None:
        if not paths:
            raise ValueError("at least one file path is required")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if maximum_lag_ms < 0:
            raise ValueError("maximum_lag_ms must be non-negative")
        self.paths = tuple(Path(path) for path in paths)
        self.capture_factory = capture_factory
        self.session_id = session_id or f"file-{uuid4().hex}"
        self.frame_stride = frame_stride
        self.realtime = realtime
        self.maximum_lag_ms = maximum_lag_ms
        self.loop = loop
        self._capture: Capture | None = None
        self._path_index = -1
        self._frame_index = 0
        self._offset_ms = 0
        self._last_timestamp_ms: int | None = None
        self._last_local_timestamp_ms = 0
        self._opened = False
        self._playback_started_at: float | None = None
        self._first_timestamp_ms: int | None = None
        self._paused_at: float | None = None
        self._stop_requested = Event()
        self._fps = DEFAULT_FPS
        self._width = 0
        self._height = 0
        self._frame_count = 0
        self.exhausted = False

    def open(self) -> None:
        self.close()
        self._stop_requested.clear()
        self._path_index = -1
        self._frame_index = 0
        self._offset_ms = 0
        self._last_timestamp_ms = None
        self._last_local_timestamp_ms = 0
        self.exhausted = False
        self._opened = True
        self._playback_started_at = monotonic()
        self._first_timestamp_ms = None
        self._paused_at = None
        try:
            self._open_next()
        except Exception:
            self._opened = False
            raise

    def read(self) -> FramePacket | None:
        if self._stop_requested.is_set():
            return None
        if not self._opened:
            raise RuntimeError("file source is not open")
        while self._capture is not None and not self._stop_requested.is_set():
            ok, image = self._capture.read()
            if self._stop_requested.is_set():
                return None
            if ok and image is not None:
                local_ms = max(
                    0,
                    round(self._capture.get(cv2.CAP_PROP_POS_MSEC)),
                )
                self._last_local_timestamp_ms = local_ms
                frame_index = self._frame_index
                self._frame_index += 1
                timestamp_ms = _strict_timestamp(
                    self._last_timestamp_ms,
                    self._offset_ms + local_ms,
                )
                self._last_timestamp_ms = timestamp_ms
                if self._first_timestamp_ms is None:
                    self._first_timestamp_ms = timestamp_ms
                if self.realtime and self._playback_started_at is not None:
                    target_elapsed_s = (
                        timestamp_ms - self._first_timestamp_ms
                    ) / 1000.0
                    delay_s = target_elapsed_s - (
                        monotonic() - self._playback_started_at
                    )
                    if delay_s > 0.0:
                        if self._stop_requested.wait(timeout=delay_s):
                            return None
                    elif delay_s < -(self.maximum_lag_ms / 1000.0):
                        # Decode faster than the pipeline consumes: skip this
                        # frame instead of baking latency into the stream.
                        continue
                height, width = image.shape[:2]
                return FramePacket(
                    source_id=self.source_id,
                    session_id=self.session_id,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    received_at_monotonic_ms=monotonic_ns() // 1_000_000,
                    image=image,
                    original_width=width,
                    original_height=height,
                    inference_eligible=(frame_index % self.frame_stride == 0),
                )
            self._offset_ms += self._current_duration_ms()
            self._open_next()
        if not self._stop_requested.is_set():
            self.exhausted = True
        return None

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
        self._playback_started_at = None
        self._paused_at = None

    def pause(self) -> None:
        if self.realtime and self._paused_at is None:
            self._paused_at = monotonic()

    def resume(self) -> None:
        if (
            self.realtime
            and self._paused_at is not None
            and self._playback_started_at is not None
        ):
            self._playback_started_at += monotonic() - self._paused_at
        self._paused_at = None

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
        """Frames in the current file; the whole sequence may be longer."""
        return self._frame_count

    def _open_next(self) -> None:
        previous = self._capture
        self._capture = None
        if previous is not None:
            previous.release()
        self._path_index += 1
        if self._path_index >= len(self.paths):
            if not self.loop:
                self.exhausted = True
                return
            self._path_index = 0
        capture = self.capture_factory(str(self.paths[self._path_index]))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"cannot open video source: {self.paths[self._path_index]}",
            )
        self._capture = capture
        self._read_capture_properties(capture)

    def _read_capture_properties(self, capture: Capture) -> None:
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

    def _current_duration_ms(self) -> int:
        if self._capture is None:
            return max(1, self._last_local_timestamp_ms + 1)
        frame_count = self._capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        if frame_count > 0.0 and fps > 0.0:
            return max(1, round(frame_count / fps * 1000.0))
        return max(1, self._last_local_timestamp_ms + 1)


class RtspSource:
    """Pull a live RTSP stream, reconnecting with exponential backoff."""

    source_id = "rtsp"
    pace_to_processing_fps = True

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

    @classmethod
    def from_env(
        cls,
        variable: str,
        *,
        capture_factory: CaptureFactory = _default_capture_factory,
        session_id: str | None = None,
        reconnect_initial_s: float = 0.5,
        reconnect_max_s: float = 8.0,
        rtsp_transport: str | None = "tcp",
    ) -> RtspSource:
        url = os.environ.get(variable)
        if not url:
            raise ValueError(f"RTSP environment variable is not set: {variable}")
        return cls(
            url,
            capture_factory=capture_factory,
            session_id=session_id,
            reconnect_initial_s=reconnect_initial_s,
            reconnect_max_s=reconnect_max_s,
            rtsp_transport=rtsp_transport,
        )

    def open(self) -> None:
        self.close()
        if self.rtsp_transport is not None:
            # OpenCV reads this when the capture is created, not per frame.
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                f"rtsp_transport;{self.rtsp_transport}",
            )
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
        received_ms = monotonic_ns() // 1_000_000
        timestamp_ms = _strict_timestamp(self._last_timestamp_ms, received_ms)
        self._last_timestamp_ms = timestamp_ms
        height, width = image.shape[:2]
        packet = FramePacket(
            source_id=self.source_id,
            session_id=self.session_id,
            frame_index=self._frame_index,
            timestamp_ms=timestamp_ms,
            received_at_monotonic_ms=received_ms,
            image=image,
            original_width=width,
            original_height=height,
        )
        self._frame_index += 1
        return packet

    def close(self) -> None:
        self._closed = True
        self.connection_state = "closed"
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()

    def _connect(self) -> bool:
        capture = self.capture_factory(self._url)
        if not capture.isOpened():
            capture.release()
            self._schedule_reconnect("rtsp_open_failed")
            return False
        # Keep only the newest decoded frame; a backlog is latency, not content.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
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
