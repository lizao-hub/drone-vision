"""Local video file source: OpenCV capture -> FramePacket."""

from __future__ import annotations

from pathlib import Path

import cv2

from algorithm.engine.packet import FramePacket


class VideoFileSource:
    """Sequential reader for a local mp4; timestamps are made strictly increasing."""

    def __init__(self, path: Path, *, stride: int = 1) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"video does not exist: {path}")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.path = path
        self.stride = stride
        self._capture: cv2.VideoCapture | None = None
        self._fps = 0.0
        self._frame_index = 0
        self._last_timestamp_ms = 0
        self._width = 0
        self._height = 0
        self._frame_count = 0

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> "VideoFileSource":
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open video: {self.path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        self._capture = capture
        self._fps = fps if fps > 0 else 25.0
        self._width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_index = 0
        self._last_timestamp_ms = 0
        return self

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoFileSource":
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- reading -----------------------------------------------------------
    def read(self) -> FramePacket | None:
        if self._capture is None:
            raise RuntimeError("video source is not open")
        frame = None
        while True:
            ok, candidate = self._capture.read()
            if not ok or candidate is None:
                return None
            index = self._frame_index
            self._frame_index += 1
            if index % self.stride == 0:
                frame = candidate
                break
        return FramePacket(
            frame=frame,
            frame_index=index,
            timestamp_ms=self._timestamp_ms(index),
            fps=self._fps,
        )

    # -- properties --------------------------------------------------------
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

    def _timestamp_ms(self, frame_index: int) -> int:
        if self._capture is None:
            raise RuntimeError("video source is not open")
        candidate = int(round(self._capture.get(cv2.CAP_PROP_POS_MSEC)))
        if candidate <= 0:
            candidate = int(round(frame_index * 1000.0 / self._fps))
        timestamp = max(self._last_timestamp_ms, candidate)
        self._last_timestamp_ms = timestamp + 1
        return timestamp
