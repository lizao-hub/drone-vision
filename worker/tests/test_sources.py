"""Sources are driven through an injected capture, so no video files are needed."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from algorithm.runtime.frame_packet import FramePacket
from algorithm.runtime.sources import FileSequenceSource, RtspSource


class FakeCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        fps: float = 25.0,
        opened: bool = True,
    ) -> None:
        self.frames = frames
        self.fps = fps
        self.opened = opened
        self.index = 0
        self.position_ms = 0.0
        self.released = False
        self.settings: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= len(self.frames):
            return False, None
        image = self.frames[self.index]
        self.index += 1
        self.position_ms = (self.index - 1) * (1000.0 / self.fps)
        return True, image

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return self.position_ms
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.frames[0].shape[1]) if self.frames else 0.0
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.frames[0].shape[0]) if self.frames else 0.0
        return 0.0

    def set(self, property_id: int, value: float) -> bool:
        self.settings.append((property_id, value))
        return True

    def release(self) -> None:
        self.released = True


def _frames(count: int) -> list[np.ndarray]:
    return [np.zeros((4, 4, 3), np.uint8) for _ in range(count)]


def _factory(captures: list[FakeCapture]):
    pending = list(captures)

    def factory(source: str) -> FakeCapture:
        if not pending:
            raise AssertionError(f"unexpected capture request for {source}")
        return pending.pop(0)

    return factory


def test_file_source_keeps_timestamps_increasing() -> None:
    source = FileSequenceSource(
        Path("a.mp4"),
        capture_factory=_factory([FakeCapture(_frames(2))]),
    )
    source.open()

    packets = [source.read() for _ in range(3)]

    assert all(packet is not None for packet in packets[:2])
    assert [packet.timestamp_ms for packet in packets[:2]] == [0, 40]
    assert [packet.frame_index for packet in packets[:2]] == [0, 1]
    assert packets[2] is None
    assert source.exhausted


def test_file_source_marks_sampled_frames_instead_of_dropping_them() -> None:
    source = FileSequenceSource(
        Path("a.mp4"),
        capture_factory=_factory([FakeCapture(_frames(7))]),
        frame_stride=3,
    )
    source.open()

    packets = [packet for packet in iter(source.read, None)]

    assert [packet.frame_index for packet in packets] == list(range(7))
    assert [packet.inference_eligible for packet in packets] == [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]


def test_file_source_reports_the_capture_properties() -> None:
    source = FileSequenceSource(
        Path("a.mp4"),
        capture_factory=_factory([FakeCapture(_frames(10), fps=30.0)]),
    )
    source.open()

    assert (source.fps, source.width, source.height, source.frame_count) == (
        30.0,
        4,
        4,
        10,
    )


def test_file_source_validates_its_settings() -> None:
    with pytest.raises(ValueError):
        FileSequenceSource(Path("a.mp4"), frame_stride=0)


def test_file_source_raises_when_the_capture_will_not_open() -> None:
    source = FileSequenceSource(
        Path("missing.mp4"),
        capture_factory=_factory([FakeCapture([], opened=False)]),
    )

    with pytest.raises(RuntimeError):
        source.open()
    with pytest.raises(RuntimeError):
        source.read()


def test_rtsp_source_reconnects_with_exponential_backoff() -> None:
    capture = FakeCapture(_frames(1))
    source = RtspSource(
        "rtsp://host/feed",
        capture_factory=_factory([capture, FakeCapture(_frames(2))]),
        reconnect_initial_s=0.2,
        reconnect_max_s=0.4,
    )
    source.open()

    first = source.read()
    assert first is not None
    assert source.connection_state == "connected"
    assert source.stream_generation == 1

    assert source.read() is None  # the fake stream ended
    assert source.connection_state == "reconnecting"
    assert source.last_error == "rtsp_read_failed"
    assert source.read() is None  # still inside the backoff window

    time.sleep(0.21)
    second = source.read()
    assert second is not None
    assert source.connection_state == "connected"
    assert source.stream_generation == 2
    assert source.last_error is None


def test_rtsp_source_sets_the_transport_and_buffer_before_connecting() -> None:
    capture = FakeCapture(_frames(1))
    source = RtspSource(
        "rtsp://host/feed",
        capture_factory=_factory([capture]),
        reconnect_initial_s=0.01,
        reconnect_max_s=0.02,
    )
    source.open()

    assert (cv2.CAP_PROP_BUFFERSIZE, 1) in capture.settings


def test_rtsp_source_reports_the_stream_frame_rate() -> None:
    source = RtspSource(
        "rtsp://host/feed",
        capture_factory=_factory([FakeCapture(_frames(1), fps=30.0)]),
    )
    source.open()

    assert source.fps == 30.0


def test_rtsp_source_validates_its_settings() -> None:
    with pytest.raises(ValueError):
        RtspSource("")
    with pytest.raises(ValueError):
        RtspSource("rtsp://host/feed", rtsp_transport="sctp")
    with pytest.raises(ValueError):
        RtspSource("rtsp://host/feed", reconnect_initial_s=-1.0)
    with pytest.raises(ValueError):
        RtspSource("rtsp://host/feed", reconnect_initial_s=4.0, reconnect_max_s=1.0)


def test_a_packet_requires_dense_metadata() -> None:
    image = np.zeros((4, 4, 3), np.uint8)

    with pytest.raises(ValueError):
        FramePacket("s", "s", 0, 0, image, 5, 4)
    with pytest.raises(ValueError):
        FramePacket("s", "s", -1, 0, image, 4, 4)
    with pytest.raises(ValueError):
        FramePacket("", "s", 0, 0, image, 4, 4)
    with pytest.raises(TypeError):
        FramePacket("s", "s", 0, 0, image, 4, 4, inference_eligible=1)  # type: ignore[arg-type]
