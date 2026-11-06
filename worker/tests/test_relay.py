import threading

import numpy as np
import pytest

from algorithm.engine.packet import FrameGeometryChanged, FramePacket
from algorithm.runtime.live import LatestFrameQueue, LiveSource
from algorithm.runtime.rtmp_sink import FfmpegSink, build_ffmpeg_args
from drone_worker.relay import _parse_args, _relay_feed


def _packet(index: int) -> FramePacket:
    return FramePacket(np.zeros((4, 4, 3), np.uint8), index, index, 25.0)


def test_the_queue_keeps_only_the_newest_frame():
    queue = LatestFrameQueue()
    queue.put_latest(_packet(0))
    queue.put_latest(_packet(1))

    assert queue.get(timeout=0.0).frame_index == 1
    assert queue.get(timeout=0.01) is None
    assert (queue.received, queue.dropped) == (2, 1)


def test_a_closed_queue_wakes_the_consumer():
    queue = LatestFrameQueue()
    queue.close()

    assert queue.get(timeout=1.0) is None
    assert queue.closed


def test_ffmpeg_args_carry_size_rate_and_destination():
    args = build_ffmpeg_args(
        width=640,
        height=368,
        fps=10.0,
        url="rtmp://127.0.0.1:1935/drone1_yolo",
        bitrate="2M",
    )

    assert args[-1] == "rtmp://127.0.0.1:1935/drone1_yolo"
    assert "640x368" in args
    assert args[args.index("-f") + 1] == "rawvideo"
    assert "flv" in args
    assert "2M" in args


def test_ffmpeg_args_reject_a_bad_size():
    with pytest.raises(ValueError):
        build_ffmpeg_args(width=0, height=10, fps=10.0, url="rtmp://x")
    with pytest.raises(ValueError):
        build_ffmpeg_args(width=10, height=10, fps=0.0, url="rtmp://x")


def test_the_sink_rejects_a_frame_of_the_wrong_size():
    sink = FfmpegSink(
        "rtmp://127.0.0.1/x", width=640, height=368, fps=10.0, ffmpeg="ffmpeg"
    )

    with pytest.raises(ValueError):
        sink.write(np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError):
        sink.write(np.zeros((368, 640), np.uint8))
    with pytest.raises(RuntimeError):
        sink.write(np.zeros((368, 640, 3), np.uint8))


def test_the_source_rejects_bad_settings():
    with pytest.raises(ValueError):
        LiveSource("")
    with pytest.raises(ValueError):
        LiveSource("rtsp://host/path", rtsp_transport="sctp")


def test_the_relay_requires_a_source_and_a_destination():
    with pytest.raises(SystemExit):
        _parse_args([])
    with pytest.raises(SystemExit):
        _parse_args(["--source", "rtsp://host/drone1"])


def test_the_relay_rejects_non_positive_rates():
    with pytest.raises(SystemExit):
        _parse_args(["--source", "rtsp://a", "--dest", "rtmp://b", "--fps", "0"])
    with pytest.raises(SystemExit):
        _parse_args(["--source", "rtsp://a", "--dest", "rtmp://b", "--out-fps", "-1"])


def test_relay_defaults_pace_the_output():
    args = _parse_args(["--source", "rtsp://host/drone1", "--dest", "rtmp://host/out"])

    assert args.out_fps == 10.0
    assert args.fps == 25.0
    assert args.rtsp_transport == "tcp"


class _FrameSource:
    """The only ``LiveSource`` behaviour the feed needs: ``get``."""

    def __init__(self) -> None:
        self.sent = 0

    def get(self, timeout=None):
        self.sent += 1
        return _packet(self.sent)


class _Engine:
    """A Pipeline stand-in that records its lifecycle."""

    def __init__(self, name: str, *, fail_on_geometry: bool = False) -> None:
        self.name = name
        self.fail_on_geometry = fail_on_geometry
        self.closes = 0

    def run(self, frames):
        if self.fail_on_geometry:
            next(iter(frames))  # the frame that crossed the change is dropped
            raise FrameGeometryChanged("frame size changed")
        for packet in frames:
            yield packet.frame, self.name

    def close(self) -> None:
        self.closes += 1


def test_the_relay_rebuilds_the_pipeline_when_the_frame_size_changes():
    # One loop serves both engines: the serial one never raises, the mp one does.
    first = _Engine("first", fail_on_geometry=True)
    second = _Engine("second")
    built = [first, second]

    feed = _relay_feed(_FrameSource(), threading.Event(), build=lambda: built.pop(0))
    try:
        _, observation = next(feed)
    finally:
        feed.close()

    assert observation == "second"
    assert (first.closes, second.closes) == (1, 1)
