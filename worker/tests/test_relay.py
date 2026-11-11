import time

import numpy as np
import pytest

from algorithm.runtime.frame_packet import FramePacket
from algorithm.runtime.live import LatestFrameQueue, ThreadedSource
from algorithm.runtime.rtmp_sink import FfmpegSink, build_ffmpeg_args
from drone_worker.relay import _parse_args, output_fps


def _packet(index: int) -> FramePacket:
    return FramePacket(
        source_id="live",
        session_id="test",
        frame_index=index,
        timestamp_ms=index,
        image=np.zeros((4, 4, 3), np.uint8),
        original_width=4,
        original_height=4,
    )


def test_the_queue_keeps_only_the_newest_frame():
    queue = LatestFrameQueue()
    queue.put_latest(_packet(0))
    queue.put_latest(_packet(1))

    assert queue.get(timeout=0.0).frame_index == 1
    assert queue.get(timeout=0.01) is None
    assert queue.dropped == 1


def test_a_closed_queue_wakes_the_consumer():
    queue = LatestFrameQueue()
    queue.close()

    assert queue.get(timeout=1.0) is None


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


def test_nvenc_args_carry_the_low_latency_codec():
    args = build_ffmpeg_args(
        width=640, height=368, fps=10.0, url="rtmp://x", encoder="h264_nvenc"
    )

    assert "h264_nvenc" in args
    assert "libx264" not in args
    # Still the x264 latency profile: no B-frames, one keyframe per second.
    assert args[args.index("-bf") + 1] == "0"
    assert args[args.index("-g") + 1] == "10"


def test_an_unknown_encoder_is_rejected():
    with pytest.raises(ValueError):
        build_ffmpeg_args(
            width=640, height=368, fps=10.0, url="rtmp://x", encoder="h264_qsv"
        )


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


class _StubSource:
    """A FrameSource that always has a frame ready."""

    source_id = "stub"

    def __init__(self) -> None:
        self.opened = 0
        self.closed = 0

    def open(self) -> None:
        self.opened += 1

    def read(self) -> FramePacket | None:
        return _packet(0)

    def close(self) -> None:
        self.closed += 1


def test_the_threaded_source_keeps_only_the_newest_frame():
    source = ThreadedSource(_StubSource()).start()
    try:
        time.sleep(0.05)
        assert source.get(timeout=1.0) is not None
        # The reader outraces the consumer, so frames are dropped by design.
        assert source.dropped >= 1
    finally:
        source.close()


def test_the_threaded_source_rejects_a_bad_retry_delay():
    with pytest.raises(ValueError):
        ThreadedSource(_StubSource(), reopen_delay_s=0.0)
    with pytest.raises(ValueError):
        ThreadedSource(_StubSource(), idle_poll_s=0.0)


def test_the_relay_only_takes_a_live_rtsp_source():
    args = _parse_args(["--source", "rtsp://host/drone1", "--dest", "rtmp://b"])

    assert args.source == "rtsp://host/drone1"
    # A local video is a batch/pipeline input, not a live feed.
    with pytest.raises(SystemExit):
        _parse_args(["--source", "clip.mp4", "--dest", "rtmp://b"])


def test_a_local_path_is_rejected_before_the_models_load():
    with pytest.raises(SystemExit):
        _parse_args(["--source", "/tmp/clip.mp4", "--dest", "rtmp://b"])


def test_the_relay_requires_a_source_and_a_destination():
    with pytest.raises(SystemExit):
        _parse_args([])
    with pytest.raises(SystemExit):
        _parse_args(["--source", "rtsp://host/drone1"])


def test_the_relay_rejects_non_positive_timeouts():
    with pytest.raises(SystemExit):
        _parse_args(["--source", "rtsp://a", "--dest", "rtmp://b", "--track-timeout-ms", "0"])


def test_the_output_rate_is_derived_not_configured():
    # The flag is gone on purpose: a second number that must match the stride is
    # the one thing that could silently make the stream play fast or slow.
    with pytest.raises(SystemExit):
        _parse_args(["--source", "rtsp://a", "--dest", "rtmp://b", "--out-fps", "15"])


def test_output_fps_follows_the_source_rate_and_stride():
    assert output_fps(30.0, 1) == 30.0
    assert output_fps(30.0, 2) == 15.0
    assert output_fps(30.0, 3) == 10.0


def test_output_fps_falls_back_when_the_source_reports_nothing():
    assert output_fps(0.0, 1) == 25.0
    assert output_fps(0.0, 2) == 12.5


def test_output_fps_rejects_a_bad_stride():
    with pytest.raises(ValueError):
        output_fps(30.0, 0)


def test_relay_defaults_target_the_first_gpu():
    args = _parse_args(["--source", "rtsp://host/drone1", "--dest", "rtmp://host/out"])

    assert args.device == (0,)
    assert args.rtsp_transport == "tcp"
    # A live feed is inferred on every frame unless the operator says otherwise.
    assert args.stride == 1
    # Untouched knobs stay None so the PipelineSettings defaults win.
    assert args.segment_imgsz is None
    assert args.track_timeout_ms == 3000


def test_the_relay_drives_a_single_device():
    with pytest.raises(SystemExit):
        _parse_args(["--source", "rtsp://a", "--dest", "rtmp://b", "--device", "0,1"])


def test_the_relay_still_accepts_an_explicit_cpu():
    args = _parse_args(["--source", "rtsp://a", "--dest", "rtmp://b", "--device", "cpu"])

    assert args.device == ("cpu",)
