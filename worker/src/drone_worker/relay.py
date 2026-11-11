"""Live relay: RTSP in, annotated RTMP out.

Only an RTSP stream is accepted: a local video is a batch/pipeline input, not a
live source, so the relay has one code path and one timing model.

Three stages run concurrently, which is what makes the numbers printable at the
bottom meaningful::

    [reader thread]   decode  鈹€鈹€鈻?LatestFrameQueue (newest wins)
                                        鈹?get()
    [inference loop]  pipeline.process 鈹€鈹? (stride-sampled on frame_index)
                                        鈹?    [render thread]   annotate + ffmpeg 鈹? 鈹€鈹€鈻?drone1_yolo (RTMP)

The inference loop is the only stage on the critical path, so its per-frame
number plus the render/write numbers are what decide ``--stride``: with a 30 fps
source, ``stride = ceil(single_frame_ms / 33.3)`` keeps the output at a steady
``30 / stride`` fps instead of a jittery drop-everything pace.
"""

from __future__ import annotations

import argparse
import signal
import threading
from dataclasses import replace
from time import monotonic, sleep
from typing import Sequence

import numpy as np

from algorithm.config import DEFAULT_SETTINGS, PipelineSettings
from algorithm.domain.observations import Observation
from algorithm.domain.vehicle_speed_runtime import VehicleSpeedRuntime
from algorithm.domain.viaduct_runtime import ViaductRuntime, ViaductUpdate
from algorithm.runtime.live import ThreadedSource
from algorithm.runtime.rtmp_sink import DEFAULT_BITRATE, FfmpegSink
from algorithm.runtime.sources import DEFAULT_FPS, RtspSource
from drone_worker.args import (
    add_device,
    add_model_overrides,
    add_sampling,
    add_source,
    settings_from_args,
)
from drone_worker.render import annotate


# How far the observed rate may sit from the nominal one before we say so.
FPS_DRIFT_WARN = 0.10


def output_fps(source_fps: float, stride: int) -> float:
    """The relay's output rate: the source rate divided by the sampling stride.

    ``--stride 2`` on a 30 fps feed means 15 fps out, and the encoder uses this
    same number for its timestamps, so the stream plays at the right speed.
    Deriving it (instead of taking a second flag) removes the one setting that
    could silently make the video play fast or slow.
    """
    if stride <= 0:
        raise ValueError("stride must be positive")
    fps = source_fps if source_fps > 0.0 else DEFAULT_FPS
    return max(fps / stride, 1.0)


class _RenderQueue:
    """Single-slot handoff to the render thread: the newest job wins."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._job: tuple[np.ndarray, Observation, ViaductUpdate] | None = None
        self._closed = False
        self._dropped = 0

    def put(self, job: tuple[np.ndarray, Observation, ViaductUpdate]) -> None:
        with self._condition:
            if self._closed:
                return
            if self._job is not None:
                self._dropped += 1
            self._job = job
            self._condition.notify()

    def get(
        self, timeout: float | None = None
    ) -> tuple[np.ndarray, Observation, ViaductUpdate] | None:
        with self._condition:
            if self._job is None and not self._closed:
                self._condition.wait(timeout)
            job, self._job = self._job, None
            return job

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def dropped(self) -> int:
        with self._condition:
            return self._dropped


class _Renderer:
    """Annotate and encode on its own thread, keeping the newest result.

    When inference or the encoder falls behind, the frame that gets dropped is
    the rendered one -- the *inference* result that was queued, not the source
    frame -- so the stream stays current instead of growing latency.
    """

    def __init__(self, args: argparse.Namespace, stop: threading.Event) -> None:
        self.args = args
        self.stop = stop
        self.queue = _RenderQueue()
        # Filled by the inference loop once the source reports its frame rate.
        self.out_fps: float | None = None
        self.encoder: str | None = None
        self.render_ms = 0.0
        self.write_ms = 0.0
        self.rendered = 0
        self._sink: FfmpegSink | None = None
        self._size: tuple[int, int] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> "_Renderer":
        self._thread = threading.Thread(
            target=self._loop, name="relay-render", daemon=True
        )
        self._thread.start()
        return self

    def submit(
        self, frame: np.ndarray, observation: Observation, viaduct: ViaductUpdate
    ) -> None:
        self.queue.put((frame, observation, viaduct))

    def close(self) -> None:
        self.queue.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sink is not None:
            self._sink.close()
            self._sink = None

    # -- thread body -------------------------------------------------------
    def _loop(self) -> None:
        while not self.stop.is_set():
            job = self.queue.get(timeout=0.5)
            if job is None:
                continue
            frame, observation, viaduct = job
            height, width = frame.shape[:2]
            if self._size != (width, height):
                self._open_sink(width, height)
            assert self._sink is not None

            started = monotonic()
            annotated = annotate(frame, observation, viaduct=viaduct)
            drawn = monotonic()
            try:
                self._sink.write(annotated)
            except (BrokenPipeError, OSError) as exc:
                print(f"relay: sink lost ({exc}); restarting", flush=True)
                self._sink.close()
                self._sink = None
                self._size = None
                continue
            written = monotonic()

            self.render_ms += (drawn - started) * 1000.0
            self.write_ms += (written - drawn) * 1000.0
            self.rendered += 1

    def _open_sink(self, width: int, height: int) -> None:
        if self._sink is not None:
            self._sink.close()
            self._sink = None
        self._sink = FfmpegSink(
            self.args.dest,
            width=width,
            height=height,
            fps=self.out_fps or DEFAULT_FPS,
            bitrate=self.args.bitrate,
            ffmpeg=self.args.ffmpeg,
        )
        self._sink.start()
        self._size = (width, height)
        self.encoder = self._sink.encoder
        print(
            f"relay: {width}x{height} -> {self.args.dest}"
            f" at {self.out_fps or DEFAULT_FPS:g} fps (encoder {self.encoder})",
            flush=True,
        )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m drone_worker relay",
        description="Pull a live RTSP feed, annotate it, push the result to RTMP.",
    )
    add_source(parser, help_text="RTSP URL of the live feed")
    add_device(parser, pair=False)
    add_sampling(parser, stride_default=1)
    add_model_overrides(parser)
    parser.add_argument("--dest", required=True, help="RTMP URL for the annotated stream")
    parser.add_argument(
        "--bitrate", default=DEFAULT_BITRATE,
        help=f"x264 target bitrate (default: {DEFAULT_BITRATE})",
    )
    parser.add_argument("--rtsp-transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument(
        "--ffmpeg", default=None,
        help="ffmpeg binary; default: $FFMPEG_BIN, PATH, then imageio-ffmpeg",
    )
    parser.add_argument(
        "--track-timeout-ms", type=int, default=3000,
        help="ByteTrack lost timeout; raise it when the pipeline is slow (default: 3000)",
    )
    parser.add_argument(
        "--speed-track-timeout-ms", type=int, default=3000,
        help="vehicle/lane speed track timeout (default: 3000)",
    )
    parser.add_argument(
        "--scale-window-ms", type=int, default=5000,
        help="car-width scale window (default: 5000)",
    )
    parser.add_argument(
        "--min-scale-samples", type=int, default=5,
        help="width samples needed for the pixel scale (default: 5)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="stop after N processed frames (0 = run forever)",
    )
    args = parser.parse_args(argv)
    if not args.source.startswith("rtsp://"):
        parser.error("relay reads a live stream: --source must start with rtsp://")
    if len(args.device) != 1:
        parser.error("relay drives one device; pass a single index or 'cpu'")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")
    for name in ("track_timeout_ms", "speed_track_timeout_ms", "scale_window_ms", "min_scale_samples"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _settings_from_args(args: argparse.Namespace) -> PipelineSettings:
    """DEFAULT_SETTINGS plus the shared knobs and the live-only timeouts."""
    settings = settings_from_args(DEFAULT_SETTINGS, args)
    # Live frames arrive unevenly and the drop-oldest queue widens the gaps, so
    # tracks are kept alive longer than the batch default.
    return replace(settings, track_max_lost_ms=args.track_timeout_ms)


def _install_stop_handlers(stop: threading.Event) -> None:
    def handler(signum: int, frame: object) -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):  # not the main thread / unsupported platform
            pass


def main_relay(argv: Sequence[str] | None = None) -> int:
    # Imported here so parsing ``relay --help`` does not load the models.
    from algorithm.runtime.factory import build_pipeline

    args = _parse_args(argv)
    stop = threading.Event()
    _install_stop_handlers(stop)

    settings = _settings_from_args(args)
    device = args.device[0]
    print(f"relay: loading models (device={device})", flush=True)
    try:
        pipeline = build_pipeline(settings)
    except Exception as exc:  # noqa: BLE001 - a missing GPU must not kill the demo
        if device == "cpu":
            raise
        print(
            f"relay: device {device} unavailable ({exc}); falling back to cpu",
            flush=True,
        )
        settings = replace(settings, device="cpu")
        pipeline = build_pipeline(settings)

    source = ThreadedSource(
        RtspSource(args.source, rtsp_transport=args.rtsp_transport)
    ).start()
    renderer = _Renderer(args, stop).start()
    print(
        f"relay: {args.source} -> {args.dest}"
        f" (stride {args.stride}, {args.bitrate})",
        flush=True,
    )

    runtime: ViaductRuntime | None = None
    frame_size: tuple[int, int] | None = None
    last_viaduct: ViaductUpdate | None = None
    processed = 0
    infer_ms = 0.0
    started = monotonic()
    # Both are fixed by the first frame, once the source reports its rate.
    out_fps: float | None = None
    period = 0.0
    deadline = monotonic()
    drifted = False

    try:
        while not stop.is_set():
            packet = source.get(timeout=1.0)
            if packet is None:
                continue

            # Sample on the source's own frame numbers: dropping frames to keep
            # up must not change which frames get inferred.
            if args.stride > 1 and packet.frame_index % args.stride != 0:
                continue

            frame = packet.image
            height, width = frame.shape[:2]
            if frame_size != (width, height):
                # First frame, or the source resolution changed: rebuild the
                # pixel-scale runtime.  The sink follows the rendered frames.
                runtime = ViaductRuntime(
                    speed=VehicleSpeedRuntime(
                        frame_width=width,
                        frame_height=height,
                        car_width_m=settings.car_width_m,
                        scale_window_ms=args.scale_window_ms,
                        min_scale_samples=args.min_scale_samples,
                        track_timeout_ms=args.speed_track_timeout_ms,
                    )
                )
                frame_size = (width, height)
                print(f"relay: frame size {width}x{height}", flush=True)
            if out_fps is None:
                out_fps = output_fps(source.fps, args.stride)
                renderer.out_fps = out_fps
                period = 1.0 / out_fps
                print(
                    f"relay: source {source.fps:g} fps / stride {args.stride}"
                    f" -> out {out_fps:g} fps",
                    flush=True,
                )
            assert runtime is not None

            infer_started = monotonic()
            observation = pipeline.process(
                frame, packet.timestamp_ms, packet.frame_index
            )
            viaduct = runtime.update(observation)
            infer_ms += (monotonic() - infer_started) * 1000.0
            last_viaduct = viaduct
            renderer.submit(frame, observation, viaduct)

            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break
            if period > 0.0:
                deadline = max(deadline + period, monotonic())
                delay = deadline - monotonic()
                if delay > 0.0:
                    sleep(delay)

            if args.log_every and processed % args.log_every == 0:
                elapsed = monotonic() - started
                actual = processed / elapsed
                rendered = max(renderer.rendered, 1)
                detail = ""
                if last_viaduct.valid:
                    detail = ", ".join(
                        f"{item.deck_id} {item.density:.1%} n{item.vehicles}"
                        + (
                            ""
                            if item.mean_speed_kmh is None
                            else f" {item.mean_speed_kmh:.0f}km/h"
                        )
                        + f" {item.congestion.value}"
                        for item in last_viaduct.decks
                    )
                else:
                    detail = "invalid: " + ", ".join(last_viaduct.reasons)
                print(
                    f"relay: {processed} frames, {actual:.1f} fps"
                    f" (nominal {out_fps or 0.0:.1f}),"
                    f" dropped {source.dropped}, render_dropped {renderer.queue.dropped}"
                    f" | infer {infer_ms / processed:.1f}"
                    f" render {renderer.render_ms / rendered:.1f}"
                    f" write {renderer.write_ms / rendered:.1f} ms"
                    f" | {detail}",
                    flush=True,
                )
                # A slower-than-nominal output means the stream plays fast.  Say
                # it once, after the warm-up, instead of hiding it in the numbers.
                if (
                    not drifted
                    and out_fps is not None
                    and elapsed >= 15.0
                    and abs(actual - out_fps) / out_fps > FPS_DRIFT_WARN
                ):
                    drifted = True
                    print(
                        f"relay: WARNING actual {actual:.1f} fps against nominal"
                        f" {out_fps:.1f} -- stride {args.stride} is too small for"
                        f" the current load (infer {infer_ms / processed:.1f} ms)",
                        flush=True,
                    )
    finally:
        source.close()
        renderer.close()

    elapsed = monotonic() - started
    print(
        f"relay: stopped after {processed} frames in {elapsed:.0f}s"
        f" (dropped {source.dropped}, render_dropped {renderer.queue.dropped})",
        flush=True,
    )
    return 0
