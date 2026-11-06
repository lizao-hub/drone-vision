"""Live relay: RTSP in, annotated RTMP out.

The batch CLI processes a file and writes an mp4; this entry point keeps the same
pipeline but runs it against a live stream::

    drone1 (RTSP) 鈹€鈹€鈻?pipeline + ViaductRuntime 鈹€鈹€鈻?annotate 鈹€鈹€鈻?ffmpeg 鈹€鈹€鈻?drone1_yolo (RTMP)

A reader thread only keeps the newest source frame, so a slow pipeline drops
frames instead of accumulating latency, and the annotated stream is paced to
``--out-fps``.  Both directions reconnect on failure: the source reopens the
capture, the sink restarts ffmpeg.
"""

from __future__ import annotations

import argparse
import signal
import threading
from dataclasses import replace
from time import monotonic, sleep
from typing import TYPE_CHECKING, Sequence

from algorithm.domain.vehicle_speed_runtime import VehicleSpeedRuntime
from algorithm.domain.viaduct_runtime import ViaductRuntime
from algorithm.engine.packet import FrameGeometryChanged
from algorithm.runtime.live import LiveSource
from algorithm.runtime.rtmp_sink import DEFAULT_BITRATE, FfmpegSink
from drone_worker.args import (
    add_model_arguments,
    add_mp_arguments,
    build_pipeline_from,
    pipeline_settings_from,
    validate_pipeline_arguments,
)
from drone_worker.render import annotate

if TYPE_CHECKING:
    from algorithm.engine.pipeline import MultiProcessPipeline


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m drone_worker relay",
        description="Pull a live RTSP feed, annotate it, push the result to RTMP.",
    )
    add_model_arguments(parser)
    add_mp_arguments(parser)
    parser.add_argument("--source", required=True,
                        help="RTSP URL, or a local video file (replayed in real time, looped)")
    parser.add_argument("--dest", required=True, help="RTMP URL for the annotated stream")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="nominal source fps handed to the trackers (default: 25)")
    parser.add_argument("--out-fps", type=float, default=10.0,
                        help="annotated stream rate, also the send pacing (default: 10)")
    parser.add_argument("--bitrate", default=DEFAULT_BITRATE,
                        help=f"x264 target bitrate (default: {DEFAULT_BITRATE})")
    parser.add_argument("--rtsp-transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--ffmpeg", default=None,
                        help="ffmpeg binary; default: $FFMPEG_BIN, PATH, then imageio-ffmpeg")
    parser.add_argument("--track-timeout-ms", type=int, default=3000,
                        help="ByteTrack lost timeout; raise it when the pipeline is slow (default: 3000)")
    parser.add_argument("--speed-track-timeout-ms", type=int, default=3000,
                        help="vehicle/lane speed track timeout (default: 3000)")
    parser.add_argument("--scale-window-ms", type=int, default=5000,
                        help="car-width scale window (default: 5000)")
    parser.add_argument("--min-scale-samples", type=int, default=5,
                        help="width samples needed for the pixel scale (default: 5)")
    args = parser.parse_args(argv)
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.out_fps <= 0.0:
        parser.error("--out-fps must be positive")
    for name in (
        "track_timeout_ms",
        "speed_track_timeout_ms",
        "scale_window_ms",
        "min_scale_samples",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    validate_pipeline_arguments(parser, args)
    return args


def _install_stop_handlers(stop: threading.Event) -> None:
    def handler(signum: int, frame: object) -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):  # not the main thread / unsupported platform
            pass


def _relay_feed(
    source: LiveSource,
    stop: threading.Event,
    *,
    build,
):
    """Yield ``(frame, observation)`` until ``stop``, rebuilding on a size change.

    The engine fixes the frame geometry when it starts (the shared-memory slots
    are allocated at that size), so a source that comes back at another
    resolution restarts it, dropping the frame that crossed the change.
    """
    pipeline = build()
    try:
        while not stop.is_set():

            def packets():
                while not stop.is_set():
                    packet = source.get(timeout=1.0)
                    if packet is None:
                        continue
                    yield packet

            try:
                yield from pipeline.run(packets())
            except FrameGeometryChanged:
                print(
                    "relay: frame size changed; restarting the pipeline",
                    flush=True,
                )
                pipeline.close()
                pipeline = build()
    finally:
        pipeline.close()


def main_relay(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    stop = threading.Event()
    _install_stop_handlers(stop)

    settings = replace(
        pipeline_settings_from(args),
        track_max_lost_ms=args.track_timeout_ms,
    )
    print("relay: loading models", flush=True)

    def build() -> "MultiProcessPipeline":
        return build_pipeline_from(args, settings, fps=args.fps)

    source = LiveSource(args.source, rtsp_transport=args.rtsp_transport).start()
    print(
        f"relay: {args.source} -> {args.dest}"
        f" (out {args.out_fps:g} fps, {args.bitrate})",
        flush=True,
    )

    runtime: ViaductRuntime | None = None
    frame_size: tuple[int, int] | None = None
    sink: FfmpegSink | None = None
    last_viaduct = None
    processed = 0
    started = monotonic()
    # The multi-process start-up is ~25 s of model loading; reporting fps from
    # before the first frame would make a healthy stream look broken.
    streaming_since: float | None = None
    period = 1.0 / args.out_fps
    deadline = monotonic()

    try:
        feed = _relay_feed(source, stop, build=build)
        try:
            for frame, observation in feed:
                height, width = frame.shape[:2]
                if frame_size != (width, height):
                    # First frame, or the source resolution changed: rebuild the
                    # pixel-scale runtime and the ffmpeg sink for the new size.
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
                    if sink is not None:
                        sink.close()
                        sink = None
                    print(f"relay: frame size {width}x{height}", flush=True)
                assert runtime is not None

                viaduct = runtime.update(observation)
                last_viaduct = viaduct
                annotated = annotate(frame, observation, viaduct=viaduct)

                if sink is None:
                    sink = FfmpegSink(
                        args.dest,
                        width=width,
                        height=height,
                        fps=args.out_fps,
                        bitrate=args.bitrate,
                        ffmpeg=args.ffmpeg,
                    )
                    sink.start()
                try:
                    sink.write(annotated)
                except (BrokenPipeError, OSError) as exc:
                    print(f"relay: sink lost ({exc}); restarting", flush=True)
                    sink.close()
                    sink = None

                processed += 1
                if streaming_since is None:
                    streaming_since = monotonic()
                if args.max_frames and processed >= args.max_frames:
                    break
                deadline = max(deadline + period, monotonic())
                delay = deadline - monotonic()
                if delay > 0.0:
                    sleep(delay)

                if args.log_every and processed % args.log_every == 0:
                    elapsed = monotonic() - (streaming_since or started)
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
                        f"relay: {processed} frames, {processed / elapsed:.1f} fps,"
                        f" dropped {source.dropped}, {detail}",
                        flush=True,
                    )
        finally:
            feed.close()
    finally:
        source.close()
        if sink is not None:
            sink.close()

    elapsed = monotonic() - started
    print(
        f"relay: stopped after {processed} frames in {elapsed:.0f}s"
        f" (dropped {source.dropped})",
        flush=True,
    )
    return 0
