"""Per-stage latency and throughput of the multi-process pipeline.

    python scripts/bench_stages.py --video videos/test_Video1.mp4 --size 1920x1080
    python scripts/bench_stages.py --segment-gpus 0 --vehicle-gpu 1 --line-gpu 2
    python scripts/bench_stages.py --segment-gpus 0,4 --vehicle-gpu 5 --line-gpu 6

Reads the first frames of a video, optionally resizing them (a 640x368 clip is a
fine timing stand-in for a 1080p stream: ``imgsz`` -- not the source resolution --
sets the model cost, though mask upsampling does scale with the source).  Prints
mean/p95/max of ``segment_ms`` / ``detect_ms`` / ``line_ms`` / ``processing_ms``
plus end-to-end fps over the timed frames.

``processing_ms`` is one frame's critical-path work, not its queueing, so it can
exceed ``1000/fps``; the fps line is the real throughput.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Sequence

import cv2
import numpy as np

from algorithm.config import MpPipelineSettings
from algorithm.engine.packet import FramePacket
from algorithm.runtime.factory import (
    DEFAULT_VIDEO,
    build_pipeline,
    default_settings,
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--size", default="1920x1080",
                        help="resize every frame to WxH (default: 1920x1080); 'native' keeps the file size")
    parser.add_argument("--frames", type=int, default=40, help="frames to time (default: 40)")
    parser.add_argument("--warmup", type=int, default=5, help="frames to run before timing (default: 5)")
    parser.add_argument("--segment-imgsz", type=int, default=960)
    parser.add_argument("--detect-imgsz", type=int, default=640)
    parser.add_argument("--line-imgsz", type=int, default=640)
    parser.add_argument("--segment-gpus", default="0",
                        help="one segment worker per GPU (default: 0)")
    parser.add_argument("--vehicle-gpu", type=int, default=1, help="vehicle detector GPU (default: 1)")
    parser.add_argument("--line-gpu", type=int, default=2, help="lane detector GPU (default: 2)")
    parser.add_argument("--window", type=int, default=4)
    return parser.parse_args(argv)


def _frames(args: argparse.Namespace) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    width, height = 0, 0
    if args.size.strip().lower() != "native":
        width, height = (int(value) for value in args.size.lower().split("x"))
    frames: list[np.ndarray] = []
    while len(frames) < args.warmup + args.frames:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    capture.release()
    if len(frames) <= args.warmup:
        raise SystemExit(f"{args.video} has too few frames")
    return frames


def _summary(values: Sequence[float]) -> str:
    if not values:
        return "-"
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return f"mean {statistics.fmean(values):6.1f} p95 {p95:6.1f} max {max(values):6.1f}"


def _report(title: str, observations: Sequence[object], elapsed_s: float) -> None:
    diagnostics = [item.diagnostics for item in observations]  # type: ignore[attr-defined]
    print(f"{title} frames={len(observations)} wall={elapsed_s:.2f}s fps={len(observations) / elapsed_s:.1f}")
    print(f"  segment  {_summary([item.segment_ms for item in diagnostics])}")
    print(f"  detect   {_summary([item.detect_ms for item in diagnostics])}")
    print(f"  line     {_summary([item.line_ms for item in diagnostics])}")
    print(f"  total    {_summary([item.processing_ms for item in diagnostics])}")


def _bench(args: argparse.Namespace, frames: list[np.ndarray]) -> None:
    settings = default_settings(
        segment_imgsz=args.segment_imgsz,
        detect_imgsz=args.detect_imgsz,
        line_imgsz=args.line_imgsz,
    )
    mp_settings = MpPipelineSettings(
        segment_gpus=tuple(int(value) for value in args.segment_gpus.split(",")),
        vehicle_gpu=args.vehicle_gpu,
        line_gpu=args.line_gpu,
        window=args.window,
    )
    pipeline = build_pipeline(settings, mp_settings, fps=25.0)
    items = (
        FramePacket(frame, index, 1000 + index * 40, 25.0)
        for index, frame in enumerate(frames)
    )
    observations = []
    started = 0
    try:
        for index, (_, observation) in enumerate(pipeline.run(items)):
            if index == args.warmup:
                started = perf_counter_ns()
            if index >= args.warmup:
                observations.append(observation)
    finally:
        pipeline.close()
    elapsed = (perf_counter_ns() - started) / 1e9
    title = (
        f"segment_gpus={list(mp_settings.segment_gpus)}"
        f" vehicle={mp_settings.vehicle_gpu} line={mp_settings.line_gpu}"
        f" window={mp_settings.window}"
    )
    _report(title, observations, elapsed)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _bench(args, _frames(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
