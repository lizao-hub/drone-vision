"""Offline batch: a video file in, an annotated mp4 and a JSONL out.

This is the only entry point: it reads the file sequentially at its own pace,
runs the shared pipeline, and exits at the end of the file.  Every tuning knob
comes from ``PipelineSettings``; the flags here only choose the input, where the
two outputs go, and whether each of them is written.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import cv2

from algorithm.config import DEFAULT_SETTINGS
from algorithm.domain.vehicle_speed_runtime import VehicleSpeedRuntime
from algorithm.domain.viaduct_runtime import ViaductRuntime
from algorithm.runtime.factory import build_pipeline
from algorithm.runtime.snapshots import record
from algorithm.runtime.sources import FileSequenceSource
from algorithm.runtime.telemetry import FrameTelemetry
from drone_worker.render import annotate


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m drone_worker",
        description="Segment the viaduct deck, black out the rest, track vehicles on it.",
    )
    parser.add_argument("--source", type=Path, required=True, help="source video file")
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="run inference on every Nth frame; one Observation/JSON line per sampled frame (default: 3)",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write the annotated mp4 (default: on)",
    )
    parser.add_argument(
        "--save-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write the per-frame JSONL (default: on)",
    )
    args = parser.parse_args(argv)
    return args


def main_batch(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = DEFAULT_SETTINGS
    input_path = args.source.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"source does not exist: {input_path}")
    # Both outputs go to the configured project directory; there is no flag for
    # it, because changing it is a settings change, not a per-run choice.
    project = settings.project
    project.mkdir(parents=True, exist_ok=True)
    annotated_path = project / f"{input_path.stem}_pipeline.mp4"
    json_path = project / f"{input_path.stem}.jsonl"

    source = FileSequenceSource(
        [input_path],
        frame_stride=args.stride,
        realtime=False,
    )
    source.open()
    pipeline = build_pipeline(settings)
    viaduct_runtime = ViaductRuntime(
        speed=VehicleSpeedRuntime(
            frame_width=source.width,
            frame_height=source.height,
            car_width_m=settings.car_width_m,
        )
    )
    writer: cv2.VideoWriter | None = None
    if args.save_video:
        writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(source.fps / args.stride, 1.0),
            (source.width, source.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer: {annotated_path}")

    telemetry = FrameTelemetry(
        log_every=args.log_every,
        total_frames=source.frame_count or None,
        stride=args.stride,
    )
    json_stream = json_path.open("w", encoding="utf-8") if args.save_json else None
    started_at = time.perf_counter()

    try:
        while True:
            packet = source.read()
            if packet is None:
                break
            if not packet.inference_eligible:
                continue
            observation = pipeline.process(
                packet.image,
                packet.timestamp_ms,
                packet.frame_index,
            )
            viaduct = viaduct_runtime.update(observation)
            telemetry.observe(observation, viaduct)
            if writer is not None:
                writer.write(
                    annotate(
                        packet.image,
                        observation,
                        viaduct=viaduct,
                    )
                )
            if json_stream is not None:
                json_stream.write(
                    json.dumps(
                        record(observation, viaduct),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        source.close()
        if writer is not None:
            writer.release()
        if json_stream is not None:
            json_stream.close()

    telemetry.summary(elapsed_s=time.perf_counter() - started_at)
    if writer is not None:
        print(f"video: {annotated_path}")
    if json_stream is not None:
        print(f"jsonl: {json_path}")
    return 0

