"""Offline batch: a video file in, a per-frame JSONL out (mp4 optional).

Every tuning knob comes from ``PipelineSettings``; the flags here only choose the
input, the device, the sampling stride and whether the annotated video is written
as well.  The mp4 is **off by default** -- it is an eyeball/demo artifact, while
the JSONL is what the regression and the platform consume.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from drone_worker.args import (
    add_device,
    add_model_overrides,
    add_sampling,
    add_source,
    settings_from_args,
)
from drone_worker.render import annotate


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m drone_worker",
        description="Segment the viaduct deck, black out the rest, track vehicles on it.",
    )
    add_source(parser, help_text="source video file")
    add_device(parser, pair=False)
    add_sampling(parser)
    add_model_overrides(parser)
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="also write the annotated mp4 (default: off; the JSONL is always written)",
    )
    args = parser.parse_args(argv)
    if len(args.device) != 1:
        parser.error("batch drives one device; use `pipeline` for two cards")
    return args


def main_batch(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = settings_from_args(DEFAULT_SETTINGS, args)
    input_path = Path(args.source).resolve()
    if not input_path.is_file():
        print(f"error: source does not exist: {input_path}", file=sys.stderr)
        return 2
    # Both outputs go to the configured project directory; there is no flag for
    # it, because changing it is a settings change, not a per-run choice.
    project = settings.project
    project.mkdir(parents=True, exist_ok=True)
    annotated_path = project / f"{input_path.stem}_pipeline.mp4"
    json_path = project / f"{input_path.stem}.jsonl"

    source = FileSequenceSource(input_path, frame_stride=args.stride)
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
    json_stream = json_path.open("w", encoding="utf-8")
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
        json_stream.close()

    telemetry.summary(elapsed_s=time.perf_counter() - started_at)
    if writer is not None:
        print(f"video: {annotated_path}")
    print(f"jsonl: {json_path}")
    return 0
