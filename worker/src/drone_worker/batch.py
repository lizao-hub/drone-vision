"""Offline batch: a video file in, an annotated mp4 (and optional JSONL) out.

The live counterpart is :mod:`drone_worker.relay`.  Both share the same pipeline,
runtime and renderer; only the loop around them differs -- this one reads the file
sequentially at its own pace and exits at the end, the relay drops stale frames
and never stops until signalled.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Sequence

import cv2

from algorithm.domain.vehicle_speed_runtime import VehicleSpeedRuntime
from algorithm.domain.viaduct_runtime import ViaductRuntime
from algorithm.engine.packet import FramePacket
from algorithm.engine.pipeline import MultiProcessPipeline
from algorithm.runtime.factory import DEFAULT_VIDEO, WORKER_ROOT
from algorithm.runtime.snapshots import record
from algorithm.runtime.sources import VideoFileSource
from drone_worker.args import (
    add_model_arguments,
    add_mp_arguments,
    build_pipeline_from,
    pipeline_settings_from,
    validate_pipeline_arguments,
)
from drone_worker.render import annotate


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m drone_worker",
        description="Segment the viaduct deck, black out the rest, track vehicles on it.",
        epilog="For the live relay use: python -m drone_worker relay --source <rtsp> --dest <rtmp>",
    )
    add_model_arguments(parser)
    add_mp_arguments(parser)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=3,
                        help="run inference on every Nth frame; one Observation/JSON line per sampled frame (default: 3)")
    parser.add_argument("--json", type=Path, default=None, help="optional JSONL output")
    parser.add_argument("--no-video", action="store_true", help="skip the annotated mp4")
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error("--stride must be a positive integer")
    validate_pipeline_arguments(parser, args)
    return args


def _packets(source: VideoFileSource) -> Iterator[FramePacket]:
    """The source as the engines consume it: one packet per sampled frame."""
    while True:
        packet = source.read()
        if packet is None:
            return
        yield packet


def _observations(source: VideoFileSource, pipeline: MultiProcessPipeline):
    """Yield ``(frame, observation)``; the engine decides how far it reads ahead."""
    yield from pipeline.run(_packets(source))


def main_batch(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    video = args.video.resolve()
    output = args.output or WORKER_ROOT / "output" / f"{video.stem}_pipeline.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    settings = pipeline_settings_from(args)

    source = VideoFileSource(video, stride=args.stride)
    source.open()
    pipeline = build_pipeline_from(args, settings, fps=source.fps)
    # Load the models (and warm them) before the clock starts, so ``wall``/``fps``
    # measure the stream and not the one-off start-up: this allocates the shared
    # memory and starts the three workers at the source geometry.
    pipeline.start(source.height, source.width)
    viaduct_runtime = ViaductRuntime(
        speed=VehicleSpeedRuntime(
            frame_width=source.width,
            frame_height=source.height,
            car_width_m=settings.car_width_m,
        )
    )
    writer: cv2.VideoWriter | None = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(source.fps / args.stride, 1.0),
            (source.width, source.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer: {output}")

    processed = 0
    valid = 0
    processing_samples: list[float] = []
    track_ids: set[str] = set()
    line_track_ids: set[str] = set()
    labels: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    car_speed_samples = 0
    car_speed_total = 0.0
    car_speed_min: float | None = None
    car_speed_max: float | None = None
    car_speed_frames = 0
    viaduct_frames = 0
    viaduct_reasons: Counter[str] = Counter()
    deck_densities: dict[str, list[float]] = {}
    deck_speeds: dict[str, list[float]] = {}
    deck_congestion: dict[str, Counter[str]] = {}
    json_stream = None
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        json_stream = args.json.open("w", encoding="utf-8")
    started_at = time.perf_counter()

    try:
        for frame, observation in _observations(source, pipeline):
            processed += 1
            processing_samples.append(observation.diagnostics.processing_ms)
            valid += int(observation.frame_valid)
            for track in observation.vehicle_tracks:
                track_ids.add(track.track_id)
                labels[track.label] += 1
            line_track_ids.update(track.track_id for track in observation.line_tracks)
            for reason in observation.unknown_reasons:
                reasons[reason] += 1
            viaduct = viaduct_runtime.update(observation)
            car_speeds = viaduct.speed
            if car_speeds.valid and car_speeds.speeds:
                car_speed_frames += 1
                values = [item.speed_kmh for item in car_speeds.speeds]
                car_speed_samples += len(values)
                car_speed_total += sum(values)
                low, high = min(values), max(values)
                car_speed_min = low if car_speed_min is None else min(car_speed_min, low)
                car_speed_max = high if car_speed_max is None else max(car_speed_max, high)
                print(
                    f"car_speed t={observation.timestamp_ms}ms: {len(values)} cars"
                    f" (min {low:.1f} max {high:.1f} km/h)",
                    flush=True,
                )
            if viaduct.valid:
                viaduct_frames += 1
                for item in viaduct.decks:
                    deck_densities.setdefault(item.deck_id, []).append(item.density)
                    deck_congestion.setdefault(item.deck_id, Counter())[
                        item.congestion.value
                    ] += 1
                    if item.mean_speed_kmh is not None:
                        deck_speeds.setdefault(item.deck_id, []).append(item.mean_speed_kmh)
                detail = ", ".join(
                    f"{item.deck_id} {item.density:.1%} n{item.vehicles}"
                    + (
                        ""
                        if item.mean_speed_kmh is None
                        else f" {item.mean_speed_kmh:.1f}km/h"
                    )
                    + f" {item.congestion.value}"
                    for item in viaduct.decks
                )
                print(f"viaduct t={observation.timestamp_ms}ms: {detail}", flush=True)
            else:
                viaduct_reasons.update(viaduct.reasons)
            if writer is not None:
                writer.write(
                    annotate(
                        frame,
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
            if args.max_frames and processed >= args.max_frames:
                break
            if args.log_every and processed % args.log_every == 0:
                print(
                    f"[{processed}/{source.frame_count or '?'}] valid={valid} "
                    f"tracks={len(observation.vehicle_tracks)} "
                    f"proc={observation.diagnostics.processing_ms:.0f}ms",
                    flush=True,
                )
    finally:
        source.close()
        pipeline.close()
        if writer is not None:
            writer.release()
        if json_stream is not None:
            json_stream.close()

    elapsed = time.perf_counter() - started_at
    # Median, not mean: the first frame pays cuDNN/cuBLAS autotuning and would
    # otherwise dominate the average of a short clip (a 30-frame clip can look
    # twice as slow as the steady state).
    median_ms = statistics.median(processing_samples) if processing_samples else 0.0
    throughput = processed / elapsed if elapsed else 0.0
    print(
        f"frames={processed} (1/{args.stride}) valid={valid} median_proc={median_ms:.1f}ms "
        f"wall={elapsed:.1f}s fps={throughput:.1f}"
    )
    if track_ids:
        print(f"unique_tracks={len(track_ids)} " + ", ".join(
            f"{label}:{count}" for label, count in sorted(labels.items())
        ))
    if line_track_ids:
        print(f"unique_line_tracks={len(line_track_ids)}")
    if reasons:
        print("reasons: " + ", ".join(f"{key}:{value}" for key, value in sorted(reasons.items())))
    if car_speed_samples:
        mean_car = car_speed_total / car_speed_samples
        print(
            f"car_speed: samples={car_speed_samples} mean={mean_car:.1f} km/h"
            f" min={car_speed_min:.1f} max={car_speed_max:.1f}"
            f" frames={car_speed_frames}/{processed}"
        )
    if viaduct_frames:
        print(
            f"viaduct: frames={viaduct_frames}/{processed}"
            f" decks={len(deck_densities)}"
        )
        for deck_id in sorted(deck_densities):
            densities = deck_densities[deck_id]
            line = (
                f"  {deck_id}: density mean={_mean(densities):.1%}"
                f" min={min(densities):.1%} max={max(densities):.1%}"
            )
            speeds = deck_speeds.get(deck_id, [])
            if speeds:
                line += f" | speed mean={_mean(speeds):.1f} km/h (frames={len(speeds)})"
            congestion = deck_congestion.get(deck_id)
            if congestion:
                line += " | congestion " + ", ".join(
                    f"{key}:{value}" for key, value in sorted(congestion.items())
                )
            print(line)
    if viaduct_reasons:
        print(
            "viaduct_unavailable: "
            + ", ".join(f"{key}:{value}" for key, value in sorted(viaduct_reasons.items()))
        )
    if writer is not None:
        print(f"video: {output}")
    if args.json is not None:
        print(f"jsonl: {args.json}")
    return 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
