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
from collections import Counter
from pathlib import Path
from typing import Sequence

import cv2

from algorithm.config import DEFAULT_SETTINGS
from algorithm.domain.vehicle_speed_runtime import VehicleSpeedRuntime
from algorithm.domain.viaduct_runtime import ViaductRuntime
from algorithm.runtime.factory import build_pipeline
from algorithm.runtime.snapshots import record
from algorithm.runtime.sources import FileSequenceSource
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

    processed = 0
    valid = 0
    total_ms = 0.0
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
            processed += 1
            total_ms += observation.diagnostics.processing_ms
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
            if args.log_every and processed % args.log_every == 0:
                print(
                    f"[{processed}/{source.frame_count or '?'}] valid={valid} "
                    f"tracks={len(observation.vehicle_tracks)} "
                    f"proc={observation.diagnostics.processing_ms:.0f}ms",
                    flush=True,
                )
    finally:
        source.close()
        if writer is not None:
            writer.release()
        if json_stream is not None:
            json_stream.close()

    elapsed = time.perf_counter() - started_at
    average_ms = total_ms / processed if processed else 0.0
    throughput = processed / elapsed if elapsed else 0.0
    print(
        f"frames={processed} (1/{args.stride}) valid={valid} avg_proc={average_ms:.1f}ms "
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
        print(f"video: {annotated_path}")
    if json_stream is not None:
        print(f"jsonl: {json_path}")
    return 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)

