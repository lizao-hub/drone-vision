"""Offline batch: a video file in, a JSONL of observations (and optionally an
annotated mp4) out.

The live counterpart is :mod:`drone_worker.relay`.  Both share the same pipeline,
runtime and renderer; only the loop around them differs -- this one reads the file
sequentially at its own pace and exits at the end, the relay drops stale frames
and never stops until signalled.  Both outputs land in ``--project``; the JSONL
is written by default (one line per sampled frame), the annotated mp4 is opt-in
via ``--save-video`` because encoding it costs roughly a fifth of the wall time
at 1080p.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Sequence

import cv2

from algorithm.domain.vehicle_speed_runtime import VehicleSpeedRuntime
from algorithm.domain.viaduct_runtime import ViaductRuntime
from algorithm.runtime.factory import build_pipeline
from algorithm.runtime.profile import DEFAULT_PROFILE, load_profile
from algorithm.runtime.provenance import code_revision
from algorithm.runtime.snapshots import record
from algorithm.runtime.sources import VideoFileSource
from drone_worker.args import parse_device
from drone_worker.render import annotate

# .../worker/src/drone_worker/batch.py -> worker/
WORKER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT = WORKER_ROOT / "output"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m drone_worker",
        description="Segment the viaduct deck, black out the rest, track vehicles on it.",
        epilog="For the live relay use: python -m drone_worker relay --source <rtsp> --dest <rtmp>",
    )
    parser.add_argument("--source", type=Path, required=True,
                        help="input video file")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT,
                        help="directory both outputs are written to (default: worker/output)")
    parser.add_argument("--device", default="0", help="integer GPU index or 'cpu'")
    parser.add_argument("--stride", type=int, default=3,
                        help="run inference on every Nth frame; one Observation/JSON line per sampled frame (default: 3)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N sampled frames")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False,
                        help="write <project>/<stem>_pipeline.mp4 (default: off)")
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=True,
                        help="write <project>/<stem>.jsonl, one line per sampled frame (default: on)")
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error("--stride must be a positive integer")
    return args


def main_batch(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    video = args.source.resolve()
    project = args.project
    project.mkdir(parents=True, exist_ok=True)
    json_path = project / f"{video.stem}.jsonl" if args.save_json else None
    video_path = project / f"{video.stem}_pipeline.mp4"

    profile = load_profile()
    settings = profile.to_settings(device=parse_device(args.device))

    source = VideoFileSource(video, stride=args.stride)
    source.open()
    print(
        f"run: revision={code_revision()} profile={DEFAULT_PROFILE.name} "
        f"device={args.device} stride={args.stride} source={video} project={project} "
        f"{source.width}x{source.height} @ {source.fps:.2f}fps",
        flush=True,
    )
    pipeline = build_pipeline(settings, fps=source.fps)
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
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(source.fps / args.stride, 1.0),
            (source.width, source.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer: {video_path}")

    processed = 0
    valid = 0
    total_ms = 0.0
    decode_total_ms = 0.0
    sink_total_ms = 0.0
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
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_stream = json_path.open("w", encoding="utf-8")
    started_at = time.perf_counter()

    try:
        while True:
            read_started = perf_counter()
            packet = source.read()
            if packet is None:
                break
            decode_ms = (perf_counter() - read_started) * 1000.0
            observation = pipeline.process(
                packet.frame,
                packet.timestamp_ms,
                packet.frame_index,
                decode_ms=decode_ms,
            )
            processed += 1
            total_ms += observation.diagnostics.processing_ms
            decode_total_ms += decode_ms
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
            sink_ms = 0.0
            if writer is not None:
                sink_started = perf_counter()
                writer.write(
                    annotate(
                        packet.frame,
                        observation,
                        viaduct=viaduct,
                    )
                )
                sink_ms = (perf_counter() - sink_started) * 1000.0
                sink_total_ms += sink_ms
            if json_stream is not None:
                json_stream.write(
                    json.dumps(
                        record(observation, viaduct, sink_ms=sink_ms),
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
                    f"proc={observation.diagnostics.processing_ms:.0f}ms "
                    f"decode={decode_ms:.0f}ms sink={sink_ms:.0f}ms",
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
    if processed:
        print(
            f"timing: decode_avg={decode_total_ms / processed:.1f}ms "
            f"sink_avg={sink_total_ms / processed:.1f}ms"
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
        print(f"video: {video_path}")
    if json_path is not None:
        print(f"jsonl: {json_path}")
    return 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
