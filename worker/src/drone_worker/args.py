"""Shared command line helpers for the worker entry points.

``batch`` and ``relay`` differ in how they pace frames, not in what they can
configure.  So the model knobs, the process layout, the validations and the
wiring live here once: one definition, one set of rules, and no way for the two
runners to drift apart.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from algorithm.config import MpPipelineSettings

if TYPE_CHECKING:  # argparse must stay importable without pulling torch in
    from algorithm.config import PipelineSettings
    from algorithm.engine.pipeline import MultiProcessPipeline


def parse_gpus(value: str) -> tuple[int, ...]:
    """``"2"`` -> ``(2,)``, ``"0,1"`` -> ``(0, 1)``: physical GPU ids, one per worker."""
    text = value.strip()
    if not text:
        raise ValueError("at least one GPU id is required")
    try:
        gpus = tuple(int(part) for part in text.split(","))
    except ValueError as exc:
        raise ValueError(f"GPU ids must be integers, got {value!r}") from exc
    if any(gpu < 0 for gpu in gpus):
        raise ValueError(f"GPU ids must be non-negative, got {value!r}")
    return gpus


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Frame pacing is the runner's business; the model knobs are not."""
    parser.add_argument("--segment-imgsz", type=int, default=960)
    parser.add_argument("--detect-imgsz", type=int, default=640)
    parser.add_argument("--line-imgsz", type=int, default=640)
    parser.add_argument("--viaduct-conf", type=float, default=0.25)
    parser.add_argument("--vehicle-conf", type=float, default=0.30)
    parser.add_argument("--line-conf", type=float, default=0.30)
    parser.add_argument(
        "--car-width-m",
        type=float,
        default=1.95,
        help="real width of a car, used to scale pixels to metres (default: 1.95)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="print one status line every N processed frames (0 = off)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="stop after N processed frames (0 = run to the end)",
    )


def add_mp_arguments(parser: argparse.ArgumentParser) -> None:
    """Which GPU each model gets, and how deep the in-flight ring is."""
    parser.add_argument(
        "--segment-gpus",
        default="0",
        help="multi-process: one segment worker per GPU, e.g. 0 or 0,1 (default: 0)",
    )
    parser.add_argument(
        "--vehicle-gpu",
        type=int,
        default=1,
        help="multi-process: GPU for the vehicle detector (default: 1)",
    )
    parser.add_argument(
        "--line-gpu",
        type=int,
        default=2,
        help="multi-process: GPU for the lane-line detector (default: 2)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=4,
        help=(
            "multi-process: frames in flight, i.e. shared-memory slots; the "
            "throughput/latency/memory knob (default: 4)"
        ),
    )
    parser.add_argument(
        "--mask-capacity",
        type=int,
        default=32,
        help=(
            "multi-process: viaduct instances per frame; each costs H*W bytes of "
            "shared memory (default: 32)"
        ),
    )


def validate_pipeline_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject a bad layout before it reaches a worker process."""
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if args.window < 1:
        parser.error("--window must be a positive integer")
    if args.mask_capacity < 1:
        parser.error("--mask-capacity must be a positive integer")
    try:
        parse_gpus(args.segment_gpus)
    except ValueError as exc:
        parser.error(str(exc))
    if args.vehicle_gpu < 0 or args.line_gpu < 0:
        parser.error("--vehicle-gpu and --line-gpu must be non-negative")


def pipeline_settings_from(args: argparse.Namespace) -> "PipelineSettings":
    """The parsed CLI as pipeline settings.

    Imported lazily: ``factory`` reaches into ``tracking``, which imports
    ultralytics and therefore torch, and ``--help`` must not pay for that.
    """
    from algorithm.runtime.factory import default_settings  # noqa: PLC0415

    return default_settings(
        segment_imgsz=args.segment_imgsz,
        detect_imgsz=args.detect_imgsz,
        line_imgsz=args.line_imgsz,
        viaduct_confidence=args.viaduct_conf,
        vehicle_confidence=args.vehicle_conf,
        line_confidence=args.line_conf,
        car_width_m=args.car_width_m,
    )


def mp_settings_from(args: argparse.Namespace) -> MpPipelineSettings:
    """The process layout, with the GPU list string parsed once."""
    return MpPipelineSettings(
        segment_gpus=parse_gpus(args.segment_gpus),
        vehicle_gpu=args.vehicle_gpu,
        line_gpu=args.line_gpu,
        window=args.window,
        mask_capacity=args.mask_capacity,
    )


def mp_layout(settings: MpPipelineSettings) -> str:
    """One line describing where the three models landed: same text everywhere."""
    return (
        f"segment GPUs {list(settings.segment_gpus)},"
        f" vehicle GPU {settings.vehicle_gpu}, line GPU {settings.line_gpu},"
        f" window {settings.window}, mask capacity {settings.mask_capacity}"
    )


def build_pipeline_from(
    args: argparse.Namespace, settings: "PipelineSettings", *, fps: float
) -> "MultiProcessPipeline":
    """The one engine, wired from the parsed CLI: the entry points call this."""
    from algorithm.runtime.factory import build_pipeline  # noqa: PLC0415

    mp_settings = mp_settings_from(args)
    print(f"[mp] {mp_layout(mp_settings)}", flush=True)
    return build_pipeline(settings, mp_settings, fps=fps)
