"""``pipeline`` subcommand: the two-card split run over a local video.

Same input as ``batch``, same JSONL shape, two extra cards' worth of throughput.
The only knob is ``--devices``: the first index segments, the second runs both
detectors, and a third value is refused rather than silently ignored.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from algorithm.config import DEFAULT_SETTINGS, PipelineSettings
from algorithm.runtime.coordinator import WorkerDied, run_parallel
from algorithm.runtime.sources import FileSequenceSource


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m drone_worker pipeline",
        description="Two-card split pipeline: segment on the first GPU, detect on the second.",
    )
    parser.add_argument("--source", type=Path, required=True, help="source video file")
    parser.add_argument(
        "--devices",
        required=True,
        help="two GPU indices, e.g. 0,1: segmentation card first, detection card second",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="run inference on every Nth frame (default: 3)",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="not supported: the split pipeline writes JSONL only",
    )
    return parser.parse_args(argv)


def _settings_with_devices(raw: str) -> PipelineSettings:
    """Parse ``--devices`` and let the settings object enforce the contract."""
    try:
        devices = tuple(int(part) for part in raw.split(","))
    except ValueError as error:
        raise ValueError(
            f"--devices must be comma-separated GPU indices, got {raw!r}"
        ) from error
    return replace(DEFAULT_SETTINGS, devices=devices)


def main_parallel(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.save_video:
        print(
            "error: pipeline does not write video in this version; drop --save-video",
            file=sys.stderr,
        )
        return 2
    try:
        settings = _settings_with_devices(args.devices)
    except (TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    input_path = args.source.resolve()
    if not input_path.is_file():
        print(f"error: source does not exist: {input_path}", file=sys.stderr)
        return 2
    project = settings.project
    project.mkdir(parents=True, exist_ok=True)
    json_path = project / f"{input_path.stem}_par.jsonl"

    source = FileSequenceSource(
        [input_path],
        frame_stride=args.stride,
        realtime=False,
    )
    try:
        run_parallel(
            settings,
            source,
            json_path,
            log_every=args.log_every,
            stride=args.stride,
        )
    except WorkerDied as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"jsonl: {json_path}")
    return 0
