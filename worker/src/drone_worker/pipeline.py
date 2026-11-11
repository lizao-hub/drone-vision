"""``pipeline`` subcommand: the two-card split run over a local video.

Same input and the same JSONL shape as ``batch``, one extra card's worth of
throughput.  ``--device 0,1`` splits the work -- segmentation on the first card,
both detectors on the second -- while a single device keeps both stages on one
card.  The split run writes JSONL only; the per-frame mp4 belongs to ``batch``.
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
from drone_worker.args import (
    add_device,
    add_model_overrides,
    add_sampling,
    add_source,
    settings_from_args,
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m drone_worker pipeline",
        description="Two-card split pipeline: segment on the first device, detect on the second.",
    )
    add_source(parser, help_text="source video file")
    add_device(parser, pair=True)
    add_sampling(parser)
    add_model_overrides(parser)
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="not supported: the split pipeline writes JSONL only",
    )
    args = parser.parse_args(argv)
    if len(args.device) > 2:
        parser.error("--device takes at most two indices here: segment card, detect card")
    return args


def _settings_with_devices(args: argparse.Namespace) -> PipelineSettings:
    """Shared knobs plus ``devices`` when two cards were named."""
    settings = settings_from_args(DEFAULT_SETTINGS, args)
    if len(args.device) == 2:
        settings = replace(settings, devices=tuple(args.device))
    return settings


def main_pipeline(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.save_video:
        print(
            "error: pipeline does not write video; drop --save-video",
            file=sys.stderr,
        )
        return 2
    settings = _settings_with_devices(args)

    input_path = Path(args.source).resolve()
    if not input_path.is_file():
        print(f"error: source does not exist: {input_path}", file=sys.stderr)
        return 2
    project = settings.project
    project.mkdir(parents=True, exist_ok=True)
    json_path = project / f"{input_path.stem}_par.jsonl"

    source = FileSequenceSource(input_path, frame_stride=args.stride)
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
