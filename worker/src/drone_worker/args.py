"""Shared command line helpers for the worker entry points.

The three entries differ in topology, not in vocabulary: all of them take a
``--source``, a ``--device`` and the same sampling knobs, so a command line can
move between ``batch``, ``pipeline`` and ``relay`` without re-reading ``--help``.

Device spec: ``0`` is one card, ``cpu`` is the CPU, ``0,1`` is two cards (the
split pipeline only).  Each entry validates how many devices it can drive.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from algorithm.config import PipelineSettings


def parse_device(value: str) -> int | str:
    """``"0"`` -> ``0``, ``"cpu"`` -> ``"cpu"``: ultralytics accepts both."""
    text = value.strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def parse_device_spec(raw: str) -> tuple[int | str, ...]:
    """``"0"`` -> ``(0,)``; ``"cpu"`` -> ``("cpu",)``; ``"0,1"`` -> ``(0, 1)``."""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            "expected a GPU index, 'cpu', or a pair such as 0,1"
        )
    return tuple(parse_device(part) for part in parts)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _unit_float(raw: str) -> float:
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be within [0, 1]")
    return value


# -- flag groups -----------------------------------------------------------
def add_source(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument("--source", required=True, help=help_text)


def add_device(
    parser: argparse.ArgumentParser, *, pair: bool = False, default: str = "0"
) -> None:
    if pair:
        help_text = (
            "GPU index, 'cpu', or two comma-separated indices such as 0,1 "
            f"(default: {default})"
        )
    else:
        help_text = f"GPU index or 'cpu' (default: {default})"
    parser.add_argument(
        "--device",
        default=default,
        type=parse_device_spec,
        metavar="DEVICE",
        help=help_text,
    )


def add_sampling(parser: argparse.ArgumentParser, *, stride_default: int = 3) -> None:
    parser.add_argument(
        "--stride",
        type=_positive_int,
        default=stride_default,
        help=f"run inference on every Nth source frame (default: {stride_default})",
    )
    parser.add_argument(
        "--log-every", type=int, default=50, help="progress line interval"
    )


def add_model_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--segment-imgsz", type=_positive_int, default=None,
        help="segmentation input size (default: the PipelineSettings value)",
    )
    parser.add_argument(
        "--detect-imgsz", type=_positive_int, default=None,
        help="vehicle detection input size (default: the PipelineSettings value)",
    )
    parser.add_argument(
        "--line-imgsz", type=_positive_int, default=None,
        help="lane-line detection input size (default: the PipelineSettings value)",
    )
    parser.add_argument("--viaduct-conf", type=_unit_float, default=None,
                        help="segmentation confidence (default: the settings value)")
    parser.add_argument("--vehicle-conf", type=_unit_float, default=None,
                        help="vehicle confidence (default: the settings value)")
    parser.add_argument("--line-conf", type=_unit_float, default=None,
                        help="lane-line confidence (default: the settings value)")


# -- settings assembly -----------------------------------------------------
def settings_from_args(settings: PipelineSettings, args: argparse.Namespace) -> PipelineSettings:
    """``settings`` plus the shared knobs the CLI was given.

    Anything the CLI did not mention keeps its ``PipelineSettings`` value, so
    tuning stays in :mod:`algorithm.config`.  ``args.device`` is the parsed
    device spec; the first entry becomes ``device``, and the caller adds
    ``devices`` when it drives a split run.
    """
    overrides: dict[str, object] = {"device": args.device[0]}
    for field, value in (
        ("segment_imgsz", getattr(args, "segment_imgsz", None)),
        ("detect_imgsz", getattr(args, "detect_imgsz", None)),
        ("line_imgsz", getattr(args, "line_imgsz", None)),
        ("viaduct_confidence", getattr(args, "viaduct_conf", None)),
        ("vehicle_confidence", getattr(args, "vehicle_conf", None)),
        ("line_confidence", getattr(args, "line_conf", None)),
    ):
        if value is not None:
            overrides[field] = value
    return replace(settings, **overrides)
