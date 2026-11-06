"""How many viaduct instances does the segmenter actually emit per frame?

The multi-process pipeline pre-allocates a fixed number of mask slots per frame,
so the default capacity has to cover the real distribution, not the typical
two-deck picture.  Run this after changing the viaduct weights or threshold::

    python scripts/count_instances.py --video /path/to/1080p.mp4 --frames 300
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

import cv2

from algorithm.perception.segment_model import ViaductSegmentAdapter
from algorithm.runtime.factory import DEFAULT_VIADUCT_WEIGHT, default_settings


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = default_settings(
        segment_imgsz=args.imgsz, viaduct_confidence=args.confidence
    )
    adapter = ViaductSegmentAdapter(
        DEFAULT_VIADUCT_WEIGHT,
        device=args.device,
        imgsz=settings.segment_imgsz,
        confidence=settings.viaduct_confidence,
        expected_label=settings.viaduct_label,
    )
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    counts: Counter[int] = Counter()
    index = kept = 0
    while kept < args.frames:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        index += 1
        if index % args.stride:
            continue
        kept += 1
        instances = adapter.predict(frame)
        counts[len(instances)] += 1
    capture.release()

    total = sum(counts.values())
    print(f"frames={total} video={args.video} threshold={args.confidence}")
    for count in sorted(counts):
        share = counts[count] / total
        print(f"  {count:2d} instances: {counts[count]:5d} frames ({share:6.1%})")
    print(f"max={max(counts)} p99={_percentile(counts, total, 0.99)}")
    return 0


def _percentile(counts: Counter[int], total: int, quantile: float) -> int:
    target = max(1, int(round(quantile * total)))
    seen = 0
    for count in sorted(counts):
        seen += counts[count]
        if seen >= target:
            return count
    return max(counts)


if __name__ == "__main__":
    raise SystemExit(main())
