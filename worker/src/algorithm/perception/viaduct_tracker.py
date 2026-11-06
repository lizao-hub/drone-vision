"""Deck identity across frames: greedy mask-IoU matching of viaduct instances.

The segmenter has no notion of "this is the same deck as last frame", yet the
deck id is what makes per-deck statistics comparable over time.  Instances are
matched to the previous tracks by **mask IoU** (greedy, best pair first); an
unmatched instance starts a new id, and a track that disappears is remembered
for ``max_gap_ms`` so a one-frame segmentation miss does not mint a new id.

Tracks carry the instance mask: downstream consumers (density) need the pixels,
not just the box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from algorithm.domain.observations import DetectedObject, TrackedObject


@dataclass(slots=True)
class _DeckState:
    track_id: str
    mask: np.ndarray
    seen_ms: int


class ViaductTracker:
    def __init__(self, *, match_threshold: float = 0.3, max_gap_ms: int = 750) -> None:
        if not 0.0 < match_threshold <= 1.0:
            raise ValueError("match_threshold must be within (0, 1]")
        if max_gap_ms <= 0:
            raise ValueError("max_gap_ms must be positive")
        self.match_threshold = match_threshold
        self.max_gap_ms = max_gap_ms
        self._decks: dict[str, _DeckState] = {}
        self._counter = 0

    def update(
        self, timestamp_ms: int, instances: tuple[DetectedObject, ...]
    ) -> tuple[TrackedObject, ...]:
        candidates = [item for item in instances if item.mask is not None]
        ids = self._assign_ids(timestamp_ms, candidates)
        self._remember(timestamp_ms, candidates, ids)
        return tuple(
            TrackedObject(
                track_id=ids[index],
                label=item.label,
                confidence=item.confidence,
                box=item.box,
                mask=item.mask,
            )
            for index, item in enumerate(candidates)
        )

    def reset(self) -> None:
        self._decks.clear()

    # -- internals ---------------------------------------------------------
    def _assign_ids(
        self, timestamp_ms: int, instances: list[DetectedObject]
    ) -> dict[int, str]:
        ids: dict[int, str] = {}
        used: set[str] = set()
        scored: list[tuple[float, int, str]] = []
        for index, instance in enumerate(instances):
            for deck in self._decks.values():
                if timestamp_ms - deck.seen_ms > self.max_gap_ms:
                    continue
                iou = _mask_iou(instance.mask.data, deck.mask)
                if iou >= self.match_threshold:
                    scored.append((iou, index, deck.track_id))
        scored.sort(reverse=True)
        for _, index, track_id in scored:
            if index in ids or track_id in used:
                continue
            ids[index] = track_id
            used.add(track_id)
        for index in range(len(instances)):
            if index not in ids:
                self._counter += 1
                ids[index] = f"deck-{self._counter:06d}"
        return ids

    def _remember(
        self,
        timestamp_ms: int,
        instances: list[DetectedObject],
        ids: dict[int, str],
    ) -> None:
        states = {
            ids[index]: _DeckState(ids[index], instance.mask.data, timestamp_ms)
            for index, instance in enumerate(instances)
        }
        for track_id, state in self._decks.items():
            if track_id in states:
                continue
            if timestamp_ms - state.seen_ms <= self.max_gap_ms:
                states[track_id] = state
        self._decks = states


_IOU_STRIDE = 4
_IOU_MIN_SIDE = 128


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    # Full-resolution comparison is memory-bandwidth bound at 1080p (2 MB per
    # mask, four whole-array passes per pair).  Deck masks are large solid
    # regions, so a strided view is plenty for the 0.3 threshold; small masks
    # (unit tests, odd inputs) are compared exactly.
    if first.shape != second.shape:
        return 0.0
    if min(first.shape) >= _IOU_MIN_SIDE:
        first = first[::_IOU_STRIDE, ::_IOU_STRIDE]
        second = second[::_IOU_STRIDE, ::_IOU_STRIDE]
    intersection = int(np.count_nonzero(first & second))
    if not intersection:
        return 0.0
    return intersection / int(np.count_nonzero(first | second))
