import numpy as np
import pytest

from algorithm.domain.geometry import Box, Mask
from algorithm.domain.observations import DetectedObject
from algorithm.perception.viaduct_tracker import ViaductTracker

FRAME = 10


def _mask(columns: slice) -> np.ndarray:
    mask = np.zeros((FRAME, FRAME), dtype=np.bool_)
    mask[:, columns] = True
    return mask


def _left() -> np.ndarray:
    return _mask(slice(0, 4))


def _right() -> np.ndarray:
    return _mask(slice(6, 10))


def _deck(mask: np.ndarray | None) -> DetectedObject:
    if mask is None:
        return DetectedObject(label="viaduct", confidence=0.9, box=Box(0.0, 0.0, 1.0, 1.0))
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    box = Box(
        columns[0] / FRAME,
        rows[0] / FRAME,
        (columns[-1] + 1) / FRAME,
        (rows[-1] + 1) / FRAME,
    )
    return DetectedObject(label="viaduct", confidence=0.9, box=box, mask=Mask(mask))


def test_a_single_instance_gets_an_id_and_keeps_its_mask():
    tracker = ViaductTracker()
    instance = _deck(_left())
    [track] = tracker.update(0, (instance,))

    assert track.track_id == "deck-000001"
    assert track.label == "viaduct"
    assert track.mask is instance.mask


def test_ids_follow_the_masks_not_the_instance_order():
    tracker = ViaductTracker()
    tracker.update(0, (_deck(_left()), _deck(_right())))
    right, left = tracker.update(125, (_deck(_right()), _deck(_left())))

    assert right.track_id == "deck-000002"
    assert left.track_id == "deck-000001"


def test_a_deck_that_drops_out_for_a_frame_keeps_its_id():
    tracker = ViaductTracker()
    tracker.update(0, (_deck(_left()), _deck(_right())))
    tracker.update(125, (_deck(_left()),))
    left, right = tracker.update(250, (_deck(_left()), _deck(_right())))

    assert left.track_id == "deck-000001"
    assert right.track_id == "deck-000002"


def test_a_deck_gone_longer_than_the_gap_gets_a_new_id():
    tracker = ViaductTracker(max_gap_ms=100)
    tracker.update(0, (_deck(_left()),))
    [later] = tracker.update(200, (_deck(_left()),))

    assert later.track_id == "deck-000002"


def test_a_new_deck_gets_a_new_id():
    tracker = ViaductTracker()
    tracker.update(0, (_deck(_left()),))
    left, right = tracker.update(125, (_deck(_left()), _deck(_right())))

    assert left.track_id == "deck-000001"
    assert right.track_id == "deck-000002"


def test_non_overlapping_masks_do_not_match():
    tracker = ViaductTracker()
    tracker.update(0, (_deck(_left()),))
    [other] = tracker.update(125, (_deck(_right()),))

    assert other.track_id == "deck-000002"


def test_instances_without_a_mask_are_not_tracked():
    tracker = ViaductTracker()
    tracks = tracker.update(0, (_deck(None), _deck(_left())))

    assert [item.track_id for item in tracks] == ["deck-000001"]


def test_reset_drops_the_known_decks():
    tracker = ViaductTracker()
    tracker.update(0, (_deck(_left()),))
    tracker.reset()
    [again] = tracker.update(125, (_deck(_left()),))

    assert again.track_id == "deck-000002"


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        ViaductTracker(match_threshold=0.0)
    with pytest.raises(ValueError):
        ViaductTracker(max_gap_ms=0)
