import numpy as np
import pytest

from algorithm.domain.geometry import Box, Mask
from algorithm.domain.observations import DetectedObject
from algorithm.perception.masking import blackout_off_deck


def _detection(mask: np.ndarray, confidence: float = 0.9) -> DetectedObject:
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    box = Box(
        float(columns[0]) / mask.shape[1],
        float(rows[0]) / mask.shape[0],
        float(columns[-1] + 1) / mask.shape[1],
        float(rows[-1] + 1) / mask.shape[0],
    )
    return DetectedObject(label="viaduct", confidence=confidence, box=box, mask=Mask(mask))


def _frame(height: int = 4, width: int = 6) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_every_instance_keeps_its_pixels():
    left = np.zeros((4, 6), dtype=np.bool_)
    left[:, :3] = True
    right = np.zeros((4, 6), dtype=np.bool_)
    right[:, 3:] = True

    masked = blackout_off_deck(_frame(), (_detection(left), _detection(right)))

    # Taking only the most confident instance would black out half the road.
    assert (masked == 255).all()


def test_confidence_does_not_filter_instances_off():
    # Filtering happens once in the pipeline; the blackout must not drop again.
    mask = np.zeros((4, 6), dtype=np.bool_)
    mask[1, 1] = True

    masked = blackout_off_deck(_frame(), (_detection(mask, confidence=0.01),))

    assert int(masked[1, 1, 0]) == 255
    assert int(masked.sum()) == 255 * 3


def test_no_instances_yields_no_image():
    assert blackout_off_deck(_frame(), ()) is None


def test_empty_masks_yield_no_image():
    mask = np.zeros((4, 6), dtype=np.bool_)
    instance = DetectedObject(
        label="viaduct", confidence=0.9, box=Box(0.0, 0.0, 1.0, 1.0), mask=Mask(mask)
    )
    assert blackout_off_deck(_frame(), (instance,)) is None


def test_geometry_mismatch_is_rejected():
    mask = np.zeros((3, 6), dtype=np.bool_)
    mask[1, 1] = True
    with pytest.raises(ValueError):
        blackout_off_deck(_frame(), (_detection(mask),))


def test_every_instance_must_carry_a_mask():
    without_mask = DetectedObject(
        label="viaduct", confidence=0.9, box=Box(0.1, 0.1, 0.2, 0.2)
    )
    with pytest.raises(ValueError):
        blackout_off_deck(_frame(), (without_mask,))


def test_off_deck_pixels_are_blacked_out_in_a_copy():
    mask = np.array([[True, False, False], [True, True, False]], dtype=np.bool_)
    frame = np.full((2, 3, 3), 255, dtype=np.uint8)

    masked = blackout_off_deck(frame, (_detection(mask),))

    assert (masked[mask] == 255).all()
    assert (masked[~mask] == 0).all()
    assert (frame == 255).all()  # the source frame is never touched
