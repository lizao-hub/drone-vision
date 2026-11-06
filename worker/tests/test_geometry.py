import numpy as np
import pytest

from algorithm.domain.geometry import Box, Mask


def test_box_rejects_unordered_and_unnormalized_coordinates():
    with pytest.raises(ValueError):
        Box(0.5, 0.1, 0.4, 0.2)
    with pytest.raises(ValueError):
        Box(-0.1, 0.1, 0.4, 0.2)
    with pytest.raises(ValueError):
        Box(0.1, 0.1, 1.2, 0.2)


def test_box_intersection_and_iou():
    left = Box(0.0, 0.0, 0.5, 0.5)
    right = Box(0.25, 0.25, 0.75, 0.75)
    assert left.intersection(right) == pytest.approx(0.0625)
    assert left.iou(right) == pytest.approx(0.0625 / (0.25 + 0.25 - 0.0625))
    assert left.contains(0.5, 0.5)
    assert left.center == (0.25, 0.25)


def test_mask_copies_and_freezes_without_touching_the_caller():
    source = np.zeros((4, 6), dtype=np.bool_)
    source[1:3, 2:5] = True
    mask = Mask(source)
    assert source.flags.writeable is True
    assert mask.data.flags.writeable is False
    assert (mask.height, mask.width) == (4, 6)
    assert mask.data.sum() == 6


def test_empty_mask():
    mask = Mask(np.zeros((3, 5), dtype=np.bool_))
    assert not mask.data.any()


def test_mask_rejects_wrong_dimensionality():
    with pytest.raises(ValueError):
        Mask(np.zeros((3, 5, 1), dtype=np.bool_))
