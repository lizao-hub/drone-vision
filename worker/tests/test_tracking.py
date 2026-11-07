import pytest

from algorithm.domain.geometry import Box
from algorithm.domain.observations import DetectedObject
from algorithm.perception.tracking import ByteTracker

WIDTH, HEIGHT = 640, 368
FRAME_MS = 41  # 24 fps


def _detection(label: str, x: float, y: float, size: float = 40.0, confidence: float = 0.9):
    return DetectedObject(
        label=label,
        confidence=confidence,
        box=Box(x / WIDTH, y / HEIGHT, (x + size) / WIDTH, (y + size) / HEIGHT),
    )


def _update(tracker: ByteTracker, timestamp_ms: int, detections):
    return tracker.update(timestamp_ms, detections, width=WIDTH, height=HEIGHT)


def test_identity_is_stable_while_the_vehicle_moves():
    tracker = ByteTracker(labels=("car",))
    ids = []
    for frame in range(3):
        tracks = _update(tracker, frame * FRAME_MS, (_detection("car", 100 + frame * 6, 100),))
        assert len(tracks) == 1
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1


def test_each_label_owns_its_identity_namespace():
    tracker = ByteTracker(labels=("car", "truck"))
    published: list[str] = []
    for timestamp_ms, detection in (
        (0, _detection("car", 100, 100)),
        (FRAME_MS, _detection("car", 100, 100)),
        (2 * FRAME_MS, _detection("truck", 300, 200)),
        (3 * FRAME_MS, _detection("truck", 300, 200)),
    ):
        published.extend(item.track_id for item in _update(tracker, timestamp_ms, (detection,)))

    # ByteTrack publishes a new track once it is confirmed, i.e. on its second
    # sighting; the counter restarts per label, never shared between classes.
    assert published == ["car-bt-000001", "car-bt-000001", "truck-bt-000001"]


def test_weak_detections_keep_identity_but_are_not_published():
    tracker = ByteTracker(labels=("car",), detection_threshold=0.30, low_threshold=0.10)
    strong_first = _update(tracker, 0, (_detection("car", 100, 100, confidence=0.9),))
    weak = _update(tracker, FRAME_MS, (_detection("car", 102, 102, confidence=0.20),))
    strong_again = _update(tracker, 2 * FRAME_MS, (_detection("car", 104, 104, confidence=0.9),))

    assert len(strong_first) == 1
    assert weak == ()
    assert [item.track_id for item in strong_again] == [item.track_id for item in strong_first]


def test_untracked_labels_are_ignored_not_rejected():
    tracker = ByteTracker(labels=("car",))
    assert _update(tracker, 0, (_detection("bus", 100, 100),)) == ()
    assert _update(tracker, FRAME_MS, (_detection("car", 100, 100),)) == ()
    tracks = _update(tracker, 2 * FRAME_MS, (_detection("car", 100, 100),))
    assert [item.label for item in tracks] == ["car"]


def test_timestamps_must_strictly_increase():
    tracker = ByteTracker(labels=("car",))
    _update(tracker, 1000, (_detection("car", 100, 100),))
    with pytest.raises(ValueError):
        _update(tracker, 1000, (_detection("car", 100, 100),))


def test_frame_resize_rebuilds_trackers_without_crashing():
    tracker = ByteTracker(labels=("car",))
    _update(tracker, 0, (_detection("car", 100, 100),))
    resized = tracker.update(
        41,
        (_detection("car", 100, 100),),
        width=1280,
        height=720,
    )
    assert len(resized) == 1
