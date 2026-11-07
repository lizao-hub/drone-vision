import numpy as np

from algorithm.domain.geometry import Box, Mask
from algorithm.domain.observations import DetectedObject, TrackedObject
from algorithm.perception.pipeline import ObservationPipeline
from algorithm.perception.raw import RawDetection
from algorithm.perception.viaduct_tracker import ViaductTracker


def _deck_mask(height: int = 4, width: int = 6) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[:, : width // 2] = True
    return mask


def _segmentation(mask: np.ndarray, confidence: float = 0.9) -> DetectedObject:
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    box = Box(
        float(columns[0]) / mask.shape[1],
        float(rows[0]) / mask.shape[0],
        float(columns[-1] + 1) / mask.shape[1],
        float(rows[-1] + 1) / mask.shape[0],
    )
    return DetectedObject(label="viaduct", confidence=confidence, box=box, mask=Mask(mask))


class FakeSegmenter:
    def __init__(self, mask: np.ndarray | None = None, *, error: bool = False) -> None:
        self.mask = _deck_mask() if mask is None else mask
        self.error = error

    def predict(self, frame: np.ndarray) -> tuple[DetectedObject, ...]:
        if self.error:
            raise RuntimeError("segment backend exploded")
        if not self.mask.any():
            return ()
        return (_segmentation(self.mask),)


class RecordingDetector:
    def __init__(self, detections: tuple[RawDetection, ...] = (), *, error: bool = False) -> None:
        self.detections = detections
        self.error = error
        self.seen_frames: list[np.ndarray] = []

    def predict(self, frame: np.ndarray) -> tuple[RawDetection, ...]:
        self.seen_frames.append(frame.copy())
        if self.error:
            raise RuntimeError("detect backend exploded")
        return self.detections


class FakeTracker:
    def __init__(self) -> None:
        self.seen: list[tuple[DetectedObject, ...]] = []

    def update(
        self,
        timestamp_ms: int,
        detections: tuple[DetectedObject, ...],
        *,
        width: int,
        height: int,
        generation: int = 0,
    ) -> tuple[TrackedObject, ...]:
        self.seen.append(detections)
        return tuple(
            TrackedObject(
                track_id=f"{item.label}-bt-{index + 1:06d}",
                label=item.label,
                confidence=item.confidence,
                box=item.box,
            )
            for index, item in enumerate(detections)
        )


def _frame(height: int = 4, width: int = 6) -> np.ndarray:
    return np.full((height, width, 3), 180, dtype=np.uint8)


def _build(**overrides) -> ObservationPipeline:
    """Build the pipeline with fake models; all three models are mandatory."""
    parts = {
        "viaduct_segment": FakeSegmenter(),
        "vehicle_detection": RecordingDetector(),
        "vehicle_tracker": FakeTracker(),
        "line_detection": RecordingDetector(),
        "line_tracker": FakeTracker(),
        "viaduct_tracker": ViaductTracker(),
    }
    parts.update(overrides)
    return ObservationPipeline(**parts)


def test_pipeline_blacks_out_the_frame_before_detecting():
    mask = _deck_mask()
    detector = RecordingDetector()
    pipeline = _build(
        viaduct_segment=FakeSegmenter(mask),
        vehicle_detection=detector,
    )
    observation = pipeline.process(_frame(), 1000, 0)

    assert observation.frame_valid
    assert observation.vehicle_tracks == ()
    assert observation.line_tracks == ()
    assert observation.vehicle == ()
    assert len(observation.viaduct) == 1
    assert observation.viaduct[0].box == Box(0.0, 0.0, 3 / 6, 1.0)
    assert observation.viaduct[0].mask is not None
    assert [item.track_id for item in observation.viaduct_tracks] == ["deck-000001"]
    assert observation.viaduct_tracks[0].mask is observation.viaduct[0].mask
    assert observation.unknown_reasons == ()
    sent = detector.seen_frames[0]
    assert (sent[:, 3:] == 0).all()
    assert (sent[:, :3] == 180).all()


def test_pipeline_tracks_every_detection_from_the_masked_frame():
    # The masked image is the only filter: whatever the detector returns here is
    # published and tracked as-is, with no centre-in-mask post-filter.
    mask = _deck_mask()
    on_deck = RawDetection("car", 0.8, (0.0, 0.0, 0.3, 0.5))
    off_deck = RawDetection("car", 0.8, (0.6, 0.0, 0.9, 0.5))
    tracker = FakeTracker()
    pipeline = _build(
        viaduct_segment=FakeSegmenter(mask),
        vehicle_detection=RecordingDetector((on_deck, off_deck)),
        vehicle_tracker=tracker,
    )
    observation = pipeline.process(_frame(), 1000, 0)

    assert observation.vehicle == (on_deck.to_detected(), off_deck.to_detected())
    assert [item.track_id for item in observation.vehicle_tracks] == [
        "car-bt-000001",
        "car-bt-000002",
    ]
    assert len(tracker.seen[0]) == 2


def test_segmentation_error_invalidates_the_frame():
    pipeline = _build(viaduct_segment=FakeSegmenter(error=True))
    observation = pipeline.process(_frame(), 1000, 0)

    assert not observation.frame_valid
    assert observation.unknown_reasons == ("viaduct_adapter_error",)
    assert observation.viaduct == ()
    assert observation.vehicle == ()
    assert observation.vehicle_tracks == ()
    assert observation.line_tracks == ()


def test_detection_error_invalidates_the_frame():
    pipeline = _build(
        viaduct_segment=FakeSegmenter(_deck_mask()),
        vehicle_detection=RecordingDetector(error=True),
    )
    observation = pipeline.process(_frame(), 1000, 0)

    assert not observation.frame_valid
    assert observation.unknown_reasons == ("vehicle_adapter_error",)


def test_empty_segmentation_skips_detection_without_invalidating():
    detector = RecordingDetector()
    line_detector = RecordingDetector()
    pipeline = _build(
        viaduct_segment=FakeSegmenter(np.zeros((4, 6), dtype=np.bool_)),
        vehicle_detection=detector,
        line_detection=line_detector,
    )
    observation = pipeline.process(_frame(), 1000, 0)

    assert observation.frame_valid
    assert observation.viaduct == ()
    assert observation.vehicle == ()
    assert observation.line_tracks == ()
    assert observation.unknown_reasons == ("viaduct_not_detected",)
    assert detector.seen_frames == []
    assert line_detector.seen_frames == []


def test_empty_frame_after_segmentation_skips_detection():
    # No hold-over: an empty segmentation frame is simply not detected.
    segmenter = FakeSegmenter(_deck_mask())
    detector = RecordingDetector()
    pipeline = _build(viaduct_segment=segmenter, vehicle_detection=detector)
    pipeline.process(_frame(), 1000, 0)
    segmenter.mask = np.zeros((4, 6), dtype=np.bool_)
    observation = pipeline.process(_frame(), 1200, 1)

    assert observation.frame_valid
    assert observation.unknown_reasons == ("viaduct_not_detected",)
    assert observation.viaduct == ()
    assert len(detector.seen_frames) == 1


def test_invalid_frame_geometry_is_rejected():
    pipeline = _build()
    observation = pipeline.process(np.zeros((4, 6), dtype=np.uint8), 1000, 0)

    assert not observation.frame_valid
    assert observation.unknown_reasons == ("frame_invalid",)


def test_every_viaduct_instance_keeps_its_own_mask():
    left = np.zeros((4, 6), dtype=np.bool_)
    left[:, :2] = True
    right = np.zeros((4, 6), dtype=np.bool_)
    right[:, 4:] = True

    class TwoInstances:
        def predict(self, frame: np.ndarray) -> tuple[DetectedObject, ...]:
            return (_segmentation(left), _segmentation(right))

    pipeline = _build(viaduct_segment=TwoInstances())
    observation = pipeline.process(_frame(), 1000, 0)

    assert len(observation.viaduct) == 2
    assert [item.mask.data.sum() for item in observation.viaduct] == [8, 8]
    assert all(item.label == "viaduct" for item in observation.viaduct)


def test_line_tracks_are_reported_and_see_the_blacked_out_frame():
    marking = RawDetection("line", 0.8, (0.0, 0.0, 0.2, 1.0))
    line_detector = RecordingDetector((marking,))
    line_tracker = FakeTracker()
    pipeline = _build(
        viaduct_segment=FakeSegmenter(_deck_mask()),
        line_detection=line_detector,
        line_tracker=line_tracker,
    )
    observation = pipeline.process(_frame(), 1000, 0)

    assert observation.frame_valid
    assert [item.track_id for item in observation.line_tracks] == ["line-bt-000001"]
    assert observation.line_tracks[0].label == "line"
    assert observation.vehicle_tracks == ()
    assert line_tracker.seen[0] == (marking.to_detected(),)
    # Lane markings are read from the same blacked-out frame as the vehicles.
    assert (line_detector.seen_frames[0][:, 3:] == 0).all()


def test_line_adapter_error_invalidates_the_frame():
    pipeline = _build(
        viaduct_segment=FakeSegmenter(_deck_mask()),
        line_detection=RecordingDetector(error=True),
    )
    observation = pipeline.process(_frame(), 1000, 0)

    assert not observation.frame_valid
    assert observation.unknown_reasons == ("line_adapter_error",)
    assert observation.line_tracks == ()


def test_line_tracker_error_invalidates_the_frame():
    class BrokenTracker(FakeTracker):
        def update(self, timestamp_ms, detections, *, width, height, generation=0):
            raise RuntimeError("tracker exploded")

    pipeline = _build(
        viaduct_segment=FakeSegmenter(_deck_mask()),
        line_tracker=BrokenTracker(),
    )
    observation = pipeline.process(_frame(), 1000, 0)

    assert not observation.frame_valid
    assert observation.unknown_reasons == ("line_tracker_error",)


def test_deck_ids_survive_an_instance_order_swap():
    left = np.zeros((4, 6), dtype=np.bool_)
    left[:, :2] = True
    right = np.zeros((4, 6), dtype=np.bool_)
    right[:, 4:] = True

    class SwappingSegmenter:
        def __init__(self) -> None:
            self.swap = False

        def predict(self, frame: np.ndarray) -> tuple[DetectedObject, ...]:
            first, second = _segmentation(left), _segmentation(right)
            return (second, first) if self.swap else (first, second)

    segmenter = SwappingSegmenter()
    pipeline = _build(viaduct_segment=segmenter)
    first = pipeline.process(_frame(), 1000, 0)
    segmenter.swap = True
    second = pipeline.process(_frame(), 1125, 3)

    first_ids = {item.track_id for item in first.viaduct_tracks}
    second_ids = {item.track_id for item in second.viaduct_tracks}
    assert first_ids == second_ids == {"deck-000001", "deck-000002"}
    # Instance 0 is the right deck on the second frame, so it keeps the right id.
    assert second.viaduct_tracks[0].track_id == "deck-000002"
    assert second.viaduct_tracks[0].mask is second.viaduct[0].mask


def test_viaduct_tracker_error_invalidates_the_frame():
    class BrokenDeckTracker(ViaductTracker):
        def update(self, timestamp_ms, instances):
            raise RuntimeError("deck tracker exploded")

    pipeline = _build(viaduct_segment=FakeSegmenter(_deck_mask()), viaduct_tracker=BrokenDeckTracker())
    observation = pipeline.process(_frame(), 1000, 0)

    assert not observation.frame_valid
    assert observation.unknown_reasons == ("viaduct_tracker_error",)
    assert observation.viaduct_tracks == ()
