"""Perception: model adapters, image post-processing, tracking, and the policy.

Everything that knows what the models see lives here, and nothing here knows how
the models are scheduled -- the engine (:mod:`algorithm.engine`) drives these
adapters from its worker processes and imports this package, never the other way
round.

    raw             SDK-independent helpers (tensor conversion, weight checks)
    segment_model   YOLO(seg) -> raw deck masks
    detect_models   YOLO(detect) -> DetectedObject (vehicles and lane lines)
    masking         deck union and blackout
    tracking        ByteTracker per label
    viaduct_tracker greedy mask IoU, stable deck ids
    assembly        StageResult + ObservationAssembler: merge the stage outputs
                    and decide the frame's validity
"""
