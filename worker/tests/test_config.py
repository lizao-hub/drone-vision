from dataclasses import replace

import pytest

from algorithm.config import DEFAULT_SETTINGS, WORKER_ROOT, PipelineSettings


def test_default_settings_carry_the_bundled_weights_and_config_defaults():
    # The only deployment-dependent values are the three weight paths.
    assert DEFAULT_SETTINGS == PipelineSettings(
        viaduct_weight=WORKER_ROOT / "models" / "viaduct" / "weights" / "best.pt",
        vehicle_weight=WORKER_ROOT / "models" / "vehicle" / "weights" / "best.pt",
        line_weight=WORKER_ROOT / "models" / "line" / "weights" / "best.pt",
    )
    assert (DEFAULT_SETTINGS.segment_imgsz, DEFAULT_SETTINGS.detect_imgsz) == (960, 640)
    assert DEFAULT_SETTINGS.line_imgsz == 640
    assert DEFAULT_SETTINGS.viaduct_confidence == 0.25
    assert DEFAULT_SETTINGS.vehicle_confidence == 0.30
    assert DEFAULT_SETTINGS.line_confidence == 0.30
    assert DEFAULT_SETTINGS.car_width_m == 1.95
    assert DEFAULT_SETTINGS.track_max_lost_ms == 1000
    assert DEFAULT_SETTINGS.device == 0
    assert DEFAULT_SETTINGS.project == WORKER_ROOT / "output"


def test_settings_still_reject_invalid_values():
    with pytest.raises(ValueError):
        replace(DEFAULT_SETTINGS, mask_min_confidence=2.0)
