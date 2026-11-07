from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from algorithm.runtime.profile import (
    DEFAULT_PROFILE,
    PipelineProfile,
    load_profile,
)


def test_shipped_profile_loads_and_maps_onto_pipeline_settings():
    profile = load_profile()
    settings = profile.to_settings(device=0)

    assert settings.segment_imgsz == 960
    assert settings.detect_imgsz == 640
    assert settings.line_imgsz == 640
    assert settings.viaduct_confidence == 0.25
    assert settings.vehicle_confidence == 0.30
    assert settings.line_confidence == 0.30
    assert settings.car_width_m == 1.95
    assert settings.device == 0
    assert settings.track_max_lost_ms == 1000
    assert settings.viaduct_weight.is_file()
    assert settings.vehicle_weight.is_file()
    assert settings.line_weight.is_file()


def test_device_and_tracker_timeout_are_per_run_overrides():
    profile = load_profile()

    assert profile.to_settings(device="cpu").device == "cpu"
    assert profile.to_settings(device=0, track_max_lost_ms=3000).track_max_lost_ms == 3000


def test_unknown_keys_are_rejected():
    document = yaml.safe_load(DEFAULT_PROFILE.read_text(encoding="utf-8"))
    document["segment_imgsz"] = 1280  # was a CLI flag once; the profile owns it now

    with pytest.raises(ValidationError):
        PipelineProfile.model_validate(document)


def test_out_of_range_confidence_is_rejected():
    document = yaml.safe_load(DEFAULT_PROFILE.read_text(encoding="utf-8"))
    document["vehicle"]["confidence"] = 1.5

    with pytest.raises(ValidationError):
        PipelineProfile.model_validate(document)


def test_load_profile_rejects_a_non_mapping_document(tmp_path: Path):
    path = tmp_path / "profile.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")

    with pytest.raises(TypeError):
        load_profile(path)


def test_relative_weights_resolve_against_the_worker_root():
    profile = load_profile()

    assert not Path(profile.vehicle.weights).is_absolute()
    assert profile.vehicle.weight_path(Path("/tmp/worker")) == Path(
        "/tmp/worker/models/vehicle/weights/best.pt"
    )
