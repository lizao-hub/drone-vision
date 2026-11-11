"""The three entry points share one flag vocabulary.

``batch``, ``pipeline`` and ``relay`` differ in topology, not in how a command
line looks: every one of them takes ``--source``, ``--device``, ``--stride``,
``--log-every`` and the model overrides.  Only the number of devices each entry
can actually drive is entry-specific.
"""

import argparse

import pytest

from drone_worker.args import parse_device_spec


def test_device_spec_accepts_one_card_cpu_and_a_pair():
    assert parse_device_spec("0") == (0,)
    assert parse_device_spec("cpu") == ("cpu",)
    assert parse_device_spec("0,1") == (0, 1)
    assert parse_device_spec(" 2 , 3 ") == (2, 3)


def test_device_spec_rejects_an_empty_value():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_device_spec(" , ")


def test_batch_drives_one_device():
    from drone_worker.batch import _parse_args as batch_args

    assert batch_args(["--source", "x.mp4"]).device == (0,)
    with pytest.raises(SystemExit):
        batch_args(["--source", "x.mp4", "--device", "0,1"])


def test_pipeline_accepts_one_or_two_devices():
    from drone_worker.pipeline import _parse_args as pipeline_args

    assert pipeline_args(["--source", "x.mp4"]).device == (0,)
    assert pipeline_args(["--source", "x.mp4", "--device", "0"]).device == (0,)
    assert pipeline_args(["--source", "x.mp4", "--device", "0,1"]).device == (0, 1)
    with pytest.raises(SystemExit):
        pipeline_args(["--source", "x.mp4", "--device", "0,1,2"])


def test_pipeline_refuses_the_video_output_of_batch(tmp_path):
    from drone_worker.pipeline import main_pipeline

    # The refusal happens before the source is resolved, so a missing file is
    # fine here.
    assert main_pipeline(["--source", str(tmp_path / "missing.mp4"), "--save-video"]) == 2


def test_batch_writes_video_only_when_asked():
    from drone_worker.batch import _parse_args as batch_args

    assert batch_args(["--source", "x.mp4"]).save_video is False
    assert batch_args(["--source", "x.mp4", "--save-video"]).save_video is True


SHARED_FLAGS = (
    "source",
    "device",
    "stride",
    "log_every",
    "segment_imgsz",
    "detect_imgsz",
    "line_imgsz",
    "viaduct_conf",
    "vehicle_conf",
    "line_conf",
)


@pytest.mark.parametrize(
    ("name", "argv"),
    [
        ("batch", ["--source", "x.mp4"]),
        ("pipeline", ["--source", "x.mp4"]),
        ("relay", ["--source", "rtsp://a", "--dest", "rtmp://b"]),
    ],
)
def test_every_entry_exposes_the_shared_flags(name, argv):
    from drone_worker import batch, pipeline, relay

    parse = {
        "batch": batch._parse_args,
        "pipeline": pipeline._parse_args,
        "relay": relay._parse_args,
    }[name]

    args = parse(argv)
    for flag in SHARED_FLAGS:
        assert hasattr(args, flag), f"{name} is missing --{flag.replace('_', '-')}"


def test_car_width_is_not_a_runtime_flag():
    # The per-car metric width is a calibration constant, not a per-run knob.
    from drone_worker.relay import _parse_args as relay_args

    with pytest.raises(SystemExit):
        relay_args(["--source", "rtsp://a", "--dest", "rtmp://b", "--car-width-m", "2.0"])
