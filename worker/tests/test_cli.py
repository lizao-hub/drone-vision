"""Entry-point dispatch: ``batch`` / ``relay`` over the one engine."""

import pytest

import drone_worker.batch as batch
import drone_worker.relay as relay
from drone_worker.args import mp_settings_from
from drone_worker.cli import main


def _capture(monkeypatch, module, name):
    seen: list[list[str]] = []

    def fake(arguments):
        seen.append(list(arguments))
        return 0

    monkeypatch.setattr(module, name, fake)
    return seen


def test_batch_is_the_default_command(monkeypatch):
    seen = _capture(monkeypatch, batch, "main_batch")

    assert main(["--video", "a.mp4"]) == 0
    assert seen == [["--video", "a.mp4"]]


def test_batch_alias_passes_through(monkeypatch):
    seen = _capture(monkeypatch, batch, "main_batch")

    assert main(["batch", "--video", "a.mp4"]) == 0
    assert seen == [["--video", "a.mp4"]]


def test_relay_passes_through(monkeypatch):
    seen = _capture(monkeypatch, relay, "main_relay")

    assert main(["relay", "--source", "s", "--dest", "d"]) == 0
    assert seen == [["--source", "s", "--dest", "d"]]


def test_an_unknown_command_is_an_argument_of_the_default_runner(monkeypatch):
    # ``drone_worker --video x`` and ``drone_worker batch --video x`` must agree,
    # including when the first token is not a command at all.
    seen = _capture(monkeypatch, batch, "main_batch")

    assert main(["relayx", "--video", "a.mp4"]) == 0
    assert seen == [["relayx", "--video", "a.mp4"]]


# Every option both runners share; the two parsers must not drift apart.
_SHARED_ARGV = [
    "--segment-gpus",
    "0,1",
    "--vehicle-gpu",
    "3",
    "--line-gpu",
    "4",
    "--window",
    "2",
    "--mask-capacity",
    "8",
    "--segment-imgsz",
    "800",
    "--detect-imgsz",
    "512",
    "--line-imgsz",
    "512",
    "--viaduct-conf",
    "0.3",
    "--vehicle-conf",
    "0.4",
    "--line-conf",
    "0.5",
    "--car-width-m",
    "2.05",
    "--log-every",
    "10",
    "--max-frames",
    "7",
]
_SHARED_NAMES = (
    "segment_gpus",
    "vehicle_gpu",
    "line_gpu",
    "window",
    "mask_capacity",
    "segment_imgsz",
    "detect_imgsz",
    "line_imgsz",
    "viaduct_conf",
    "vehicle_conf",
    "line_conf",
    "car_width_m",
    "log_every",
    "max_frames",
)


def test_both_runners_accept_the_same_pipeline_options():
    batch_args = batch._parse_args(["--video", "a.mp4", *_SHARED_ARGV])
    relay_args = relay._parse_args(["--source", "s", "--dest", "d", *_SHARED_ARGV])

    for name in _SHARED_NAMES:
        assert getattr(batch_args, name) == getattr(relay_args, name), name


def test_mp_settings_come_from_the_shared_parser():
    args = batch._parse_args(
        ["--video", "a.mp4", "--segment-gpus", "2,3", "--vehicle-gpu", "1",
         "--line-gpu", "0", "--window", "6", "--mask-capacity", "16"]
    )

    settings = mp_settings_from(args)

    assert settings.segment_gpus == (2, 3)
    assert (settings.vehicle_gpu, settings.line_gpu) == (1, 0)
    assert (settings.window, settings.mask_capacity) == (6, 16)


@pytest.mark.parametrize("argv", [["--window", "0"], ["--mask-capacity", "0"],
                                  ["--segment-gpus", "a"], ["--vehicle-gpu", "-1"],
                                  ["--log-every", "-1"]])
def test_both_runners_reject_the_same_bad_layout(argv):
    with pytest.raises(SystemExit):
        batch._parse_args(["--video", "a.mp4", *argv])
    with pytest.raises(SystemExit):
        relay._parse_args(["--source", "s", "--dest", "d", *argv])
