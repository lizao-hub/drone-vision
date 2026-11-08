"""RTMP output: push raw BGR frames into an ffmpeg process.

OpenCV's ``VideoWriter`` cannot be trusted to hand an RTMP URL to a muxer, so the
sink spawns ffmpeg and writes ``bgr24`` frames to its stdin instead.  The process
is restarted by the caller when it dies (broken pipe), which is the normal failure
mode of a live stream.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

DEFAULT_BITRATE = "4M"
DEFAULT_FFMPEG = "ffmpeg"


def ffmpeg_binary(explicit: str | None = None) -> str:
    """``--ffmpeg`` > ``$FFMPEG_BIN`` > ``PATH`` > the imageio-ffmpeg wheel."""
    for candidate in (explicit, os.environ.get("FFMPEG_BIN"), shutil.which(DEFAULT_FFMPEG)):
        if candidate:
            return candidate
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "ffmpeg not found: install it, set FFMPEG_BIN, or `pip install imageio-ffmpeg`"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def build_ffmpeg_args(
    *,
    width: int,
    height: int,
    fps: float,
    url: str,
    bitrate: str = DEFAULT_BITRATE,
) -> list[str]:
    """ffmpeg arguments (without the binary) for raw BGR stdin -> FLV/RTMP."""
    if width <= 0 or height <= 0:
        raise ValueError("frame size must be positive")
    if not fps > 0.0:
        raise ValueError("fps must be positive")
    if not url:
        raise ValueError("destination url must be non-empty")
    gop = max(1, int(round(fps)))
    return [
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-y",  # RTMP never needs it; a file destination otherwise refuses to start
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-profile:v",
        "baseline",
        "-bf",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        bitrate,
        "-maxrate",
        bitrate,
        "-bufsize",
        bitrate,
        "-g",
        str(gop),
        "-f",
        "flv",
        "-rw_timeout",
        "5000000",
        url,
    ]


class FfmpegSink:
    """One ffmpeg process accepting fixed-size BGR frames on stdin."""

    def __init__(
        self,
        url: str,
        *,
        width: int,
        height: int,
        fps: float,
        bitrate: str = DEFAULT_BITRATE,
        ffmpeg: str | None = None,
    ) -> None:
        if not isinstance(url, str) or not url:
            raise ValueError("url must be a non-empty string")
        if width <= 0 or height <= 0:
            raise ValueError("frame size must be positive")
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self._binary = ffmpeg_binary(ffmpeg)
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self) -> "FfmpegSink":
        if self._proc is not None:
            return self
        self._proc = subprocess.Popen(
            [self._binary, *build_ffmpeg_args(
                width=self.width,
                height=self.height,
                fps=self.fps,
                url=self.url,
                bitrate=self.bitrate,
            )],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=None,
        )
        return self

    def write(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an H x W x 3 BGR array")
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"frame size {frame.shape[1]}x{frame.shape[0]} does not match "
                f"{self.width}x{self.height}"
            )
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("sink is not started")
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self._proc.stdin.flush()

    def close(self) -> None:
        process, self._proc = self._proc, None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - ffmpeg hung
                process.kill()
                process.wait(timeout=2.0)

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def __enter__(self) -> "FfmpegSink":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.close()
