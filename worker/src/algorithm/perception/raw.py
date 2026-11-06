"""SDK-neutral helpers that keep torch objects out of the pipeline.

Adapters convert ultralytics ``Results`` into domain ``DetectedObject`` values
here; everything below this module is numpy/domain only.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any


def sdk_to_list(value: Any) -> list[Any]:
    """Detach an SDK tensor (GPU or CPU) into a plain Python list."""
    if value is None:
        return []
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        return converted if isinstance(converted, list) else list(converted)
    return list(value)


def clamp_unit(value: float) -> float:
    """Clamp a detector coordinate into the normalized unit square."""
    return max(0.0, min(1.0, value))


def verify_weight_file(path: Path, expected_sha256: str | None = None) -> Path:
    """Fail fast on a missing or unexpected model file."""
    if not path.is_file():
        raise FileNotFoundError(f"model weight does not exist: {path}")
    if expected_sha256 is None:
        return path
    if len(expected_sha256) != 64:
        raise ValueError("expected SHA-256 must contain 64 hexadecimal characters")
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected_sha256.lower():
        raise ValueError(
            f"model weight SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return path
