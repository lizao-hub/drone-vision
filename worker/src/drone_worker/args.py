"""Shared command line helpers for the worker entry points."""

from __future__ import annotations


def parse_device(value: str) -> int | str:
    """``"0"`` -> ``0``, ``"cpu"`` -> ``"cpu"``: ultralytics accepts both."""
    text = value.strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text
