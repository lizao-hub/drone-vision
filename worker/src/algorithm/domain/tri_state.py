"""Three-valued truth for measurements that can be *unknown*.

``bool`` cannot say "not measured": ``False`` would read as a measured zero.
The vehicle-speed runtime already distinguishes ``None`` (no measurable car) from
``0.0`` (a stopped car); :class:`TriState` makes that distinction explicit at the
domain boundary instead of leaving it implicit in an ``Optional[float]``.
"""

from __future__ import annotations

from enum import StrEnum


class TriState(StrEnum):
    """``TRUE`` measured and set, ``FALSE`` measured and zero, ``UNKNOWN`` not measured."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
