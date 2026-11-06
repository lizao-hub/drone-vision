"""Annotated frames for eyeballing the pipeline: off-deck tint and per-deck stats.

The deck region is left untouched and everything off the decks is tinted blue, so
the viaduct shows up by what is not covered.  Annotation policy:

    * one colour per deck, used both for its outline and for its stats line, so
      the numbers at the bottom-left always point at a visible deck;
    * a vehicle is drawn only when it has a measured ground speed -- a box with
      the speed inside it; without a speed it gets no annotation at all;
    * lane markings are not drawn (they are still measured, just not shaded);
    * a frame with no deck measurement draws no stats line at all.
"""

from __future__ import annotations

import cv2
import numpy as np

from algorithm.domain.geometry import Box
from algorithm.domain.observations import Observation, TrackedObject
from algorithm.domain.viaduct_runtime import ViaductUpdate

_OUTSIDE_COLOUR = (255, 90, 40)  # BGR blue: everything off the viaduct decks
_CAR_COLOUR = (70, 210, 70)
_SPEED_COLOUR = (0, 255, 190)
_DECK_COLOURS = (
    (80, 200, 80),  # #1 green
    (0, 165, 255),  # #2 orange
    (255, 0, 255),  # #3 magenta
    (255, 255, 0),  # #4 cyan
)
_STAT_LINE_HEIGHT = 18


def annotate(
    frame: np.ndarray,
    observation: Observation,
    *,
    viaduct: ViaductUpdate | None = None,
) -> np.ndarray:
    canvas = frame.copy()
    decks = [track for track in observation.viaduct_tracks if track.mask is not None]
    colours = _deck_colours(viaduct, decks)
    union = _union_mask(decks)
    if union is not None:
        canvas = _tint_outside(canvas, union)
    _outline_decks(canvas, decks, colours)
    _draw_speeds(canvas, observation, viaduct)
    _draw_deck_stats(canvas, viaduct, colours)
    return canvas


# -- off-deck tint ---------------------------------------------------------
def _union_mask(decks: list[TrackedObject]) -> np.ndarray | None:
    """Union of this frame's deck masks, or ``None`` when there is no deck."""
    if not decks:
        return None
    union = np.zeros(decks[0].mask.data.shape, dtype=np.bool_)
    for deck in decks:
        union |= deck.mask.data
    return union


def _tint_outside(canvas: np.ndarray, union: np.ndarray) -> np.ndarray:
    overlay = canvas.copy()
    overlay[~union] = _OUTSIDE_COLOUR
    return cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0.0)


# -- per-deck anchor -------------------------------------------------------
def _deck_colours(
    viaduct: ViaductUpdate | None, decks: list[TrackedObject]
) -> dict[str, tuple[int, int, int]]:
    """Stable colour per deck id: the palette index follows the sorted ids."""
    if viaduct is not None and viaduct.valid:
        deck_ids = [item.deck_id for item in viaduct.decks]
    else:
        deck_ids = sorted(deck.track_id for deck in decks)
    return {
        deck_id: _DECK_COLOURS[index % len(_DECK_COLOURS)]
        for index, deck_id in enumerate(deck_ids)
    }


def _outline_decks(
    canvas: np.ndarray,
    decks: list[TrackedObject],
    colours: dict[str, tuple[int, int, int]],
) -> None:
    """Outline each deck in its colour and tag it with the same ``#id``."""
    for deck in sorted(decks, key=lambda track: track.track_id):
        colour = colours.get(deck.track_id, _DECK_COLOURS[0])
        mask = deck.mask.data.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, 2, cv2.LINE_AA)
        rows = np.flatnonzero(mask.any(axis=1))
        columns = np.flatnonzero(mask.any(axis=0))
        if rows.size and columns.size:
            origin = (int(columns[0]) + 2, max(int(rows[0]) - 5, 12))
            _text(canvas, f"#{_short_id(deck.track_id)}", origin, 0.5, colour)


# -- vehicles and their speeds ---------------------------------------------
def _draw_speeds(
    canvas: np.ndarray, observation: Observation, viaduct: ViaductUpdate | None
) -> None:
    """Box + speed for vehicles with a measured speed; nothing otherwise."""
    if viaduct is None:
        return
    speeds = {item.track_id: item.speed_kmh for item in viaduct.speed.speeds}
    for track in observation.vehicle_tracks:
        speed_kmh = speeds.get(track.track_id)
        if speed_kmh is None:
            continue
        x1, y1, x2, y2 = _pixels(track.box, canvas.shape)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _CAR_COLOUR, 2)
        scale = min(max((y2 - y1) / 90.0, 0.30), 0.55)
        text = f"{speed_kmh:.0f}km/h"
        (_, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
        )
        origin = (x1 + 2, min(y1 + text_height + 2, y2 - 2))
        _text(canvas, text, origin, scale, _SPEED_COLOUR)


# -- bottom-left deck statistics -------------------------------------------
def _draw_deck_stats(
    canvas: np.ndarray,
    viaduct: ViaductUpdate | None,
    colours: dict[str, tuple[int, int, int]],
) -> None:
    """One line per deck, in the deck's own colour; nothing when the frame has none."""
    if viaduct is None or not viaduct.valid or not viaduct.decks:
        return
    baseline = canvas.shape[0] - 10
    for index, item in enumerate(reversed(viaduct.decks)):
        colour = colours.get(item.deck_id, _DECK_COLOURS[0])
        speed = "-- km/h" if item.mean_speed_kmh is None else f"{item.mean_speed_kmh:.0f}km/h"
        text = (
            f"#{_short_id(item.deck_id)}  {item.density:.1%}  n{item.vehicles}"
            f"  {speed}  {item.congestion.value}"
        )
        origin = (10, baseline - index * _STAT_LINE_HEIGHT)
        _text(canvas, text, origin, 0.5, colour)


# -- primitives ------------------------------------------------------------
def _text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    colour: tuple[int, int, int],
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def _short_id(track_id: str) -> str:
    suffix = track_id.rsplit("-", 1)[-1]
    return suffix.lstrip("0") or "0"


def _pixels(box: Box, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = shape[0], shape[1]
    x1 = min(max(int(box.x1 * width), 0), width - 1)
    y1 = min(max(int(box.y1 * height), 0), height - 1)
    x2 = min(max(int(box.x2 * width), x1 + 1), width - 1)
    y2 = min(max(int(box.y2 * height), y1 + 1), height - 1)
    return x1, y1, x2, y2
