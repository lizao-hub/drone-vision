"""Annotated frames for eyeballing the pipeline: off-deck tint and per-deck stats.

The deck region is left untouched and everything off the decks is tinted blue, so
the viaduct shows up by what is not covered.  Annotation policy:

    * one colour per deck, used both for its outline and for its stats line, so
      the numbers at the bottom-left always point at a visible deck;
    * the ``#N`` shown to the operator is the deck's left-to-right position in
      the frame, not its global ``deck-000006`` track id: a long-lived relay
      mints a fresh id after every scene cut, and #6 on screen reads as a bug;
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
    # Left-to-right, so #1 is the deck an operator would call "the first one";
    # the track id only breaks ties, and the counter never reaches the picture.
    ordered = sorted(decks, key=lambda track: (track.box.x1, track.box.y1, track.track_id))
    numbers = {track.track_id: index + 1 for index, track in enumerate(ordered)}
    colours = _deck_colours(ordered)
    union = _union_mask(decks)
    if union is not None:
        canvas = _tint_outside(canvas, union)
    _outline_decks(canvas, decks, colours, numbers)
    _draw_speeds(canvas, observation, viaduct)
    _draw_deck_stats(canvas, viaduct, colours, numbers)
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
    """Blend a flat colour under everything outside the decks.

    The obvious implementation -- copy, boolean-index the outside, then
    ``addWeighted`` the whole frame -- costs 23 ms at 1080p because the boolean
    scatter touches 1.5M pixels one by one.  Blending the whole frame and then
    copying the deck pixels back is ~4x faster and pixel-identical.
    """
    solid = np.empty_like(canvas)
    solid[...] = _OUTSIDE_COLOUR
    tinted = cv2.addWeighted(canvas, 0.70, solid, 0.30, 0.0)
    cv2.copyTo(canvas, union.view(np.uint8), tinted)
    return tinted


# -- per-deck anchor -------------------------------------------------------
def _deck_colours(decks: list[TrackedObject]) -> dict[str, tuple[int, int, int]]:
    """Stable colour per deck id: the palette follows the display order.

    ``decks`` must already be in display order (left to right), so the same
    physical deck keeps its colour while the relay is running.
    """
    return {
        deck.track_id: _DECK_COLOURS[index % len(_DECK_COLOURS)]
        for index, deck in enumerate(decks)
    }


def _outline_decks(
    canvas: np.ndarray,
    decks: list[TrackedObject],
    colours: dict[str, tuple[int, int, int]],
    numbers: dict[str, int],
) -> None:
    """Outline each deck in its colour and tag it with the same ``#N``."""
    for deck in sorted(decks, key=lambda track: track.track_id):
        colour = colours.get(deck.track_id, _DECK_COLOURS[0])
        # A bool mask is already 1 byte per pixel, so the view is free; the
        # astype it replaces allocated and copied 2 MB per deck per frame.
        mask = deck.mask.data.view(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, 2, cv2.LINE_AA)
        rows = np.flatnonzero(mask.any(axis=1))
        columns = np.flatnonzero(mask.any(axis=0))
        if rows.size and columns.size:
            origin = (int(columns[0]) + 2, max(int(rows[0]) - 5, 12))
            number = numbers.get(deck.track_id, _short_id(deck.track_id))
            _text(canvas, f"#{number}", origin, 0.5, colour)


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
    numbers: dict[str, int],
) -> None:
    """One line per deck, in the deck's own colour; nothing when the frame has none."""
    if viaduct is None or not viaduct.valid or not viaduct.decks:
        return
    baseline = canvas.shape[0] - 10
    # Draw the highest number first (bottom line), so the block reads #1..#N
    # top-down regardless of the order the segmenter returned the decks in.
    order = sorted(
        viaduct.decks,
        key=lambda item: numbers.get(item.deck_id, -1),
        reverse=True,
    )
    for index, item in enumerate(order):
        colour = colours.get(item.deck_id, _DECK_COLOURS[0])
        number = numbers.get(item.deck_id, _short_id(item.deck_id))
        speed = "-- km/h" if item.mean_speed_kmh is None else f"{item.mean_speed_kmh:.0f}km/h"
        text = (
            f"#{number}  {item.density:.1%}  n{item.vehicles}"
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
