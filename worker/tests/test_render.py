"""The ``#N`` painted on screen is the frame's ordinal, not the deck counter.

A relay that runs for minutes mints a fresh ``deck-000006`` after every scene
cut, but an operator reading the picture must still see ``#1``/``#2``.
"""

import numpy as np

from algorithm.domain.geometry import Box, Mask
from algorithm.domain.observations import (
    DetectedObject,
    Diagnostics,
    Observation,
    TrackedObject,
)
from algorithm.domain.vehicle_speed_runtime import VehicleSpeedUpdate
from algorithm.domain.viaduct_runtime import ViaductDeck, ViaductUpdate
from algorithm.domain.congestion import CongestionLevel
from drone_worker import render

SIDE = 64


def _mask(columns: slice) -> np.ndarray:
    mask = np.zeros((SIDE, SIDE), dtype=np.bool_)
    mask[:, columns] = True
    return mask


def _box(mask: np.ndarray) -> Box:
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    return Box(
        columns[0] / SIDE,
        rows[0] / SIDE,
        (columns[-1] + 1) / SIDE,
        (rows[-1] + 1) / SIDE,
    )


def _track(track_id: str, mask: np.ndarray) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        label="viaduct",
        confidence=0.9,
        box=_box(mask),
        mask=Mask(mask),
    )


def _observation(track_ids: tuple[str, str]) -> Observation:
    tracks = (_track(track_ids[0], _mask(slice(0, 20))), _track(track_ids[1], _mask(slice(40, 60))))
    return Observation(
        timestamp_ms=1000,
        frame_sequence=1,
        generation=0,
        frame_valid=True,
        viaduct=tuple(
            DetectedObject(label=item.label, confidence=item.confidence, box=item.box, mask=item.mask)
            for item in tracks
        ),
        viaduct_tracks=tracks,
        vehicle=(),
        vehicle_tracks=(),
        line_tracks=(),
        unknown_reasons=(),
        diagnostics=Diagnostics(segment_ms=1.0, detect_ms=1.0, line_ms=1.0, processing_ms=3.0),
    )


def _update(track_ids: tuple[str, str]) -> ViaductUpdate:
    decks = tuple(
        ViaductDeck(
            deck_id=track_id,
            density=0.15,
            mean_speed_mps=8.0,
            vehicles=3,
            congestion=CongestionLevel.CLEAR,
        )
        for track_id in track_ids
    )
    return ViaductUpdate(
        timestamp_ms=1000,
        valid=True,
        decks=decks,
        speed=VehicleSpeedUpdate(timestamp_ms=1000, valid=True, speeds=()),
    )


def _painted_text(monkeypatch, observation: Observation) -> list[str]:
    texts: list[str] = []

    def fake_text(canvas, text, origin, scale, colour):
        texts.append(text)

    monkeypatch.setattr(render, "_text", fake_text)
    frame = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
    render.annotate(frame, observation, viaduct=_update(("deck-000006", "deck-000007")))
    return texts


def test_display_numbers_start_at_one_even_when_the_track_ids_are_high(monkeypatch):
    texts = _painted_text(monkeypatch, _observation(("deck-000006", "deck-000007")))

    assert "#1" in texts
    assert "#2" in texts
    # The raw track ids must never leak into the picture.
    assert not any("000006" in text or "000007" in text for text in texts)


def test_numbering_follows_the_screen_from_left_to_right(monkeypatch):
    # deck-000007 is the left-hand deck on purpose: the lower id must not win
    # the numbering, the leftmost deck must.
    painted: list[tuple[str, int]] = []

    def fake_text(canvas, text, origin, scale, colour):
        painted.append((text, origin[0]))

    monkeypatch.setattr(render, "_text", fake_text)
    observation = _observation(("deck-000007", "deck-000006"))
    render.annotate(np.zeros((SIDE, SIDE, 3), dtype=np.uint8), observation, viaduct=_update(("deck-000007", "deck-000006")))

    first = next(x for text, x in painted if text == "#1")
    second = next(x for text, x in painted if text == "#2")
    assert first < second


def test_the_same_deck_keeps_its_colour_across_track_id_drift():
    # Ordered by track_id, so deck-000006 stays the first palette colour even
    # when the counter has advanced past 1.
    tracks = [_track("deck-000006", _mask(slice(0, 20))), _track("deck-000007", _mask(slice(40, 60)))]

    colours = render._deck_colours(sorted(tracks, key=lambda track: track.track_id))

    assert colours["deck-000006"] == render._DECK_COLOURS[0]
    assert colours["deck-000007"] == render._DECK_COLOURS[1]
