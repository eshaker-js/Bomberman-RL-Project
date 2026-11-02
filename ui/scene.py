"""What to draw where, as plain data. Pure: no pygame.

A scene is a list of ``Item``s in draw order. ``key`` names a sprite (see
``ui.sprites.Sprites``) and ``x``, ``y`` are pixel offsets inside the arena.
Everything here is derived from real ``get_state()`` dicts only; the scene
never decides anything the engine decides.

Facing is the one thing the engine does not have. It is inferred from
movement between two states and kept while a character stands still.
"""

from typing import NamedTuple

from game.environment import Cell

from ui.config import TILE

LAYER_TILE = 0
LAYER_GATE = 1
LAYER_BOMB = 2
LAYER_FLAME = 3
LAYER_CHARACTER = 4

DEFAULT_FACING = "down"
_FACING_BY_DELTA = {(1, 0): "right", (-1, 0): "left", (0, -1): "up", (0, 1): "down"}
_TILE_NAMES = {Cell.EMPTY: "floor", Cell.WALL: "wall", Cell.BLOCK: "crate"}


class Item(NamedTuple):
    layer: int
    key: tuple
    x: int
    y: int


def initial_facings(state: dict) -> dict[int, str]:
    return {c["id"]: DEFAULT_FACING for c in state["characters"]}


def update_facings(facings: dict[int, str], prev: dict, cur: dict) -> dict[int, str]:
    """Facing after the tick ``prev`` -> ``cur``: the direction moved, else unchanged."""
    result = dict(facings)
    for before, after in zip(prev["characters"], cur["characters"]):
        delta = (after["x"] - before["x"], after["y"] - before["y"])
        if delta in _FACING_BY_DELTA:
            result[after["id"]] = _FACING_BY_DELTA[delta]
    return result


def gate_key(gate_state: str, steps: int) -> tuple | None:
    """Sprite of the gate, or ``None`` while it is hidden under its crate."""
    if gate_state == "hidden":
        return None
    if gate_state == "open":
        return ("gate", "open", 0)
    return ("gate", "closed", 0)


def draw_order(characters: list[dict]) -> list[dict]:
    """Characters may share a cell; draw higher IDs first so the human is on top."""
    return sorted(characters, key=lambda c: -c["id"])


def static_scene(state: dict, facings: dict[int, str]) -> list[Item]:
    """The scene exactly as ``state`` is: every object on its cell, nothing in between."""
    items = []
    for y, row in enumerate(state["grid"]):
        for x, cell in enumerate(row):
            items.append(Item(LAYER_TILE, ("tile", _TILE_NAMES[cell]), x * TILE, y * TILE))
    gate = state["gate"]
    key = gate_key(gate["state"], state["steps"])
    if key is not None:
        items.append(Item(LAYER_GATE, key, gate["x"] * TILE, gate["y"] * TILE))
    for bomb in state["bombs"]:
        items.append(Item(LAYER_BOMB, ("bomb", 0), bomb["x"] * TILE, bomb["y"] * TILE))
    for x, y in sorted(state["explosion"]):
        items.append(Item(LAYER_FLAME, ("flame", "steady"), x * TILE, y * TILE))
    for c in draw_order(state["characters"]):
        if not c["dead"]:
            facing = facings.get(c["id"], DEFAULT_FACING)
            items.append(Item(LAYER_CHARACTER, ("char", c["id"], facing, 0),
                              c["x"] * TILE, c["y"] * TILE))
    return items
