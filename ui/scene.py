"""What to draw where, as plain data. Pure: no pygame.

A scene is a list of ``Item``s in draw order. ``key`` names a sprite (see
``ui.sprites.Sprites``) and ``x``, ``y`` are pixel offsets inside the arena.
Everything here is derived from real ``get_state()`` dicts only; the scene
never decides anything the engine decides.

``static_scene(cur)`` is the state exactly as it is. ``scene(prev, cur, t)``
animates the tick from ``prev`` to ``cur`` with ``t`` from 0 to 1. Every
effect ends by ``t = 1``, where ``scene`` equals ``static_scene(cur)``, so
at the end of every tick the screen matches the state. "Died this tick",
"crate broke" and "gate changed" are differences between ``prev`` and
``cur``, never guesses.

Facing is the one thing the engine does not have. It is inferred from
movement between two states and kept while a character stands still.
"""

from typing import NamedTuple

from game.environment import BOMB_FUSE, Cell

from ui.art import FLAME_FADE_LEVELS
from ui.config import TILE

LAYER_TILE = 0
LAYER_GATE = 1
LAYER_BOMB = 2
LAYER_FLAME = 3
LAYER_EFFECT = 4  # crate debris and gate blinks, drawn over the flames
LAYER_CHARACTER = 5

DEFAULT_FACING = "down"
_FACING_BY_DELTA = {(1, 0): "right", (-1, 0): "left", (0, -1): "up", (0, 1): "down"}
_TILE_NAMES = {Cell.EMPTY: "floor", Cell.WALL: "wall", Cell.BLOCK: "crate"}

# Animation timing, as fractions of one tick.
WALK_STRIDE_END = 0.5   # stride frame while t < this, standing frame after
FLAME_GROW = (0.15, 0.35)  # small burst until the first, medium until the second
CRATE_BREAK_SPLIT = 0.3  # cracked crate until this, debris until CRATE_BREAK_END
CRATE_BREAK_END = 0.6
GATE_BLINK_END = 0.5
GATE_BLINKS = 4  # on/off blinks per tick while t < GATE_BLINK_END
DEATH_FRAME_COUNT = 3


class Item(NamedTuple):
    layer: int
    key: tuple
    x: int
    y: int


# ------------------------------------------------------------------ facing

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


# ------------------------------------------------------- animation helpers

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def slide(prev_cell: tuple[int, int], cur_cell: tuple[int, int], t: float) -> tuple[int, int]:
    """Pixel position between two cells, rounded to whole pixels."""
    return (round(lerp(prev_cell[0] * TILE, cur_cell[0] * TILE, t)),
            round(lerp(prev_cell[1] * TILE, cur_cell[1] * TILE, t)))


def walk_frame(moved: bool, t: float) -> int:
    """Two-frame walk cycle: stride during the first half of a move, then stand."""
    return 1 if moved and t < WALK_STRIDE_END else 0


def bomb_pulses_per_tick(countdown: int) -> int:
    """1 pulse per tick at a full fuse, one more each tick it burns down."""
    return max(1, BOMB_FUSE + 1 - countdown)


def bomb_frame(countdown: int, t: float) -> int:
    """Pulse frame during the second half of each pulse; normal at ``t`` 0 and 1."""
    phase = (t * bomb_pulses_per_tick(countdown)) % 1.0
    return 1 if phase >= 0.5 else 0


def flame_frame(t: float) -> int | str:
    if t < FLAME_GROW[0]:
        return 0
    if t < FLAME_GROW[1]:
        return 1
    return "steady"


def flame_fade_level(ticks_left: int, flame_ticks: int) -> int:
    """0 for a flame burning its first tick, then one level dimmer per tick burned.

    Capped at ``FLAME_FADE_LEVELS``. It depends on the state only, so a
    lingering flame looks the same all through a tick.
    """
    return min(max(flame_ticks - ticks_left, 0), FLAME_FADE_LEVELS)


def flame_key(level: int, grow: int | str = "steady") -> tuple:
    """Sprite of a flame: growing/steady at level 0, else a faded steady flame."""
    if level == 0:
        return ("flame", grow)
    return ("flame", "fade", level)


def death_frame(t: float) -> int:
    return min(int(t * DEATH_FRAME_COUNT), DEATH_FRAME_COUNT - 1)


def broken_crates(prev: dict, cur: dict) -> list[tuple[int, int]]:
    """Cells that were crates in ``prev`` and are floor in ``cur``."""
    return [
        (x, y)
        for y, (old_row, new_row) in enumerate(zip(prev["grid"], cur["grid"]))
        for x, (old, new) in enumerate(zip(old_row, new_row))
        if old is Cell.BLOCK and new is not Cell.BLOCK
    ]


def gate_changed(prev: dict, cur: dict) -> bool:
    return prev["gate"]["state"] != cur["gate"]["state"]


def died_this_tick(prev: dict, cur: dict) -> list[tuple[dict, dict]]:
    """``(before, after)`` of every character alive in ``prev`` and dead in ``cur``."""
    return [
        (before, after)
        for before, after in zip(prev["characters"], cur["characters"])
        if not before["dead"] and after["dead"]
    ]


def gate_key(gate_state: str, steps: int) -> tuple | None:
    """Sprite of the gate, or ``None`` while it is hidden under its crate.

    The open gate shimmers once per tick, so its frame comes from the tick
    counter in the state, never from wall-clock time.
    """
    if gate_state == "hidden":
        return None
    if gate_state == "open":
        return ("gate", "open", steps % 2)
    return ("gate", "closed", 0)


def draw_order(characters: list[dict]) -> list[dict]:
    """Characters may share a cell; draw higher IDs first so the human is on top."""
    return sorted(characters, key=lambda c: -c["id"])


# ------------------------------------------------------------------ scenes

def _tiles(state: dict) -> list[Item]:
    return [
        Item(LAYER_TILE, ("tile", _TILE_NAMES[cell]), x * TILE, y * TILE)
        for y, row in enumerate(state["grid"])
        for x, cell in enumerate(row)
    ]


def _gate(state: dict) -> list[Item]:
    gate = state["gate"]
    key = gate_key(gate["state"], state["steps"])
    if key is None:
        return []
    return [Item(LAYER_GATE, key, gate["x"] * TILE, gate["y"] * TILE)]


def static_scene(state: dict, facings: dict[int, str]) -> list[Item]:
    """The scene exactly as ``state`` is: every object on its cell, nothing in between."""
    items = _tiles(state) + _gate(state)
    for bomb in state["bombs"]:
        items.append(Item(LAYER_BOMB, ("bomb", 0), bomb["x"] * TILE, bomb["y"] * TILE))
    flame_ticks = state["rules"]["flame_ticks"]
    for flame in state["flames"]:
        level = flame_fade_level(flame["ticks_left"], flame_ticks)
        items.append(Item(LAYER_FLAME, flame_key(level), flame["x"] * TILE, flame["y"] * TILE))
    for c in draw_order(state["characters"]):
        if not c["dead"]:
            facing = facings.get(c["id"], DEFAULT_FACING)
            items.append(Item(LAYER_CHARACTER, ("char", c["id"], facing, 0),
                              c["x"] * TILE, c["y"] * TILE))
    return items


def scene(prev: dict, cur: dict, t: float, facings: dict[int, str]) -> list[Item]:
    """The tick from ``prev`` to ``cur`` at ``t`` in [0, 1]. ``facings`` are after the tick.

    Bombs, flames, crates and the gate are those of ``cur`` from the start of
    the tick (they are the result of the tick); only characters move between
    the two states. Freshly burned cells grow into a steady flame; flames
    still burning from earlier ticks show their fade level the whole tick;
    a flame that went out (in ``prev`` but not ``cur``) is gone from the
    start. Short effects (crate debris, gate blink, death) end within the
    tick.
    """
    t = min(max(t, 0.0), 1.0)
    items = _tiles(cur) + _gate(cur)

    for bomb in cur["bombs"]:
        items.append(Item(LAYER_BOMB, ("bomb", bomb_frame(bomb["countdown"], t)),
                          bomb["x"] * TILE, bomb["y"] * TILE))

    grow = flame_frame(t)
    flame_ticks = cur["rules"]["flame_ticks"]
    fresh = set(cur["explosion"])
    for flame in cur["flames"]:
        cell = (flame["x"], flame["y"])
        level = flame_fade_level(flame["ticks_left"], flame_ticks)
        key = flame_key(level, grow if cell in fresh else "steady")
        items.append(Item(LAYER_FLAME, key, cell[0] * TILE, cell[1] * TILE))

    if t < CRATE_BREAK_END:
        frame = 0 if t < CRATE_BREAK_SPLIT else 1
        for x, y in broken_crates(prev, cur):
            items.append(Item(LAYER_EFFECT, ("debris", frame), x * TILE, y * TILE))
    if (gate_changed(prev, cur) and t < GATE_BLINK_END
            and int(t / GATE_BLINK_END * GATE_BLINKS) % 2 == 0):
        gate = cur["gate"]
        items.append(Item(LAYER_EFFECT, ("gate", "flash", 0), gate["x"] * TILE, gate["y"] * TILE))

    before_by_id = {c["id"]: c for c in prev["characters"]}
    for c in draw_order(cur["characters"]):
        before = before_by_id[c["id"]]
        if before["dead"]:
            continue
        old_cell, new_cell = (before["x"], before["y"]), (c["x"], c["y"])
        x, y = slide(old_cell, new_cell, t)
        if c["dead"]:
            if t < 1.0:
                items.append(Item(LAYER_CHARACTER, ("death", c["id"], death_frame(t)), x, y))
            continue
        facing = facings.get(c["id"], DEFAULT_FACING)
        frame = walk_frame(old_cell != new_cell, t)
        items.append(Item(LAYER_CHARACTER, ("char", c["id"], facing, frame), x, y))
    return items
