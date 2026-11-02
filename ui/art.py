"""All pixel art as text grids. Pure data: no pygame.

Each sprite is a tuple of equal-length strings, one string per pixel row.
A character is one PICO-8 palette index as a hex digit (``0``-``f``, see
``ui.palette``), ``.`` is transparent, and the placeholders ``P`` (main
colour) and ``Q`` (shade) are replaced per character by
``CHARACTER_COLORS``. Map tiles and character frames are ``TILE`` x
``TILE`` (16 x 16) pixels.

All designs are original to this project.
"""

from ui.config import TILE

TRANSPARENT = "."
PLACEHOLDERS = "PQ"
_HEX = "0123456789abcdef"

# Main and shade colour of each character: 0 blue (the human or agent),
# 1 red, 2 green, 3 orange.
CHARACTER_COLORS = {
    0: {"P": "c", "Q": "1"},
    1: {"P": "8", "Q": "2"},
    2: {"P": "b", "Q": "3"},
    3: {"P": "9", "Q": "4"},
}
# Portrait colours on the card of a dead character.
DEAD_COLORS = {"P": "5", "Q": "0"}


def apply_colors(rows: tuple[str, ...], mapping: dict[str, str]) -> tuple[str, ...]:
    """Return ``rows`` with every character in ``mapping`` replaced."""
    return tuple("".join(mapping.get(ch, ch) for ch in row) for row in rows)


def validate(name: str, rows: tuple[str, ...], size: tuple[int, int] | None = None,
             allow_placeholders: bool = False) -> None:
    """Raise ``ValueError`` if ``rows`` is not a well-formed sprite grid."""
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError(f"sprite {name!r}: rows must all have the same length")
    if size is not None and (len(rows[0]), len(rows)) != size:
        raise ValueError(f"sprite {name!r}: expected {size}, got {(len(rows[0]), len(rows))}")
    allowed = set(_HEX + TRANSPARENT + (PLACEHOLDERS if allow_placeholders else ""))
    for row in rows:
        bad = set(row) - allowed
        if bad:
            raise ValueError(f"sprite {name!r}: unknown characters {sorted(bad)}")


# ---------------------------------------------------------------- map tiles

FLOOR = (
    "3333333333333333",
    "3333333333333333",
    "33b3333333333333",
    "3333333333333b33",
    "3333333333333333",
    "333333b333333333",
    "3333333333333333",
    "3333333333333333",
    "33333333333b3333",
    "3b33333333333333",
    "3333333333333333",
    "3333333b33333333",
    "3333333333333333",
    "333333333333333b",
    "3333b33333333333",
    "3333333333333333",
)

WALL = (
    "7777777777777776",
    "7666666666666665",
    "7666666666666665",
    "7666566666666665",
    "7666666666666665",
    "7666666666656665",
    "7666666666666665",
    "7666666666666665",
    "7666666666666665",
    "7665666666666665",
    "7666666666666665",
    "7666666666666565",
    "7666666666666665",
    "7666666666666665",
    "7655555555555555",
    "6555555555555550",
)

CRATE = (
    "2222222222222222",
    "2999999999999942",
    "2944444444444442",
    "2949444444449442",
    "2944944444494442",
    "2944494444944442",
    "2944449449444442",
    "2944444994444442",
    "2944444994444442",
    "2944449449444442",
    "2944494444944442",
    "2944944444494442",
    "2949444444449442",
    "2944444444444442",
    "2444444444444442",
    "2222222222222222",
)

# --------------------------------------------------------------------- bomb

BOMB = (
    "................",
    "...........a....",
    "..........9a9...",
    "..........4.....",
    ".........4......",
    ".....000000.....",
    "...0011111100...",
    "..011776111110..",
    "..017611111110..",
    ".01161111111110.",
    ".01111111111110.",
    ".01111111111110.",
    "..011111111110..",
    "..011111111110..",
    "...0011111100...",
    ".....000000.....",
)
BOMB_FRAMES = (BOMB,)

# ------------------------------------------------------------------- flames
# One colour scheme for every blast. The last frame is the steady flame that
# stays on a burned cell until the next tick.

FLAME_STEADY = (
    "..8.88888888.8..",
    ".88899999999888.",
    "8899aaaaaaaa9988",
    "899aaaaaaaaaa998",
    "89aaa777777aaa98",
    "89aa77777777aa98",
    "89aa77777777aa98",
    "89aa77777777aa98",
    "89aa77777777aa98",
    "89aa77777777aa98",
    "89aa77777777aa98",
    "89aaa777777aaa98",
    "899aaaaaaaaaa998",
    "8899aaaaaaaa9988",
    ".88899999999888.",
    "..8.88888888.8..",
)
FLAME_FRAMES = (FLAME_STEADY,)

# --------------------------------------------------------------------- gate
# A hatch in the floor: barred while closed, a glowing portal once open.
# A hidden gate is not drawn at all: its crate is still on the grid.

GATE_CLOSED = (
    "0555555555555550",
    "5666666666666665",
    "5655555555555565",
    "565d00d00d00d565",
    "565d00d00d00d565",
    "565d00d00d00d565",
    "565d00d00d00d565",
    "565dddddddddd565",
    "565d00d00d00d565",
    "565d00d00d00d565",
    "565d00d00d00d565",
    "565d00d00d00d565",
    "5655555555555565",
    "5666666666666665",
    "5555555555555555",
    "0555555555555550",
)

_GATE_OPEN_BASE = (
    "0333333333333330",
    "3bbbbbbbbbbbbbb3",
    "3b333333333333b3",
    "3b311111111113b3",
    "3b3111cccc1113b3",
    "3b311cc77cc113b3",
    "3b31cc7777cc13b3",
    "3b31c777777c13b3",
    "3b31c777777c13b3",
    "3b31cc7777cc13b3",
    "3b311cc77cc113b3",
    "3b3111cccc1113b3",
    "3b311111111113b3",
    "3b333333333333b3",
    "3bbbbbbbbbbbbbb3",
    "0333333333333330",
)
GATE_OPEN_FRAMES = (_GATE_OPEN_BASE,)

# --------------------------------------------------------------- characters
# One original design for all four: a round hooded figure with a dark visor.
# Facings are "down", "up" and "right"; "left" is drawn as "right" mirrored.

_CHAR_DOWN = (
    "................",
    "......0000......",
    "....00PPPP00....",
    "...0PPPPPPPP0...",
    "..0PPPPPPPPPP0..",
    "..0PP000000PP0..",
    "..0P07700770P0..",
    "..0P00000000P0..",
    "..0QP000000PQ0..",
    "...0QQPPPPQQ0...",
    "..0PPQQQQQQPP0..",
    ".0f0PPPaaPPP0f0.",
    "..00PPPPPPPP00..",
    "...0QPPPPPPQ0...",
    "....0QQ00QQ0....",
    "....000..000....",
)

_CHAR_UP = (
    "................",
    "......0000......",
    "....00PPPP00....",
    "...0PPPPPPPP0...",
    "..0PPPPPPPPPP0..",
    "..0PPPPPPPPPP0..",
    "..0PPPPQQPPPP0..",
    "..0PPPQQQQPPP0..",
    "..0QPPPQQPPPQ0..",
    "...0QQPPPPQQ0...",
    "..0PPQQQQQQPP0..",
    ".0f0PPPPPPPP0f0.",
    "..00PPPPPPPP00..",
    "...0QPPPPPPQ0...",
    "....0QQ00QQ0....",
    "....000..000....",
)

_CHAR_RIGHT = (
    "................",
    ".....00000......",
    "...00PPPPP00....",
    "..0PPPPPPPPP0...",
    ".0PPPPPPPPPPP0..",
    ".0PPPPP000000P0.",
    ".0QPPP0000770P0.",
    ".0QPPP0000000P0.",
    ".0QQPPP000000Q0.",
    "..0QQQPPPPPPQ0..",
    "..0PPQQQQQQPP0..",
    "...0PPPPPPf0....",
    "...0PPPaaPP0....",
    "...0QPPPPPQ0....",
    "....0QQ0QQ0.....",
    "....000.000.....",
)

FACINGS = ("down", "up", "left", "right")

# (facing, walk frame) -> grid. Frame 0 is the standing pose.
CHAR_FRAMES = {
    ("down", 0): _CHAR_DOWN,
    ("up", 0): _CHAR_UP,
    ("right", 0): _CHAR_RIGHT,
}

# ------------------------------------------------------------- panel icons

ICON_BOMB = (
    "....a..",
    "...4...",
    "..000..",
    ".01100.",
    ".01000.",
    ".00000.",
    "..000..",
)
# The same icon, dimmed, for a character whose bomb is on the board.
ICON_BOMB_ARMED = apply_colors(ICON_BOMB, {"a": "5", "4": "5", "1": "5", "0": "5"})

TILE_SPRITES = {
    "floor": FLOOR,
    "wall": WALL,
    "crate": CRATE,
}


def all_tile_sprites() -> dict[str, tuple[str, ...]]:
    """Every TILE x TILE grid by name, for validation."""
    sprites = dict(TILE_SPRITES)
    sprites.update({f"bomb{i}": rows for i, rows in enumerate(BOMB_FRAMES)})
    sprites.update({f"flame{i}": rows for i, rows in enumerate(FLAME_FRAMES)})
    sprites["gate_closed"] = GATE_CLOSED
    sprites.update({f"gate_open{i}": rows for i, rows in enumerate(GATE_OPEN_FRAMES)})
    return sprites


def validate_all() -> None:
    """Check every grid in this module. Called once when sprites are built."""
    for name, rows in all_tile_sprites().items():
        validate(name, rows, (TILE, TILE))
    for (facing, frame), rows in CHAR_FRAMES.items():
        validate(f"char_{facing}{frame}", rows, (TILE, TILE), allow_placeholders=True)
    validate("icon_bomb", ICON_BOMB)
    validate("icon_bomb_armed", ICON_BOMB_ARMED)
