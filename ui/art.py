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
    "5555555555555555",
    "5555555555555555",
    "5505555555555555",
    "5555555555555055",
    "5555555555555555",
    "5555550555555555",
    "5555555555555555",
    "5555555555555555",
    "5555555555505555",
    "5055555555555555",
    "5555555555555555",
    "5555555055555555",
    "5555555555555555",
    "5555555555555550",
    "5555055555555555",
    "5555555555555555",
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
# Pulse frame: the body glows red. Bombs pulse faster as their countdown drops.
BOMB_PULSE = apply_colors(BOMB, {"1": "2", "6": "8", "7": "e", "a": "7"})
BOMB_FRAMES = (BOMB, BOMB_PULSE)

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
FLAME_SMALL = (
    "................",
    "................",
    "................",
    "......8888......",
    ".....899998.....",
    "....89aaaa98....",
    "...89aa77aa98...",
    "...89a7777a98...",
    "...89a7777a98...",
    "...89aa77aa98...",
    "....89aaaa98....",
    ".....899998.....",
    "......8888......",
    "................",
    "................",
    "................",
)

FLAME_MEDIUM = (
    "................",
    "................",
    "....88888888....",
    "...8899999988...",
    "..899aaaaaa998..",
    "..89aa7777aa98..",
    "..89a777777a98..",
    "..89a777777a98..",
    "..89a777777a98..",
    "..89a777777a98..",
    "..89aa7777aa98..",
    "..899aaaaaa998..",
    "...8899999988...",
    "....88888888....",
    "................",
    "................",
)
# Growing burst, then the steady flame.
FLAME_FRAMES = (FLAME_SMALL, FLAME_MEDIUM, FLAME_STEADY)

# A lingering flame fades as its remaining ticks drop: the steady flame
# recoloured, level 1 (slightly dimmer) to FLAME_FADE_LEVELS (embers).
FLAME_FADE_COLORS = (
    {"7": "a", "a": "9"},
    {"7": "9", "a": "9", "9": "8"},
    {"7": "8", "a": "8", "9": "2", "8": "2"},
)
FLAME_FADE_LEVELS = len(FLAME_FADE_COLORS)


def flame_fade(level: int) -> tuple[str, ...]:
    """The steady flame at fade ``level`` (1 to ``FLAME_FADE_LEVELS``)."""
    return apply_colors(FLAME_STEADY, FLAME_FADE_COLORS[level - 1])

# ----------------------------------------------------------- breaking crate
# Drawn over the floor (and over the flames) on the tick a crate is destroyed.

CRATE_BREAK = (
    (
        "2222222..2222222",
        "29999994.9999942",
        "2944444..4444442",
        "2949444.44449442",
        "294494.44449442.",
        "2944494..94444.2",
        "294444.949444.2.",
        "..44444994444...",
        "..444449944442..",
        "2944449.4944442.",
        "29444944.494442.",
        "2944944..4494442",
        "2949444.44449442",
        "294444.444444442",
        "244444..44444442",
        "222222..22222222",
    ),
    (
        "................",
        ".29.........942.",
        "..94......494...",
        "...4..........2.",
        "................",
        "......2.........",
        ".24.......94....",
        "..9.............",
        "..........4.2...",
        "....42..........",
        "...2.........94.",
        "................",
        ".9.....24.......",
        "..4.........2...",
        "................",
        "................",
    ),
)

# --------------------------------------------------------------------- gate
# A hatch in the floor: barred while closed, a glowing portal once open.
# A hidden gate is not drawn at all: its crate is still on the grid.

GATE_CLOSED = (
    "0000000000000000",
    "0666666666666660",
    "0655555555555560",
    "065d00d00d00d560",
    "065d00d00d00d560",
    "065d00d00d00d560",
    "065d00d00d00d560",
    "065dddddddddd560",
    "065d00d00d00d560",
    "065d00d00d00d560",
    "065d00d00d00d560",
    "065d00d00d00d560",
    "0655555555555560",
    "0666666666666660",
    "0666666666666660",
    "0000000000000000",
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
# The open portal shimmers: one frame per tick, alternating.
GATE_OPEN_FRAMES = (
    _GATE_OPEN_BASE,
    apply_colors(_GATE_OPEN_BASE, {"1": "c", "c": "7", "7": "1"}),
)
# White blink drawn over the gate on the tick it is revealed or opens.
GATE_FLASH = apply_colors(
    GATE_CLOSED, {ch: "7" for ch in "123456789abcdef"}
)

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


def _with_rows(rows: tuple[str, ...], replacements: dict[int, str]) -> tuple[str, ...]:
    """``rows`` with some rows replaced, for walk frames that only move the feet."""
    return tuple(replacements.get(i, row) for i, row in enumerate(rows))


_STRIDE = {14: "...0QQ0..0QQ0...", 15: "...000....000..."}

# (facing, walk frame) -> grid. Frame 0 is the standing pose, frame 1 the stride.
CHAR_FRAMES = {
    ("down", 0): _CHAR_DOWN,
    ("down", 1): _with_rows(_CHAR_DOWN, _STRIDE),
    ("up", 0): _CHAR_UP,
    ("up", 1): _with_rows(_CHAR_UP, _STRIDE),
    ("right", 0): _CHAR_RIGHT,
    ("right", 1): _with_rows(_CHAR_RIGHT, _STRIDE),
}

# Death, all within one tick: a white flash of the body, a puff in the
# character's colour, then fading sparks. After the tick the character is gone.
DEATH_FRAMES = (
    apply_colors(_CHAR_DOWN, {ch: "7" for ch in "0123456789abcdefPQ"}),
    (
        "................",
        "................",
        "................",
        "................",
        ".....7....7.....",
        "...7..6PP6..7...",
        "....66P77P66....",
        "..7.6P7777P6.7..",
        "...6P777777P6...",
        "...6P777777P6...",
        "..7.6P7777P6.7..",
        "....66P77P66....",
        "...7..6PP6..7...",
        ".....7....7.....",
        "................",
        "................",
    ),
    (
        "................",
        "................",
        "................",
        "................",
        "................",
        ".....7....7.....",
        "................",
        "...7...PP...7...",
        ".......PP.......",
        "...7........7...",
        "................",
        ".....7....7.....",
        "................",
        "................",
        "................",
        "................",
    ),
)

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

# The same flame shapes in darker ember colours: a cell whose fire outlasts
# ``flame_ticks`` (the burn buff) is drawn with these while its extra ticks
# last, so a player can see where the fire will stay longer.
EMBER_COLORS = {"7": "e", "a": "8", "9": "2", "8": "0"}
EMBER_FRAMES = tuple(apply_colors(rows, EMBER_COLORS) for rows in FLAME_FRAMES)

# -------------------------------------------------------------------- buffs
# A buff lies on the floor as a round badge with a black rim (readable on the
# grey floor), each in its own colour with its own symbol: blue with a
# four-way arrow (range), green with a bomb and a plus (extra bomb), red with
# a flame (burn). Frame 1 is the same badge one pixel higher: buffs bob once
# per tick.

BUFF_RANGE = (
    "................",
    "....00000000....",
    "...0111111110...",
    "..01cccccccc10..",
    ".01cccc77cccc10.",
    ".01ccc7777ccc10.",
    ".01cc7c77c7cc10.",
    ".01c77777777c10.",
    ".01c77777777c10.",
    ".01cc7c77c7cc10.",
    ".01ccc7777ccc10.",
    ".01cccc77cccc10.",
    "..01cccccccc10..",
    "...0111111110...",
    "....00000000....",
    "................",
)

BUFF_BOMB = (
    "................",
    "....00000000....",
    "...0333333330...",
    "..03bbbbbbbb30..",
    ".03bb7bbbbabb30.",
    ".03b777bb4bbb30.",
    ".03bb7b0000bb30.",
    ".03bbb000000b30.",
    ".03bbb070000b30.",
    ".03bbb000000b30.",
    ".03bbbb0000bb30.",
    ".03bbbbbbbbbb30.",
    "..03bbbbbbbb30..",
    "...0333333330...",
    "....00000000....",
    "................",
)

BUFF_BURN = (
    "................",
    "....00000000....",
    "...0222222220...",
    "..028888888820..",
    ".028888a8888820.",
    ".02888aa8a88820.",
    ".02888a9aa88820.",
    ".0288a99a9a8820.",
    ".0288a9799a8820.",
    ".0288a9779a8820.",
    ".02888a77a88820.",
    ".028888aa888820.",
    "..028888888820..",
    "...0222222220...",
    "....00000000....",
    "................",
)


def _raised(rows: tuple[str, ...]) -> tuple[str, ...]:
    """``rows`` one pixel higher (the top row must be empty)."""
    return rows[1:] + (TRANSPARENT * len(rows[0]),)


BUFF_ICONS = {"range": BUFF_RANGE, "bomb_count": BUFF_BOMB, "burn": BUFF_BURN}
BUFF_FRAMES = {kind: (rows, _raised(rows)) for kind, rows in BUFF_ICONS.items()}

# Effects, drawn over the cell for part of one tick: a white burst while a
# dropped buff pops in, a yellow sparkle when it is picked up, a grey puff
# when a blast destroys it.
BUFF_POP = (
    (
        "................",
        "................",
        "................",
        "................",
        "................",
        "......7..7......",
        ".......77.......",
        "......7777......",
        "......7777......",
        ".......77.......",
        "......7..7......",
        "................",
        "................",
        "................",
        "................",
        "................",
    ),
    (
        "................",
        "................",
        "...7........7...",
        "....7......7....",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "....7......7....",
        "...7........7...",
        "................",
        "................",
    ),
)

BUFF_SPARKLE = (
    (
        "................",
        ".......a........",
        ".......a........",
        "................",
        "..a....7....a...",
        "...a..777..a....",
        "......777.......",
        ".aa..77777..aa..",
        "......777.......",
        "...a..777..a....",
        "..a....7....a...",
        "................",
        ".......a........",
        ".......a........",
        "................",
        "................",
    ),
    (
        ".......7........",
        "................",
        ".a...........a..",
        "................",
        "................",
        ".......a........",
        "................",
        "7.....a.a.....7.",
        "................",
        ".......a........",
        "................",
        "................",
        ".a...........a..",
        "................",
        ".......7........",
        "................",
    ),
)

BUFF_PUFF = (
    (
        "................",
        "................",
        "................",
        ".....666666.....",
        "...6666666666...",
        "..666655556666..",
        "..665555555566..",
        ".66655555555666.",
        ".66655555555666.",
        "..665555555566..",
        "..666655556666..",
        "...6666666666...",
        ".....666666.....",
        "................",
        "................",
        "................",
    ),
    (
        "..66........66..",
        ".6556......6556.",
        ".6556......6556.",
        "..66........66..",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "..66........66..",
        ".6556......6556.",
        ".6556......6556.",
        "..66........66..",
    ),
)
BUFF_EFFECTS = {"pop": BUFF_POP, "sparkle": BUFF_SPARKLE, "puff": BUFF_PUFF}

# ------------------------------------------------------------- panel icons

ICON_RANGE = (
    "...c...",
    "..ccc..",
    ".c.c.c.",
    "ccccccc",
    ".c.c.c.",
    "..ccc..",
    "...c...",
)
ICON_BURN = (
    "...a...",
    "..aa.a.",
    "..a9aa.",
    ".a999a.",
    ".a989a.",
    ".a989a.",
    "..aaa..",
)
# Panel icon of each buff, next to its level pips.
BUFF_PANEL_ICONS = {"range": ICON_RANGE, "bomb_count": ICON_BOMB, "burn": ICON_BURN}

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
    sprites.update({f"flame_fade{level}": flame_fade(level)
                    for level in range(1, FLAME_FADE_LEVELS + 1)})
    sprites["gate_closed"] = GATE_CLOSED
    sprites["gate_flash"] = GATE_FLASH
    sprites.update({f"crate_break{i}": rows for i, rows in enumerate(CRATE_BREAK)})
    sprites.update({f"gate_open{i}": rows for i, rows in enumerate(GATE_OPEN_FRAMES)})
    sprites.update({f"ember{i}": rows for i, rows in enumerate(EMBER_FRAMES)})
    for kind, frames in BUFF_FRAMES.items():
        sprites.update({f"buff_{kind}{i}": rows for i, rows in enumerate(frames)})
    for kind, frames in BUFF_EFFECTS.items():
        sprites.update({f"buff_{kind}{i}": rows for i, rows in enumerate(frames)})
    return sprites


def validate_all() -> None:
    """Check every grid in this module. Called once when sprites are built."""
    for name, rows in all_tile_sprites().items():
        validate(name, rows, (TILE, TILE))
    for (facing, frame), rows in CHAR_FRAMES.items():
        validate(f"char_{facing}{frame}", rows, (TILE, TILE), allow_placeholders=True)
    for i, rows in enumerate(DEATH_FRAMES):
        validate(f"death{i}", rows, (TILE, TILE), allow_placeholders=True)
    validate("icon_bomb", ICON_BOMB)
    validate("icon_bomb_armed", ICON_BOMB_ARMED)
    validate("icon_range", ICON_RANGE, (7, 7))
    validate("icon_burn", ICON_BURN, (7, 7))
