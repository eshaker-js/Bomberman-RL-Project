"""A 3x5 pixel font defined in code. Pure: no pygame import.

Drawing uses only ``surface.fill(color, rect)``, so any pygame ``Surface``
works and the glyph data stays testable without pygame. Text is uppercased;
unknown characters are drawn as ``?``.
"""

from ui.palette import rgb

GLYPH_W = 3
GLYPH_H = 5
# Pixels from the start of one glyph to the next, at scale 1.
ADVANCE = GLYPH_W + 1

GLYPHS = {
    "A": (".#.", "#.#", "###", "#.#", "#.#"),
    "B": ("##.", "#.#", "##.", "#.#", "##."),
    "C": (".##", "#..", "#..", "#..", ".##"),
    "D": ("##.", "#.#", "#.#", "#.#", "##."),
    "E": ("###", "#..", "##.", "#..", "###"),
    "F": ("###", "#..", "##.", "#..", "#.."),
    "G": (".##", "#..", "#.#", "#.#", ".##"),
    "H": ("#.#", "#.#", "###", "#.#", "#.#"),
    "I": ("###", ".#.", ".#.", ".#.", "###"),
    "J": ("..#", "..#", "..#", "#.#", ".#."),
    "K": ("#.#", "#.#", "##.", "#.#", "#.#"),
    "L": ("#..", "#..", "#..", "#..", "###"),
    "M": ("#.#", "###", "###", "#.#", "#.#"),
    "N": ("##.", "#.#", "#.#", "#.#", "#.#"),
    "O": (".#.", "#.#", "#.#", "#.#", ".#."),
    "P": ("##.", "#.#", "##.", "#..", "#.."),
    "Q": (".#.", "#.#", "#.#", "##.", ".##"),
    "R": ("##.", "#.#", "##.", "#.#", "#.#"),
    "S": (".##", "#..", ".#.", "..#", "##."),
    "T": ("###", ".#.", ".#.", ".#.", ".#."),
    "U": ("#.#", "#.#", "#.#", "#.#", "###"),
    "V": ("#.#", "#.#", "#.#", "#.#", ".#."),
    "W": ("#.#", "#.#", "###", "###", "#.#"),
    "X": ("#.#", "#.#", ".#.", "#.#", "#.#"),
    "Y": ("#.#", "#.#", ".#.", ".#.", ".#."),
    "Z": ("###", "..#", ".#.", "#..", "###"),
    "0": ("###", "#.#", "#.#", "#.#", "###"),
    "1": (".#.", "##.", ".#.", ".#.", "###"),
    "2": ("##.", "..#", ".#.", "#..", "###"),
    "3": ("##.", "..#", ".#.", "..#", "##."),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "##.", "..#", "##."),
    "6": (".##", "#..", "###", "#.#", "###"),
    "7": ("###", "..#", ".#.", ".#.", ".#."),
    "8": ("###", "#.#", "###", "#.#", "###"),
    "9": ("###", "#.#", "###", "..#", "##."),
    " ": ("...", "...", "...", "...", "..."),
    ":": ("...", ".#.", "...", ".#.", "..."),
    "-": ("...", "...", "###", "...", "..."),
    "+": ("...", ".#.", "###", ".#.", "..."),
    "!": (".#.", ".#.", ".#.", "...", ".#."),
    "?": ("##.", "..#", ".#.", "...", ".#."),
    ".": ("...", "...", "...", "...", ".#."),
    ",": ("...", "...", "...", ".#.", "#.."),
    "/": ("..#", "..#", ".#.", "#..", "#.."),
    "(": ("..#", ".#.", ".#.", ".#.", "..#"),
    ")": ("#..", ".#.", ".#.", ".#.", "#.."),
    ">": ("#..", ".#.", "..#", ".#.", "#.."),
    "<": ("..#", ".#.", "#..", ".#.", "..#"),
}


def text_width(text: str, scale: int = 1) -> int:
    """Width in pixels of ``text`` drawn at ``scale``, without trailing space."""
    if not text:
        return 0
    return (len(text) * ADVANCE - 1) * scale


def draw_text(surface, text: str, x: int, y: int, color: int, scale: int = 1) -> None:
    """Draw ``text`` with its top-left corner at ``(x, y)`` in palette ``color``."""
    fill = rgb(color)
    for i, ch in enumerate(text.upper()):
        glyph = GLYPHS.get(ch, GLYPHS["?"])
        gx = x + i * ADVANCE * scale
        for row, line in enumerate(glyph):
            for col, bit in enumerate(line):
                if bit == "#":
                    surface.fill(fill, (gx + col * scale, y + row * scale, scale, scale))


def draw_text_centered(surface, text: str, cx: int, y: int, color: int, scale: int = 1) -> None:
    """Draw ``text`` horizontally centred on ``cx``."""
    draw_text(surface, text, cx - text_width(text, scale) // 2, y, color, scale)
