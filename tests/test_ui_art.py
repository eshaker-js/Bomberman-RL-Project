"""Headless checks for the pixel art, palette and font data. Pygame is not required."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui import art, config  # noqa: E402
from ui.font import GLYPH_H, GLYPH_W, GLYPHS, text_width  # noqa: E402
from ui.palette import PALETTE  # noqa: E402


class TestPalette(unittest.TestCase):
    def test_sixteen_pico8_colours(self):
        self.assertEqual(len(PALETTE), 16)
        self.assertEqual(PALETTE[0], (0, 0, 0))
        self.assertEqual(PALETTE[7], (255, 241, 232))
        self.assertEqual(PALETTE[15], (255, 204, 170))
        for color in PALETTE:
            self.assertTrue(all(0 <= c <= 255 for c in color))


class TestSprites(unittest.TestCase):
    def test_every_sprite_is_well_formed(self):
        art.validate_all()

    def test_tiles_are_16_by_16(self):
        for name, rows in art.all_tile_sprites().items():
            with self.subTest(name=name):
                self.assertEqual(len(rows), 16)
                self.assertTrue(all(len(row) == 16 for row in rows))

    def test_characters_have_all_facings_with_placeholders(self):
        for facing in ("down", "up", "right"):
            rows = art.CHAR_FRAMES[(facing, 0)]
            self.assertTrue(any("P" in row for row in rows))

    def test_each_character_has_its_own_colour(self):
        self.assertEqual(sorted(art.CHARACTER_COLORS), [0, 1, 2, 3])
        mains = {colors["P"] for colors in art.CHARACTER_COLORS.values()}
        self.assertEqual(len(mains), 4)

    def test_recoloured_character_has_no_placeholders_left(self):
        for colors in [*art.CHARACTER_COLORS.values(), art.DEAD_COLORS]:
            rows = art.apply_colors(art.CHAR_FRAMES[("down", 0)], colors)
            art.validate("recoloured", rows, (16, 16))

    def test_validator_rejects_bad_grids(self):
        with self.assertRaises(ValueError):
            art.validate("ragged", ("..", "..."))
        with self.assertRaises(ValueError):
            art.validate("letter", ("x.",))
        with self.assertRaises(ValueError):
            art.validate("size", ("..",), (16, 16))
        with self.assertRaises(ValueError):
            art.validate("placeholder", ("P.",))


class TestFont(unittest.TestCase):
    def test_glyphs_are_3_by_5(self):
        for ch, glyph in GLYPHS.items():
            with self.subTest(ch=ch):
                self.assertEqual(len(glyph), GLYPH_H)
                self.assertTrue(all(len(line) == GLYPH_W for line in glyph))
                self.assertTrue(set("".join(glyph)) <= {"#", "."})

    def test_covers_letters_digits_and_ui_text(self):
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :-!?.<>+":
            self.assertIn(ch, GLYPHS)
        for text in (config.GAME_NAME, "READY...", "GO!", "NOT READY"):
            self.assertTrue(set(text) <= set(GLYPHS), text)

    def test_text_width(self):
        self.assertEqual(text_width(""), 0)
        self.assertEqual(text_width("A"), 3)
        self.assertEqual(text_width("AB"), 7)
        self.assertEqual(text_width("AB", scale=3), 21)


class TestLayout(unittest.TestCase):
    def test_window_is_three_times_the_canvas(self):
        self.assertEqual(config.SCALE, 3)
        self.assertEqual(config.CANVAS_SIZE, (288, 192))
        self.assertEqual(config.WINDOW_SIZE, (864, 576))
        self.assertEqual((config.ARENA_W, config.ARENA_H), (176, 176))

    def test_game_name_defined_once(self):
        self.assertEqual(config.GAME_NAME, "BOMB ARENA")
        hits = [
            path.name
            for path in (ROOT / "ui").glob("*.py")
            if '"BOMB ARENA"' in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(hits, ["config.py"])


if __name__ == "__main__":
    unittest.main()
