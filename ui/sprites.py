"""Turns the text grids of ``ui.art`` into pygame surfaces, once each."""

import pygame

from ui import art
from ui.palette import rgb


def surface_from_rows(rows: tuple[str, ...]) -> pygame.Surface:
    """A transparent surface with one pixel per grid character."""
    surface = pygame.Surface((len(rows[0]), len(rows)), pygame.SRCALPHA)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != art.TRANSPARENT:
                surface.set_at((x, y), rgb(int(ch, 16)))
    return surface


class Sprites:
    """Builds every sprite on first use and caches it by key.

    Keys (also used by ``ui.scene``):
    - ``("tile", "floor" | "wall" | "crate")``
    - ``("bomb", frame)``
    - ``("flame", "steady")``
    - ``("gate", "closed", 0)`` and ``("gate", "open", frame)``
    - ``("char", id, facing, frame)`` with facing down, up, left or right
    - ``("portrait", id, dead)``
    - ``("icon", "bomb", ready)``
    """

    def __init__(self) -> None:
        art.validate_all()
        self._cache: dict[tuple, pygame.Surface] = {}

    def get(self, key: tuple) -> pygame.Surface:
        surface = self._cache.get(key)
        if surface is None:
            surface = self._build(key)
            self._cache[key] = surface
        return surface

    def _build(self, key: tuple) -> pygame.Surface:
        kind = key[0]
        if kind == "tile":
            return surface_from_rows(art.TILE_SPRITES[key[1]])
        if kind == "bomb":
            return surface_from_rows(art.BOMB_FRAMES[key[1]])
        if kind == "flame":
            return surface_from_rows(art.FLAME_STEADY)
        if kind == "gate":
            if key[1] == "closed":
                return surface_from_rows(art.GATE_CLOSED)
            return surface_from_rows(art.GATE_OPEN_FRAMES[key[2]])
        if kind == "char":
            _, character_id, facing, frame = key
            return self._character(art.CHARACTER_COLORS[character_id], facing, frame)
        if kind == "portrait":
            _, character_id, dead = key
            colors = art.DEAD_COLORS if dead else art.CHARACTER_COLORS[character_id]
            return self._character(colors, "down", 0)
        if kind == "icon":
            return surface_from_rows(art.ICON_BOMB if key[2] else art.ICON_BOMB_ARMED)
        raise KeyError(f"unknown sprite key {key!r}")

    @staticmethod
    def _character(colors: dict[str, str], facing: str, frame: int) -> pygame.Surface:
        source = "right" if facing == "left" else facing
        rows = art.apply_colors(art.CHAR_FRAMES[(source, frame)], colors)
        surface = surface_from_rows(rows)
        if facing == "left":
            surface = pygame.transform.flip(surface, True, False)
        return surface
