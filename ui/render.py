"""Drawing only: blits scenes, the side panel, boxes and menus. No game logic.

Everything is drawn onto the logical canvas (``config.CANVAS_SIZE``), which
the app scales up ``config.SCALE`` times without smoothing.
"""

import pygame

from ui import palette as pal
from ui.art import CHARACTER_COLORS
from ui.config import (
    ARENA_H,
    ARENA_W,
    ARENA_X,
    ARENA_Y,
    GAME_NAME,
    PANEL_H,
    PANEL_W,
    PANEL_X,
    PANEL_Y,
)
from ui.font import draw_text, draw_text_centered, text_width
from ui.menu import Menu
from ui.palette import rgb
from ui.scene import Item
from ui.sprites import Sprites

CARD_H = 26
CARD_GAP = 2
INFO_Y = PANEL_Y + 126
BANNER_Y = PANEL_Y + 152
OUTCOME_TEXT = {"win": "WIN!", "loss": "LOSS", "draw": "DRAW"}
OUTCOME_COLOR = {"win": pal.GREEN, "loss": pal.RED, "draw": pal.ORANGE}


def frame_rect(surface, rect, fill: int, border: int) -> None:
    """A filled rectangle with a 1-pixel border, both palette colours."""
    rect = pygame.Rect(rect)
    surface.fill(rgb(border), rect)
    surface.fill(rgb(fill), rect.inflate(-2, -2))


def draw_scene(canvas, sprites: Sprites, items: list[Item]) -> None:
    """Draw the arena frame and every scene item, in list order."""
    frame_rect(canvas, (ARENA_X - 2, ARENA_Y - 2, ARENA_W + 4, ARENA_H + 4),
               pal.BLACK, pal.DARK_GREY)
    for item in items:
        canvas.blit(sprites.get(item.key), (ARENA_X + item.x, ARENA_Y + item.y))


def draw_panel(canvas, sprites: Sprites, session, state: dict) -> None:
    """Character cards, tick counter, gate state and the outcome banner."""
    frame_rect(canvas, (PANEL_X, PANEL_Y, PANEL_W, PANEL_H), pal.DARK_BLUE, pal.DARK_GREY)
    cx = PANEL_X + PANEL_W // 2
    draw_text_centered(canvas, GAME_NAME, cx, PANEL_Y + 3, pal.YELLOW)

    bomb_owners = {bomb["owner"] for bomb in state["bombs"]}
    for i, character in enumerate(state["characters"]):
        y = PANEL_Y + 11 + i * (CARD_H + CARD_GAP)
        _draw_card(canvas, sprites, session.label(character["id"]), character,
                   character["id"] not in bomb_owners, PANEL_X + 3, y)

    x = PANEL_X + 4
    draw_text(canvas, f"TICK {state['steps']}", x, INFO_Y, pal.WHITE)
    draw_text(canvas, f"GATE {state['gate']['state']}", x, INFO_Y + 8,
              pal.GREEN if state["gate"]["state"] == "open" else pal.LIGHT_GREY)
    draw_text(canvas, f"SEED {session.seed}", x, INFO_Y + 16, pal.DARK_GREY)

    outcome = state["outcome"]
    if outcome is not None:
        frame_rect(canvas, (PANEL_X + 3, BANNER_Y, PANEL_W - 6, 18),
                   pal.BLACK, OUTCOME_COLOR[outcome])
        draw_text_centered(canvas, OUTCOME_TEXT[outcome], cx, BANNER_Y + 4,
                           OUTCOME_COLOR[outcome], scale=2)


def _draw_card(canvas, sprites: Sprites, label: str, character: dict,
               bomb_ready: bool, x: int, y: int) -> None:
    dead = character["dead"]
    border = pal.DARK_GREY if dead else _card_color(character["id"])
    frame_rect(canvas, (x, y, PANEL_W - 6, CARD_H), pal.BLACK, border)
    canvas.blit(sprites.get(("portrait", character["id"], dead)), (x + 3, y + 5))
    tx = x + 22
    draw_text(canvas, label, tx, y + 3, pal.WHITE)
    if dead:
        draw_text(canvas, "DEAD", tx, y + 10, pal.RED)
        return
    draw_text(canvas, "ALIVE", tx, y + 10, pal.GREEN)
    canvas.blit(sprites.get(("icon", "bomb", bomb_ready)), (tx, y + 17))
    draw_text(canvas, "READY" if bomb_ready else "ARMED", tx + 9, y + 18,
              pal.YELLOW if bomb_ready else pal.DARK_GREY)


def _card_color(character_id: int) -> int:
    return int(CHARACTER_COLORS[character_id]["P"], 16)


def draw_box(canvas, lines: list[tuple[str, int, int]], cx: int, cy: int) -> None:
    """A centred box with lines of ``(text, palette colour, scale)``."""
    width = max(text_width(text, scale) for text, _, scale in lines) + 16
    height = sum(5 * scale + 4 for _, _, scale in lines) + 10
    top = cy - height // 2
    frame_rect(canvas, (cx - width // 2, top, width, height), pal.BLACK, pal.WHITE)
    y = top + 7
    for text, color, scale in lines:
        draw_text_centered(canvas, text, cx, y, color, scale)
        y += 5 * scale + 4


def draw_dialog(canvas, title: str, color: int, menu: Menu, cx: int, cy: int) -> None:
    """A centred box with a big ``title`` and a menu under it."""
    spacing = 10
    width = max(text_width(title, 2),
                *(text_width(f"> {option} <") for option in menu.options)) + 16
    height = 7 + 10 + 8 + len(menu.options) * spacing + 4
    top = cy - height // 2
    frame_rect(canvas, (cx - width // 2, top, width, height), pal.BLACK, pal.WHITE)
    draw_text_centered(canvas, title, cx, top + 7, color, scale=2)
    draw_menu(canvas, menu, cx, top + 7 + 10 + 8, spacing)


def draw_menu(canvas, menu: Menu, cx: int, y: int, spacing: int = 10) -> None:
    """The options centred on ``cx``; the selected one in yellow with arrows."""
    for i, option in enumerate(menu.options):
        if i == menu.index:
            draw_text_centered(canvas, f"> {option} <", cx, y + i * spacing, pal.YELLOW)
        else:
            draw_text_centered(canvas, option, cx, y + i * spacing, pal.LIGHT_GREY)
