"""Every UI constant in one place. Pure: no pygame.

All sizes are in logical pixels. Everything is drawn once at 1x onto a
``CANVAS_SIZE`` surface, then scaled up ``SCALE`` times without smoothing.
"""

from game.environment import MAP_HEIGHT, MAP_WIDTH

# Placeholder name until the team decides. Change it here only.
GAME_NAME = "BOMB ARENA"

# One game tick (one env.step) every TICK_MS while a game runs.
TICK_MS = 250
FPS = 60
# "READY..." then "GO!" before the first tick of every game, clock frozen.
READY_MS = 1000
GO_MS = 300

# Every new game draws its seed from the UI's own RNG in range(SEED_COUNT).
SEED_COUNT = 1_000_000

TILE = 16
SCALE = 3
MARGIN = 8

ARENA_X = MARGIN
ARENA_Y = MARGIN
ARENA_W = MAP_WIDTH * TILE
ARENA_H = MAP_HEIGHT * TILE

PANEL_X = ARENA_X + ARENA_W + MARGIN
PANEL_W = 88
PANEL_Y = MARGIN
PANEL_H = ARENA_H

CANVAS_W = PANEL_X + PANEL_W + MARGIN
CANVAS_H = ARENA_Y + ARENA_H + MARGIN
CANVAS_SIZE = (CANVAS_W, CANVAS_H)
WINDOW_SIZE = (CANVAS_W * SCALE, CANVAS_H * SCALE)
