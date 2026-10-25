"""Deterministic one-player Bomberman rules.

The simulation advances only inside ``step``. There is no display, keyboard,
or wall-clock timer here, so an agent can drive the same rules as the
manual client in ``play.py``.

Coordinates: ``x`` is the column from the left, ``y`` is the row from the top.
Both start at 0. Read a cell as ``grid[y][x]``.
"""

from dataclasses import dataclass
from enum import Enum

# Rules a later assignment can tune without hunting through the logic.
MAP_WIDTH = 11
MAP_HEIGHT = 11
BOMB_FUSE = 4
BLAST_RANGE = 2


class Cell(Enum):
    """What occupies one map tile, ignoring the player and the bomb."""

    EMPTY = "empty"
    WALL = "wall"
    BLOCK = "block"


class Action(Enum):
    """One decision. Passing any of these to ``step`` spends one turn."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    WAIT = "wait"
    BOMB = "bomb"


# y grows downward, so UP decreases y.
_MOVE = {
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}

_BLAST_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))

_SYMBOLS = {
    "#": Cell.WALL,
    ".": Cell.EMPTY,
    "B": Cell.BLOCK,
}

# Fixed board. "#" wall, "." floor, "B" breakable block, "P" player start.
# The start cell is floor. Three floor cells to the right of P are open, so
# a bomb at the start (blast length 2) can be escaped by walking right.
MAP_ROWS = (
    "###########",
    "#P...B.B..#",
    "#.#.#.#.#.#",
    "#.B.B.B.B.#",
    "#.#.#.#.#.#",
    "#B.B.B.B.B#",
    "#.#.#.#.#.#",
    "#.B.B.B.B.#",
    "#.#.#.#.#.#",
    "#..B.B.B..#",
    "###########",
)


@dataclass(frozen=True)
class Bomb:
    x: int
    y: int
    countdown: int


def _inside(x: int, y: int) -> bool:
    return 0 <= x < MAP_WIDTH and 0 <= y < MAP_HEIGHT


def _parse_map(rows: tuple[str, ...]) -> tuple[list[list[Cell]], tuple[int, int]]:
    if len(rows) != MAP_HEIGHT:
        raise ValueError(f"map must have {MAP_HEIGHT} rows, found {len(rows)}")

    grid: list[list[Cell]] = []
    start: tuple[int, int] | None = None
    for y, row in enumerate(rows):
        if len(row) != MAP_WIDTH:
            raise ValueError(
                f"row {y} must have {MAP_WIDTH} columns, found {len(row)}"
            )
        cells: list[Cell] = []
        for x, symbol in enumerate(row):
            if symbol == "P":
                if start is not None:
                    raise ValueError("map has more than one player start")
                start = (x, y)
                cells.append(Cell.EMPTY)
            elif symbol in _SYMBOLS:
                cells.append(_SYMBOLS[symbol])
            else:
                raise ValueError(f"unknown map symbol {symbol!r} at {(x, y)}")
        grid.append(cells)

    if start is None:
        raise ValueError("map is missing the player start 'P'")
    return grid, start


_INITIAL_GRID, PLAYER_START = _parse_map(MAP_ROWS)
PLAYER_START_X, PLAYER_START_Y = PLAYER_START


class BombermanEnv:
    """Board, player, and the single bomb.

    ``reset`` restores the initial board. ``step`` resolves one action, then
    counts down a bomb that was already on the board. A bomb placed on this
    action keeps the full fuse, so the player has ``BOMB_FUSE`` later actions
    before it explodes. Their move on the exploding action is applied first.

    Once ``dead`` is true, further ``step`` calls leave the state unchanged
    until ``reset``. Flames are remembered in ``explosion`` so a renderer can
    show them until the next action. They are not checked again and do not
    pass time on their own.

    Public attributes:
    - ``grid``: cells indexed as ``grid[y][x]``
    - ``player_x``, ``player_y``: player column and row
    - ``dead``: whether an explosion has hit the player
    - ``bomb``: the active ``Bomb``, or ``None``
    - ``explosion``: cells burned by the latest explosion, or an empty list
    - ``steps``: actions resolved since the last reset
    """

    def __init__(self) -> None:
        self.grid: list[list[Cell]] = []
        self.player_x = 0
        self.player_y = 0
        self.dead = False
        self.bomb: Bomb | None = None
        self.explosion: list[tuple[int, int]] = []
        self.steps = 0
        self.reset()

    @property
    def player(self) -> tuple[int, int]:
        return (self.player_x, self.player_y)

    def reset(self) -> dict:
        """Restore the map, player, bomb, flames, death flag, and step count."""
        self.grid = [row[:] for row in _INITIAL_GRID]
        self.player_x, self.player_y = PLAYER_START
        self.dead = False
        self.bomb = None
        self.explosion = []
        self.steps = 0
        return self.get_state()

    def get_state(self) -> dict:
        """Copy of the current state. Mutating the copy does not change the env."""
        bomb = None
        if self.bomb is not None:
            bomb = {
                "x": self.bomb.x,
                "y": self.bomb.y,
                "countdown": self.bomb.countdown,
            }
        return {
            "grid": [row[:] for row in self.grid],
            "player": self.player,
            "dead": self.dead,
            "bomb": bomb,
            "explosion": list(self.explosion),
            "steps": self.steps,
        }

    def step(self, action: Action) -> dict:
        """Resolve one action and return ``get_state()``.

        Movement that hits a wall, a block, or the active bomb still spends
        the turn. Placing a bomb while one is already active does too.
        """
        if not isinstance(action, Action):
            raise TypeError(f"action must be an Action, not {type(action).__name__}")
        if self.dead:
            return self.get_state()

        self.explosion = []
        self.steps += 1
        placed_bomb = self._apply_action(action)
        # A bomb placed on this action does not tick. An older bomb does,
        # after the player has already moved.
        if self.bomb is not None and not placed_bomb:
            self._tick_bomb()
        return self.get_state()

    def _apply_action(self, action: Action) -> bool:
        """Apply the action. Return True when a new bomb was placed."""
        if action is Action.WAIT:
            return False

        if action is Action.BOMB:
            if self.bomb is not None:
                return False
            self.bomb = Bomb(self.player_x, self.player_y, BOMB_FUSE)
            return True

        dx, dy = _MOVE[action]
        new_x = self.player_x + dx
        new_y = self.player_y + dy
        if self._is_walkable(new_x, new_y):
            self.player_x = new_x
            self.player_y = new_y
        return False

    def _is_walkable(self, x: int, y: int) -> bool:
        if not _inside(x, y):
            return False
        if self.grid[y][x] is not Cell.EMPTY:
            return False
        if self.bomb is not None and (x, y) == (self.bomb.x, self.bomb.y):
            return False
        return True

    def _tick_bomb(self) -> None:
        assert self.bomb is not None
        self.bomb = Bomb(self.bomb.x, self.bomb.y, self.bomb.countdown - 1)
        if self.bomb.countdown <= 0:
            self._explode()

    def _explode(self) -> None:
        """Burn the bomb cell and up to ``BLAST_RANGE`` cells each way.

        A wall ends that ray and stays. A breakable block burns, becomes
        floor, and ends that ray. The player dies on a burned cell.
        """
        assert self.bomb is not None
        origin_x = self.bomb.x
        origin_y = self.bomb.y
        cells = [(origin_x, origin_y)]

        for dx, dy in _BLAST_DELTAS:
            for distance in range(1, BLAST_RANGE + 1):
                x = origin_x + dx * distance
                y = origin_y + dy * distance
                if not _inside(x, y):
                    break
                tile = self.grid[y][x]
                if tile is Cell.WALL:
                    break
                cells.append((x, y))
                if tile is Cell.BLOCK:
                    self.grid[y][x] = Cell.EMPTY
                    break

        self.bomb = None
        self.explosion = cells
        if (self.player_x, self.player_y) in cells:
            self.dead = True
