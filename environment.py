"""Deterministic Bomberman rules for the agent and optional NPC opponents.

The simulation advances only inside ``step``. There is no display, keyboard,
or wall-clock timer here, so an agent can drive the same rules as the
manual client in ``play.py``.

By default there is one character, the agent. ``BombermanEnv(with_npcs=True)``
adds three NPC opponents that follow the same rules. The engine only applies
the actions it is given; NPC decision-making lives elsewhere.

Coordinates: ``x`` is the column from the left, ``y`` is the row from the top.
Both start at 0. Read a cell as ``grid[y][x]``.
"""

from collections.abc import Mapping
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


@dataclass
class Character:
    """One character on the board. ``id`` 0 is the agent, 1-3 are NPCs."""

    id: int
    x: int
    y: int
    dead: bool = False


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

# Character 0 is the RL agent. NPCs exist only with ``with_npcs=True``.
AGENT_ID = 0
NPC_IDS = (1, 2, 3)
# The other three corners. Each NPC escapes its own opening bomb through a
# three-cell floor corridor: down for NPC 1, up for NPCs 2 and 3.
NPC_STARTS = {1: (9, 1), 2: (1, 9), 3: (9, 9)}


class BombermanEnv:
    """Board, characters, and their bombs.

    ``reset`` restores the initial board. ``step`` lets every living
    character act in ID order (agent first, then NPCs 1, 2, 3), then counts
    down the bombs that were already on the board. A bomb placed on this
    action keeps the full fuse, so its owner has ``BOMB_FUSE`` later actions
    before it explodes. Moves on the exploding action are applied first.

    Each character has at most one bomb. Bombs and living characters block
    movement; dead characters do not. A bomb outlives its owner's death.

    Once the agent is dead, further ``step`` calls leave the state unchanged
    until ``reset``. Flames are remembered in ``explosion`` so a renderer can
    show them until the next action. They are not checked again and do not
    pass time on their own.

    Public attributes:
    - ``grid``: cells indexed as ``grid[y][x]``
    - ``characters``: ``Character`` list indexed by ID; only the agent by default
    - ``bombs``: active ``Bomb`` per owner ID
    - ``player_x``, ``player_y``: agent column and row
    - ``dead``: whether an explosion has hit the agent
    - ``bomb``: the agent's active ``Bomb``, or ``None``
    - ``explosion``: cells burned by any bomb on the latest action, or an empty list
    - ``events``: ``death`` and ``block_destroyed`` events of the latest action
    - ``steps``: actions resolved since the last reset
    """

    def __init__(self, *, with_npcs: bool = False) -> None:
        self.with_npcs = with_npcs
        self.grid: list[list[Cell]] = []
        self.characters: list[Character] = []
        self.bombs: dict[int, Bomb] = {}
        self.explosion: list[tuple[int, int]] = []
        self.events: list[dict] = []
        self.steps = 0
        self.reset()

    # The agent-only attributes below were plain attributes before NPCs
    # existed. They now read and write character 0 and its bomb.
    @property
    def player(self) -> tuple[int, int]:
        return (self.player_x, self.player_y)

    @property
    def player_x(self) -> int:
        return self.characters[AGENT_ID].x

    @player_x.setter
    def player_x(self, value: int) -> None:
        self.characters[AGENT_ID].x = value

    @property
    def player_y(self) -> int:
        return self.characters[AGENT_ID].y

    @player_y.setter
    def player_y(self, value: int) -> None:
        self.characters[AGENT_ID].y = value

    @property
    def dead(self) -> bool:
        return self.characters[AGENT_ID].dead

    @dead.setter
    def dead(self, value: bool) -> None:
        self.characters[AGENT_ID].dead = value

    @property
    def bomb(self) -> Bomb | None:
        return self.bombs.get(AGENT_ID)

    @bomb.setter
    def bomb(self, value: Bomb | None) -> None:
        if value is None:
            self.bombs.pop(AGENT_ID, None)
        else:
            self.bombs[AGENT_ID] = value

    @property
    def all_npcs_dead(self) -> bool:
        """True when NPCs exist and all are dead. Not a terminal state."""
        npcs = self.characters[1:]
        return bool(npcs) and all(npc.dead for npc in npcs)

    def reset(self) -> dict:
        """Restore the map, characters, bombs, flames, events, and step count."""
        self.grid = [row[:] for row in _INITIAL_GRID]
        self.characters = [Character(AGENT_ID, *PLAYER_START)]
        if self.with_npcs:
            self.characters += [
                Character(npc_id, *NPC_STARTS[npc_id]) for npc_id in NPC_IDS
            ]
        self.bombs = {}
        self.explosion = []
        self.events = []
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
            "characters": [
                {"id": c.id, "x": c.x, "y": c.y, "dead": c.dead}
                for c in self.characters
            ],
            "bombs": [
                {"owner": owner, "x": b.x, "y": b.y, "countdown": b.countdown}
                for owner, b in sorted(self.bombs.items())
            ],
            # Event values are ints and tuples, so a shallow copy is enough.
            "events": [dict(event) for event in self.events],
            "all_npcs_dead": self.all_npcs_dead,
        }

    def step(
        self, action: Action, npc_actions: Mapping[int, Action] | None = None
    ) -> dict:
        """Resolve one action for every living character and return ``get_state()``.

        ``action`` is the agent's. ``npc_actions`` maps NPC IDs to actions;
        a missing NPC waits and an action for a dead NPC is ignored. Passing
        ``npc_actions`` without NPCs raises ``ValueError``.

        Movement that hits a wall, a block, a bomb, or a living character
        still spends the turn. Placing a bomb while one is already active
        does too. Moves resolve one character at a time, so a character can
        enter a cell vacated earlier in the same action, but two neighbours
        cannot swap.
        """
        if not isinstance(action, Action):
            raise TypeError(f"action must be an Action, not {type(action).__name__}")
        self._check_npc_actions(npc_actions)
        if self.dead:
            return self.get_state()

        self.explosion = []
        self.events = []
        self.steps += 1
        placed: set[int] = set()
        for character in self.characters:
            if character.dead:
                continue
            if character.id == AGENT_ID:
                chosen = action
            else:
                chosen = (npc_actions or {}).get(character.id, Action.WAIT)
            if self._apply_action(character, chosen):
                placed.add(character.id)
        # A bomb placed on this action does not tick. Older bombs do, after
        # every character has already moved.
        due = self._tick_bombs(placed)
        if due:
            self._explode(due)
        return self.get_state()

    def _check_npc_actions(self, npc_actions: Mapping[int, Action] | None) -> None:
        """Reject bad NPC input before anything changes."""
        if npc_actions is None:
            return
        if not self.with_npcs:
            raise ValueError("npc_actions given, but this env has no NPCs")
        if not isinstance(npc_actions, Mapping):
            raise TypeError(
                f"npc_actions must be a mapping, not {type(npc_actions).__name__}"
            )
        for npc_id, npc_action in npc_actions.items():
            if npc_id not in NPC_IDS:
                raise ValueError(f"unknown NPC id {npc_id!r}")
            if not isinstance(npc_action, Action):
                raise TypeError(
                    f"action for NPC {npc_id} must be an Action, "
                    f"not {type(npc_action).__name__}"
                )

    def _apply_action(self, character: Character, action: Action) -> bool:
        """Apply one character's action. Return True when a new bomb was placed."""
        if action is Action.WAIT:
            return False

        if action is Action.BOMB:
            if character.id in self.bombs:
                return False
            self.bombs[character.id] = Bomb(character.x, character.y, BOMB_FUSE)
            return True

        dx, dy = _MOVE[action]
        new_x = character.x + dx
        new_y = character.y + dy
        if self._is_walkable(new_x, new_y):
            character.x = new_x
            character.y = new_y
        return False

    def _is_walkable(self, x: int, y: int) -> bool:
        if not _inside(x, y):
            return False
        if self.grid[y][x] is not Cell.EMPTY:
            return False
        if any((bomb.x, bomb.y) == (x, y) for bomb in self.bombs.values()):
            return False
        if any(not c.dead and (c.x, c.y) == (x, y) for c in self.characters):
            return False
        return True

    def _tick_bombs(self, placed: set[int]) -> list[int]:
        """Count down bombs not placed this action. Return owners now due, in ID order."""
        due = []
        for owner in sorted(self.bombs):
            if owner in placed:
                continue
            bomb = self.bombs[owner]
            self.bombs[owner] = Bomb(bomb.x, bomb.y, bomb.countdown - 1)
            if self.bombs[owner].countdown <= 0:
                due.append(owner)
        return due

    def _explode(self, owners: list[int]) -> None:
        """Burn every due bomb at once.

        Each bomb burns its own cell and up to ``BLAST_RANGE`` cells each way.
        A wall ends that ray and stays. A breakable block burns and ends that
        ray. Any other bomb's cell, exploding or not, burns and ends that ray;
        the bomb there is not triggered and keeps counting down.

        All rays are traced against the board as it was before this action's
        blasts, so blocks become floor only afterwards. Then every living
        character on a burned cell dies. Events list destroyed blocks in burn
        order, then deaths in ID order, each naming every contributing owner.
        """
        bomb_cells = {(bomb.x, bomb.y) for bomb in self.bombs.values()}
        # Burned cell -> owners whose blast reached it, in first-burn order.
        burned: dict[tuple[int, int], set[int]] = {}

        for owner in owners:
            origin_x = self.bombs[owner].x
            origin_y = self.bombs[owner].y
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
                    if tile is Cell.BLOCK or (x, y) in bomb_cells:
                        break

            for cell in cells:
                burned.setdefault(cell, set()).add(owner)

        for (x, y), by in burned.items():
            if self.grid[y][x] is Cell.BLOCK:
                self.grid[y][x] = Cell.EMPTY
                self.events.append(
                    {
                        "type": "block_destroyed",
                        "cell": (x, y),
                        "destroyed_by": tuple(sorted(by)),
                    }
                )

        for owner in owners:
            del self.bombs[owner]
        self.explosion = list(burned)

        for character in self.characters:
            position = (character.x, character.y)
            if not character.dead and position in burned:
                character.dead = True
                self.events.append(
                    {
                        "type": "death",
                        "character": character.id,
                        "killed_by": tuple(sorted(burned[position])),
                    }
                )
