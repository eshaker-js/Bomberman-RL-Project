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

import random
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType

# Rules a later assignment can tune without hunting through the logic.
MAP_WIDTH = 11
MAP_HEIGHT = 11
BOMB_FUSE = 4
BLAST_RANGE = 2
# Bombs a character may have on the board at once without buffs.
BOMB_COUNT = 1
# Burn level without buffs: no cell burns longer than ``flame_ticks``.
BURN_LEVEL = 0
# Default number of ticks a burned cell keeps burning, counting the tick of
# the explosion. 1 is the original rule: flames last for one tick only.
FLAME_TICKS = 1


@dataclass(frozen=True)
class BuffType:
    """One kind of buff: its level at the start, its highest level, and the
    chance that a destroyed crate drops it.

    ``name`` is both the buff's type on the floor and the level key of a
    character (``"range"``, ``"bomb_count"``, ``"burn"``).
    """

    name: str
    start: int
    max: int
    drop_chance: float


@dataclass(frozen=True)
class BuffRules:
    """Every buff setting of a game, in one immutable object.

    The order of ``types`` is the order of the drop roll: one uniform draw
    per destroyed crate is compared against the cumulative drop chances in
    this order, and anything above their sum drops nothing. A new buff type
    only needs a new ``BuffType`` here and its effect in the engine.
    """

    types: tuple[BuffType, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.types, tuple) or not self.types:
            raise ValueError("types must be a non-empty tuple of BuffType")
        names = set()
        total = 0.0
        for buff in self.types:
            if not isinstance(buff, BuffType):
                raise TypeError(f"{buff!r} is not a BuffType")
            if not isinstance(buff.name, str) or not buff.name or buff.name in names:
                raise ValueError(f"buff names must be unique non-empty strings: {buff.name!r}")
            names.add(buff.name)
            for value in (buff.start, buff.max):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"buff {buff.name!r}: levels must be integers >= 0")
            if buff.start > buff.max:
                raise ValueError(f"buff {buff.name!r}: start {buff.start} > max {buff.max}")
            chance = buff.drop_chance
            if isinstance(chance, bool) or not isinstance(chance, (int, float)) \
                    or not 0.0 <= chance <= 1.0:
                raise ValueError(f"buff {buff.name!r}: drop_chance must be in [0, 1]")
            total += chance
        if total > 1.0 + 1e-9:
            raise ValueError(f"drop chances add up to {total}, more than 1")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(buff.name for buff in self.types)

    def get(self, name: str) -> BuffType | None:
        """The ``BuffType`` called ``name``, or ``None``."""
        for buff in self.types:
            if buff.name == name:
                return buff
        return None

    def as_dict(self) -> dict:
        """Plain data for ``get_state()["buff_rules"]``, in roll order."""
        return {
            buff.name: {"start": buff.start, "max": buff.max, "drop_chance": buff.drop_chance}
            for buff in self.types
        }


# Levels every character has when buffs are off, and the effect names the
# engine understands. With buffs on, a character's levels come from the
# BuffRules (a type missing there keeps its base level).
BASE_LEVELS = MappingProxyType(
    {"range": BLAST_RANGE, "bomb_count": BOMB_COUNT, "burn": BURN_LEVEL}
)

# The three buffs of the real game. Every number is v0 (provisional): range
# +1 per pickup up to 5 (15% drop chance), one more bomb at once up to 3
# (15%), burn level up to 2 (10%). Burn level L makes the blast cells within
# L tiles of the bomb's cell burn ``flame_ticks + L`` ticks.
BUFF_RULES = BuffRules(
    (
        BuffType("range", start=BLAST_RANGE, max=5, drop_chance=0.15),
        BuffType("bomb_count", start=BOMB_COUNT, max=3, drop_chance=0.15),
        BuffType("burn", start=BURN_LEVEL, max=2, drop_chance=0.10),
    )
)

# Rules of the real game. Everything that runs the real game (the UI, the
# Gym wrapper, the evaluation with ``--game-rules``) creates its env with
# ``BombermanEnv(**GAME_RULES)``. The engine defaults stay at BOMB_FUSE and
# FLAME_TICKS without buffs so the original rules remain testable.
# A 4-tick fuse (1 s at 250 ms per tick) and flames that stay deadly for
# 4 ticks (1 s). Provisional (v0) until the whole team confirms.
GAME_RULES = MappingProxyType({"bomb_fuse": 4, "flame_ticks": 4})


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
    """A bomb on the board. ``range`` and ``burn`` are its owner's levels at
    the moment it was placed; buffs picked up later never change them.
    The defaults are the levels without buffs."""

    x: int
    y: int
    countdown: int
    range: int = BLAST_RANGE
    burn: int = BURN_LEVEL


@dataclass(frozen=True)
class Gate:
    """The exit under the crate at ``GATE_POSITION``.

    ``state`` is ``"hidden"`` while the crate stands, ``"closed"`` once it is
    destroyed while an NPC lives, and ``"open"`` once it is destroyed and
    every NPC is dead. The gate is ordinary floor in every state.
    """

    x: int
    y: int
    state: str


@dataclass
class Character:
    """One character on the board. ``id`` 0 is the agent, 1-3 are NPCs."""

    id: int
    x: int
    y: int
    dead: bool = False
    # Buff levels by name; empty when buffs are off (the base levels apply).
    # They stay frozen when the character dies.
    levels: dict[str, int] = field(default_factory=dict)


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

# The gate hides under the central crate. The grid keeps using Cell.BLOCK and
# Cell.EMPTY there; the gate's state is tracked separately in ``env.gate``.
GATE_POSITION = (5, 5)

# Random-crate mode keeps the walls and the crate count of the fixed map.
CRATE_COUNT = 18
# Each corner and the three cells next to it along both walls never hold a
# random crate, so every spawn can escape its own opening bomb either way.
SAFE_CELLS = frozenset(
    {
        (1, 1), (2, 1), (3, 1), (4, 1), (1, 2), (1, 3), (1, 4),
        (9, 1), (8, 1), (7, 1), (6, 1), (9, 2), (9, 3), (9, 4),
        (1, 9), (2, 9), (3, 9), (4, 9), (1, 8), (1, 7), (1, 6),
        (9, 9), (8, 9), (7, 9), (6, 9), (9, 8), (9, 7), (9, 6),
    }
)
# Sorted, so a seed always maps to the same layout.
_RANDOM_CRATE_CELLS = tuple(
    sorted(
        (x, y)
        for y in range(MAP_HEIGHT)
        for x in range(MAP_WIDTH)
        if _INITIAL_GRID[y][x] is not Cell.WALL
        and (x, y) not in SAFE_CELLS
        and (x, y) != GATE_POSITION
    )
)


class BombermanEnv:
    """Board, characters, and their bombs.

    ``reset`` restores the initial board: the fixed map by default, or with
    ``random_crates=True`` the same walls with ``CRATE_COUNT`` crates, one on
    the gate and the rest drawn outside ``SAFE_CELLS``. The draw uses the
    env's own RNG: ``reset(seed=...)`` reseeds it, a plain ``reset()``
    continues it, as in Gymnasium. ``step`` lets every living
    character act in ID order (agent first, then NPCs 1, 2, 3), then counts
    down the bombs that were already on the board. A bomb placed on this
    action keeps the full fuse, so its owner has ``bomb_fuse`` later actions
    before it explodes. Moves on the exploding action are applied first.

    Without buffs each character has at most one bomb. Each cell holds at
    most one bomb. Characters never block each other and may share a cell.
    Bombs block movement for everyone, but only from the tick after they
    were placed. A bomb outlives its owner's death.

    ``outcome`` is decided at the end of every action: ``"win"`` when the
    living agent stands on the open gate, ``"draw"`` when the agent and the
    last living NPCs die on the same action, ``"loss"`` for any other agent
    death, otherwise ``None``. All NPCs dead is not an outcome by itself.
    Once there is an outcome (or the agent is dead), further ``step`` calls
    leave the state unchanged until ``reset``.

    Burned cells keep burning for ``flame_ticks`` ticks, counting the tick
    of the explosion (longer with the burn buff); ``flames`` maps each
    burning cell to its remaining ticks (counting the tick just resolved,
    never 0) and the owners of the blast that last burned it. Order inside
    ``step``:

    1. flames from earlier ticks age by one tick; those at 0 go out
    2. every living character acts in ID order (flames never block)
    3. bombs count down and due bombs explode, with chain reactions; every
       freshly burned cell (re)starts burning with the fresh blast's owners,
       for the longer of its time left and the fresh blast's time
    4. every living character on a burning cell dies, fresh or lingering
    5. with buffs: every living character on a floor buff picks it up
    6. the gate opens if it is revealed and every NPC is dead
    7. the outcome is decided

    Lingering flames never stop or start blast rays and never set off bombs;
    only fresh blasts do. With ``flame_ticks=1`` step 1 puts out every older
    flame, so only this tick's blast cells burn: exactly the original rules.
    ``explosion`` keeps listing only the cells burned on the latest action.

    ``bomb_fuse`` and ``flame_ticks`` must be integers of at least 1.

    Buffs (``buffs=BuffRules``, off with ``None``, the default): every
    character starts with the rules' start levels. A crate destroyed by a
    blast may drop a buff (never the gate's crate); a living character
    standing on it at the end of a tick picks it up and gains one level, up
    to the max. ``range`` is the blast range of the bombs it places,
    ``bomb_count`` the bombs it may have on the board at once, ``burn`` the
    burn level of its bombs. A bomb keeps the range and burn its owner had
    when it was placed. A fresh blast destroys the floor buffs on the cells
    it burns, except those dropped on this tick. Buffs never block anything.
    Drops and pickup ties use a second RNG seeded from the same seed, so the
    crates never depend on buffs. ``start_buffs`` (for example
    ``{0: {"range": 3}}``) sets absolute start levels for some characters;
    the ``reset`` argument overrides the constructor's for one game. With
    buffs off the engine behaves exactly as without this feature.

    Public attributes:
    - ``grid``: cells indexed as ``grid[y][x]``
    - ``characters``: ``Character`` list indexed by ID; only the agent by default
    - ``bombs``: each owner's oldest active ``Bomb`` by owner ID (read-only
      view; ``active_bombs()`` lists every bomb when an owner has several)
    - ``player_x``, ``player_y``: agent column and row
    - ``dead``: whether an explosion has hit the agent
    - ``bomb``: the agent's oldest active ``Bomb``, or ``None``
    - ``explosion``: cells burned by any bomb on the latest action, or an empty list
    - ``flames``: burning cell -> ``(ticks_left, owners)``, see above
    - ``bomb_fuse``, ``flame_ticks``: this env's timing settings
    - ``buff_rules``: the ``BuffRules`` in use, or ``None``
    - ``floor_buffs``: cell -> buff type of every buff lying on the floor
    - ``events``: events of the latest action, see ``_explode`` for their order
    - ``steps``: actions resolved since the last reset
    - ``gate``: the ``Gate`` at ``GATE_POSITION``
    - ``outcome``: ``None``, ``"win"``, ``"loss"``, or ``"draw"``
    """

    def __init__(
        self,
        *,
        with_npcs: bool = False,
        random_crates: bool = False,
        bomb_fuse: int = BOMB_FUSE,
        flame_ticks: int = FLAME_TICKS,
        buffs: BuffRules | None = None,
        start_buffs: Mapping[int, Mapping[str, int]] | None = None,
    ) -> None:
        for name, value in (("bomb_fuse", bomb_fuse), ("flame_ticks", flame_ticks)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1, not {value!r}")
        if buffs is not None and not isinstance(buffs, BuffRules):
            raise TypeError(f"buffs must be a BuffRules or None, not {type(buffs).__name__}")
        self.with_npcs = with_npcs
        self.random_crates = random_crates
        self.bomb_fuse = bomb_fuse
        self.flame_ticks = flame_ticks
        self.buff_rules = buffs
        self.start_buffs = start_buffs
        # Unseeded until the first reset(seed=...), like a Gymnasium env.
        self._rng = random.Random()
        self._buff_rng = random.Random()
        self.grid: list[list[Cell]] = []
        self.characters: list[Character] = []
        # Bomb cell -> (owner, bomb), in placement order. One bomb per cell.
        self._bombs: dict[tuple[int, int], tuple[int, Bomb]] = {}
        self.floor_buffs: dict[tuple[int, int], str] = {}
        self.explosion: list[tuple[int, int]] = []
        self.flames: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {}
        self.events: list[dict] = []
        self.steps = 0
        self.gate = Gate(*GATE_POSITION, "hidden")
        self.outcome: str | None = None
        self.reset()

    @property
    def buffs_on(self) -> bool:
        return self.buff_rules is not None

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
    def bombs(self) -> dict[int, Bomb]:
        """Each owner's oldest active bomb, by owner ID.

        Without buffs every owner has at most one bomb, so this is every
        bomb on the board. It is a fresh dict: changing it changes nothing.
        """
        result: dict[int, Bomb] = {}
        for owner, bomb in self._bombs.values():
            result.setdefault(owner, bomb)
        return result

    def active_bombs(self) -> list[tuple[int, Bomb]]:
        """Every bomb on the board as ``(owner, bomb)``, in placement order."""
        return list(self._bombs.values())

    @property
    def bomb(self) -> Bomb | None:
        return self.bombs.get(AGENT_ID)

    @bomb.setter
    def bomb(self, value: Bomb | None) -> None:
        """Replace all of the agent's bombs with ``value`` (or none)."""
        self._bombs = {
            cell: entry for cell, entry in self._bombs.items() if entry[0] != AGENT_ID
        }
        if value is not None:
            self._bombs[(value.x, value.y)] = (AGENT_ID, value)

    @property
    def all_npcs_dead(self) -> bool:
        """True when NPCs exist and all are dead. Not a terminal state."""
        npcs = self.characters[1:]
        return bool(npcs) and all(npc.dead for npc in npcs)

    def reset(
        self,
        *,
        seed: int | None = None,
        start_buffs: Mapping[int, Mapping[str, int]] | None = None,
    ) -> dict:
        """Restore the map, characters, bombs, buffs, flames, events, steps, gate, and outcome.

        ``seed`` reseeds the env's RNGs first: the crate RNG with ``seed``
        and the buff RNG with ``"buffs:<seed>"``, so the crate layout of a
        seed is the same with or without buffs. Without buffs the seed only
        changes anything in random-crate mode; the fixed map ignores it.
        ``start_buffs`` overrides the constructor's ``start_buffs`` for this
        game (buffs must be on).
        """
        chosen = start_buffs if start_buffs is not None else self.start_buffs
        start_levels = self._checked_start_buffs(chosen)
        if seed is not None:
            self._rng = random.Random(seed)
            # A string seed is hashed with SHA-512, independent of PYTHONHASHSEED.
            self._buff_rng = random.Random(f"buffs:{seed}")
        if self.random_crates:
            self.grid = self._random_crate_grid()
        else:
            self.grid = [row[:] for row in _INITIAL_GRID]
        self.characters = [Character(AGENT_ID, *PLAYER_START)]
        if self.with_npcs:
            self.characters += [
                Character(npc_id, *NPC_STARTS[npc_id]) for npc_id in NPC_IDS
            ]
        if self.buffs_on:
            for character in self.characters:
                character.levels = {buff.name: buff.start for buff in self.buff_rules.types}
                character.levels.update(start_levels.get(character.id, {}))
        self._bombs = {}
        self.floor_buffs = {}
        self.explosion = []
        self.flames = {}
        self.events = []
        self.steps = 0
        self.gate = Gate(*GATE_POSITION, "hidden")
        self.outcome = None
        return self.get_state()

    def _checked_start_buffs(self, start_buffs) -> dict[int, dict[str, int]]:
        """Validated ``start_buffs`` as a plain dict. Raises before anything changes."""
        if start_buffs is None:
            return {}
        if not self.buffs_on:
            raise ValueError("start_buffs given, but buffs are off")
        if not isinstance(start_buffs, Mapping):
            raise TypeError(f"start_buffs must be a mapping, not {type(start_buffs).__name__}")
        ids = (AGENT_ID, *NPC_IDS) if self.with_npcs else (AGENT_ID,)
        result: dict[int, dict[str, int]] = {}
        for character_id, levels in start_buffs.items():
            if character_id not in ids:
                raise ValueError(f"start_buffs: no character {character_id!r} in this game")
            if not isinstance(levels, Mapping):
                raise TypeError(f"start_buffs[{character_id}] must be a mapping")
            for name, level in levels.items():
                buff = self.buff_rules.get(name)
                if buff is None:
                    raise ValueError(f"start_buffs: unknown buff {name!r}; "
                                     f"known: {list(self.buff_rules.names)}")
                if isinstance(level, bool) or not isinstance(level, int) \
                        or not buff.start <= level <= buff.max:
                    raise ValueError(f"start_buffs: {name} level must be an integer "
                                     f"in [{buff.start}, {buff.max}], not {level!r}")
            result[character_id] = dict(levels)
        return result

    def _random_crate_grid(self) -> list[list[Cell]]:
        """The fixed map's walls with a new random crate layout."""
        grid = [
            [Cell.EMPTY if cell is Cell.BLOCK else cell for cell in row]
            for row in _INITIAL_GRID
        ]
        gate_x, gate_y = GATE_POSITION
        grid[gate_y][gate_x] = Cell.BLOCK
        for x, y in self._rng.sample(_RANDOM_CRATE_CELLS, CRATE_COUNT - 1):
            grid[y][x] = Cell.BLOCK
        return grid

    # ----------------------------------------------------------- buff levels

    @staticmethod
    def _level(character: Character, name: str) -> int:
        """A character's level of ``name``; the base level when it has none."""
        return character.levels.get(name, BASE_LEVELS[name])

    def _own_bombs(self, owner: int) -> int:
        return sum(1 for bomb_owner, _ in self._bombs.values() if bomb_owner == owner)

    def _bombs_left(self, character: Character) -> int:
        """How many more bombs ``character`` may place right now (never negative)."""
        return max(self._level(character, "bomb_count") - self._own_bombs(character.id), 0)

    def get_state(self) -> dict:
        """Copy of the current state. Mutating the copy does not change the env.

        With buffs on, every character entry also has its levels (one key per
        buff type) and ``"bombs_left"``, and every bomb entry its ``"range"``
        and ``"burn"``. ``"buffs"`` lists the floor buffs (empty without
        buffs) and ``"buff_rules"`` is ``BuffRules.as_dict()`` or ``None``.
        """
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
            "characters": [self._character_entry(c) for c in self.characters],
            # Sorted by owner; a stable sort keeps one owner's bombs in placement order.
            "bombs": [
                self._bomb_entry(owner, b)
                for owner, b in sorted(self._bombs.values(), key=lambda entry: entry[0])
            ],
            # Event values are ints and tuples, so a shallow copy is enough.
            "events": [dict(event) for event in self.events],
            "all_npcs_dead": self.all_npcs_dead,
            "gate": {"x": self.gate.x, "y": self.gate.y, "state": self.gate.state},
            "outcome": self.outcome,
            "flames": [
                {"x": x, "y": y, "ticks_left": ticks_left, "owners": owners}
                for (x, y), (ticks_left, owners) in sorted(self.flames.items())
            ],
            "rules": {
                "bomb_fuse": self.bomb_fuse,
                "blast_range": BLAST_RANGE,
                "flame_ticks": self.flame_ticks,
            },
            "buffs": [
                {"x": x, "y": y, "type": kind}
                for (x, y), kind in sorted(self.floor_buffs.items())
            ],
            "buff_rules": self.buff_rules.as_dict() if self.buffs_on else None,
        }

    def _character_entry(self, c: Character) -> dict:
        entry = {"id": c.id, "x": c.x, "y": c.y, "dead": c.dead}
        if self.buffs_on:
            entry.update(c.levels)
            entry["bombs_left"] = self._bombs_left(c)
        return entry

    def _bomb_entry(self, owner: int, b: Bomb) -> dict:
        entry = {"owner": owner, "x": b.x, "y": b.y, "countdown": b.countdown}
        if self.buffs_on:
            entry["range"] = b.range
            entry["burn"] = b.burn
        return entry

    def step(
        self, action: Action, npc_actions: Mapping[int, Action] | None = None
    ) -> dict:
        """Resolve one action for every living character and return ``get_state()``.

        ``action`` is the agent's. ``npc_actions`` maps NPC IDs to actions;
        a missing NPC waits and an action for a dead NPC is ignored. Passing
        ``npc_actions`` without NPCs raises ``ValueError``.

        Movement that hits a wall, a block, or a bomb still spends the turn.
        Placing a bomb with no bomb left (one active bomb without buffs),
        or on a cell that already holds a bomb, does too.

        Every move is checked against the bombs that existed at the start of
        this action, so a character may walk onto a cell where a bomb is
        placed in the same action. Characters never block each other, so the
        ID order only breaks one tie: two characters placing a bomb on the
        same cell, where the lower ID places it and the other is refused.
        """
        if not isinstance(action, Action):
            raise TypeError(f"action must be an Action, not {type(action).__name__}")
        self._check_npc_actions(npc_actions)
        if self.dead or self.outcome is not None:
            return self.get_state()

        self.explosion = []
        self.events = []
        self.steps += 1
        npcs_alive_before = self._npcs_alive()
        self._age_flames()
        # Bombs placed during this action start blocking on the next one.
        blocking = set(self._bombs)
        placed: set[tuple[int, int]] = set()
        for character in self.characters:
            if character.dead:
                continue
            if character.id == AGENT_ID:
                chosen = action
            else:
                chosen = (npc_actions or {}).get(character.id, Action.WAIT)
            if self._apply_action(character, chosen, blocking):
                # A new bomb lies on the cell of the character that placed it.
                placed.add((character.x, character.y))
        # A bomb placed on this action does not tick. Older bombs do, after
        # every character has already moved.
        due = self._tick_bombs(placed)
        if due:
            self._explode(due)
        self._burn_characters()
        self._pick_up_buffs()
        self._maybe_open_gate()
        self._decide_outcome(npcs_alive_before)
        return self.get_state()

    def _age_flames(self) -> None:
        """Flames from earlier ticks burn one tick less; those at 0 go out."""
        self.flames = {
            cell: (ticks_left - 1, owners)
            for cell, (ticks_left, owners) in self.flames.items()
            if ticks_left > 1
        }

    def _burn_characters(self) -> None:
        """Every living character on a burning cell dies, in ID order.

        Runs after all moves, explosions and chain reactions of the action,
        for fresh and lingering flames alike. ``killed_by`` names the owners
        of the blast that last burned the cell.
        """
        for character in self.characters:
            position = (character.x, character.y)
            if not character.dead and position in self.flames:
                character.dead = True
                self.events.append(
                    {
                        "type": "death",
                        "character": character.id,
                        "killed_by": self.flames[position][1],
                    }
                )

    def _pick_up_buffs(self) -> None:
        """Every floor buff under a living character is picked up, in cell order.

        Runs after deaths, so a character killed on this tick picks up
        nothing. When several living characters share the buff's cell, one
        is chosen with the buff RNG (never by ID, which would favour the
        agent). The level rises by one up to the max; a pickup at the max
        is consumed and changes nothing. ``level`` in the event is the level
        after the pickup.
        """
        if not self.floor_buffs:
            return
        for cell in sorted(self.floor_buffs):
            takers = [c for c in self.characters if not c.dead and (c.x, c.y) == cell]
            if not takers:
                continue
            taker = takers[0] if len(takers) == 1 else self._buff_rng.choice(takers)
            kind = self.floor_buffs.pop(cell)
            top = self.buff_rules.get(kind).max
            taker.levels[kind] = min(taker.levels[kind] + 1, top)
            self.events.append(
                {
                    "type": "buff_picked",
                    "character": taker.id,
                    "x": cell[0],
                    "y": cell[1],
                    "buff": kind,
                    "level": taker.levels[kind],
                }
            )

    def _maybe_open_gate(self) -> None:
        """Open a revealed gate once every NPC is dead. Checked every action."""
        # Single-player has no NPCs to defeat, so its gate never opens.
        if self.gate.state == "closed" and self.all_npcs_dead:
            self.gate = Gate(self.gate.x, self.gate.y, "open")
            self.events.append({"type": "gate_opened"})

    def _npcs_alive(self) -> bool:
        return any(not npc.dead for npc in self.characters[1:])

    def _decide_outcome(self, npcs_alive_before: bool) -> None:
        """Set ``outcome`` after explosions, deaths, and gate updates.

        The agent's death is a draw only when it takes the last living NPCs
        with it; otherwise, including in single-player mode, it is a loss.
        """
        agent = self.characters[AGENT_ID]
        if agent.dead:
            if npcs_alive_before and not self._npcs_alive():
                self.outcome = "draw"
            else:
                self.outcome = "loss"
        elif self.gate.state == "open" and (agent.x, agent.y) == GATE_POSITION:
            self.outcome = "win"

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

    def _apply_action(
        self, character: Character, action: Action, blocking: set[tuple[int, int]]
    ) -> bool:
        """Apply one character's action. Return True when a new bomb was placed.

        ``blocking`` holds the bomb cells from the start of the action. The
        one-bomb-per-cell check uses the live bombs instead, so it also sees
        a bomb a lower ID placed earlier in this action. A floor buff never
        stops a bomb from being placed.
        """
        if action is Action.WAIT:
            return False

        if action is Action.BOMB:
            if self._bombs_left(character) <= 0:
                return False
            here = (character.x, character.y)
            if here in self._bombs:
                return False
            # The bomb keeps its owner's range and burn of this moment.
            self._bombs[here] = (
                character.id,
                Bomb(
                    character.x,
                    character.y,
                    self.bomb_fuse,
                    range=self._level(character, "range"),
                    burn=self._level(character, "burn"),
                ),
            )
            return True

        dx, dy = _MOVE[action]
        new_x = character.x + dx
        new_y = character.y + dy
        if self._is_walkable(new_x, new_y, blocking):
            character.x = new_x
            character.y = new_y
        return False

    def _is_walkable(self, x: int, y: int, blocking: set[tuple[int, int]]) -> bool:
        if not _inside(x, y):
            return False
        if self.grid[y][x] is not Cell.EMPTY:
            return False
        return (x, y) not in blocking

    def _bomb_order(self, cells) -> list[tuple[int, int]]:
        """Bomb cells sorted by owner, then placement order (oldest first).

        Without buffs every owner has one bomb, so this is owner order.
        """
        placement = {cell: i for i, cell in enumerate(self._bombs)}
        return sorted(cells, key=lambda cell: (self._bombs[cell][0], placement[cell]))

    def _tick_bombs(self, placed: set[tuple[int, int]]) -> list[tuple[int, int]]:
        """Count down bombs not placed this action. Return the cells of bombs now due."""
        due = []
        for cell, (owner, bomb) in list(self._bombs.items()):
            if cell in placed:
                continue
            # Same key, so the bomb keeps its place in the placement order.
            self._bombs[cell] = (owner, replace(bomb, countdown=bomb.countdown - 1))
            if bomb.countdown - 1 <= 0:
                due.append(cell)
        return self._bomb_order(due)

    def _blast_cells(
        self, bomb: Bomb, bomb_cells: Mapping[tuple[int, int], int]
    ) -> list[tuple[tuple[int, int], int]]:
        """Cells one bomb burns, with their distance from the bomb along the ray.

        Its own cell (distance 0) first, then each ray right, left, down, up,
        up to the bomb's own ``range``.
        """
        cells = [((bomb.x, bomb.y), 0)]
        for dx, dy in _BLAST_DELTAS:
            for distance in range(1, bomb.range + 1):
                x = bomb.x + dx * distance
                y = bomb.y + dy * distance
                if not _inside(x, y):
                    break
                tile = self.grid[y][x]
                if tile is Cell.WALL:
                    break
                cells.append(((x, y), distance))
                if tile is Cell.BLOCK or (x, y) in bomb_cells:
                    break
        return cells

    def _explode(self, due: list[tuple[int, int]]) -> None:
        """Burn every due bomb, and every bomb their blasts set off.

        Each bomb burns its own cell and up to its ``range`` cells each way
        (``BLAST_RANGE`` without buffs). A wall ends that ray and stays. A
        breakable block burns and ends that ray. Any other bomb's cell burns
        and ends that ray, and a bomb there that has not exploded yet is set
        off in the same action, with its own stored range and burn.

        Bombs explode in waves: first the due bombs, then the bombs their
        rays reached, and so on until no new bomb is set off. Within a wave
        bombs go in owner order (then placement order). Every ray of the
        action is traced against the board as it was before this action's
        blasts: blocks become floor only afterwards, and every bomb on the
        board when the explosions began still ends rays, even after it
        exploded in an earlier wave. A burned cell is credited to the owner
        of every bomb that burned it.

        Fresh timer of a burned cell: the longest over the bombs that burned
        it of ``flame_ticks``, plus the bomb's burn level L when the cell is
        within L tiles of the bomb's cell along its ray. The cell then burns
        for the longer of its time left and that fresh timer (with buffs off
        this is always ``flame_ticks``, as before).

        With buffs on, after the blocks: every floor buff on a burned cell is
        destroyed, then every destroyed crate except the gate's rolls once
        for a drop, in the order of the ``block_destroyed`` events.

        Then the gate is revealed if its crate burned, and every burned cell
        starts (or restarts) burning with the owners whose blasts reached it.
        Deaths, pickups and the gate opening follow in ``step``
        (``_burn_characters``, ``_pick_up_buffs``, ``_maybe_open_gate``), so
        lingering flames kill and open the gate the same way.

        Events of one action, in this order:
        1. ``bomb_triggered`` by wave, then owner, naming the owners whose
           rays in the previous wave reached the bomb (with buffs also the
           bomb's ``x`` and ``y``, since an owner may have several)
        2. ``block_destroyed`` in first-burn order, naming every owner whose
           blast reached the block
        3. ``buff_destroyed`` in first-burn order (``x``, ``y``, ``buff``,
           ``destroyed_by``)
        4. ``buff_dropped`` in ``block_destroyed`` order (``x``, ``y``, ``buff``)
        5. ``gate_revealed`` when the gate's crate was destroyed
        6. ``death`` in ID order (added by ``_burn_characters``), naming the
           owners of the blast that last burned the cell
        7. ``buff_picked`` in cell order (``_pick_up_buffs``)
        8. ``gate_opened`` when the gate became open (``_maybe_open_gate``)
        """
        # One bomb per cell, so each bomb cell names exactly one owner.
        bomb_cells = {cell: owner for cell, (owner, _) in self._bombs.items()}
        # Burned cell -> owners whose blast reached it, in first-burn order.
        burned: dict[tuple[int, int], set[int]] = {}
        # Burned cell -> longest fresh timer of this action's blasts there.
        fresh: dict[tuple[int, int], int] = {}
        exploded: list[tuple[int, int]] = []

        wave = due
        while wave:
            exploded += wave
            # Bomb cell set off by this wave -> owners whose rays reached it.
            reached: dict[tuple[int, int], set[int]] = {}
            for bomb_cell in wave:
                owner, bomb = self._bombs[bomb_cell]
                for cell, distance in self._blast_cells(bomb, bomb_cells):
                    burned.setdefault(cell, set()).add(owner)
                    timer = self.flame_ticks + (bomb.burn if distance <= bomb.burn else 0)
                    fresh[cell] = max(fresh.get(cell, 0), timer)
                    if cell in bomb_cells and cell not in exploded:
                        reached.setdefault(cell, set()).add(owner)
            wave = self._bomb_order(reached)
            for bomb_cell in wave:
                event = {
                    "type": "bomb_triggered",
                    "owner": bomb_cells[bomb_cell],
                    "triggered_by": tuple(sorted(reached[bomb_cell])),
                }
                if self.buffs_on:
                    event["x"], event["y"] = bomb_cell
                self.events.append(event)

        destroyed: list[tuple[int, int]] = []
        for (x, y), by in burned.items():
            if self.grid[y][x] is Cell.BLOCK:
                self.grid[y][x] = Cell.EMPTY
                destroyed.append((x, y))
                self.events.append(
                    {
                        "type": "block_destroyed",
                        "cell": (x, y),
                        "destroyed_by": tuple(sorted(by)),
                    }
                )

        if self.buffs_on:
            # Destroy before dropping, so this action's drops survive it.
            self._destroy_buffs(burned)
            self._drop_buffs(destroyed)

        for bomb_cell in exploded:
            del self._bombs[bomb_cell]
        self.explosion = list(burned)
        for cell, by in burned.items():
            left = self.flames[cell][0] if cell in self.flames else 0
            self.flames[cell] = (max(left, fresh[cell]), tuple(sorted(by)))

        gate_x, gate_y = GATE_POSITION
        if self.gate.state == "hidden" and self.grid[gate_y][gate_x] is Cell.EMPTY:
            self.gate = Gate(gate_x, gate_y, "closed")
            self.events.append({"type": "gate_revealed"})

    def _destroy_buffs(self, burned: Mapping[tuple[int, int], set[int]]) -> None:
        """A fresh blast destroys the floor buffs on every cell it burns."""
        for cell, by in burned.items():
            kind = self.floor_buffs.pop(cell, None)
            if kind is not None:
                self.events.append(
                    {
                        "type": "buff_destroyed",
                        "x": cell[0],
                        "y": cell[1],
                        "buff": kind,
                        "destroyed_by": tuple(sorted(by)),
                    }
                )

    def _drop_buffs(self, destroyed: list[tuple[int, int]]) -> None:
        """Each destroyed crate but the gate's rolls once for a buff, in order.

        One uniform draw from the buff RNG is compared with the cumulative
        drop chances in ``BuffRules.types`` order; above their sum nothing
        drops. The gate's crate draws nothing.
        """
        for cell in destroyed:
            if cell == GATE_POSITION:
                continue
            roll = self._buff_rng.random()
            edge = 0.0
            for buff in self.buff_rules.types:
                edge += buff.drop_chance
                if roll < edge:
                    self.floor_buffs[cell] = buff.name
                    self.events.append(
                        {"type": "buff_dropped", "x": cell[0], "y": cell[1],
                         "buff": buff.name}
                    )
                    break
