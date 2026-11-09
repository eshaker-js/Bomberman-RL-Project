"""The tiny stage for tabular Q-learning, and its discrete state encoding.

``TinyStageEnv`` is a small Gymnasium env of its own (not a mode of the
full wrapper): single-player, the fixed map, ``GAME_RULES`` timing. An
episode ends when the first crate is destroyed or the agent dies
(``terminated``), or after ``max_steps`` steps (``truncated``). Reward (v0,
from ``TabularConfig``): ``reward_death`` if the agent died on this step,
otherwise ``reward_crate`` if a crate was destroyed, plus ``reward_step``
every step. A crate destroyed by the blast that also kills the agent only
counts as a death.

State (one integer, see ``encode_state``):

- the agent's cell;
- its active bomb: cell and countdown (1..bomb_fuse), or none;
- its burning flames: the cell of the bomb that made them, and their ticks
  left (1..flame_ticks), or none.

Until the first crate is destroyed the map never changes, so on the tiny
stage this state is Markov without the grid. The flames' cell is needed
because a bomb that hits no crate leaves deadly flames the agent could walk
into. A new bomb and the old flames can exist together, hence two fields.

Cells are indexed over every non-wall cell of the board (65), not only the
fixed map's 47 floor cells, so the same encoding also works in the full game
with random crates (``evaluate_agent --stage full``):
65 x (65*9 + 1) x (65*4 + 1) = 9,941,490 possible indices. Only visited
states are ever stored.
"""

import gymnasium as gym
from gymnasium import spaces

from game.environment import (
    AGENT_ID,
    GAME_RULES,
    MAP_HEIGHT,
    MAP_WIDTH,
    Action,
    BombermanEnv,
    Cell,
)
from rl.config import TabularConfig

ACTIONS = tuple(Action)

# Every cell a character can ever stand on (walls are the same on every
# board), in a fixed (row, column) order.
_GRID = BombermanEnv().get_state()["grid"]
CELLS = tuple(
    (x, y)
    for y in range(MAP_HEIGHT)
    for x in range(MAP_WIDTH)
    if _GRID[y][x] is not Cell.WALL
)
CELL_INDEX = {cell: i for i, cell in enumerate(CELLS)}
FUSE = GAME_RULES["bomb_fuse"]
FLAME_TICKS = GAME_RULES["flame_ticks"]
BOMB_VALUES = len(CELLS) * FUSE + 1  # + "no bomb"
FLAME_VALUES = len(CELLS) * FLAME_TICKS + 1  # + "no flames"
N_STATES = len(CELLS) * BOMB_VALUES * FLAME_VALUES


def encode_state(cell, bomb=None, flames=None) -> int:
    """Index of (my cell, my bomb ``(x, y, countdown)``, my flames ``(x, y, ticks)``)."""
    b = 0 if bomb is None else 1 + CELL_INDEX[bomb[:2]] * FUSE + (bomb[2] - 1)
    f = 0 if flames is None else 1 + CELL_INDEX[flames[:2]] * FLAME_TICKS + (flames[2] - 1)
    return (CELL_INDEX[tuple(cell)] * BOMB_VALUES + b) * FLAME_VALUES + f


def decode_state(index: int) -> tuple:
    """Inverse of ``encode_state``: ``(cell, bomb or None, flames or None)``."""
    rest, f = divmod(index, FLAME_VALUES)
    p, b = divmod(rest, BOMB_VALUES)
    bomb = None if b == 0 else (*CELLS[(b - 1) // FUSE], (b - 1) % FUSE + 1)
    flames = None if f == 0 else (*CELLS[(f - 1) // FLAME_TICKS], (f - 1) % FLAME_TICKS + 1)
    return CELLS[p], bomb, flames


class TabularStateTracker:
    """Turns successive engine states into ``encode_state`` indices for one character.

    The engine state shows burning cells but not which bomb made them, so
    the tracker remembers the character's bomb: when it disappears, its
    cell becomes the origin of the character's flames. Call ``reset`` at
    the start of every episode and ``observe`` on every state, in order.
    Assumes the game uses ``GAME_RULES`` timing.
    """

    def __init__(self, me: int = AGENT_ID) -> None:
        self.me = me
        self.reset()

    def reset(self) -> None:
        self._bomb = None
        self._flame_origin = None

    def observe(self, state: dict) -> int:
        me = state["characters"][self.me]
        bomb = next(((b["x"], b["y"], b["countdown"]) for b in state["bombs"]
                     if b["owner"] == self.me), None)
        if self._bomb is not None and bomb is None:
            self._flame_origin = self._bomb[:2]  # my bomb exploded
        self._bomb = bomb
        ticks = max((f["ticks_left"] for f in state["flames"] if self.me in f["owners"]),
                    default=0)
        flames = None
        if ticks and self._flame_origin is not None:
            flames = (*self._flame_origin, ticks)
        return encode_state((me["x"], me["y"]), bomb, flames)


class TinyStageEnv(gym.Env):
    """Single-player, fixed map: bomb the first crate without dying."""

    metadata = {"render_modes": []}

    def __init__(self, config: TabularConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else TabularConfig()
        # GAME_RULES timing without buffs: the tiny stage has no buffs.
        self.engine = BombermanEnv(random_crates=False, **{**GAME_RULES, "buffs": None})
        self.tracker = TabularStateTracker(AGENT_ID)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Discrete(N_STATES)
        self._state = self.engine.get_state()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # the fixed map has no randomness
        self._state = self.engine.reset()
        self.tracker.reset()
        return self.tracker.observe(self._state), {"result": None, "steps": 0}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r} for {self.action_space}")
        state = self._state = self.engine.step(ACTIONS[int(action)])
        cfg = self.config
        died = state["dead"]
        crate = any(e["type"] == "block_destroyed" for e in state["events"])
        terms = {"death": cfg.reward_death if died else 0.0,
                 "crate": cfg.reward_crate if crate and not died else 0.0,
                 "step": cfg.reward_step}
        terminated = died or crate
        truncated = not terminated and state["steps"] >= cfg.max_steps
        result = "death" if died else "crate" if crate else "timeout" if truncated else None
        info = {"result": result, "steps": state["steps"],
                "events": [dict(e) for e in state["events"]], "reward_terms": terms}
        reward = float(sum(terms.values()))
        return self.tracker.observe(state), reward, terminated, truncated, info

    @property
    def state(self) -> dict:
        """The engine state after the latest reset or step. Read it, don't mutate it."""
        return self._state
