"""Numeric observation of the full game (v0), from one character's point of view.

``encode_observation(state, me)`` turns ``BombermanEnv.get_state()`` into a
``(N_CHANNELS, MAP_HEIGHT, MAP_WIDTH)`` float32 array indexed ``obs[c, y, x]``
with every value in [0, 1]. Only ``me`` decides which character is "me" and
which are "opponents", so a frozen copy of the agent playing NPC 2 can call
``encode_observation(state, 2)`` and see itself in the ``ME`` channel.
The board is never flipped or rotated (v0).
"""

import numpy as np

from game.environment import MAP_HEIGHT, MAP_WIDTH, Cell

# Channel indices (v0).
WALLS = 0  # 1 on walls
CRATES = 1  # 1 on crates (breakable blocks)
BOMBS = 2  # countdown / bomb_fuse on every bomb, any owner: (0, 1]
FLAMES = 3  # ticks_left / flame_ticks on every burning cell: (0, 1]
ME = 4  # 1 on my cell while I am alive, all zero once I am dead
OPPONENTS = 5  # 1 on every cell with a living character other than me
GATE_HIDDEN = 6  # 1 on the gate cell while its crate stands
GATE_CLOSED = 7  # 1 on the gate cell once revealed, while an NPC lives
GATE_OPEN = 8  # 1 on the gate cell once open
BOMB_READY = 9  # full plane: 1 while I am alive and have no active bomb

N_CHANNELS = 10
CHANNEL_NAMES = (
    "walls", "crates", "bombs", "flames", "me", "opponents",
    "gate_hidden", "gate_closed", "gate_open", "bomb_ready",
)
OBS_SHAPE = (N_CHANNELS, MAP_HEIGHT, MAP_WIDTH)

_GATE_CHANNEL = {"hidden": GATE_HIDDEN, "closed": GATE_CLOSED, "open": GATE_OPEN}


def encode_observation(state: dict, me: int = 0) -> np.ndarray:
    """The v0 observation of ``state`` as seen by character ``me``."""
    ids = [c["id"] for c in state["characters"]]
    if me not in ids:
        raise ValueError(f"character {me!r} is not in this game (ids {ids})")

    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    for y, row in enumerate(state["grid"]):
        for x, cell in enumerate(row):
            if cell is Cell.WALL:
                obs[WALLS, y, x] = 1.0
            elif cell is Cell.BLOCK:
                obs[CRATES, y, x] = 1.0

    rules = state["rules"]
    for bomb in state["bombs"]:
        obs[BOMBS, bomb["y"], bomb["x"]] = bomb["countdown"] / rules["bomb_fuse"]
    for flame in state["flames"]:
        obs[FLAMES, flame["y"], flame["x"]] = flame["ticks_left"] / rules["flame_ticks"]

    for character in state["characters"]:
        if character["dead"]:
            continue
        channel = ME if character["id"] == me else OPPONENTS
        obs[channel, character["y"], character["x"]] = 1.0

    gate = state["gate"]
    obs[_GATE_CHANNEL[gate["state"]], gate["y"], gate["x"]] = 1.0

    alive = not state["characters"][ids.index(me)]["dead"]
    armed = any(bomb["owner"] == me for bomb in state["bombs"])
    if alive and not armed:
        obs[BOMB_READY] = 1.0
    return obs
