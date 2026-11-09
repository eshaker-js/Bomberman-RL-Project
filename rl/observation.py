"""Numeric observation of the full game (v1), from one character's point of view.

``encode_observation(state, me)`` turns ``BombermanEnv.get_state()`` into a
``(N_CHANNELS, MAP_HEIGHT, MAP_WIDTH)`` float32 array indexed ``obs[c, y, x]``
with every value in [0, 1]. Only ``me`` decides which character is "me" and
which are "opponents", so a frozen copy of the agent playing NPC 2 can call
``encode_observation(state, 2)`` and see itself in the ``ME`` channel.
The board is never flipped or rotated.

Levels are divided by the highest level the game's ``buff_rules`` allow
(never by a literal), so every value stays in [0, 1]. A state without buffs
(``buff_rules`` is ``None``) uses the engine's base levels as both start and
max: my range plane is then 1, my burn plane 0 (a max of 0 always gives 0),
and ``MY_BOMBS_LEFT`` is exactly the old "my bomb is ready" plane.
"""

import numpy as np

from game.environment import BASE_LEVELS, MAP_HEIGHT, MAP_WIDTH, Cell

# Channel indices (v1).
WALLS = 0  # 1 on walls
CRATES = 1  # 1 on crates (breakable blocks)
BOMBS = 2  # countdown / bomb_fuse on every bomb, any owner: (0, 1]
BOMB_RANGE = 3  # the bomb's range / max range, on every bomb cell
FLAMES = 4  # ticks_left / (flame_ticks + max burn level) on every burning cell
ME = 5  # 1 on my cell while I am alive, all zero once I am dead
OPPONENTS = 6  # 1 on every cell with a living character other than me
GATE_HIDDEN = 7  # 1 on the gate cell while its crate stands
GATE_CLOSED = 8  # 1 on the gate cell once revealed, while an NPC lives
GATE_OPEN = 9  # 1 on the gate cell once open
BUFF_RANGE = 10  # 1 on every range buff lying on the floor
BUFF_BOMB = 11  # 1 on every extra-bomb buff lying on the floor
BUFF_BURN = 12  # 1 on every burn buff lying on the floor
MY_BOMBS_LEFT = 13  # full plane: my bombs left / max bomb count (0 once dead)
MY_RANGE = 14  # full plane: my range / max range (0 once dead)
MY_BURN = 15  # full plane: my burn level / max burn level (0 once dead)
OPP_RANGE = 16  # on each living opponent's cell: its range / max range
OPP_BOMBS_LEFT = 17  # on each living opponent's cell: its bombs left / max bomb count
OPP_BURN = 18  # on each living opponent's cell: its burn level / max burn level
BOMB_BURN = 19  # the bomb's burn level / max burn level, on every bomb cell

N_CHANNELS = 20
CHANNEL_NAMES = (
    "walls", "crates", "bombs", "bomb_range", "flames", "me", "opponents",
    "gate_hidden", "gate_closed", "gate_open",
    "buff_range", "buff_bomb", "buff_burn",
    "my_bombs_left", "my_range", "my_burn",
    "opp_range", "opp_bombs_left", "opp_burn", "bomb_burn",
)
OBS_SHAPE = (N_CHANNELS, MAP_HEIGHT, MAP_WIDTH)

_GATE_CHANNEL = {"hidden": GATE_HIDDEN, "closed": GATE_CLOSED, "open": GATE_OPEN}
# Floor buff type -> its channel. A new buff type needs a new channel here.
_BUFF_CHANNEL = {"range": BUFF_RANGE, "bomb_count": BUFF_BOMB, "burn": BUFF_BURN}


def _ratio(value: float, top: float) -> float:
    """``value / top`` clipped to [0, 1]; 0 when ``top`` is 0."""
    if top <= 0:
        return 0.0
    return float(min(max(value / top, 0.0), 1.0))


def max_levels(state: dict) -> dict:
    """Highest level of every buff in this game (the base level without buffs)."""
    base = dict(BASE_LEVELS, range=state["rules"]["blast_range"])
    rules = state.get("buff_rules")
    if rules is None:
        return base
    return {name: rules[name]["max"] if name in rules else level
            for name, level in base.items()}


def character_levels(state: dict, character: dict) -> dict:
    """``range``, ``bomb_count``, ``burn`` and ``bombs_left`` of one character.

    Without buffs the keys are missing: the base levels apply, and a
    character has one bomb left unless it already has a bomb on the board.
    """
    levels = {"range": character.get("range", state["rules"]["blast_range"]),
              "bomb_count": character.get("bomb_count", BASE_LEVELS["bomb_count"]),
              "burn": character.get("burn", BASE_LEVELS["burn"])}
    if "bombs_left" in character:
        levels["bombs_left"] = character["bombs_left"]
    else:
        owned = sum(1 for b in state["bombs"] if b["owner"] == character["id"])
        levels["bombs_left"] = max(levels["bomb_count"] - owned, 0)
    return levels


def encode_observation(state: dict, me: int = 0) -> np.ndarray:
    """The v1 observation of ``state`` as seen by character ``me``."""
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
    top = max_levels(state)
    for bomb in state["bombs"]:
        x, y = bomb["x"], bomb["y"]
        obs[BOMBS, y, x] = _ratio(bomb["countdown"], rules["bomb_fuse"])
        obs[BOMB_RANGE, y, x] = _ratio(bomb.get("range", rules["blast_range"]), top["range"])
        obs[BOMB_BURN, y, x] = _ratio(bomb.get("burn", BASE_LEVELS["burn"]), top["burn"])
    longest_burn = rules["flame_ticks"] + top["burn"]
    for flame in state["flames"]:
        obs[FLAMES, flame["y"], flame["x"]] = _ratio(flame["ticks_left"], longest_burn)

    for buff in state.get("buffs", ()):
        channel = _BUFF_CHANNEL.get(buff["type"])
        if channel is not None:
            obs[channel, buff["y"], buff["x"]] = 1.0

    for character in state["characters"]:
        if character["dead"]:
            continue
        x, y = character["x"], character["y"]
        levels = character_levels(state, character)
        if character["id"] == me:
            obs[ME, y, x] = 1.0
            obs[MY_BOMBS_LEFT] = _ratio(levels["bombs_left"], top["bomb_count"])
            obs[MY_RANGE] = _ratio(levels["range"], top["range"])
            obs[MY_BURN] = _ratio(levels["burn"], top["burn"])
        else:
            # Opponents may share a cell: each plane keeps the highest value.
            obs[OPPONENTS, y, x] = 1.0
            obs[OPP_RANGE, y, x] = max(obs[OPP_RANGE, y, x],
                                       _ratio(levels["range"], top["range"]))
            obs[OPP_BOMBS_LEFT, y, x] = max(obs[OPP_BOMBS_LEFT, y, x],
                                            _ratio(levels["bombs_left"], top["bomb_count"]))
            obs[OPP_BURN, y, x] = max(obs[OPP_BURN, y, x],
                                      _ratio(levels["burn"], top["burn"]))

    gate = state["gate"]
    obs[_GATE_CHANNEL[gate["state"]], gate["y"], gate["x"]] = 1.0
    return obs
