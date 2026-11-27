"""The danger map (Task 13): where blasts will land and when, as two extra planes.

The v1 observation (``rl.observation``) shows a bomb's countdown and range
only on the bomb's own cell; which cells burn, and when, is left to the
network. These two planes give it the answer the rule bot works out every
turn. They are only added when ``EnvConfig.danger_obs`` is on (off by
default, so the observation stays the v1 one, bit for bit).

``danger_planes(state, me)`` returns a ``(2, MAP_HEIGHT, MAP_WIDTH)``
float32 array, indexed ``[c, y, x]`` like the observation, every value in
[0, 1]:

- ``DANGER_SOON`` (plane 0): for every cell that will burn, how soon:
  ``(bomb_fuse + 1 - t) / bomb_fuse``, where ``t`` is the first action
  (1 = the next one) on which the cell burns. A cell burning on the next
  action is 1.0, one burning on the 4th next action (a bomb placed just now,
  fuse 4) 0.25, a cell no known blast reaches 0. It covers every bomb on
  the board, whoever owns it, with chain reactions (a bomb in another's
  blast goes off with it), each bomb's own range and burn level, and flames
  that are still burning (they kill too). Bombs that are not placed yet are
  not in it.
- ``BOMB_ESCAPE`` (plane 1): a full plane, 1.0 when a bomb placed on my
  cell right now would still leave me a way out, else 0. "A way out" is the
  rule bot's own check, in its cautious form: the escape must also survive
  every bomb an opponent could drop on this action. It is 0 when I cannot
  bomb (no bomb left, or a bomb already on my cell; ``MY_BOMBS_LEFT``
  already shows the first).

Both planes are all zero once ``me`` is dead.

Everything comes from ``agents.rule_bot`` (Joelle's code), used read-only,
so "safe" means exactly what it means for the rule bot: ``plan_bombs``,
``danger_times``, ``bombs_from_state``, ``levels_of``, and the bot's own
``_possible_bombs`` and ``_escape_after_bomb`` (called as they are, not
copied, so the two can never drift apart). Its timing is the engine's: a
bomb shown with ``countdown`` c explodes during the c-th next action.
"""

import numpy as np

from agents.rule_bot import RuleBot, bombs_from_state, danger_times, levels_of, plan_bombs
from game.environment import MAP_HEIGHT, MAP_WIDTH

DANGER_SOON = 0
BOMB_ESCAPE = 1
N_DANGER_CHANNELS = 2
DANGER_CHANNEL_NAMES = ("danger_soon", "bomb_escape")

# The rule bot's escape check only reads its arguments (no bot state, no
# random numbers), so one shared planner is enough.
_PLANNER = RuleBot(seed=0)


def burning_cells(state: dict) -> dict:
    """Cells still burning -> their ``ticks_left``, as the rule bot reads them."""
    return {(f["x"], f["y"]): f["ticks_left"] for f in state["flames"]}


def first_burn(state: dict) -> dict:
    """Cell -> the first action (1 = next) on which it burns, from the board's bombs and flames."""
    rules = state["rules"]
    planned = plan_bombs(state["grid"], bombs_from_state(state), rules["blast_range"])
    danger = danger_times(planned, rules["flame_ticks"], burning_cells(state))
    return {cell: min(times) for cell, times in danger.items() if times}


def can_escape_bomb(state: dict, me: int) -> bool:
    """Would a bomb on ``me``'s cell now still leave a way out (cautious)?

    ``False`` when ``me`` is dead or cannot bomb (the engine would refuse).
    The same call as ``RuleBot.act`` makes before it bombs.
    """
    character = state["characters"][me]
    if character["dead"]:
        return False
    here = (character["x"], character["y"])
    bombs = bombs_from_state(state)
    my_range, my_burn, bombs_left = levels_of(state, character)
    if bombs_left <= 0 or here in {(b[0], b[1]) for b in bombs}:
        return False
    possible = RuleBot._possible_bombs(state, me)
    return _PLANNER._escape_after_bomb(state["grid"], here, bombs, state["rules"],
                                       burning_cells(state), my_range, my_burn, possible)


def danger_planes(state: dict, me: int = 0) -> np.ndarray:
    """The two danger planes of ``state`` as seen by character ``me``."""
    planes = np.zeros((N_DANGER_CHANNELS, MAP_HEIGHT, MAP_WIDTH), dtype=np.float32)
    if state["characters"][me]["dead"]:
        return planes
    fuse = state["rules"]["bomb_fuse"]
    for (x, y), t in first_burn(state).items():
        planes[DANGER_SOON, y, x] = min(max((fuse + 1 - t) / fuse, 0.0), 1.0)
    if can_escape_bomb(state, me):
        planes[BOMB_ESCAPE] = 1.0
    return planes
