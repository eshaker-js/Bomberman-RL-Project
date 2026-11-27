"""The safety filter (Task 13): which of the agent's actions still leave a way out.

``action_mask(state, me, mode)`` returns 6 booleans, one per action in the
order of ``rl.bomberman_env.ACTIONS`` (UP, DOWN, LEFT, RIGHT, WAIT, BOMB):
``True`` = allowed. With ``EnvConfig.action_mask`` on, the agent's policy
only picks among the allowed actions (``sb3_contrib.MaskablePPO``, see
``rl.train_ppo``). Known as "shielding" in safe RL (M. Alshiekh et al.,
"Safe Reinforcement Learning via Shielding", AAAI 2018): PPO still learns
where to go, when to bomb and how to win; the filter only removes actions
after which no escape exists.

Modes (``MASK_MODES``):

- ``"off"`` (default): every action allowed, and nothing is computed.
- ``"board"``: a move or WAIT is allowed when, after it, some path still
  escapes every bomb on the board (whoever owns it, with chain reactions,
  range and burn levels, and flames still burning). BOMB is allowed when the
  same holds with the new bomb added.
- ``"cautious"``: the rule bot's own rule. Moves and WAIT must also survive
  every bomb an opponent could drop on this action; if none does, it falls
  back to ``"board"`` for them. BOMB must escape the possible opponent
  bombs too.

Two edge cases, both as decided for Task 13:

- A BOMB the engine would refuse (no bomb left, or a bomb already on my
  cell) leaves me standing still, like WAIT, so it gets WAIT's value. The
  filter only removes deadly actions.
- If no action is safe (trapped), every action is allowed: the agent dies
  anyway, and ``MaskablePPO`` cannot sample from an all-False mask.

Once ``me`` is dead or the game is over every action is allowed (the
engine ignores them).

Everything comes from ``agents.rule_bot`` (Joelle's code), used read-only
and not copied: ``plan_bombs``, ``cautious_plan``, ``steps_to_safety``,
``bombs_from_state``, ``levels_of``, ``RuleBot._possible_bombs`` and
``RuleBot._escape_after_bomb``. "Safe" means exactly what it means for the
rule bot (``RuleBot.act``, steps 0 and 3).
"""

import numpy as np

from agents.rule_bot import (
    STEP_ACTIONS,
    RuleBot,
    bombs_from_state,
    cautious_plan,
    levels_of,
    plan_bombs,
    steps_to_safety,
)
from game.environment import Action

MASK_MODES = ("off", "board", "cautious")
# Mask index of each action: the order of the Action enum (= ACTIONS).
_INDEX = {action: i for i, action in enumerate(Action)}
N_ACTIONS = len(_INDEX)

# The escape check only reads its arguments (no bot state, no random numbers).
_PLANNER = RuleBot(seed=0)


def check_mask_mode(mode) -> None:
    """``ValueError`` unless ``mode`` is one of ``MASK_MODES``."""
    if mode not in MASK_MODES:
        raise ValueError(f"unknown action_mask {mode!r}; choose from {list(MASK_MODES)}")


def _safe_steps(grid, here, planned, flame_ticks, burning) -> set:
    return {a for a in STEP_ACTIONS
            if steps_to_safety(grid, here, planned, a, flame_ticks, burning) is not None}


def action_mask(state: dict, me: int, mode: str) -> np.ndarray:
    """The allowed actions of ``me`` in ``state`` under ``mode`` (see the module docstring)."""
    check_mask_mode(mode)
    mask = np.ones(N_ACTIONS, dtype=bool)
    character = state["characters"][me]
    if mode == "off" or character["dead"] or state["outcome"] is not None:
        return mask

    grid = state["grid"]
    here = (character["x"], character["y"])
    rules = state["rules"]
    blast_range, flame_ticks = rules["blast_range"], rules["flame_ticks"]
    burning = {(f["x"], f["y"]): f["ticks_left"] for f in state["flames"]}
    bombs = bombs_from_state(state)
    planned = plan_bombs(grid, bombs, blast_range)

    possible = RuleBot._possible_bombs(state, me) if mode == "cautious" else []
    if mode == "cautious":
        safe = _safe_steps(grid, here, cautious_plan(grid, bombs, possible, blast_range),
                           flame_ticks, burning)
        if not safe:  # the rule bot's fallback: the bombs really on the board
            safe = _safe_steps(grid, here, planned, flame_ticks, burning)
    else:
        safe = _safe_steps(grid, here, planned, flame_ticks, burning)

    my_range, my_burn, bombs_left = levels_of(state, character)
    if bombs_left <= 0 or here in {(b[0], b[1]) for b in bombs}:
        bomb_ok = Action.WAIT in safe  # the engine refuses: it is a WAIT
    else:
        bomb_ok = _PLANNER._escape_after_bomb(grid, here, bombs, rules, burning,
                                              my_range, my_burn, possible)

    for action in STEP_ACTIONS:
        mask[_INDEX[action]] = action in safe
    mask[_INDEX[Action.BOMB]] = bomb_ok
    if not mask.any():
        mask[:] = True  # trapped: nothing survives
    return mask
