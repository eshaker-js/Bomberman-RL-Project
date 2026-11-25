"""Gymnasium wrapper of the full game for the RL agent (character 0).

The wrapper builds a ``BombermanEnv`` with the real game's rules
(``GAME_RULES``, random crates) and never changes the engine: it only reads
``get_state()`` and passes actions to ``step``.

Opponents (``opponents=``):

- a ladder level name from ``agents.noisy_bot.LADDER`` (``"random"``,
  ``"noisy50"``, ...): three fresh bots of that level play NPCs 1-3;
- ``"idle"`` (Task 11), a training-only opponent from
  ``agents.idle_bot.TRAINING_ONLY_OPPONENTS``: three bots that always WAIT.
  It is not a ladder level, so the evaluation protocol never plays it;
- a list of 1-3 controllers, any object with ``reset(seed)`` and
  ``act(state, me)`` (ladder bots now, frozen agents later): they play
  NPCs 1, 2, 3 in list order. The engine always has three NPCs, so **an NPC
  without a controller always WAITs**: it stays alive until a blast kills
  it, and the gate only opens once it is dead;
- ``None`` or ``[]``: single-player. The gate never opens without NPCs, so a
  single-player episode can only end in death (``"loss"``) or truncation.

Seeding: ``reset(seed=s)`` uses ``s`` directly as the engine seed, so it
gives the same crate layout and bot seeds as ``agents.match.play_game(...,
seed=s)``. ``reset()`` without a seed draws the next episode seed from the
wrapper's ``np_random`` (seeded by the last explicit seed).

Each ``step``: the NPC controllers choose from the state *before* the step
(like ``play_game``), the engine resolves one action for everyone, and the
reward is computed from the new state only (``step_reward``). ``terminated``
is True once the engine has an outcome; ``truncated`` is True at
``max_steps`` engine steps unless the episode terminated on that step.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agents.idle_bot import TRAINING_ONLY_OPPONENTS
from agents.match import bot_seed
from agents.noisy_bot import LADDER, make_bot
from game.environment import AGENT_ID, GAME_RULES, NPC_IDS, Action, BombermanEnv
from rl.config import EnvConfig, RewardConfig
from rl.observation import OBS_SHAPE, encode_observation

# Discrete action index -> engine action, in the order of the Action enum:
# 0 UP, 1 DOWN, 2 LEFT, 3 RIGHT, 4 WAIT, 5 BOMB.
ACTIONS = tuple(Action)
MOVE_ACTIONS = frozenset({Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT})

# The Task 10 terms come last, so the sum of the v0 terms keeps its order
# (with both at 0.0 every total is bit-identical to before they existed).
REWARD_TERMS = ("outcome", "kill", "crate", "per_step", "bump", "buff",
                "wait", "bomb_survived")
_SEED_BOUND = 2**31 - 1
# Every opponent name training accepts (Task 11): the ladder levels, then the
# training-only opponents. Evaluation accepts ladder levels only.
TRAINING_OPPONENTS = (*LADDER, *TRAINING_ONLY_OPPONENTS)


def resolve_opponents(opponents) -> tuple[bool, dict]:
    """``(with_npcs, {npc_id: controller})`` for an ``opponents`` argument."""
    if opponents is None:
        return False, {}
    if isinstance(opponents, str):
        if opponents in TRAINING_ONLY_OPPONENTS:
            factory = TRAINING_ONLY_OPPONENTS[opponents]
            return True, {npc_id: factory() for npc_id in NPC_IDS}
        if opponents not in LADDER:
            raise ValueError(f"unknown opponents {opponents!r}; choose a ladder level "
                             f"{list(LADDER)} or a training-only opponent "
                             f"{list(TRAINING_ONLY_OPPONENTS)}")
        return True, {npc_id: make_bot(opponents) for npc_id in NPC_IDS}
    controllers = list(opponents)
    if not controllers:
        return False, {}
    if len(controllers) > len(NPC_IDS):
        raise ValueError(f"at most {len(NPC_IDS)} NPC controllers, got {len(controllers)}")
    for controller in controllers:
        if not (callable(getattr(controller, "reset", None))
                and callable(getattr(controller, "act", None))):
            raise TypeError(f"{controller!r} needs reset(seed) and act(state, me)")
    return True, dict(zip(NPC_IDS, controllers))


def step_reward(prev: dict, state: dict, action: Action, me: int,
                cfg: RewardConfig) -> tuple[float, dict]:
    """Reward of one step for character ``me``, and its terms (v0).

    Uses only the step's ``events``, ``outcome`` and the positions before and
    after the step:

    - ``outcome``: ``win`` / ``loss`` / ``draw`` on the step it is decided
      (a death by my own bomb is simply a loss);
    - ``kill``: per ``death`` event of another character with ``me`` in
      ``killed_by`` (lingering flames credit their blast's owners);
    - ``crate``: per ``block_destroyed`` event with ``me`` in ``destroyed_by``
      (a crate several blasts reached credits every owner in full);
    - ``per_step``: every step;
    - ``bump``: a move action after which my position is unchanged (wall,
      crate, bomb). A refused bomb is not a bump.
    - ``buff``: per ``buff_picked`` event of ``me`` that raised my level
      (a pickup at the max level earns nothing).
    - ``wait`` (Task 10, 0.0 by default): the action was ``WAIT``, or a
      ``BOMB`` the engine refused (no bomb left, or a bomb already on my
      cell), which also leaves me standing still (``bomb_refused``).
    - ``bomb_survived`` (Task 10, 0.0 by default): per own bomb that
      exploded and broke at least one crate, while I am alive after the
      step. Counted as ``min(exploded, crates)``: ``exploded`` = my bombs in
      ``prev["bombs"]`` and not in ``state["bombs"]`` (matched by cell; in a
      step a bomb leaves the list only by exploding, by its fuse or a chain
      reaction), ``crates`` = this step's ``block_destroyed`` events with
      ``me`` in ``destroyed_by``. The events don't say which bomb broke
      which crate, so this is exact when one of my bombs explodes. With two
      or more (extra-bomb buff) it is a known approximation: one bomb
      breaking 2 crates and another breaking none pays twice (one too many);
      two bombs that both reach the same single crate pay once (one too
      few). It never pays more often than crates were broken (at most 18 per
      game). A bomb placed and set off on the same step is not in ``prev``
      and is never counted, but its placer stands on its burning cell and
      dies, so it would pay nothing anyway.
    """
    terms = dict.fromkeys(REWARD_TERMS, 0.0)
    my_crates = 0
    outcome = state["outcome"]
    if outcome is not None:
        terms["outcome"] = {"win": cfg.win, "loss": cfg.loss, "draw": cfg.draw}[outcome]
    for event in state["events"]:
        if event["type"] == "death":
            if event["character"] != me and me in event["killed_by"]:
                terms["kill"] += cfg.kill
        elif event["type"] == "block_destroyed":
            if me in event["destroyed_by"]:
                terms["crate"] += cfg.crate
                my_crates += 1
        elif event["type"] == "buff_picked" and event["character"] == me:
            before = prev["characters"][me].get(event["buff"], event["level"])
            if event["level"] > before:
                terms["buff"] += cfg.buff
    terms["per_step"] = cfg.per_step
    if action in MOVE_ACTIONS:
        before = prev["characters"][me]
        after = state["characters"][me]
        if (before["x"], before["y"]) == (after["x"], after["y"]):
            terms["bump"] = cfg.bump
    if action is Action.WAIT or (action is Action.BOMB and bomb_refused(prev, me)):
        terms["wait"] = cfg.wait
    exploded = own_bomb_cells(prev, me) - own_bomb_cells(state, me)
    if exploded and not state["characters"][me]["dead"]:
        terms["bomb_survived"] = cfg.bomb_survived * min(len(exploded), my_crates)
    return float(sum(terms.values())), terms


def own_bomb_cells(state: dict, me: int) -> set:
    """The cells ``(x, y)`` of ``me``'s bombs in ``state`` (one bomb per cell).

    A state without ``"bombs"`` (a hand-made test state) has no bombs; the
    engine's states always have the key.
    """
    return {(b["x"], b["y"]) for b in state.get("bombs", ()) if b["owner"] == me}


def bomb_refused(prev: dict, me: int) -> bool:
    """Would the engine refuse a ``BOMB`` of ``me`` in the state ``prev``?

    It refuses when ``me`` has no bomb left or a bomb already lies on its
    cell (``BombermanEnv._apply_action``). Exact from ``prev`` for the agent:
    it acts first (lowest ID), so no one places a bomb on its cell before it
    in the same step. Without buffs there is no ``bombs_left``; each
    character then has one bomb.
    """
    me_entry = prev["characters"][me]
    own = own_bomb_cells(prev, me)
    left = me_entry.get("bombs_left", 0 if own else 1)
    occupied = any((b["x"], b["y"]) == (me_entry["x"], me_entry["y"])
                   for b in prev.get("bombs", ()))
    return left <= 0 or occupied


class BombermanGymEnv(gym.Env):
    """The full game as a Gymnasium env: ``Discrete(6)`` actions, v1 grid observation."""

    metadata = {"render_modes": []}

    def __init__(self, opponents="random", config: EnvConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else EnvConfig()
        with_npcs, self.controllers = resolve_opponents(opponents)
        self.engine = BombermanEnv(with_npcs=with_npcs, random_crates=True, **GAME_RULES)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(0.0, 1.0, shape=OBS_SHAPE, dtype=np.float32)
        self.episode_seed: int | None = None
        self._state = self.engine.get_state()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.episode_seed = int(seed)
        else:
            self.episode_seed = int(self.np_random.integers(0, _SEED_BOUND))
        self._state = self.engine.reset(seed=self.episode_seed)
        for npc_id, controller in self.controllers.items():
            controller.reset(bot_seed(self.episode_seed, npc_id))
        info = {"outcome": None, "steps": 0, "events": [],
                "episode_seed": self.episode_seed}
        return encode_observation(self._state, AGENT_ID), info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r} for {self.action_space}")
        chosen = ACTIONS[int(action)]
        prev = self._state
        npc_actions = {
            c["id"]: self.controllers[c["id"]].act(prev, c["id"])
            for c in prev["characters"]
            if c["id"] in self.controllers and not c["dead"]
        }
        self._state = self.engine.step(chosen, npc_actions if self.engine.with_npcs else None)
        state = self._state

        reward, terms = step_reward(prev, state, chosen, AGENT_ID, self.config.reward)
        terminated = state["outcome"] is not None
        truncated = not terminated and state["steps"] >= self.config.max_steps
        info = {
            "outcome": state["outcome"],
            "steps": state["steps"],
            "events": [dict(event) for event in state["events"]],
            "episode_seed": self.episode_seed,
            "reward_terms": terms,
        }
        return encode_observation(state, AGENT_ID), reward, terminated, truncated, info

    @property
    def state(self) -> dict:
        """The engine state after the latest reset or step. Read it, don't mutate it."""
        return self._state
