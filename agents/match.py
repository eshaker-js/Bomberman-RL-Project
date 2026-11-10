"""Play one seeded game between bots and collect what happened.

This is evaluation tooling, not a game rule. ``max_steps`` only stops a game
that would otherwise never end (for example two careful bots dodging each
other forever) and reports it as ``"timeout"``. The real episode limit for
training belongs to the Gym wrapper (track C).

``rules`` picks the timing: ``None`` (the default) keeps the engine defaults
(fuse 4, flames 1), and ``rules=GAME_RULES`` plays the real game's timing.
"""

from dataclasses import dataclass, field

from game.environment import AGENT_ID, NPC_IDS, BombermanEnv

DEFAULT_MAX_STEPS = 800


@dataclass
class GameResult:
    seed: int
    outcome: str  # "win", "loss", "draw" or "timeout", from the agent's side
    steps: int
    death_step: dict = field(default_factory=dict)  # character -> action it died on
    kills: dict = field(default_factory=dict)  # owner -> other characters it killed
    suicides: dict = field(default_factory=dict)  # character -> 1 if hit by own blast
    crates: dict = field(default_factory=dict)  # owner -> crates its blasts destroyed


def bot_seed(game_seed: int, character_id: int) -> int:
    """A different but reproducible seed for each bot in each game."""
    return game_seed * 1000 + character_id


def play_game(
    bots: dict,
    seed: int,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    random_crates: bool = True,
    rules=None,
) -> GameResult:
    """Play one game. ``bots`` maps every character ID (0-3) to a bot.

    ``rules`` is passed to ``BombermanEnv`` as keyword arguments, for example
    ``GAME_RULES``. ``None`` uses the engine defaults, exactly as before.
    """
    missing = {AGENT_ID, *NPC_IDS} - set(bots)
    if missing:
        raise ValueError(f"no bot for characters {sorted(missing)}")

    env = BombermanEnv(with_npcs=True, random_crates=random_crates, **(rules or {}))
    state = env.reset(seed=seed)
    for character_id, bot in bots.items():
        bot.reset(bot_seed(seed, character_id))

    result = GameResult(seed=seed, outcome="timeout", steps=0)
    for _ in range(max_steps):
        actions = {
            c["id"]: bots[c["id"]].act(state, c["id"])
            for c in state["characters"]
            if not c["dead"]
        }
        agent_action = actions.pop(AGENT_ID)
        state = env.step(agent_action, actions)
        for event in state["events"]:
            _record(result, event, state["steps"])
        if state["outcome"] is not None:
            result.outcome = state["outcome"]
            break
    result.steps = state["steps"]
    return result


def _record(result: GameResult, event: dict, step: int) -> None:
    if event["type"] == "block_destroyed":
        for owner in event["destroyed_by"]:
            result.crates[owner] = result.crates.get(owner, 0) + 1
    elif event["type"] == "death":
        victim = event["character"]
        result.death_step[victim] = step
        for owner in event["killed_by"]:
            if owner == victim:
                result.suicides[victim] = 1
            else:
                result.kills[owner] = result.kills.get(owner, 0) + 1
