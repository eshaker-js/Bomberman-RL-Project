"""One game: the environment, the bots, the seed and the last two states. Pure: no pygame.

The bots are called exactly as ``agents/match.py`` calls them: each one is
reset with ``bot_seed(seed, id)``, and every tick each living bot gets the
same current state and its own ID. The human always controls character 0.
"""

from agents import RandomBot, RuleBot
from agents.match import bot_seed
from game.environment import AGENT_ID, NPC_IDS, Action, BombermanEnv

MODE_FREE = "free"
MODE_BOTS = "bots"

# Bot class per NPC seat 1, 2, 3 for each opponent choice.
OPPONENTS = {
    "random": (RandomBot, RandomBot, RandomBot),
    "rule": (RuleBot, RuleBot, RuleBot),
    "mixed": (RuleBot, RuleBot, RandomBot),
}
BOT_NAMES = {RandomBot: "RANDOM", RuleBot: "RULE"}


class GameSession:
    """Owns one ``BombermanEnv`` and its bots for one mode and seed.

    ``prev`` and ``cur`` are the states before and after the latest tick;
    the renderer only ever interpolates between these two real states.
    """

    def __init__(self, mode: str, seed: int, opponents: str = "rule") -> None:
        if mode == MODE_FREE:
            self.env = BombermanEnv(random_crates=True)
            self.bots = {}
        elif mode == MODE_BOTS:
            if opponents not in OPPONENTS:
                raise ValueError(f"unknown opponents {opponents!r}")
            self.env = BombermanEnv(with_npcs=True, random_crates=True)
            self.bots = {
                npc_id: bot_class()
                for npc_id, bot_class in zip(NPC_IDS, OPPONENTS[opponents])
            }
        else:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.opponents = opponents if mode == MODE_BOTS else None
        self.seed = seed
        self.restart()

    def restart(self) -> None:
        """Start the same game again: same seed, same crates, same bot seeds."""
        state = self.env.reset(seed=self.seed)
        for character_id, bot in self.bots.items():
            bot.reset(bot_seed(self.seed, character_id))
        self.prev = state
        self.cur = state

    @property
    def over(self) -> bool:
        return self.cur["outcome"] is not None

    def advance(self, action: Action) -> dict:
        """Run one tick with the human's ``action`` and return the new state."""
        npc_actions = {
            c["id"]: self.bots[c["id"]].act(self.cur, c["id"])
            for c in self.cur["characters"]
            if c["id"] != AGENT_ID and not c["dead"]
        }
        self.prev = self.cur
        self.cur = self.env.step(action, npc_actions if self.bots else None)
        return self.cur

    def label(self, character_id: int) -> str:
        """Short name for a character's card."""
        if character_id == AGENT_ID:
            return "YOU"
        return f"BOT {character_id} {BOT_NAMES[type(self.bots[character_id])]}"
