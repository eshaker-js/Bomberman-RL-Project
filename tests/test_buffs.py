"""Buffs, several bombs per character, per-bomb range and burn (Task 7).

Headless; pygame is not required. Most tests build a fixed-map game with
``buffs=BUFF_RULES`` and set up the exact situation by hand. Fixed map row 1
is ``#P...B.B..#``: floor at x 1-4, 6, 8, 9 and crates at (5, 1) and (7, 1);
(2, 2), (4, 2), (6, 2) are pillars.
"""

import random
import unittest

from game.environment import (
    AGENT_ID,
    BLAST_RANGE,
    BUFF_RULES,
    GATE_POSITION,
    Action,
    Bomb,
    BombermanEnv,
    BuffRules,
    BuffType,
    Cell,
)

U, D, L, R, W, B = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT,
                    Action.WAIT, Action.BOMB)
FUSE, FLAMES = 4, 4
# Rules where every destroyed crate drops a range buff.
ALWAYS_RANGE = BuffRules((BuffType("range", BLAST_RANGE, 5, 1.0),))


def buff_env(buffs=BUFF_RULES, **kwargs) -> BombermanEnv:
    settings = {"with_npcs": True, "bomb_fuse": FUSE, "flame_ticks": FLAMES, **kwargs}
    return BombermanEnv(buffs=buffs, **settings)


def step(env, action=W, **npc):
    """One action; NPCs given as ``n1=Action.LEFT`` etc., the rest wait."""
    return env.step(action, {int(k[1:]): a for k, a in npc.items()})


def run(env, actions):
    state = env.get_state()
    for action in actions:
        state = env.step(action)
    return state


def events(state, kind):
    return [e for e in state["events"] if e["type"] == kind]


def clear_row_one(env):
    """Make every crate of row 1 floor, for long rays."""
    for x in range(env_width(env)):
        if env.grid[1][x] is Cell.BLOCK:
            env.grid[1][x] = Cell.EMPTY


def env_width(env):
    return len(env.grid[0])


def park_agent(env, cell=(1, 3)):
    """Move the agent out of the way of row 1 blasts."""
    env.player_x, env.player_y = cell


BUFF_KEYS = {"range", "bomb_count", "burn", "bombs_left"}


def without_buffs(state: dict) -> dict:
    """``state`` with every key that only exists with buffs removed."""
    result = dict(state)
    result.pop("buffs")
    result.pop("buff_rules")
    result["characters"] = [{k: v for k, v in c.items() if k not in BUFF_KEYS}
                            for c in state["characters"]]
    result["bombs"] = [{k: v for k, v in b.items() if k not in ("range", "burn")}
                       for b in state["bombs"]]
    result["events"] = [
        {k: v for k, v in e.items() if not (e["type"] == "bomb_triggered" and k in "xy")}
        for e in state["events"]
    ]
    return result


def random_actions(rng, state):
    return {c["id"]: rng.choice(tuple(Action)) for c in state["characters"] if not c["dead"]}


class TestBuffsOffIsUnchanged(unittest.TestCase):
    def test_same_crates_with_and_without_buffs(self):
        for seed in range(200):
            off = BombermanEnv(with_npcs=True, random_crates=True).reset(seed=seed)
            on = buff_env(random_crates=True).reset(seed=seed)
            self.assertEqual(off["grid"], on["grid"], seed)

    def test_reset_without_seed_continues_the_same_crates(self):
        off = BombermanEnv(with_npcs=True, random_crates=True)
        on = buff_env(random_crates=True)
        off.reset(seed=7)
        on.reset(seed=7)
        for _ in range(5):
            self.assertEqual(off.reset()["grid"], on.reset()["grid"])

    def test_same_game_until_the_first_buff_event(self):
        """Same seed and actions: buffs on and off agree until a buff drops."""
        compared = 0
        for seed in range(40):
            off = BombermanEnv(with_npcs=True, random_crates=True,
                               bomb_fuse=FUSE, flame_ticks=FLAMES)
            on = buff_env(random_crates=True)
            s_off, s_on = off.reset(seed=seed), on.reset(seed=seed)
            rng = random.Random(seed)
            while s_off["outcome"] is None:
                actions = random_actions(rng, s_off)
                agent = actions.pop(0, W)
                s_off, s_on = off.step(agent, actions), on.step(agent, actions)
                if any(e["type"].startswith("buff_") for e in s_on["events"]):
                    break
                self.assertEqual((s_off["buffs"], s_off["buff_rules"]), ([], None))
                self.assertEqual(without_buffs(s_on), without_buffs(s_off))
                compared += 1
        self.assertGreater(compared, 300)

    def test_state_keys_without_buffs(self):
        env = BombermanEnv(with_npcs=True)
        state = step(env, B)
        self.assertEqual(state["buffs"], [])
        self.assertIsNone(state["buff_rules"])
        self.assertEqual(set(state["characters"][0]), {"id", "x", "y", "dead"})
        self.assertEqual(set(state["bombs"][0]), {"owner", "x", "y", "countdown"})
        self.assertEqual(env.bombs, {0: Bomb(1, 1, 4)})


class TestStateAndEvents(unittest.TestCase):
    def test_state_keys_with_buffs(self):
        env = buff_env()
        state = step(env, B)
        self.assertEqual(state["buff_rules"], BUFF_RULES.as_dict())
        self.assertEqual(state["buff_rules"]["burn"], {"start": 0, "max": 2, "drop_chance": 0.10})
        self.assertEqual(state["characters"][0],
                         {"id": 0, "x": 1, "y": 1, "dead": False, "range": 2,
                          "bomb_count": 1, "burn": 0, "bombs_left": 0})
        self.assertEqual(state["characters"][1]["bombs_left"], 1)
        self.assertEqual(state["bombs"],
                         [{"owner": 0, "x": 1, "y": 1, "countdown": 4, "range": 2, "burn": 0}])
        self.assertEqual(state["bomb"], {"x": 1, "y": 1, "countdown": 4})
        self.assertEqual(state["rules"], {"bomb_fuse": 4, "blast_range": 2, "flame_ticks": 4})

    def test_state_is_a_copy(self):
        env = buff_env()
        env.floor_buffs[(2, 1)] = "range"
        state = env.get_state()
        state["buffs"].clear()
        state["characters"][0]["range"] = 9
        state["buff_rules"]["range"]["max"] = 99
        again = env.get_state()
        self.assertEqual(again["buffs"], [{"x": 2, "y": 1, "type": "range"}])
        self.assertEqual(again["characters"][0]["range"], 2)
        self.assertEqual(again["buff_rules"]["range"]["max"], 5)

    def test_bomb_triggered_has_the_bomb_cell_with_buffs_only(self):
        for buffs, expected in ((None, {"type", "owner", "triggered_by"}),
                                (BUFF_RULES, {"type", "owner", "triggered_by", "x", "y"})):
            env = BombermanEnv(with_npcs=True, buffs=buffs)
            env.bomb = Bomb(1, 1, 1)
            env.characters[1].x, env.characters[1].y = 3, 1
            step(env, D, n1=B)  # NPC 1's new bomb at (3, 1) is in the agent's blast
            triggered = events(env.get_state(), "bomb_triggered")
            self.assertEqual(len(triggered), 1)
            self.assertEqual(set(triggered[0]), expected)
            if buffs is not None:
                self.assertEqual((triggered[0]["x"], triggered[0]["y"]), (3, 1))


class TestBuffRules(unittest.TestCase):
    def test_game_buff_values(self):
        self.assertEqual(
            [(b.name, b.start, b.max, b.drop_chance) for b in BUFF_RULES.types],
            [("range", 2, 5, 0.15), ("bomb_count", 1, 3, 0.15), ("burn", 0, 2, 0.10)])

    def test_bad_rules_rejected(self):
        bad = [
            (),
            (BuffType("range", 2, 5, 0.5), BuffType("range", 2, 5, 0.1)),
            (BuffType("range", 3, 2, 0.1),),
            (BuffType("range", -1, 2, 0.1),),
            (BuffType("range", 2, 5, 1.5),),
            (BuffType("range", 2, 5, 0.6), BuffType("burn", 0, 2, 0.6)),
            (BuffType("", 0, 1, 0.1),),
            (BuffType("range", True, 5, 0.1),),
        ]
        for types in bad:
            with self.subTest(types=types):
                with self.assertRaises(ValueError):
                    BuffRules(types)
        with self.assertRaises(TypeError):
            BombermanEnv(buffs={"range": 1})

    def test_rules_are_immutable(self):
        with self.assertRaises(AttributeError):
            BUFF_RULES.types = ()  # type: ignore[misc]


class TestDrops(unittest.TestCase):
    def test_drop_chances_over_many_crates(self):
        env = buff_env()
        env.reset(seed=123)
        counts = {"range": 0, "bomb_count": 0, "burn": 0, None: 0}
        n = 20000
        for _ in range(n):
            env.floor_buffs = {}
            env._drop_buffs([(1, 3)])
            counts[env.floor_buffs.get((1, 3))] += 1
        for kind, chance in (("range", 0.15), ("bomb_count", 0.15), ("burn", 0.10),
                             (None, 0.60)):
            # About 4 standard deviations at n = 20000.
            self.assertAlmostEqual(counts[kind] / n, chance, delta=0.015, msg=kind)

    def test_drops_over_real_games_match_the_chances(self):
        """Every crate destroyed in many random games rolls once, at about the set rates."""
        drops = {"range": 0, "bomb_count": 0, "burn": 0}
        crates = 0
        for seed in range(2000):
            env = buff_env(random_crates=True)
            state = env.reset(seed=seed)
            rng = random.Random(seed)
            for _ in range(300):
                actions = random_actions(rng, state)
                state = env.step(actions.pop(0, W), actions)
                crates += sum(1 for e in events(state, "block_destroyed")
                              if e["cell"] != GATE_POSITION)
                for e in events(state, "buff_dropped"):
                    drops[e["buff"]] += 1
                if state["outcome"]:
                    break
        self.assertGreater(crates, 800)
        for kind, chance in (("range", 0.15), ("bomb_count", 0.15), ("burn", 0.10)):
            self.assertAlmostEqual(drops[kind] / crates, chance, delta=0.05, msg=kind)

    def test_crate_drop_appears_after_its_block_event(self):
        env = buff_env(ALWAYS_RANGE)
        park_agent(env)
        env.bomb = Bomb(3, 1, 1)
        state = step(env)
        kinds = [e["type"] for e in state["events"]]
        self.assertEqual(kinds, ["block_destroyed", "buff_dropped"])
        self.assertEqual(events(state, "buff_dropped"),
                         [{"type": "buff_dropped", "x": 5, "y": 1, "buff": "range"}])
        self.assertEqual(state["buffs"], [{"x": 5, "y": 1, "type": "range"}])

    def test_gate_crate_never_drops_and_draws_nothing(self):
        env = buff_env(ALWAYS_RANGE)
        before = env._buff_rng.getstate()
        env.bomb = Bomb(5, 4, 1)  # its down ray destroys the gate's crate only
        state = step(env)
        self.assertEqual([e["cell"] for e in events(state, "block_destroyed")], [GATE_POSITION])
        self.assertEqual(events(state, "buff_dropped"), [])
        self.assertEqual(state["buffs"], [])
        self.assertEqual(env._buff_rng.getstate(), before)

    def test_several_crates_in_one_tick_roll_in_block_order(self):
        """A bomb at (6, 1) destroys (7, 1) then (5, 1): two draws, in that order."""
        rolled = set()
        for seed in range(60):
            env = buff_env()
            env.reset(seed=seed)
            park_agent(env)
            env.bomb = Bomb(6, 1, 1)
            state = step(env)
            blocks = [e["cell"] for e in events(state, "block_destroyed")]
            self.assertEqual(blocks, [(7, 1), (5, 1)])
            stream = random.Random(f"buffs:{seed}")
            expected = []
            for cell in blocks:
                roll, edge = stream.random(), 0.0
                for buff in BUFF_RULES.types:
                    edge += buff.drop_chance
                    if roll < edge:
                        expected.append({"type": "buff_dropped", "x": cell[0], "y": cell[1],
                                         "buff": buff.name})
                        break
            self.assertEqual(events(state, "buff_dropped"), expected)
            rolled.add(len(expected))
            # The same seed gives the same drops again.
            again = buff_env()
            again.reset(seed=seed)
            park_agent(again)
            again.bomb = Bomb(6, 1, 1)
            self.assertEqual(step(again)["buffs"], state["buffs"])
        self.assertTrue({0, 1, 2} <= rolled)


class TestPickup(unittest.TestCase):
    def test_pickup_at_the_end_of_the_tick(self):
        env = buff_env()
        env.floor_buffs[(2, 1)] = "range"
        state = step(env, R)
        self.assertEqual(state["buffs"], [])
        self.assertEqual(state["characters"][0]["range"], 3)
        self.assertEqual(events(state, "buff_picked"),
                         [{"type": "buff_picked", "character": 0, "x": 2, "y": 1,
                           "buff": "range", "level": 3}])

    def test_each_buff_raises_its_own_level(self):
        env = buff_env()
        env.floor_buffs.update({(2, 1): "bomb_count", (3, 1): "burn", (4, 1): "range"})
        state = run(env, [R, R, R])
        c = state["characters"][0]
        self.assertEqual((c["bomb_count"], c["burn"], c["range"]), (2, 1, 3))
        self.assertEqual(c["bombs_left"], 2)

    def test_pickup_at_max_is_consumed_and_does_nothing(self):
        env = buff_env(start_buffs={0: {"range": 5}})
        env.floor_buffs[(2, 1)] = "range"
        state = step(env, R)
        self.assertEqual(state["buffs"], [])
        self.assertEqual(state["characters"][0]["range"], 5)
        self.assertEqual(events(state, "buff_picked")[0]["level"], 5)

    def test_dead_character_picks_up_nothing(self):
        env = buff_env()
        env.floor_buffs[(2, 1)] = "range"
        env.flames[(2, 1)] = (3, (1,))  # test-only: a lingering flame on the buff
        state = step(env, R)
        self.assertTrue(state["dead"])
        self.assertEqual(state["buffs"], [{"x": 2, "y": 1, "type": "range"}])
        self.assertEqual(events(state, "buff_picked"), [])
        self.assertEqual(state["characters"][0]["range"], 2)  # frozen on death

    def test_tie_is_decided_by_the_buff_stream(self):
        winners = {}
        for seed in range(100):
            env = buff_env()
            env.reset(seed=seed)
            env.characters[1].x, env.characters[1].y = 3, 1
            env.floor_buffs[(2, 1)] = "burn"
            state = step(env, R, n1=L)  # both arrive on (2, 1)
            picked = events(state, "buff_picked")
            self.assertEqual(len(picked), 1)
            # The first draw of the stream picks between IDs 0 and 1.
            expected = random.Random(f"buffs:{seed}").choice([0, 1])
            self.assertEqual(picked[0]["character"], expected)
            winners[expected] = winners.get(expected, 0) + 1
        self.assertGreater(winners.get(0, 0), 20)
        self.assertGreater(winners.get(1, 0), 20)

    def test_tie_is_reproducible(self):
        results = []
        for _ in range(2):
            env = buff_env()
            env.reset(seed=11)
            for npc in env.characters[1:]:
                npc.x, npc.y = 3, 1
            env.floor_buffs[(2, 1)] = "range"
            results.append(step(env, R, n1=L, n2=L, n3=L)["events"])
        self.assertEqual(results[0], results[1])

    def test_lone_pickup_draws_nothing(self):
        env = buff_env()
        before = env._buff_rng.getstate()
        env.floor_buffs[(2, 1)] = "range"
        step(env, R)
        self.assertEqual(env._buff_rng.getstate(), before)


class TestBuffsBlockNothing(unittest.TestCase):
    def test_walk_onto_a_buff(self):
        env = buff_env()
        env.floor_buffs[(2, 1)] = "range"
        self.assertEqual(step(env, R)["player"], (2, 1))

    def test_rays_pass_over_buffs(self):
        env = buff_env()
        park_agent(env)
        env.floor_buffs[(2, 1)] = "range"
        env.bomb = Bomb(1, 1, 1)
        state = step(env)
        self.assertIn((3, 1), state["explosion"])

    def test_bomb_placed_on_a_buff(self):
        env = buff_env()
        env.floor_buffs[(1, 1)] = "range"  # test-only: a buff under the agent
        state = step(env, B)
        self.assertEqual(len(state["bombs"]), 1)
        self.assertEqual(state["bombs"][0]["range"], 2)  # placed before the pickup
        self.assertEqual(state["characters"][0]["range"], 3)


class TestBuffDestruction(unittest.TestCase):
    def test_fresh_blast_destroys_floor_buffs(self):
        env = buff_env()
        park_agent(env)
        env.floor_buffs[(3, 1)] = "burn"
        env.bomb = Bomb(1, 1, 1)
        state = step(env)
        self.assertEqual(state["buffs"], [])
        self.assertEqual(events(state, "buff_destroyed"),
                         [{"type": "buff_destroyed", "x": 3, "y": 1, "buff": "burn",
                           "destroyed_by": (0,)}])

    def test_drop_of_the_same_tick_survives_and_the_next_blast_destroys_it(self):
        env = buff_env(ALWAYS_RANGE)
        park_agent(env)
        env.bomb = Bomb(3, 1, 1)
        state = step(env)
        self.assertEqual(state["buffs"], [{"x": 5, "y": 1, "type": "range"}])
        self.assertEqual(events(state, "buff_destroyed"), [])
        # NPC 1 blasts (5, 1) on a later tick: a fresh blast destroys it.
        run(env, [W] * FLAMES)
        env.characters[1].x, env.characters[1].y = 9, 3
        env._bombs[(6, 1)] = (1, Bomb(6, 1, 1))
        state = step(env)
        self.assertEqual(state["buffs"], [{"x": 7, "y": 1, "type": "range"}])
        self.assertEqual(events(state, "buff_destroyed"),
                         [{"type": "buff_destroyed", "x": 5, "y": 1, "buff": "range",
                           "destroyed_by": (1,)}])

    def test_lingering_flames_do_not_destroy_buffs(self):
        env = buff_env()
        env.floor_buffs[(3, 1)] = "range"
        env.flames[(3, 1)] = (3, (2,))  # test-only
        state = run(env, [W, W, W])
        self.assertEqual(state["buffs"], [{"x": 3, "y": 1, "type": "range"}])
        self.assertEqual(events(state, "buff_destroyed"), [])


class TestRange(unittest.TestCase):
    def test_bomb_keeps_its_range_from_placement(self):
        env = buff_env(start_buffs={0: {"range": 3, "bomb_count": 2}})
        step(env, B)
        env.characters[0].levels["range"] = 5  # as if picked up now
        state = step(env, D)
        self.assertEqual(state["bombs"][0]["range"], 3)
        self.assertEqual(env.bombs[0], Bomb(1, 1, 3, range=3, burn=0))

    def test_range_five_stops_at_range_crates_and_walls(self):
        env = buff_env()
        clear_row_one(env)
        park_agent(env, (1, 7))
        env.bomb = Bomb(1, 1, 1, range=5)
        cells = set(step(env)["explosion"])
        self.assertEqual({c for c in cells if c[1] == 1}, {(x, 1) for x in range(1, 7)})
        # Down: (1, 2), (1, 3), (1, 4), then the crate at (1, 5) burns and stops it.
        self.assertEqual({c for c in cells if c[0] == 1}, {(1, y) for y in range(1, 6)})

    def test_wall_and_crate_stop_long_rays(self):
        env = buff_env()
        clear_row_one(env)
        env.grid[1][3] = Cell.BLOCK
        park_agent(env)
        env.bomb = Bomb(6, 1, 1, range=5)
        cells = {c for c in step(env)["explosion"] if c[1] == 1}
        # Right: 7, 8, 9, then the wall at 10. Left: 5, 4, then the crate at 3.
        self.assertEqual(cells, {(x, 1) for x in range(3, 10)})


class TestExtraBombs(unittest.TestCase):
    def test_two_bombs_then_refused(self):
        # A long fuse, so no bomb explodes during the test.
        env = buff_env(start_buffs={0: {"bomb_count": 2}}, bomb_fuse=9)
        state = run(env, [B, R, B, R, B])
        self.assertEqual([(b["x"], b["y"]) for b in state["bombs"]], [(1, 1), (2, 1)])
        self.assertEqual(state["characters"][0]["bombs_left"], 0)
        self.assertEqual(env.bombs, {0: env.active_bombs()[0][1]})  # oldest
        self.assertEqual(state["bomb"]["x"], 1)

    def test_three_bombs(self):
        env = buff_env(start_buffs={0: {"bomb_count": 3}}, bomb_fuse=9)
        state = run(env, [B, D, B, D, B])
        self.assertEqual([(b["x"], b["y"]) for b in state["bombs"]],
                         [(1, 1), (1, 2), (1, 3)])
        self.assertEqual([b["countdown"] for b in state["bombs"]], [5, 7, 9])
        self.assertEqual(state["characters"][0]["bombs_left"], 0)
        state = run(env, [D, B])  # a fourth bomb is refused
        self.assertEqual(len(state["bombs"]), 3)

    def test_still_one_bomb_per_cell(self):
        env = buff_env(start_buffs={0: {"bomb_count": 3}})
        state = run(env, [B, B])
        self.assertEqual(len(state["bombs"]), 1)
        self.assertEqual(state["characters"][0]["bombs_left"], 2)

    def test_chain_between_bombs_of_one_owner(self):
        env = buff_env(start_buffs={0: {"bomb_count": 2}})
        run(env, [B, R, B, R])  # bombs at (1, 1) step 1 and (2, 1) step 3
        state = step(env, R)  # step 5: the first bomb explodes and sets off the second
        self.assertEqual(events(state, "bomb_triggered"),
                         [{"type": "bomb_triggered", "owner": 0, "triggered_by": (0,),
                           "x": 2, "y": 1}])
        self.assertEqual(state["bombs"], [])
        self.assertIn((4, 1), state["explosion"])  # the second bomb's ray

    def test_explosion_order_is_owner_then_placement(self):
        env = buff_env(start_buffs={0: {"bomb_count": 2}, 1: {"bomb_count": 2}})
        park_agent(env, (1, 3))
        env._bombs = {}
        env._bombs[(9, 3)] = (1, Bomb(9, 3, 1))
        env._bombs[(3, 1)] = (0, Bomb(3, 1, 1))
        env._bombs[(9, 1)] = (1, Bomb(9, 1, 1))
        env._bombs[(1, 1)] = (0, Bomb(1, 1, 1))
        self.assertEqual(env._bomb_order(list(env._bombs)),
                         [(3, 1), (1, 1), (9, 3), (9, 1)])


class TestBurn(unittest.TestCase):
    def flames_after(self, bomb, *, clear=True):
        env = buff_env()
        if clear:
            clear_row_one(env)
        park_agent(env, (1, 7))
        env.bomb = bomb
        state = step(env)
        return {(f["x"], f["y"]): f["ticks_left"] for f in state["flames"]}

    def test_burn_one_and_two(self):
        # (4, 1): left 3, 2, 1; right 5, 6, 7 with range 3; up and down are walls.
        one = self.flames_after(Bomb(4, 1, 1, range=3, burn=1))
        self.assertEqual(one, {(4, 1): 5, (3, 1): 5, (5, 1): 5,
                               (2, 1): 4, (6, 1): 4, (1, 1): 4, (7, 1): 4})
        two = self.flames_after(Bomb(4, 1, 1, range=3, burn=2))
        self.assertEqual(two, {(4, 1): 6, (3, 1): 6, (5, 1): 6, (2, 1): 6, (6, 1): 6,
                               (1, 1): 4, (7, 1): 4})

    def test_walls_cut_burn(self):
        flames = self.flames_after(Bomb(3, 1, 1, range=2, burn=2))
        # Down from (3, 1): (3, 2), (3, 3); up is the wall.
        self.assertEqual(flames[(3, 2)], 6)
        self.assertEqual(flames[(3, 3)], 6)
        self.assertNotIn((3, 0), flames)

    def test_burn_never_exceeds_the_blast(self):
        flames = self.flames_after(Bomb(4, 1, 1, range=1, burn=2))
        self.assertEqual(flames, {(4, 1): 6, (3, 1): 6, (5, 1): 6})

    def test_crate_stops_burn_like_the_blast(self):
        flames = self.flames_after(Bomb(3, 1, 1, range=3, burn=2), clear=False)
        # Right: 4, then the crate at 5 (distance 2 <= burn: 6 ticks) stops it.
        self.assertEqual(flames[(5, 1)], 6)
        self.assertNotIn((6, 1), flames)

    def test_flames_count_down_from_the_burn_time(self):
        env = buff_env()
        park_agent(env, (1, 7))
        env.bomb = Bomb(3, 1, 1, range=2, burn=2)
        run(env, [W])
        lefts = []
        for _ in range(6):
            lefts.append(env.flames.get((3, 1), (0,))[0])
            step(env)
        self.assertEqual(lefts, [6, 5, 4, 3, 2, 1])
        self.assertNotIn((3, 1), env.flames)

    def test_overlapping_blasts_keep_the_longest_timer_and_both_owners(self):
        env = buff_env()
        clear_row_one(env)
        park_agent(env, (1, 7))
        env._bombs[(2, 1)] = (0, Bomb(2, 1, 1, range=2, burn=0))
        env._bombs[(4, 1)] = (1, Bomb(4, 1, 1, range=2, burn=1))
        state = step(env)
        flames = {(f["x"], f["y"]): (f["ticks_left"], f["owners"]) for f in state["flames"]}
        self.assertEqual(flames[(3, 1)], (5, (0, 1)))  # distance 1 from the burn bomb
        self.assertEqual(flames[(2, 1)], (4, (0, 1)))  # distance 2 from it: normal

    def test_normal_blast_on_a_burn_cell_keeps_the_longer_timer(self):
        env = buff_env()
        clear_row_one(env)
        park_agent(env, (1, 7))
        env._bombs[(4, 1)] = (1, Bomb(4, 1, 1, range=1, burn=2))
        step(env)
        self.assertEqual(env.flames[(3, 1)], (6, (1,)))
        env._bombs[(2, 1)] = (0, Bomb(2, 1, 1))  # a normal bomb next tick
        state = step(env)
        flames = {(f["x"], f["y"]): (f["ticks_left"], f["owners"]) for f in state["flames"]}
        self.assertEqual(flames[(3, 1)], (5, (0,)))  # longer timer kept, owners taken over
        self.assertEqual(flames[(1, 1)], (4, (0,)))

    def test_normal_blast_after_the_burn_time_left_runs_short(self):
        env = buff_env()
        clear_row_one(env)
        park_agent(env, (1, 7))
        env._bombs[(4, 1)] = (1, Bomb(4, 1, 1, range=1, burn=1))
        run(env, [W, W, W])  # (3, 1): 5, 4, 3 ticks left
        env._bombs[(2, 1)] = (0, Bomb(2, 1, 1))
        state = step(env)
        flames = {(f["x"], f["y"]): f["ticks_left"] for f in state["flames"]}
        self.assertEqual(flames[(3, 1)], FLAMES)  # the fresh 4 beats the 2 left


class TestChainsKeepStoredValues(unittest.TestCase):
    def test_bomb_set_off_uses_its_own_range_and_burn(self):
        env = buff_env(start_buffs={0: {"bomb_count": 2}})
        step(env, B)  # (1, 1): range 2, burn 0
        env.characters[0].levels.update(range=3, burn=1)  # as if picked up
        run(env, [R, B])  # (2, 1): range 3, burn 1, placed on step 3
        run(env, [D])  # step 4: (2, 1) is a pillar, the agent stays
        state = step(env)  # step 5: the first bomb sets off the second
        flames = {(f["x"], f["y"]): f["ticks_left"] for f in state["flames"]}
        # The second bomb reaches the crate at (5, 1) (range 3) and burns its
        # own cell and distance 1 for 5 ticks.
        self.assertIn({"type": "block_destroyed", "cell": (5, 1), "destroyed_by": (0,)},
                      state["events"])
        self.assertEqual(flames[(2, 1)], 5)
        self.assertEqual(flames[(3, 1)], 5)
        self.assertEqual(flames[(1, 1)], 5)  # also distance 1 from the second bomb
        self.assertEqual(flames[(4, 1)], 4)
        self.assertEqual(flames[(1, 3)], 4)  # the first bomb's down ray, range 2
        self.assertNotIn((1, 4), flames)

    def test_npc_bomb_set_off_by_the_agent_keeps_npc_values(self):
        env = buff_env(start_buffs={1: {"range": 4}})
        env.characters[1].x, env.characters[1].y = 3, 1
        park_agent(env, (1, 7))
        env.bomb = Bomb(1, 1, 2)
        step(env, n1=B)  # NPC 1 bombs (3, 1) with range 4
        env.characters[1].x, env.characters[1].y = 9, 9
        state = step(env)  # the agent's bomb (range 2) sets off NPC 1's
        cells = set(state["explosion"])
        # NPC 1's ray down reaches (3, 4) and the crate (3, 5): range 4, not 2.
        self.assertTrue({(3, 2), (3, 3), (3, 4), (3, 5)} <= cells)
        self.assertIn({"type": "block_destroyed", "cell": (3, 5), "destroyed_by": (1,)},
                      state["events"])
        self.assertIn({"type": "block_destroyed", "cell": (5, 1), "destroyed_by": (1,)},
                      state["events"])


class TestOutcomeUnchangedWithBuffs(unittest.TestCase):
    def test_lingering_burn_flame_kills_and_credits_the_owner(self):
        env = buff_env()
        park_agent(env, (1, 7))
        env.characters[1].x, env.characters[1].y = 4, 1
        env._bombs[(2, 1)] = (2, Bomb(2, 1, 1, range=1, burn=1))
        run(env, [W] * 4)  # (3, 1) burns 5 ticks: still burning on the 5th
        state = step(env, n1=L)
        self.assertEqual(events(state, "death"),
                         [{"type": "death", "character": 1, "killed_by": (2,)}])

    def test_own_burn_bomb_is_a_loss(self):
        env = buff_env(start_buffs={0: {"burn": 2}})
        state = run(env, [B, W, W, W, W])
        self.assertEqual(state["outcome"], "loss")
        self.assertEqual(events(state, "death"),
                         [{"type": "death", "character": 0, "killed_by": (0,)}])


class TestStartBuffs(unittest.TestCase):
    def test_start_levels(self):
        env = buff_env(start_buffs={0: {"range": 3}, 2: {"burn": 2, "bomb_count": 3}})
        state = env.get_state()
        self.assertEqual(state["characters"][0]["range"], 3)
        self.assertEqual(state["characters"][1]["range"], 2)
        self.assertEqual((state["characters"][2]["burn"], state["characters"][2]["bomb_count"]),
                         (2, 3))

    def test_reset_override_and_constructor_default(self):
        env = buff_env(start_buffs={0: {"range": 4}})
        self.assertEqual(env.reset(start_buffs={0: {"burn": 1}})["characters"][0]["range"], 2)
        self.assertEqual(env.get_state()["characters"][0]["burn"], 1)
        self.assertEqual(env.reset()["characters"][0]["range"], 4)

    def test_levels_do_not_leak_between_games(self):
        env = buff_env()
        env.floor_buffs[(2, 1)] = "range"
        step(env, R)
        self.assertEqual(env.reset()["characters"][0]["range"], 2)

    def test_bad_start_buffs(self):
        cases = [
            (BombermanEnv, {0: {"range": 3}}, ValueError),  # buffs off
            (buff_env, {5: {"range": 3}}, ValueError),
            (buff_env, {0: {"speed": 1}}, ValueError),
            (buff_env, {0: {"range": 6}}, ValueError),
            (buff_env, {0: {"range": 1}}, ValueError),
            (buff_env, {0: {"range": True}}, ValueError),
            (buff_env, [0], TypeError),
            (buff_env, {0: 3}, TypeError),
        ]
        for factory, start, error in cases:
            with self.subTest(start=start):
                with self.assertRaises(error):
                    factory(start_buffs=start)
        single = BombermanEnv(buffs=BUFF_RULES)
        with self.assertRaises(ValueError):
            single.reset(start_buffs={1: {"range": 3}})  # no NPC 1 in single player

    def test_bad_reset_changes_nothing(self):
        env = buff_env()
        step(env, R)
        before = env.get_state()
        with self.assertRaises(ValueError):
            env.reset(seed=3, start_buffs={0: {"range": 9}})
        self.assertEqual(env.get_state(), before)


class TestReproducible(unittest.TestCase):
    def test_same_seed_and_actions_same_game_with_buffs(self):
        seen_buffs = 0
        for seed in range(25):
            games = []
            for _ in range(2):
                env = buff_env(random_crates=True)
                state = env.reset(seed=seed)
                rng = random.Random(seed)
                history = [state]
                for _ in range(200):
                    actions = random_actions(rng, state)
                    state = env.step(actions.pop(0, W), actions)
                    history.append(state)
                    if state["outcome"]:
                        break
                games.append(history)
            self.assertEqual(games[0], games[1])
            seen_buffs += sum(len(s["buffs"]) for s in games[0])
        self.assertGreater(seen_buffs, 0)

    def test_bomb_setter_still_works_with_buffs(self):
        env = buff_env(start_buffs={0: {"bomb_count": 2}})
        run(env, [B, R, B])
        env.bomb = Bomb(4, 1, 2)
        self.assertEqual(env.active_bombs(), [(AGENT_ID, Bomb(4, 1, 2))])
        env.bomb = None
        self.assertEqual(env.bombs, {})


if __name__ == "__main__":
    unittest.main()
