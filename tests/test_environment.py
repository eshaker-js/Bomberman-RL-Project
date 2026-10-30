"""Headless checks for the Bomberman rules. Pygame is not required."""

import ast
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.environment import (  # noqa: E402
    BLAST_RANGE,
    BOMB_FUSE,
    MAP_HEIGHT,
    MAP_WIDTH,
    PLAYER_START,
    Action,
    BombermanEnv,
    Cell,
)


def run(actions, env=None) -> BombermanEnv:
    if env is None:
        env = BombermanEnv()
    for action in actions:
        env.step(action)
    return env


class TestMap(unittest.TestCase):
    def test_configuration_and_opening_escape_geometry(self):
        self.assertEqual((MAP_WIDTH, MAP_HEIGHT), (11, 11))
        self.assertEqual(BOMB_FUSE, 4)
        self.assertEqual(BLAST_RANGE, 2)

        env = BombermanEnv()
        self.assertEqual(len(env.grid), MAP_HEIGHT)
        self.assertTrue(all(len(row) == MAP_WIDTH for row in env.grid))
        for x in range(MAP_WIDTH):
            self.assertIs(env.grid[0][x], Cell.WALL)
            self.assertIs(env.grid[MAP_HEIGHT - 1][x], Cell.WALL)
        for y in range(MAP_HEIGHT):
            self.assertIs(env.grid[y][0], Cell.WALL)
            self.assertIs(env.grid[y][MAP_WIDTH - 1], Cell.WALL)

        self.assertEqual(env.player, PLAYER_START)
        self.assertEqual(PLAYER_START, (1, 1))
        # Floor far enough from a bomb on the start cell to survive a range-2 blast.
        for position in ((1, 1), (2, 1), (3, 1), (4, 1)):
            x, y = position
            self.assertIs(env.grid[y][x], Cell.EMPTY)
        self.assertIs(env.grid[2][2], Cell.WALL)
        self.assertIs(env.grid[1][5], Cell.BLOCK)


class TestMovement(unittest.TestCase):
    def test_walls_and_blocks_block_movement_but_spend_the_step(self):
        env = BombermanEnv()
        env.step(Action.LEFT)
        env.step(Action.UP)
        self.assertEqual(env.player, (1, 1))
        self.assertEqual(env.steps, 2)

        run([Action.RIGHT, Action.RIGHT, Action.RIGHT], env)
        self.assertEqual(env.player, (4, 1))
        env.step(Action.RIGHT)
        self.assertEqual(env.player, (4, 1))
        self.assertIs(env.grid[1][5], Cell.BLOCK)
        self.assertEqual(env.steps, 6)

    def test_blocked_move_still_ticks_an_active_bomb(self):
        env = BombermanEnv()
        env.step(Action.BOMB)
        env.step(Action.UP)
        self.assertEqual(env.player, (1, 1))
        self.assertEqual(env.steps, 2)
        self.assertEqual(env.bomb.countdown, 3)

    def test_rejects_non_action_values(self):
        env = BombermanEnv()
        with self.assertRaises(TypeError):
            env.step("up")
        with self.assertRaises(TypeError):
            env.step(0)
        self.assertEqual(env.steps, 0)


class TestBombs(unittest.TestCase):
    def test_fuse_waits_four_actions_then_explodes(self):
        env = BombermanEnv()
        state = env.step(Action.BOMB)
        self.assertEqual(state["bomb"]["countdown"], 4)
        self.assertEqual(env.explosion, [])
        self.assertFalse(env.dead)

        for expected in (3, 2, 1):
            env.step(Action.WAIT)
            self.assertEqual(env.bomb.countdown, expected)
            self.assertFalse(env.dead)
            self.assertEqual(env.explosion, [])

        state = env.step(Action.WAIT)
        self.assertIsNone(env.bomb)
        self.assertIsNone(state["bomb"])
        self.assertTrue(state["dead"])
        self.assertIn((1, 1), state["explosion"])
        self.assertEqual(env.steps, 5)

    def test_second_place_does_not_replace_the_active_bomb(self):
        env = BombermanEnv()
        env.step(Action.BOMB)
        state = env.step(Action.BOMB)
        self.assertEqual(state["bomb"], {"x": 1, "y": 1, "countdown": 3})
        self.assertEqual(env.steps, 2)

    def test_player_can_leave_a_bomb_but_not_reenter_until_it_explodes(self):
        env = BombermanEnv()
        env.step(Action.BOMB)
        env.step(Action.RIGHT)
        self.assertEqual(env.player, (2, 1))
        self.assertEqual((env.bomb.x, env.bomb.y), (1, 1))

        env.step(Action.LEFT)
        self.assertEqual(env.player, (2, 1))
        self.assertEqual(env.bomb.countdown, 2)

        # Two more actions finish the fuse. (4, 1) is outside a range-2 blast.
        env.step(Action.RIGHT)
        env.step(Action.RIGHT)
        self.assertEqual(env.player, (4, 1))
        self.assertFalse(env.dead)
        self.assertIsNone(env.bomb)

        env.step(Action.LEFT)
        self.assertEqual(env.player, (3, 1))
        self.assertEqual(env.explosion, [])
        self.assertFalse(env.dead)
        env.step(Action.LEFT)
        env.step(Action.LEFT)
        self.assertEqual(env.player, (1, 1))
        self.assertFalse(env.dead)


class TestExplosions(unittest.TestCase):
    def test_walls_stop_the_blast_and_remain(self):
        env = run([Action.BOMB, Action.WAIT, Action.WAIT, Action.WAIT, Action.WAIT])
        self.assertNotIn((1, 0), env.explosion)
        self.assertNotIn((0, 1), env.explosion)
        self.assertIs(env.grid[0][1], Cell.WALL)
        self.assertIs(env.grid[1][0], Cell.WALL)
        self.assertIn((3, 1), env.explosion)
        self.assertNotIn((4, 1), env.explosion)

        env.reset()
        # Pillar (2, 2) is immediately right of the bomb. (3, 2) is behind it.
        run(
            [Action.DOWN, Action.BOMB, Action.UP, Action.RIGHT, Action.WAIT, Action.WAIT],
            env,
        )
        self.assertEqual(env.player, (2, 1))
        self.assertFalse(env.dead)
        self.assertNotIn((2, 2), env.explosion)
        self.assertNotIn((3, 2), env.explosion)
        self.assertIs(env.grid[2][2], Cell.WALL)
        self.assertIn((1, 2), env.explosion)
        self.assertIn((1, 4), env.explosion)
        self.assertNotIn((1, 5), env.explosion)
        self.assertIs(env.grid[5][1], Cell.BLOCK)

    def test_breakable_block_is_destroyed_and_stops_the_blast(self):
        env = run([Action.RIGHT, Action.RIGHT, Action.RIGHT])
        self.assertEqual(env.player, (4, 1))
        self.assertIs(env.grid[1][5], Cell.BLOCK)
        behind = env.grid[1][6]

        env.step(Action.BOMB)
        run([Action.LEFT, Action.LEFT, Action.LEFT, Action.WAIT], env)

        self.assertEqual(env.player, (1, 1))
        self.assertFalse(env.dead)
        self.assertIs(env.grid[1][5], Cell.EMPTY)
        self.assertIn((5, 1), env.explosion)
        self.assertIn((4, 1), env.explosion)
        self.assertNotIn((6, 1), env.explosion)
        self.assertIs(env.grid[1][6], behind)
        self.assertNotIn((4, 2), env.explosion)
        self.assertIs(env.grid[2][4], Cell.WALL)

    def test_opening_bomb_can_be_escaped(self):
        env = run([Action.BOMB, Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.WAIT])
        self.assertEqual(env.player, (4, 1))
        self.assertFalse(env.dead)
        self.assertIn((1, 1), env.explosion)
        self.assertIn((3, 1), env.explosion)
        self.assertIn((1, 3), env.explosion)
        self.assertNotIn((4, 1), env.explosion)
        self.assertNotIn((1, 4), env.explosion)

    def test_walking_out_on_the_exploding_action_survives(self):
        env = run([Action.BOMB, Action.RIGHT, Action.RIGHT, Action.WAIT])
        self.assertEqual(env.player, (3, 1))
        self.assertEqual(env.bomb.countdown, 1)
        env.step(Action.RIGHT)
        self.assertEqual(env.player, (4, 1))
        self.assertFalse(env.dead)
        self.assertIn((3, 1), env.explosion)
        self.assertNotIn((4, 1), env.explosion)

    def test_walking_into_the_blast_on_the_exploding_action_kills(self):
        env = run([Action.BOMB, Action.RIGHT, Action.RIGHT, Action.RIGHT])
        self.assertEqual(env.player, (4, 1))
        self.assertEqual(env.bomb.countdown, 1)
        self.assertFalse(env.dead)
        env.step(Action.LEFT)
        self.assertEqual(env.player, (3, 1))
        self.assertTrue(env.dead)
        self.assertIn((3, 1), env.explosion)

    def test_flames_do_not_damage_on_the_following_step(self):
        env = run([Action.BOMB, Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.WAIT])
        self.assertIn((3, 1), env.explosion)
        env.step(Action.LEFT)
        self.assertEqual(env.player, (3, 1))
        self.assertFalse(env.dead)
        self.assertEqual(env.explosion, [])

    def test_death_is_terminal_until_reset(self):
        env = run([Action.BOMB, Action.WAIT, Action.WAIT, Action.WAIT, Action.WAIT])
        self.assertTrue(env.dead)
        steps = env.steps
        explosion = list(env.explosion)
        env.step(Action.RIGHT)
        env.step(Action.WAIT)
        self.assertTrue(env.dead)
        self.assertEqual(env.player, (1, 1))
        self.assertEqual(env.steps, steps)
        self.assertEqual(env.explosion, explosion)


class TestReset(unittest.TestCase):
    def test_reset_restores_the_whole_initial_state(self):
        env = run(
            [
                Action.RIGHT,
                Action.RIGHT,
                Action.RIGHT,
                Action.BOMB,
                Action.LEFT,
                Action.LEFT,
                Action.LEFT,
                Action.WAIT,
                Action.BOMB,
                Action.WAIT,
                Action.WAIT,
                Action.WAIT,
                Action.WAIT,
            ]
        )
        self.assertTrue(env.dead)
        self.assertIs(env.grid[1][5], Cell.EMPTY)
        self.assertGreater(env.steps, 0)

        state = env.reset()
        fresh = BombermanEnv()
        self.assertEqual(env.grid, fresh.grid)
        self.assertEqual(env.player, fresh.player)
        self.assertEqual(state["player"], PLAYER_START)
        self.assertIs(env.grid[1][5], Cell.BLOCK)
        self.assertFalse(env.dead)
        self.assertIsNone(env.bomb)
        self.assertEqual(env.explosion, [])
        self.assertEqual(env.steps, 0)

        snapshot = env.get_state()
        snapshot["grid"][1][5] = Cell.EMPTY
        self.assertIs(env.grid[1][5], Cell.BLOCK)


class TestHeadlessUse(unittest.TestCase):
    def test_environment_source_does_not_import_pygame(self):
        source = (ROOT / "game" / "environment.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module.split(".")[0])
        self.assertNotIn("pygame", imported)

    def test_scripted_sequence_runs_without_pygame(self):
        script = textwrap.dedent(
            """
            import sys
            from game.environment import Action, BombermanEnv, Cell

            assert "pygame" not in sys.modules

            env = BombermanEnv()
            for _ in range(3):
                env.step(Action.RIGHT)
            env.step(Action.BOMB)
            for _ in range(3):
                env.step(Action.LEFT)
            env.step(Action.WAIT)

            assert env.dead is False
            assert env.player == (1, 1)
            assert env.grid[1][5] is Cell.EMPTY
            assert (5, 1) in env.explosion
            assert (6, 1) not in env.explosion

            env.step(Action.BOMB)
            for _ in range(4):
                env.step(Action.WAIT)
            assert env.dead is True

            env.reset()
            assert env.dead is False
            assert env.steps == 0
            assert env.player == (1, 1)
            assert env.bomb is None
            assert env.grid[1][5] is Cell.BLOCK
            assert "pygame" not in sys.modules
            print("scripted-ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("scripted-ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
