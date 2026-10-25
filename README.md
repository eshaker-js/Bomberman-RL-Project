# Bomberman RL

A step-based Bomberman sandbox. The rules live in `environment.py`. `play.py` is a Pygame window and keyboard client. There is no learning algorithm yet. A later agent can call `BombermanEnv.step(action)` and `BombermanEnv.reset()` without opening a window.

The board is a fixed 11×11 map: indestructible border and pillar walls, some breakable blocks, and one player at `(1, 1)`. Map size, bomb fuse, and blast range are constants at the top of `environment.py`.

## Install

From this directory, with Python 3.10 or newer:

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Play

```text
python play.py
```

## Controls

Each key press is one action. Holding a key does not repeat.

| Key | Action |
| --- | --- |
| Arrow keys or WASD | Move one cell |
| Space | Place a bomb |
| `N` or `.` (period) | Wait |
| `R` | Reset the whole game |
| Esc | Quit |

Walking into a wall or block, waiting, and placing a bomb while one is already active all spend the turn. After you die, the window stays open and ignores move and bomb keys until you press `R`.

## Timing

Nothing happens on its own. The window redraws on a frame clock, but the environment changes only when `step` runs.

A bomb starts with countdown 4 and does not count down on the turn it is placed. Each later action lowers the countdown by one. On the fourth action after placement, you move first and then the bomb explodes. The blast covers the bomb cell and two cells in each direction. Walls stop it and stay. Breakable blocks are destroyed and stop it. Flames are shown until the next action; they do not burn again and do not advance time while they sit on screen. The HUD shows the countdown, the step count, the controls, and a death message.

## Tests

```text
python -m unittest discover -s tests -v
```

The tests drive `environment.py` only. They do not open a window.

## Layout

- `environment.py` — board, movement, bomb, explosions, death, `reset`, `step`
- `play.py` — drawing and keyboard input
- `tests/test_environment.py` — headless rule tests
