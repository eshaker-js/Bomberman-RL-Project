# History

## 2026-10-25 — First playable environment

Added a deterministic 11×11 Bomberman sandbox with no learning code. `environment.py` owns the board, movement, the single bomb, explosions, and death. It exposes `reset()` and `step(action)` and does not import Pygame or use a wall clock. `play.py` draws colored shapes and turns one key press into one action.

A bomb starts at countdown 4 and does not tick on the turn it is placed. Each later action moves the player first, then counts the bomb down. On the fourth of those actions it explodes. The blast covers the bomb cell and two cells in each direction. Walls stop a ray and stay; breakable blocks are destroyed and stop that ray. Flames are stored on the environment for the renderer and hurt the player only on the explosion step. There is one player, at most one bomb, and no win condition. After death, `step` leaves the state unchanged until `reset`.

Verification: `python -m unittest discover -s tests -v` — 17 tests, OK. They cover collision, fuse timing, leaving and re-entering a bomb cell, blast blocking, block destruction, self-death, reset, and a scripted action sequence in a fresh process that never imports Pygame. A dummy-video pass through `play.py` also moved, planted a bomb, escaped, destroyed a block, died, ignored further play keys, and reset.

Suggested next steps: a reward and a real terminal condition, then whatever agent API the course wants around the existing `step`. Later, random solvable maps, persistent fire, and opponents.
