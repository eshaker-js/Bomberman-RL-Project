# History

## Next History entry goes here so older entries get pushed down, newest is on top.
## 2026-10-28 Multi-character engine (agent + 3 NPCs)

**Author:** Basel Swaid  
**Commits:** `71ca046` `6aaf256` (workflow) `d19e3cb` (code) `4b95d49` (merge PR #1)  
**Time spent:** ~[hours]h  
**Tests:** 50 passing (17 original unchanged + 33 new)

### What I did
* Added `WORKFLOW.md` for team handoffs
* Added the multi-character rules
* Extended the engine to one agent plus three NPCs without changing single-player behavior
* Had the plan reviewed first and the code verified after
* Ran the tests locally and merged PR #1

### How it works
* `BombermanEnv()` is the old single-player game
* `BombermanEnv(with_npcs=True)` adds NPCs 1 2 3 at `(9,1)` `(1,9)` `(9,9)` and the agent stays at `(1,1)`
* NPC actions are passed in: `env.step(Action.RIGHT, {1: Action.UP})` and missing NPCs wait
* Each step: characters act 0 then 1 2 3, bombs count down, due bombs explode together
* `get_state()` adds `characters` `bombs` `events` `all_npcs_dead`
* Events list deaths and destroyed blocks with every bomb owner involved
* New tests live in `tests/test_multi_character.py` and the 17 old tests must stay unchanged
* Set `PYTHONDONTWRITEBYTECODE=1` before running tests or git shows fake `.pyc` changes

### Decisions I made (provisional until the whole team agrees)
* **One bomb per character** to keep it simple (more bombs later)
* **Characters block each other and touching does no damage** because our NPCs are bombers not monsters
* **Agent acts first then NPCs in order** because it is simple and deterministic (simultaneous moves need extra collision rules)
* **No chain reactions yet** so a blast stops at another bomb without setting it off
* **Blasts hurt everyone including the owner** so bombing stays risky
* **A dead character's bomb still explodes** so timing stays predictable
* **Four fixed corner spawns** each with a tested escape route (random maps later)
* **All NPCs dead is reported but is not a win yet** and agent death still ends the game
* **Events keep every bomb owner** so we decide reward credit later instead of now

### AI-assisted parts
* Claude Code wrote the engine changes and the 33 new tests
* I reviewed the plan and the code and compared with Claude 20,000 random games against the old engine
* I made the design calls and applied and merged the code

### Open questions
* Teammates needs to confirm the rules above
* Still open: episode length, draws, random crates, exit door, buffs
* `README.md` still says one player and `play.py` does not draw NPCs
* Housekeeping: committed `__pycache__` and no `.gitignore`

### Next task
* If we agree on M1 below: claim one track in the chat
* If not: build the scripted NPC bots in a separate module

### Proposed milestone M1: full game + first learning agent (target 15.11)
* **A Rules:** win loss draw, episode limit, seeded random crates, chain reactions (buffs and exit door move to M2)
* **B NPC bots:** random bot and a rule bot that bombs crates and escapes with BFS
* **C RL setup:** Gym wrapper, shared evaluation script with fixed seeds, logging
* **D First results:** tabular Q-learning on a small stage and one PPO or DQN run against random bots
* **Team call first** to decide observation, reward and episode length together since that is the graded part
* **Done when** rules are frozen and tested and a trained agent beats random bots over 3 seeds
* A B C can run in parallel now with one person each and D comes after

## 2026-10-25 — First playable environment

Basic version of Bomberman, currently starts with a pre-determined 11x11 grid.

The grid has 3 types of cells:
Empty
Wall = unwalkable terrain.
Block = breakable terrain (with a bomb)

The game takes place in a class called BombermanEnv.
The class keeps data on the following:

1. Grid State.
2. Player Location
3. If the player is dead.
4. Bomb Location
5. Explosion (active cells marked as explosion that kill the player if he is in them)
6. Number of steps

At every turn the player can take an action:
1. Move (U,D,L,R)
2. Plant Bomb
3. Wait

The bomb is a class that keeps track of the bombs location and its timer. Currently a bomb is placed with a timer value of 4 (steps).

After every action/step, the env class returns the current state including all the data stated above^

Cursor also generated a file "play.py" that allows you to run the environment as a game and try it for yourself.
Controls: WASD/ARROWKEYS - Move | Space - Bomb | N or . - Wait | R - Reset | Esc - Quit 
(this is also detailed in the game window.)
I suggest running the game and experimenting after reading this.

### Next Tasks:

Right now when the player tries to walk on unwalkable terrain (block or wall) the game doesn't move the player (good) but consumes a move and doesn't provide feedback (probably bad, but might be ok, we need to decide together)

This is still a very early version, serves mainly as a skeleton we can start from.

I do think we can start running a simple RL Model and see if it can learn to move, explode a block and survive its own bomb as a first step. (that's what I'd want to do next if I'm the one progressing again)
