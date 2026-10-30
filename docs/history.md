# History

## Next History entry goes here so older entries get pushed down, newest is on top.


## 2026-10-30 Track A game rules (movement, gate, random crates, chains)

**Author:** Basel Swaid  
**Commits:** `9e88720` `8ca580e` `5fb70a9` `2b0abef` (stages 1 to 4) `7f4c3d6` (merge PR #2) `2bae4dd` (cleanup) `694814c` (merge PR #3)
**Tests:** 88 passing (17 original unchanged + 33 Task 1 of which 5 rewritten for the new rules + 38 new)

### Repository cleanup 
The repo is now organized into folders and nothing in the game or the tests behaves differently. The engine moved from `environment.py` to `game/environment.py` and the Pygame client from `play.py` to `game/play.py`, with an empty `game/__init__.py` so `game` works as a package. This file moved from `history.md` to `docs/history.md` and the checklist from `WORKFLOW.md` to `docs/WORKFLOW.md`, while `README.md` and `requirements.txt` stay in the root and all tests stay in `tests/`. Every move kept its git history, and `environment.py` and `history.md` are byte for byte the same as before. The only line changes are imports (`from environment import` became `from game.environment import`) and file paths in the tests, `play.py`, the README and the checklist, so Jan's 17 original tests check exactly what they checked before. I also added a `.gitignore` and stopped tracking the auto generated `__pycache__` files, so running the game or the tests no longer shows fake changes in git and the `PYTHONDONTWRITEBYTECODE` trick is not needed anymore. 
**What changes for everyone:** start the game with `python -m game.play` from the repo folder (`python play.py` no longer works), import the engine in new code with `from game.environment import ...`, and keep `game/__init__.py` empty. New folders like `agents/` for bots and `rl/` for the Gym wrapper will be created by the tasks that need them.

### What I did
* Settled the track A rules
* Had Claude plan all 4 stages first and implement them one commit per stage
* Claude and I verified every stage plus 20,000 single player games against the original engine and 1,500 full bot games against the new rules
* Applied the patch and merged PR #2

### How it works
* Characters pass through each other and can share a cell
* A new bomb blocks movement from the next tick so anyone stepping in on the same tick still gets in
* You can step off a bomb but never back on and each cell holds one bomb
* Gate at `(5,5)` under the center crate: hidden then closed then open once its crate is gone and all NPCs are dead
* Outcome is checked at the end of every step in `get_state()["outcome"]`
  * **win** agent alive on the open gate
  * **draw** agent dies together with the last NPC
  * **loss** any other agent death
* Any outcome freezes the game until `reset()`
* `BombermanEnv(random_crates=True)` places 18 crates with one always on the gate and keeps the 4 corner zones clear
* `reset(seed=42)` gives the same layout every time
* Chain reactions: a blast that reaches a bomb sets it off in the same tick
* New events: `bomb_triggered` `gate_revealed` `gate_opened`

### Decisions I made

- **Characters pass through each other**: they can share a cell and touching does nothing
  - Why: move order stops mattering, so frozen self-play copies behave the same from any slot
  - Alternatives considered: blocking each other (kept the order advantage and needed swap rules)
  - Status: provisional (needs team OK)

- **A new bomb blocks from the next tick**: anyone stepping into its cell on the same tick still gets in
  - Why: nobody gets trapped by a bomb dropped on their target cell in the same tick
  - Alternatives considered: blocking immediately (unfair depending on who acts first); bombs never blocking (kills bomb trapping)
  - Status: provisional (needs team OK)

- **Gate at (5,5) under the center crate**: hidden, then closed, then open once its crate is gone and all NPCs are dead
  - Why: the win needs both clearing the map and reaching a goal, like the original game
  - Alternatives considered: winning just by killing every NPC (too simple a goal)
  - Status: provisional (needs team OK)

- **The closed gate is normal floor**: you can walk on it and bomb it, and it can't be destroyed
  - Why: simplest option and it doesn't block paths through the center
  - Alternatives considered: blocking like a wall until it opens
  - Status: provisional (needs team OK)

- **Win, loss, draw**: win = alive on the open gate, draw = dying together with the last NPC, loss = any other death
  - Why: every agent death gets a clear result and any outcome freezes the game
  - Alternatives considered: counting a shared death as a loss (punishes the agent for finishing the job)
  - Status: provisional (needs team OK)

- **Random crates keep 18 crates and clear corner zones**: one crate always sits on the gate
  - Why: same difficulty as the fixed map and every spawn always has an escape
  - Alternatives considered: a random crate count (makes results harder to compare)
  - Status: provisional (needs team OK)

- **Chain reactions**: a blast that reaches a bomb sets it off in the same tick, credited to that bomb's owner
  - Why: matches the original game and keeps the event data factual for rewards later
  - Alternatives considered: blasts stopping at bombs (our 28.10 rule, now replaced)
  - Status: provisional (needs team OK)

- **The step limit lives in the Gym wrapper**, not the engine
  - Why: training stages can use different limits without touching the game rules
  - Alternatives considered: a fixed limit in the engine
  - Status: provisional (needs team OK)

**Replaces from 28.10:** characters block each other / no chain reactions / agent death ends the game (now any outcome freezes the game)

### AI-assisted parts
* Claude Code wrote the engine changes and all new tests
* The Claude supervisor reviewed the plan and every stage and ran the large game checks
* I made the design calls with the team and applied and merged the code

### Open questions
* `play.py` still shows only the single player game
* `README.md` still describes one player
* GitHub App access for the builder and `.gitignore` cleanup need the repo owner's OK

### Next task

**Track B: baseline bots.** We need simple scripted opponents so the agent has someone to play against before any trained agents exist. They train the first generation, and they stay as fixed opponents for evaluation forever, so every later generation is measured against the same thing. At minimum we want a random bot as the weakest baseline and a rule-based bot that plays sensibly: it should survive its own bombs, avoid blasts (including chain reactions), break crates and go after opponents. The bots must live outside the engine without changing `environment.py`, plug into the game through the NPC actions `env.step` already accepts, and be reproducible when seeded. How the rule bot decides what to do is up to whoever builds it. It's done when there are tests for its behavior and the rule bot clearly beats the random bot over many seeded games, since that is also our first sanity check for evaluation.

**Track C: Gym wrapper.** This turns our engine into a standard Gymnasium environment, so Stable-Baselines3 and our own tabular code can train on it. It must follow the Gymnasium API (`reset` with a seed, `step` returning observation, reward, terminated, truncated, info) and wrap `BombermanEnv` without changing it. It should also own two things the engine deliberately leaves out: the episode step limit, and the choice of who controls the NPCs. That choice should be swappable, because the NPCs start as the track B bots and later become frozen copies of our agent. For the same reason, the observation has to work from any character's point of view, so a frozen agent playing as an NPC sees the board as itself. **Before anyone codes this we need a team call** to decide what the agent observes, every reward term and its value, and the step limit, because together these are the problem formulation the course grades most heavily. It's done when it passes Gymnasium's own environment checker and has tests for the decisions we make in that call.

## 2026-10-28 Multi-character engine (agent + 3 NPCs)

**Author:** Basel Swaid  
**Commits:** `71ca046` `6aaf256` (workflow) `d19e3cb` (code) `4b95d49` (merge PR #1)  
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
