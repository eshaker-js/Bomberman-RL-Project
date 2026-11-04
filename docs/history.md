# History

## Next History entry goes here so older entries get pushed down, newest is on top.

## 2026-11-05 Stage 1 of the project setup: evaluation, opponent ladder, Gym wrapper, training scripts (Task 6)

**Author:** Basel Swaid  
**Commits:** `dfa12f0` (part 1 evaluation) `d993524` (part 2 ladder) `97ff122` (part 3 Gym wrapper) `4930028` (part 4 training scripts) `[merge hash]` (merge PR #7)  
**Tests:** 353 passing (256 before + 97 new).

### Start here (the short version)
* Stage 1 of the project is now about **setup only**: everything needed so stage 2 of the project can start training on day one. **Nothing was trained for real yet**, only tiny smoke runs to prove every script works end to end
* Four new pieces:
  1. **Evaluation can play the real game timing** (`--game-rules`)
  2. **An opponent ladder** from easy to hard: `random < noisy50 < noisy25 < noisy10 < rule`
  3. **A Gym wrapper** in `rl/` that turns our game into a standard RL environment
  4. **Training scripts** for tabular Q-learning and PPO, an agent evaluator, and run logging
* **Every MDP number (rewards, step limit, hyperparameters) lives in one file: `rl/config.py`**, marked v0. Changing a value never needs code changes

### New packages
* `requirements.txt` now has `numpy`, `gymnasium`, `torch` and `stable-baselines3` (min versions that work on Python 3.10 to 3.13)
* Install: `pip install -r requirements.txt` (on Windows this gives the small CPU torch automatically)
* On Linux run `pip install torch --index-url https://download.pytorch.org/whl/cpu` first or pip downloads several GB of GPU libraries we don't need
* You can still play the game and run the old tests without these packages, the RL tests just skip

### Part 1: evaluation with the real rules
* `agents/match.py`: `play_game(..., rules=None)`. No `rules` means exactly the old behavior (checked on 150 games against the old file)
* `agents/evaluate.py`: new `--game-rules` flag, and the output now prints which rules were used
* Run: `python -m agents.evaluate --games 200 --game-rules`
* Result with the real rules: rule bot as agent vs 3 random bots wins 100%, random bot wins 0%

### Part 2: opponent difficulty ladder
* New file `agents/noisy_bot.py`: a rule bot that takes a random action some % of the time (Joelle's `rule_bot.py` and `random_bot.py` are untouched)
* `LADDER` has 5 levels: `random`, `noisy50` (50% random), `noisy25`, `noisy10`, `rule`
* Run both ladder tables: `python -m agents.ladder_eval --games 200 --game-rules`
* It really is a ladder (200 games each, real rules):

  | Level | Win as agent vs 3 random | Rule bot's win vs 3 of this level |
  |---|---|---|
  | random | 0% | 100% |
  | noisy50 | 30% | 99% |
  | noisy25 | 59.5% | 75.5% |
  | noisy10 | 83.5% | 27% |
  | rule | 100% | 0% (90.5% timeouts) |

* Why it matters: stage 2 of the project can train in steps (random first, then harder levels) instead of jumping straight from "opponents kill themselves" to "opponents never die"

### Part 3: the Gym wrapper (`rl/`)
* `rl/bomberman_env.py`: `BombermanGymEnv(opponents=...)`, a standard Gymnasium environment around our engine. **The engine itself is untouched**
* Always plays the real game: `GAME_RULES` and random crates
* **Opponents** are a parameter:
  * a ladder level name like `"random"` or `"noisy25"` (3 bots of that level)
  * a list of your own controllers (later: frozen copies of our agent)
  * or `None` for single player (you can only lose there, the gate never opens without NPCs)
  * NPCs that get no controller just stand still and wait
* **Actions:** 6 (up, down, left, right, wait, bomb) as numbers 0 to 5
* **Observation:** 10 layers of 11×11 grids, all values between 0 and 1
  1. walls
  2. crates
  3. bombs (how long until they explode)
  4. flames (how long they still burn)
  5. me
  6. opponents
  7. gate hidden
  8. gate closed
  9. gate open
  10. "my bomb is ready" (the whole layer is 1 or 0)
* The observation can be built **from any character's point of view** (`encode_observation(state, me)`), so a frozen agent playing as an NPC sees itself as "me"
* **Reward v0** (all in `rl/config.py`):

  | Event | Reward |
  |---|---|
  | win | +1.0 |
  | loss (any death, including my own bomb) | −1.0 |
  | draw | −0.5 |
  | each opponent my blast killed (lingering flames count) | +0.3 |
  | each crate my blast destroyed | +0.02 |
  | every step | −0.001 |
  | a move that didn't move me (wall, crate, bomb) | −0.01 |

* `info["reward_terms"]` shows how much each term contributed on every step, useful in stage 2 of the project to see *why* the agent gets its reward
* **Episode ends** (`terminated`) on win, loss or draw. **Cut off** (`truncated`) after 800 steps (about 3.3 minutes of play)
* Seeds match `play_game`: `env.reset(seed=5)` is exactly the same game as the bot evaluation with seed 5, so agents and bots are compared on identical games
* Passes Gymnasium's official `check_env`

### Part 4: training scripts and logging (smoke runs only)
* **Tabular Q-learning** (our own code, no library): `python -m rl.train_tabular --episodes 3000 --seed 0`
  * Plays a **tiny practice stage** (`rl/tabular_env.py`): single player, fixed map, the episode ends when the first crate is destroyed or the agent dies
  * Reward: +1 crate destroyed while alive, −1 death, −0.001 per step
  * State: my position + my bomb (cell and countdown) + where my old flames still burn
  * Smoke run: 2 seconds, the agent destroys a crate without dying in 100% of the evaluation games
* **PPO** (Stable-Baselines3, cited in the code): `python -m rl.train_ppo --timesteps 10000 --seed 0`
  * Plays the full game vs 3 random bots, with a small network made for our 11×11 grid (SB3's default needs bigger images)
  * Smoke run: under a minute. The agent mostly learns to wait (expected after only 10,000 steps, that's what stage 2 of the project fixes)
* **Evaluate any saved agent:** `python -m rl.evaluate_agent --run runs/<folder> --opponents random` (tabular also has `--stage tiny|full`)
* **Every run gets its own folder** in `runs/` (ignored by git) with its config, seed, git commit, whether there were uncommitted changes, package versions, metrics and the saved model, so every result can be reproduced
* `docs/EXPERIMENTS.md` is ready: one row per run we want to keep (smoke runs don't go there)
* Every script takes `--config my_overrides.json` to change values without touching code, for example `{"max_steps": 600, "reward": {"kill": 0.5}}`

### Decisions I made

- **Stage 1 of the project is setup only, training moves to stage 2**
  - Why: experiments should start on a finished setup, not one that keeps changing
  - Alternatives considered: first training runs already in stage 1 (would mix building and research)
  - Status: provisional (needs team OK)

- **All MDP values in one config file, marked v0**
  - Why: the team can change any number at the meeting without anyone rewriting code
  - Alternatives considered: values spread across the scripts
  - Status: provisional (needs team OK)

- **The reward v0 table above, with the win larger than everything else combined**
  - Why: kills and crates guide the agent but never pay more than winning, so kamikaze kills don't pay off
  - Alternatives considered: only win/loss rewards (very slow to learn), bigger kill rewards (risk of suicide attacks)
  - Status: provisional (needs team OK)

- **Episode cut off at 800 steps**
  - Why: long enough to finish a real game, short enough that a waiting agent doesn't run forever
  - Alternatives considered: 400 (the evaluation limit, too short with the 9 tick fuse)
  - Status: provisional (needs team OK)

- **Tabular only on a tiny stage**
  - Why: a table can't hold the full game, the tiny stage is our "fundamentals" algorithm from the course
  - Alternatives considered: tabular on the full game (billions of states, can't work)
  - Status: provisional (needs team OK)

- **PPO first, DQN in stage 2 of the project**
  - Why: PPO is more forgiving for a first setup
  - Status: provisional (needs team OK)

- **NPCs without a controller stand still**
  - Why: lets us build an easy "target practice" stage for the curriculum later
  - Alternatives considered: removing extra NPCs (changes the game)
  - Status: provisional (needs team OK)

- **Opponent ladder levels 50%, 25% and 10% random**
  - Why: the two tables above show each step really is harder
  - Status: provisional (needs team OK)

### ANY OBJECTIONS ON:
Please reply in the group if you disagree, otherwise these become team-agreed
* The reward values, the 800 step limit and the PPO and tabular settings in `rl/config.py`
* The observation layers
* The ladder levels
* Moving all real training to stage 2 of the project

### AI-assisted parts
* Claude Code wrote all the code and the 97 new tests
* The Claude supervisor planned the spec, reviewed the plan, and verified the result on its own machine with every package installed:
  * every part's tests
  * the old evaluation unchanged on 150 games
  * the ladder order on seeds the builder never used
  * the wrapper's rewards and observation compared step by step with its **own independent implementation** over 330 games (every reward term covered)
  * its own smoke runs
* I made the design calls and applied and merged the code

### Open questions for the team
* Confirm everything under "ANY OBJECTIONS ON"
* **Buffs** (bigger blast, speed) are the last game item of stage 1 of the project. We need to decide how speed works when one step is one tile
* Literature review and report skeleton: who takes them (the guideline wants the algorithm choice justified with state of the art research)
* `agents/evaluate.py` still stops games at 400 steps: with the real timing, rule bot vs rule bot times out 90% of the time
* Give the Claude GitHub App access to the repo (every task so far needed a patch file instead of a direct push)
* README: add the `rl/` commands

### Next task for whoever picks this up

**Buffs (last item of stage 1 of the project).** Crates should sometimes drop a buff when destroyed, with a probability per buff. We planned a bigger blast range and more speed(idk if thats relevant). The engine already reads its rules from settings, so buffs should fit the same way, and the wrapper's observation will need a layer for buffs on the floor and for each character's current buffs. Speed needs a decision first, because one step is exactly one tile.

**Stage 2 of the project: first real training.** Everything is ready: pick an opponent level, run `rl.train_tabular` or `rl.train_ppo` with a real number of steps and several seeds, evaluate with `rl.evaluate_agent`, and log every run you keep in `docs/EXPERIMENTS.md`. Start with the tiny stage and PPO vs `random`, then climb the ladder.

## 2026-11-02 Retro UI and real game timing (Tasks 4 and 5) + track B review

**Author:** Basel Swaid  
**Commits:** `e95e102` (merged Joelle's PR #4) `b78935e` `05c5c60` `588e56f` (UI) `efacc27` (merge PR #5) `7a020d8` `1e887b1` `59733c2` (timing) `03cb261` (merge PR #6)  
**Tests:** 256 passing (104 before today + 109 UI + 43 timing) and UI tests skip cleanly without pygame

### What I did
* Built a playable retro game window in `ui/` (Task 4)
* Measured the original game's timing from a gameplay video and matched our game to it (Task 5)
* Updated the rule bot so it understands the new timing
* Each task was planned first, reviewed, built in stages and verified before merging

### How it works
* Start the game with `python -m ui` from the repo folder
* Start screen has three modes
  * **Free World** you alone with no bots to test the rules
  * **Player vs Bots** you against 3 bots (Random, Rule or Mixed)
  * **Agent Spectator** a NOT READY placeholder until we have a trained agent
* The game ticks every 250 ms by itself and the last key you pressed is your move (holding a key keeps walking and no key means wait)
* Characters slide between cells and flames fade out as they burn down
* The UI only reads the game state and sends actions so it never changes any rule and training never opens a window
* Real game timing lives in one place: `GAME_RULES` in `game/environment.py` (fuse 9 ticks and flames 4 ticks)
* `BombermanEnv()` still uses the old values (fuse 4 and flames 1) so all old tests are untouched and the real game uses `BombermanEnv(**GAME_RULES)`
* Flames stay deadly while they burn: walking into a burning cell kills you
* The state now has `flames` (burning cells with ticks left) and `rules` (the timing this game uses)
* `game/play.py` is still the simple debug client with the old rules

### Decisions I made

- **Auto tick every 250 ms in the UI**
  - Why: feels like a real game and matches the measured walking speed of the original (0.26 s per tile)
  - Alternatives considered: tick only when a key is pressed (turn based, does not feel like the original)
  - Status: provisional (needs team OK)

- **Bomb fuse 9 ticks (about 2.25 s)**
  - Why: measured from the original game, a longer tick would have made walking twice as slow
  - Alternatives considered: keeping 4 ticks with a longer tick (slow motion and still only 4 steps to escape)
  - Status: provisional (needs team OK)

- **Flames stay deadly for 4 ticks (about 1 s)**
  - Why: measured from the original game where flames kill while they burn
  - Alternatives considered: flames only visual after the first tick (confusing to walk through fire)
  - Status: provisional (needs team OK)

- **Fuse and flames are settings with the old values as defaults**
  - Why: old tests stay untouched and humans and the agent play the same `GAME_RULES`
  - Alternatives considered: changing the constants (would break most existing tests)
  - Status: provisional (needs team OK)

- **I approved updating the rule bot** (`agents/rule_bot.py` only, minimal lines)
  - Why: the bot had the old fuse built in and walked into burning flames, under the new rules it won only 3.5% and now wins 100% with no self kills
  - Alternatives considered: 
  - Status: 

- **Retro look: pixel art and font drawn in code with the PICO-8 palette, dark grey floor**
  - Why: our own art with no image files or licenses, the grey floor keeps bombs visible (black hid them)
  - Alternatives considered: free art packs (not ours), black floor (bombs disappeared)
  - Status: provisional (needs team OK)

- **Game name is a placeholder "BOMB ARENA"** in one constant in `ui/config.py`
  - Status: open, the team picks the real name

### ANY OBJECTIONS ON:
* **Game timing:** the bomb now explodes after 9 ticks (about 2.25 s) and flames stay deadly for 4 ticks (about 1 s), measured from the original game. The agent will learn with these values through `GAME_RULES`
* **Rule bot update:** I changed `agents/rule_bot.py` (only the needed lines) so it reads the timing from the game and avoids burning flames. Under the new timing it went from 3.5% wins to 100% and with the old timing it plays exactly as before
* **UI and how the game plays:** `python -m ui` with three modes (Free World, Player vs Bots, Agent Spectator placeholder), an automatic tick every 250 ms (your last key is your move and no key means wait), smooth movement and fading flames
* **How the game looks:** retro pixel art drawn in code with the PICO-8 colors, four original hooded characters (blue, red, green, orange), dark grey floor, black border on the closed gate, placeholder name "BOMB ARENA"

### AI-assisted parts
* Claude Code wrote the UI, the timing changes, the bot update and all new tests
* The Claude supervisor planned the specs, reviewed every plan, measured the original game's timing frame by frame from my video, and verified each patch (default rules identical to Jan's original engine on 20,000 games, the new flame rules checked over 1,200 full games)

## 2026-10-31 Track B baseline bots (random bot + rule bot)

**Author:** Joelle  
**Commits:** 755fdca (bots, match runner, evaluation, tests)  
**Tests:** 104 passing (88 earlier tests unchanged + 16 new in `tests/test_bots.py`)

### What I did
- Added the `agents/` package with a random bot and a rule-based bot
- Added `agents/match.py` to play one seeded game between any four bots and record what happened
- Added `agents/evaluate.py` to compare the bots over many seeded games
- Added `tests/test_bots.py`
- `game/environment.py` and all earlier tests are untouched

### How it works
- Every bot has `reset(seed)` and `act(state, me)`: it reads `env.get_state()` and the ID it controls and returns one `Action`, so the same bot can sit in any seat (agent or NPC)
- `RandomBot`: a uniformly random action out of all six every turn, seeded
- `RuleBot` re-plans every turn in this order: survive, take the open gate (agent only), bomb if it hits a crate or an opponent and an escape exists, walk to the nearest such bomb spot, wait
- Survival: it predicts on which action each bomb explodes, including chain reactions, and searches over (cell, time) for a way to a cell no blast will reach
- It also assumes any opponent without an active bomb might drop one this action, and prefers moves that stay safe even then
- Play a game: `play_game({0: RuleBot(), 1: RandomBot(), 2: RandomBot(), 3: RandomBot()}, seed=0)`
- Compare the bots: `python -m agents.evaluate` (add `--games 500` for more)

### Results (`python -m agents.evaluate --games 200`, seeds 0..199)
- Agent seat vs 3 random NPCs: rule bot wins 100% (0 suicides, 8.6 crates per game); random bot wins 0%, loses 96.5%, draws 3.5%, and always dies by its own bomb
- Mixed table (2 rule + 2 random, seats rotated): rule bots died 0%, random bots 87%
- 4 rule bots against each other: nobody died in 40 games, all hit the 400 step limit
- Rule bot alone (single player): clears all 18 crates and never dies (30 seeds, fixed and random maps)

### Decisions I made
- **Bot interface**: `reset(seed)` + `act(state, me)` reading only `get_state()`
  - Why: the bots stay outside the engine and the Gym wrapper can swap any of them into any NPC seat
  - Alternatives considered: giving bots the env object (they could change the game by accident)
  - Status: provisional (needs team OK)
- **Random bot uses all six actions equally, including BOMB**
  - Why: the weakest possible baseline, no knowledge at all
  - Alternatives considered: a random bot that avoids suicide (that is already a small rule bot)
  - Status: provisional (needs team OK)
- **Rule bot is cautious about opponents' next bomb**
  - Why: without it the rule bot lost 1 of 60 games (seed 0): an NPC bombed the only exit of the corner it fled into. With it: 0 losses in 200 games
  - Alternatives considered: ignoring bombs not yet placed (simpler, 1.7% losses)
  - Status: provisional (needs team OK)
- **The 400 step limit lives in `agents/match.py` only and is reported as `"timeout"`**
  - Why: evaluation needs games to end; the training episode limit is still the Gym wrapper's job (track C)
  - Alternatives considered: adding a limit to the engine (against the track A decision)
  - Status: provisional (needs team OK)
- **Each bot's seed is `game_seed * 1000 + character_id`**
  - Why: every bot in every game gets a different but reproducible seed
  - Status: provisional (needs team OK)

### AI-assisted parts
- Claude wrote the bots, the match and evaluation scripts and the tests, and found and fixed the corner-trap loss
- I reviewed the code, ran the tests and the evaluation, and committed

### Open questions for the team
- Rule bots never kill each other. If the agent later trains against rule bot NPCs it must kill all of them to open the gate, which may be close to impossible. Do we want a more aggressive variant, or a rule bot that takes a random action some percent of the time?
- Rule bots get few kills against random bots because random bots usually kill themselves first
- README.md still describes one player and does not mention `agents/`

### Next task for whoever picks this up
- Track C (Gym wrapper): team call first on observation, rewards and step limit. The NPC controller can use these bots directly: `bot.act(state, npc_id)`
- Track D can use `agents/evaluate.py` as the fixed evaluation against the random and rule bots



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
