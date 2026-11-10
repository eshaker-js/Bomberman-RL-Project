# History

## Next History entry goes here so older entries get pushed down, newest is on top.

## 2026-11-10 Stage 2, part 1: tabular Q-learning experiments on the tiny stage

**Author:** Joelle  
**Commits:** `b3202a3` (sweep, plots, tests) + the commit that adds this entry and `docs/EXPERIMENTS.md`  
**Tests:** 472 passing (457 before + 15 new in `tests/test_sweep_tabular.py`)

### Start here (the short version)
* **The tabular half of stage 2 is done:** Q-learning on the tiny stage, 4 hyperparameters, 5 to 20 seeds each, plus a confirmation on 20 seeds never used for tuning
* **Main result:** alpha = 1.0 learns about **twice as fast** as v0 (alpha = 0.1) and always finds the shortest route. Confirmed on seeds 100 to 119: faster on 19 of 20 seeds (sign test p = 0.00004)
* **The tabular agent fails completely on the full game** (0% wins, about 90% self-kills): its state cannot see crates or opponents. This is our argument for PPO
* **The PPO half of stage 2 is not started** and needs an owner
* All runs are on commit `b3202a3` with a clean working tree, logged in `docs/EXPERIMENTS.md`

### What I did
* `rl/sweep_tabular.py`: runs a hyperparameter sweep over several seeds and writes summary tables
* `rl/plot_tabular.py`: one learning-curve figure per parameter
* `tests/test_sweep_tabular.py`: 15 tests
* `experiments/tabular/`: the plan files of the alpha and confirmation runs
* `rl/train_tabular.py` (Basel's file): an optional `on_episode_end` hook so the sweep can test the agent during training. Training itself is unchanged, a test checks the Q-table is identical with and without the hook
* `requirements.txt`: added `matplotlib` (only needed for the figures)

### How it works
* `python -m rl.sweep_tabular` runs the default plan with 5 seeds. `--plan experiments/tabular/alpha.json --seeds 20` runs another plan
* **One parameter at a time:** every config is v0 with exactly one value changed. v0 itself runs once per seed and is shared by all groups (13 configs, not 16)
* Every 50 episodes the **greedy** policy (no exploration) plays one game. `learned_at` = the first checkpoint from which every later one succeeds
* Why greedy and not training success: with `epsilon_end = 0.05`, training success stays around 90% even when the policy is already perfect
* `optimal` = the agent takes the fewest steps possible. That minimum (7 steps) is not assumed: `shortest_solution` finds it by a breadth-first search over every action sequence, directly on the engine
* Every run is a normal tabular run folder inside one sweep folder, plus `plan.json`, `summary.csv`, `checkpoints.csv` and `aggregate.csv`
* `python -m rl.plot_tabular runs/<sweep folder>` writes `plots/<param>.png`: left the share of seeds whose greedy policy solves the stage, right the training success (moving average, mean and std over seeds)

### Results
Main sweep, 5 seeds (`learned_at` = mean and std over seeds, lower is faster):

| Parameter | Value | Solved | Optimal | learned_at |
|---|---|---|---|---|
| alpha | 0.01 / 0.05 / **0.1 (v0)** / 0.3 / 0.5 / 1.0 | all 5/5 | 5, 5, 4, 3, 5, 5 | 2430, 1280, 970, 640, 790, **540** |
| gamma | 0.5 / 0.9 / 0.95 / **0.99 (v0)** | all 5/5 | 5, 5, 3, 4 | 1130, 990, 980, 970 |
| epsilon_decay_fraction | 0.2 / 0.5 / **0.8 (v0)** | all 5/5 | **1**, 4, 4 | **530**, 840, 970 |
| episodes | 300 / 1000 / **3000 (v0)** | **0/5**, 5/5, 5/5 | 0, 2, 4 | n/a, 730, 970 |

Alpha with 20 seeds: 0.01: 2462, 0.05: 1335, 0.1: 1008, 0.3: 662, 0.5: 710, 1.0: 582. Optimal route 20/20 only for 0.5 and 1.0

Confirmation on new seeds 100 to 119:

| | v0 (alpha 0.1) | Tuned (alpha 1.0) |
|---|---|---|
| learned_at | 998 +- 170 | **495 +- 203** |
| Optimal route | 15/20 | **20/20** |
| Faster on the same seed | | **19 of 20** (p = 0.00004) |

v0 agent (seed 0) on the full game, 100 games each:

| Opponents | Win | Draw | Loss | Timeout | Self-kill |
|---|---|---|---|---|---|
| 3 random | 0% | 5% | 89% | 6% | 94% |
| 3 noisy25 | 0% | 0% | 97% | 3% | 89% |
| 3 rule | 0% | 0% | 100% | 0% | 91% |

### What we learned
* **alpha has the largest effect.** 0.1 and below are clearly slower. From 0.3 up it is a plateau within the noise, even with 20 seeds
* **alpha = 1.0 works because the tiny stage is deterministic:** the same action in the same state always gives the same result, so there is no noise to average. This should not carry over to the full game
* **gamma barely matters** because an episode is only about 7 steps long
* **Exploration vs exploitation:** faster epsilon decay learns sooner (530) but only 1 of 5 seeds found the optimal route
* **300 episodes are not enough.** But episodes and the epsilon schedule are coupled (epsilon decays over a fraction of the episodes), so 1000 vs 3000 mostly measures the schedule
* **Q-learning is off-policy:** the greedy policy solves the stage long before training success rises, because the agent is still exploring
* **Why tabular fails on the full game:** it memorised one route for the fixed map. On random maps the crate it expects is not there and it walks back into its own fire. Its state has no crates, opponents or other bombs, and a table cannot hold the full observation, so the full game needs function approximation (PPO)

### Decisions I made

- **Measure learning with the greedy policy at checkpoints**
  - Why: training success is capped by exploration (epsilon 0.05), so it never shows when the policy is learned
  - Alternatives considered: 90% training success over 100 episodes (v0 only reached 83%)
  - Status: provisional (needs team OK)

- **One parameter at a time, v0 shared**
  - Why: every difference comes from one value, 13 configs instead of a 216 config grid
  - Alternatives considered: a full grid (misses nothing, but 16 times more runs and harder to read)
  - Status: provisional (needs team OK)

- **Confirm on seeds never used for tuning (100 to 119)**
  - Why: choosing and confirming on the same seeds would overstate the gain
  - Status: provisional (needs team OK)

- **Tuned tiny-stage config: alpha = 1.0, everything else v0**
  - Why: fastest, optimal route on every seed, and a single change keeps the comparison clean
  - Alternatives considered: alpha 1.0 with epsilon decay 0.5 (faster maybe, but never tested together)
  - Status: provisional. `TabularConfig` in `rl/config.py` is unchanged, it is the team's call whether v0 should become 1.0

- **matplotlib in requirements.txt**
  - Why: the figures for the report. Only `rl.plot_tabular` imports it, its test skips without it
  - Status: provisional (needs team OK)

### AI-assisted parts
* Claude wrote the sweep, plot and test code and the hook, checked the 7-step optimum by exhaustive search, and checked that the results are identical on two different machines
* I chose the tuned config, ran every experiment on my machine, and committed

### Open questions for the team
* **Who takes PPO?** The M1 done-criterion (a trained agent beats random bots over 3 seeds) can only be reached with PPO
* OK to add `matplotlib`?
* **Small bug in `new_run_dir`:** folders are named by the second, so two runs of the same kind and seed started in the same second crash with `FileExistsError` (happened with `evaluate_agent` run twice quickly). The sweep avoids it with subfolders. Possible fix: add milliseconds or a counter
* The epsilon schedule depends on the episode count. Should it decay over a fixed number of episodes instead?
* Should the v0 tabular alpha become 1.0?

### Next task for whoever picks this up
**Stage 2, part 2: PPO.** First measure how long PPO training takes on your machine with a short run, then train PPO against `random` with several seeds, find how many steps and which settings it needs before it stops just waiting, evaluate on unseen seeds, log in `docs/EXPERIMENTS.md`, and climb the ladder. `rl/sweep_tabular.py` shows a pattern for running seeds and summarising them.

## 2026-11-10 Buffs, fuse 4 everywhere, last items of stage 1 of the project (Task 7)

**Author:** Basel Swaid  
**Commits:** `818911c` (part 1 engine) `9078cc6` (part 2 bots and evaluation) `596c1fa` (part 2b bot fix) `417b2ac` (part 3 observation and reward) `43eaef4` (part 4 UI) `2826b73` (part 5 README) `2ad685c` (part 6 Windows fix) `d6c25de` (merge PR #8) 
**Tests:** 457 passing (353 before + 104 new)

### Start here (the short version)
* **Stage 1 of the project is finished.** Everything is set up for training
* **The game is now fuse 4 everywhere:** humans in the UI, the bots, the evaluation and the agent all play the exact same rules
* **Three buffs:** bigger blast (range), extra bomb, and burn (fire that lasts longer around the bomb). Speed was dropped
* The rule bot understands buffs, the agent's observation shows them, and the UI draws them
* Old saved PPO models no longer load (the observation changed). They were only smoke runs anyway
* **Install the RL packages** (`pip install -r requirements.txt`), otherwise 72+ tests are skipped and you aren't really testing the RL code

### The rules now
* `GAME_RULES` = fuse **4**, flames **4**, buffs on. The engine defaults (fuse 4, flames 1, no buffs) didn't change, so the old tests still describe the original game
* Games stop at **800 steps** in both training and evaluation (evaluation used 400 before)
* With buffs off the engine plays **exactly** like before (checked on 20,000 random games against the old engine)

### Part 1: buffs in the engine
* All buff numbers live in **one place**, `BUFF_RULES` in `game/environment.py` (v0):

  | Buff | Effect per pickup | Start | Max | Drop chance |
  |---|---|---|---|---|
  | Range | +1 blast range | 2 | 5 | 15% |
  | Extra bomb | +1 bomb at the same time | 1 | 3 | 15% |
  | Burn | +1 burn level | 0 | 2 | 10% |

* **Burn:** level 1 = the bomb's cell and 1 tile each way burn **5** ticks, level 2 = 2 tiles each way burn **6** ticks. The rest of the blast burns the normal 4. It never goes past the blast
* **Drops:** each destroyed crate rolls once (40% total, about 6 to 7 buffs per game). The gate crate never drops. Drops use their **own random stream**, so a seed still gives exactly the same crates
* **Pickup:** walk onto it. If two characters land on it together, a random one gets it (not the lowest id, that would always favour our agent)
* A fresh blast destroys buffs on the floor, but not the ones it just dropped. Lingering flames don't destroy buffs
* **Each bomb remembers** the range and burn its owner had when placing it
* A fresh blast never shortens a cell that is already burning longer
* New: `state["buffs"]`, `state["buff_rules"]`, each character's levels and bombs left, each bomb's range and burn, and the events `buff_dropped`, `buff_picked`, `buff_destroyed`
* `start_buffs` lets a game start with characters already buffed (used by the measurement tool, maybe useful for curriculum later)

### Part 2: bots and evaluation
* **Rule bot**: reads each bomb's own range and burn, handles several bombs, and only places a second bomb if it can still escape everything
* **A bug the builder found (part 2b):** when ranges were different, the bot thought an opponent's *possible* bomb would stop its own longer blast, so it walked into its own fire. Fixed, with a test. Without buffs the bot plays exactly as before (240 of 240 games identical)
* New tool `agents/buff_eval.py`: the rule bot starts with a buff against 3 rule bots or 3 `noisy25` bots, and reports wins, survival, kills and self-kills
* **The ladder still gets harder level by level** (200 games each, final code):

  | Level | Win as agent vs 3 random | Rule bot's win vs 3 of this level |
  |---|---|---|
  | random | 0% | 100% |
  | noisy50 | 0.5% | 99.5% |
  | noisy25 | 19% | 99% |
  | noisy10 | 52% | 94% |
  | rule | 100% | 1% (73% timeouts) |

  The noisy levels are much weaker as agents with fuse 4 than with fuse 9 (noisy25 was 59.5%)

### Part 3: what the agent sees and gets
* The observation now has **20 layers** (was 10). New: each bomb's range and burn, buffs on the floor (one layer per type), my bombs left, my range, my burn, and each opponent's range, bombs left and burn. Everything stays between 0 and 1
* The agent still has **6 actions**
* New reward term `buff` for picking up a buff, only when it really raises a level. **v0 = 0** (off)
* The tabular tiny stage stays without buffs

### Part 4: UI
* Three original buff icons that bob on the floor, a pop when they drop, a sparkle when picked up, a puff when destroyed
* Cells burning longer than normal are drawn as **pink embers**
* The side panel shows each character's levels as pips
* Buffs are drawn above flames, otherwise a fresh drop would be hidden under its own crate's fire

### Part 5 and 6: README and a Windows fix
* README: a new section with the `rl/` commands and the buff rules, plus the sentences that became false (fuse 9, one bomb)
* **Windows bug (from Task 6):** the PPO script never closed its `progress.csv`, and Windows can't delete an open file, so 2 tests failed on Windows only. Now every log file and training env is closed, even if a run crashes. 5 new tests check it

### Decisions I made

- **Fuse 4 everywhere**
  - Why: the bomb reward arrives 4 ticks after placing instead of 9, so the agent learns faster, and one rule set means the agent never plays different rules than it trained on
  - Alternatives considered: fuse 9 for humans and 4 for training (the agent would misjudge bombs in the real game)
  - Status: provisional (needs team OK)

- **Keep the 800 step limit, also in evaluation**
  - Why: better a long game than cutting one off early
  - Status: provisional (needs team OK)

- **Three buffs, speed dropped**
  - Why: speed changes how movement works and brought too many edge cases (two moves per tick, fire on the middle tile, 22 actions). Range, extra bomb and burn only touch bombs
  - Alternatives considered: speed as 2 moves per tick
  - Status: provisional (needs team OK), speed can come back later as an extra

- **Burn 5 then 6 ticks**
  - Why: 8 ticks would trap the agent in its own fire a lot early in training
  - Alternatives considered: 6 then 8 ticks
  - Status: provisional (needs team OK)

- **Drop rates 15% / 15% / 10%**
  - Why: about 6 to 7 buffs per game, more chaotic
  - Status: provisional, to decide again with the trained agent

- **Random tie when two characters take the same buff**
  - Why: lowest id would always favour our agent
  - Status: provisional (needs team OK)

- **Buff reward off (0) for now**
  - Why: first see if the agent picks buffs up on its own
  - Status: provisional (needs team OK)

### ANY OBJECTIONS ON:
* Fuse 4 everywhere and the 800 step limit
* The three buffs and their numbers in `BUFF_RULES`
* Dropping speed
* The 20 observation layers

### AI-assisted parts
* Claude Code wrote all the code and the 104 new tests
* The Claude supervisor wrote the spec, reviewed the plan, and verified the result on its own:
  * the engine with buffs off against the old engine on 20,000 random games
  * its own model of the blast and buff rules compared step by step over 9,000 random games
  * the drop rates and the random tie
  * the observation and reward against its own version on 400 games from all 4 points of view
  * the ladder re-run on the final code
  * the new tests really fail without the Windows fix
* I made the design calls (fuse, buffs, burn, drop rates), and applied and merged the code

### Open questions for the team
* **Extra bomb max 2 or 3?** With fuse 4 you need a new cell for every bomb, so you can place one at most every other tick, and your first bomb explodes 4 ticks after you placed it. A third bomb can only go down on the tick your first one explodes, so level 3 almost does nothing
* **Drop rates:** the buff measurement only measures our simple bot (it can't escape long blasts and almost never uses a second bomb). Decide with the trained agent
* README line 5 still says there's no training code

### Next task for whoever picks this up

**Stage 2 of the project: the first real training runs.** Everything is ready. The idea is to show that our setup actually learns something before trying anything fancy: tabular Q-learning on the tiny stage, and PPO against the easiest opponents first, each with several seeds, evaluated on games it never trained on and logged in `docs/EXPERIMENTS.md`. Expect the first PPO runs to learn to just wait (that's what the smoke runs did), so a big part of this task is finding out how many steps and which settings it needs before it starts bombing crates and surviving. Then climb the ladder.

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
