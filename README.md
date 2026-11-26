# Bomberman RL

A Bomberman-style game built for our Reinforcement Learning course project, plus the tools to train an RL agent on it. The game is turn-based underneath: every character makes one move per tick. The rules engine (`game/environment.py`) runs without any window, so an agent can play thousands of games fast, while the retro game window (`ui/`) lets humans play the same rules in real time.


## Install

From this directory, with Python 3.10 or newer:

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Play

From the repository root:

```text
python -m ui
```

The start screen offers three modes:

- **Free World:** you alone on a random map, to try the rules
- **Player vs Bots:** you against 3 bots (Random, Rule or Mixed)
- **Agent Spectator:** placeholder until a trained agent exists

### Controls

| Key | Action |
| --- | --- |
| Arrow keys or WASD | Move (hold to keep walking) |
| Space | Place a bomb |
| Arrows + Enter | Choose in menus |
| Esc | Pause (resume, restart, menu); quit on the start screen |

The game ticks by itself every 250 ms. The last key you pressed during a tick is your move; no key means you wait. Each game starts with a short READY… GO!

## The game

- **Map:** 11×11 with fixed walls and pillars. Crates are either the fixed layout or placed randomly (18 crates) from a seed, always keeping the four corners clear.
- **Characters:** the player (or agent) starts at `(1, 1)`; up to three NPCs start in the other corners. Characters can pass through each other and share a cell.
- **Bombs:** one active bomb per character (more with the extra bomb buff). A bomb explodes after 4 ticks (1 s) with a blast of 2 cells in each direction (more with the range buff). Walls stop the blast, crates are destroyed and stop it, and a blast that reaches another bomb sets it off in the same tick (chain reaction). A new bomb blocks movement from the next tick; you can step off your own bomb but not back on.
- **Flames:** a blast keeps burning for 4 ticks (about 1 s). Anyone standing on or walking into a burning cell dies, including the bomb's owner.
- **Gate:** hidden under the center crate at `(5, 5)`. It is revealed when that crate is destroyed and opens once every NPC is dead.
- **Outcome:** **win** by standing on the open gate, **draw** if you die together with the last NPC, **loss** on any other death. Any outcome ends the game.

The real game's timing is defined once as `GAME_RULES` in `game/environment.py` (fuse 4, flames 4, buffs on). `BombermanEnv()` on its own still uses the original values (fuse 4, flames 1, no buffs), which the older tests rely on; the real game is `BombermanEnv(**GAME_RULES)`.

## Using the engine from code

```python
from game.environment import BombermanEnv, GAME_RULES, Action

env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
state = env.reset(seed=42)
state = env.step(Action.RIGHT, {1: Action.BOMB})  # NPCs left out wait
print(state["outcome"], state["events"])
```

`get_state()` returns the board, characters, bombs, burning flames, the gate, the outcome, the rules in use, and the events of the last tick (deaths, destroyed crates, chain reactions, gate changes).

## Bots

`agents/` has two baseline bots: a **random bot** and a **rule bot** that escapes blasts, breaks crates and goes after opponents. Compare them with:

```text
python -m agents.evaluate
```

(Without `--game-rules` the evaluation scripts use the original timing.)

## Buffs

A destroyed crate may drop a buff (never the gate's crate): **range** (+1 blast range, 2 up to 5, 15%), **extra bomb** (+1 bomb on the board at once, 1 up to 3, 15%) or **burn** (level 0 up to 2, 10%: the blast cells within that many tiles of the bomb burn that many ticks longer). Walk onto a buff to pick it up; a buff at its max level is used up and does nothing. A bomb keeps the range and burn its owner had when placing it. A new blast destroys buffs lying in its path, buffs never block anything, and a character's buffs are gone when it dies. All buff numbers are in `BUFF_RULES` in `game/environment.py` (v0).

## Training and evaluation (`rl/`)

Install the RL packages (CPU-only torch first, then the rest):

```text
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

From the repository root:

```text
python -m rl.train_tabular --episodes 3000 --seed 0
python -m rl.train_ppo --timesteps 10000 --seed 0
python -m rl.evaluate_agent --run runs/<run folder> --opponents random
python -m agents.ladder_eval --games 200 --game-rules
python -m agents.buff_eval --games 200
python -m rl.benchmark_ppo --high-priority
python -m rl.run_ppo_seeds experiments/ppo/first_runs.json
python -m rl.run_ppo_seeds --continue runs/<plan folder>
python -m rl.eval_protocol --run runs/<ppo run folder>
python -m rl.plot_ppo runs/<plan folder>
```

`rl.benchmark_ppo` measures how fast PPO trains on this computer and prints the settings and step counts for the long runs. `rl.run_ppo_seeds` trains and evaluates a plan of PPO runs (several seeds) and summarises them with 95% confidence intervals; after a stop, `--continue` finishes only what is left. `rl.eval_protocol` evaluates a PPO run against all 5 ladder levels, 200 games each, on the evaluation seeds (from 1,000,000; training never plays them). `rl.plot_ppo` draws the learning curves from the evaluation during training.

Training and agent evaluation write a new folder in `runs/` (config, seed, git commit, metrics, model). `rl/config.py` holds every training value (all v0); any script takes `--config overrides.json`. `agents.ladder_eval` checks the opponent ladder, `agents.buff_eval` measures what each buff is worth to the rule bot.

Two reward terms are 0 in v0: `wait` (per WAIT, or a BOMB that places nothing) and `bomb_survived` (per own bomb that broke at least one crate while the agent is alive). `progress.csv` has a `reward_terms/<term>` column per term (the mean per episode that finished in that rollout), plus `reward_terms/total` and `reward_terms/episodes`. `eval_opponents` is the ladder level of the evaluation during training, when it should differ from the training opponents.

Curriculum: `stages` trains the same model against each stage's opponents in turn, each for a fixed number of steps (whole rollouts, adding up to `total_timesteps`). The first stage can be `"idle"`: three training-only bots that always wait, so such a run must set `eval_opponents`. Every stage but the last ends with `checkpoints/stage<k>_<opponents>_<step>.zip`. `experiments/ppo/curriculum.json` is the curriculum plan (idle, noisy25, noisy10, rule); set its `null` values first:

```text
python -m rl.run_ppo_seeds experiments/ppo/curriculum.json
```

### Training on the GPU

Training runs on the CPU unless you ask for the GPU (`PPOConfig.device`, `"cpu"` by default). Only training uses the GPU: evaluation and loading a saved model always run on the CPU, whichever device trained it.

Check whether the GPU can be used:

```text
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`nvidia-smi` must list the GPU and its driver, and the Python line must print a CUDA version and `True`. `None` for `torch.version.cuda` means a CPU-only build of torch. To install a CUDA build on Windows, take the exact command from the official selector on [pytorch.org](https://pytorch.org/get-started/locally/) (Stable, Windows, Pip, Python, the CUDA version that suits your driver), run it inside the project's `.venv`, then `pip install -r requirements.txt` again.

Benchmark CPU and GPU (the GPU part runs only when CUDA is available, otherwise it is skipped and says so), then train on the GPU:

```text
python -m rl.benchmark_ppo --high-priority
python -m rl.train_ppo --timesteps 10000 --seed 0 --device cuda
```

In a plan for `rl.run_ppo_seeds`, set `"device": "cuda"` in `config`. Asking for `cuda` when CUDA is not available stops before anything starts, with the checks above. `meta.json` records the device used and the GPU. GPU runs are not bit-identical to CPU runs with the same seed, and not guaranteed bit-identical between reruns on the GPU.

### Continuing training from a saved model

A run can start from the weights and optimizer state of a saved model (`model.zip` or any checkpoint) instead of fresh weights. Every other setting comes from the new run's config, so rewards, `ent_coef`, opponents or stages can change between pieces; only the network sizes (`conv_channels`, `features_dim`, `net_arch_pi`, `net_arch_vf`) must match, which is checked before anything starts. The new run gets its own folder and counts its steps from 0; `meta.json` records the source file, its SHA-256 and the step it was saved at. `env_seed_offset` shifts the seeds of the training games, so the next piece does not replay the same games:

```json
{"init_model": "runs/<run folder>/checkpoints/model_2000000.zip", "env_seed_offset": 1000, "ent_coef": 0.02}
```

```text
python -m rl.train_ppo --config continue.json --seed 0
```

In a plan, `start_from` starts each seed from the same seed's model in an earlier plan folder. `file` is `"model"`, `"latest_checkpoint"`, `{"stage": k}` or `{"step": N}`; a stopped run's checkpoints work too:

```json
"start_from": {"plan_dir": "runs/<plan folder>", "variant": "curriculum", "file": "latest_checkpoint"}
```

Every seed's file is found and checked before the plan starts, `plan.json` records it, and `--continue` restarts from the same file. Paths in a plan file are relative to the repository root. Only load model files from your own runs (or from a teammate you trust): loading one unpickles it, which can run code.

## Tests

```text
python -m unittest discover -s tests -v
```

The UI tests are skipped automatically if pygame is not installed.

## Layout

- `game/environment.py`: the rules engine (board, movement, bombs, flames, gate, outcome, `reset`, `step`)
- `game/play.py`: the original simple debug client (old timing, single player)
- `ui/`: the retro game window (`python -m ui`)
- `agents/`: baseline bots, match runner and evaluation
- `tests/`: all tests
- `docs/history.md`: what each of us did and why
- `docs/WORKFLOW.md`: our checklist for every work session
