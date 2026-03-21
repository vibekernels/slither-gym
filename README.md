# slither-gym

A slither.io clone as a Gymnasium environment with a built-in DreamerV3 implementation for RL training.

## Overview

- **Game**: Slither.io-style arena — snakes eat food to grow, boost to move faster (shedding mass as pellets), die on collision with other snakes or the arena boundary. 4 NPC bot snakes provide opponents.
- **Observation**: 64×64×3 uint8 RGB image (ego-centric view centered on the player's head)
- **Action**: `Discrete(6)` — straight, turn left, turn right, boost straight, boost left, boost right
- **Reward**: Configurable. Default: +1.5 per food eaten, +10 per kill, −0.1×length on death, −0.005 per step (time pressure), −0.01 per step boosting, −1.5 per boost pellet dropped

## Setup

```bash
git clone <repo-url> && cd slither-gym
pip install -e .
pip install torch tensorboard imageio[ffmpeg]
```

For GPU training, install PyTorch with CUDA (see https://pytorch.org/get-started/locally/):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Training with DreamerV3

```bash
# GPU (recommended)
python train.py --device cuda --steps 500000

# CPU (very slow, only for smoke-testing)
python train.py --device cpu --steps 6000 --prefill 1000 --train_ratio 32 --batch_size 4 --seq_len 16
```

### Training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cpu` | `cpu` or `cuda` |
| `--steps` | 500,000 | Total environment steps |
| `--batch_size` | 32 | Sequences per training batch |
| `--seq_len` | 50 | Sequence length for world model training |
| `--train_ratio` | 512 | Gradient steps per env step (DreamerV3 default) |
| `--prefill` | 5,000 | Random exploration steps before training begins |
| `--save_every` | 50,000 | Checkpoint interval (env steps) |
| `--logdir` | `runs/slither` | TensorBoard log and checkpoint directory |
| `--resume` | None | Path to a checkpoint `.pt` file to resume from |
| `--seed` | 0 | Random seed |
| `--num_envs` | 4 | Parallel envs for async collection |
| `--no_async` | — | Disable async collection (single env) |
| `--no_amp` | — | Disable mixed precision (bf16) |
| `--food_reward` | 1.5 | Reward per food eaten |
| `--kill_reward` | 10.0 | Reward per kill |
| `--death_scale` | 0.1 | Death penalty = −death_scale × snake length |
| `--survival_bonus` | 0.0 | Per-step reward (use negative for time pressure) |

### Monitoring

```bash
tensorboard --logdir runs/
```

Key metrics to watch:
- `episode/return` — total reward per episode (should trend upward)
- `episode/length` — survival time (longer = better)
- `train/recon_loss` — image reconstruction quality (should decrease)
- `train/entropy` — policy entropy (should decrease gradually)

### Checkpoints

Checkpoints are saved to `--logdir` as `checkpoint_<steps>.pt`. Resume training:

```bash
python train.py --device cuda --resume runs/slither/checkpoint_50000.pt
```

## Evaluation

Evaluate a trained checkpoint:

```bash
# Greedy policy (deterministic action selection)
python eval.py --checkpoint runs/slither/checkpoint_final.pt --episodes 50
```

### Recording videos

```bash
python record_videos.py --checkpoint runs/slither/checkpoint_final.pt --episodes 5
```

Records MP4 videos for greedy, stochastic, and random policies into `videos/`.

## Human play

You can play the game manually with keyboard controls:

```bash
pip install pygame
python examples/human_play.py
```

| Key | Action |
|-----|--------|
| Left Arrow / A | Turn left |
| Right Arrow / D | Turn right |
| Space | Boost |
| (no key) | Go straight |
| R | Restart after death |
| ESC / Q | Quit |

## Using the environment directly

```python
import slither_gym
import gymnasium as gym

env = gym.make("Slither-v0")
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Environment constructor options

```python
env = gym.make(
    "Slither-v0",
    render_mode="human",     # or "rgb_array" (default: None)
    num_npcs=4,              # number of bot opponents
    arena_radius=1000.0,     # world size
    max_steps=4000,          # episode truncation limit
    obs_size=64,             # observation image size (NxN)
    viewport_radius=120.0,   # visible world radius around the player
)
```

Human rendering requires pygame: `pip install pygame`

## Project structure

```
slither_gym/
├── engine/          # Game simulation (NumPy-based)
│   ├── config.py    #   Game parameters
│   ├── snake.py     #   Snake with ring-buffer segments
│   ├── food.py      #   Food spawning and consumption
│   └── game.py      #   Main game loop, NPCs, collisions
├── env/             # Gymnasium interface
│   ├── slither_env.py  # Env subclass
│   └── rewards.py      # Configurable reward function
├── rendering/       # Observation rendering
│   └── numpy_renderer.py  # Ego-centric 64x64 RGB rasterizer
└── dreamer/         # DreamerV3 implementation (PyTorch)
    ├── networks.py     # RSSM, CNN encoder/decoder, actor, critic
    ├── agent.py        # Training logic (world model + actor-critic)
    └── replay_buffer.py
```
