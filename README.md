# slither-gym

A slither.io clone with high-performance RL training using PPO, a Cython vectorized engine, and CNN policies.

## Overview

- **Game**: Slither.io-style arena — snakes eat food to grow, boost to move faster (shedding mass), die on collision with other snakes or the arena boundary. 4 NPC bot snakes provide opponents.
- **Observation**: Ego-centric RGB grid (32x32 default) rendered by the Cython engine — food is green, own body is blue, NPCs are red, arena boundary is gray, head is white. Plus 3 scalar features (length, boosting, distance to edge).
- **Action**: `Discrete(6)` — straight, turn left, turn right, boost straight, boost left, boost right
- **Reward**: Delta snake length (pure length gain/loss per step)

## Results

CNN with ego-centric spatial observations dramatically outperforms vector-based approaches:

| Model | Observation | Peak snake length |
|-------|------------|-------------------|
| MLP | 54-dim vector (K-nearest) | 110 |
| MLP-LSTM | 54-dim vector | 110 |
| MLP + frame stack | 54-dim x 4 frames | 110 |
| CNN | 5-channel 32x32 grid | 315 |
| **CNN (RGB)** | **3-channel 32x32 grid** | **315** |

The spatial grid preserves geometric relationships that the K-nearest-neighbor vector observation destroys, enabling nearly 3x better performance.

Training throughput: ~10,000 SPS (CNN) on a single GPU with 256 parallel environments.

## Setup

```bash
git clone <repo-url> && cd slither-gym
pip install -e .
pip install torch tensorboard imageio[ffmpeg] cython numpy
```

Build the Cython engine:

```bash
python setup_puffer.py build_ext --inplace
```

For GPU training, install PyTorch with CUDA (see https://pytorch.org/get-started/locally/).

## Training

Use `PYTHONUNBUFFERED=1` so training metrics print in real time:

```bash
# RGB CNN (recommended)
PYTHONUNBUFFERED=1 python -m puffer_rl.train --device cuda --compile --rgb --rgb_res 32 \
    --total_steps 30000000 --num_envs 256 --rollout_len 128 \
    --hidden_dim 256 --ent_coef 0.03 --anneal_lr \
    --logdir runs/rgb_cnn

# 5-channel CNN (pre-separated entity channels)
PYTHONUNBUFFERED=1 python -m puffer_rl.train --device cuda --compile --cnn \
    --total_steps 30000000 --num_envs 256 --rollout_len 128 \
    --hidden_dim 256 --ent_coef 0.03 --anneal_lr \
    --logdir runs/spatial_cnn

# MLP baseline (vector observations)
PYTHONUNBUFFERED=1 python -m puffer_rl.train --device cuda --compile \
    --total_steps 5000000 --num_envs 256 --rollout_len 128 \
    --logdir runs/mlp_baseline
```

### Training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cpu` | `cpu` or `cuda` |
| `--total_steps` | 5,000,000 | Total environment steps |
| `--num_envs` | 256 | Parallel environments |
| `--rollout_len` | 128 | Steps per rollout |
| `--lr` | 2.5e-4 | Learning rate |
| `--anneal_lr` | off | Linear LR annealing to 0 |
| `--gamma` | 0.99 | Discount factor |
| `--clip_ratio` | 0.2 | PPO clip ratio |
| `--ppo_epochs` | 4 | PPO epochs per update |
| `--ent_coef` | 0.01 | Entropy bonus coefficient |
| `--hidden_dim` | 256 | Hidden layer size |
| `--rgb` | off | Use RGB ego-centric rendering |
| `--rgb_res` | 32 | RGB observation resolution |
| `--cnn` | off | Use 5-channel spatial grid |
| `--lstm` | off | Use MLP-LSTM (vector obs) |
| `--compile` | off | Enable torch.compile |
| `--save_every` | 5,000,000 | Checkpoint interval |
| `--logdir` | `runs/puffer_ppo` | Output directory |
| `--checkpoint` | None | Resume from checkpoint |

### Monitoring

```bash
tensorboard --logdir runs/
```

### Recording videos

```bash
# CNN checkpoint
python -m puffer_rl.record_video --checkpoint runs/rgb_cnn/final.pt \
    --cnn --device cuda --outdir videos --episodes 3

# MLP checkpoint
python -m puffer_rl.record_video --checkpoint runs/mlp/final.pt \
    --outdir videos --episodes 3
```

## Architecture

### Engine (`puffer_rl/engine.pyx`)

Cython vectorized game engine running N environments in parallel with OpenMP. Supports three observation modes:
- **Vector**: 54-dim flat observation (6 player state + 16 nearest food + 8 nearest NPC segments)
- **Spatial**: (5, 32, 32) binary grid with separate channels for food, own body, NPC body, boundary, head
- **RGB**: (3, H, W) float32 ego-centric rendering with color-coded entities

### Models (`puffer_rl/model.py`)

- **MLPPolicy**: 3-layer MLP for vector observations
- **MLPLSTMPolicy**: MLP encoder + LSTM for temporal context
- **CNNPolicy**: Conv2d encoder + scalar side-channel for spatial/RGB observations

### Training (`puffer_rl/train.py`)

PPO with:
- CPU inference during collection (no GPU transfers per step)
- GPU-accelerated GAE computation
- torch.compile and bf16 mixed precision
- Optional async double-buffer collection

## Human play

```bash
pip install pygame
python examples/human_play.py
```

## Gymnasium interface

The original Gymnasium environment is also available for integration with other RL libraries:

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

## Project structure

```
puffer_rl/
├── engine.pyx         # Cython vectorized game engine (OpenMP)
├── model.py           # MLP, LSTM, and CNN policy networks
├── train.py           # PPO training loop
└── record_video.py    # Video recording from checkpoints

slither_gym/
├── engine/            # Python game simulation
├── env/               # Gymnasium interface
├── rendering/         # Observation rendering
└── dreamer/           # DreamerV3 implementation (legacy)

setup_puffer.py        # Cython build script
```
