"""Record video episodes of the trained PPO agent using the original renderer."""

import argparse
import os

import numpy as np
import torch
import imageio.v3 as iio

import slither_gym  # noqa: F401 – registers Slither-v0
import gymnasium as gym

from puffer_rl.model import MLPLSTMPolicy

# Must match engine.pyx constants
K_FOOD = 16
K_NPC = 8
MAX_SEG = 200
OBS_DIM = 54
VIEWPORT = 200.0


def extract_obs(game_state, config):
    """Extract the same 54-dim observation the Cython engine produces."""
    player = game_state.player
    if not player.alive:
        return np.zeros(OBS_DIM, dtype=np.float32)

    px, py = player.head_pos
    d = player.direction
    cd, sd = np.cos(d), np.sin(d)
    inv_vp = 1.0 / VIEWPORT
    inv_ar = 1.0 / config.arena_radius
    inv_ml = 1.0 / MAX_SEG

    obs = np.zeros(OBS_DIM, dtype=np.float32)

    # Player state (6)
    obs[0] = player.length * inv_ml
    obs[1] = 1.0 if player.boosting else 0.5
    obs[2] = sd
    obs[3] = cd
    obs[4] = np.sqrt(px * px + py * py) * inv_ar
    obs[5] = float(player.boosting)

    # K nearest foods in ego frame (K_FOOD * 2)
    food = game_state.food
    active_mask = food.active
    if np.any(active_mask):
        fpos = food.positions[active_mask]
        dx = fpos[:, 0] - px
        dy = fpos[:, 1] - py
        dists = dx * dx + dy * dy
        k = min(K_FOOD, len(dists))
        idx = np.argpartition(dists, k)[:k]
        idx = idx[np.argsort(dists[idx])]
        for i, fi in enumerate(idx):
            ego_x = cd * dx[fi] + sd * dy[fi]
            ego_y = -sd * dx[fi] + cd * dy[fi]
            obs[6 + i * 2] = ego_x * inv_vp
            obs[6 + i * 2 + 1] = ego_y * inv_vp

    # K nearest NPC segments in ego frame (K_NPC * 2)
    all_segs = []
    for snake in game_state.snakes[1:]:
        if not snake.alive:
            continue
        segs = snake.active_segments()
        all_segs.append(segs)

    if all_segs:
        all_segs = np.concatenate(all_segs, axis=0)
        dx = all_segs[:, 0] - px
        dy = all_segs[:, 1] - py
        dists = dx * dx + dy * dy
        k = min(K_NPC, len(dists))
        idx = np.argpartition(dists, k)[:k]
        idx = idx[np.argsort(dists[idx])]
        off = 6 + K_FOOD * 2
        for i, si in enumerate(idx):
            ego_x = cd * dx[si] + sd * dy[si]
            ego_y = -sd * dx[si] + cd * dy[si]
            obs[off + i * 2] = ego_x * inv_vp
            obs[off + i * 2 + 1] = ego_y * inv_vp

    return obs


def upscale(frame, scale=8):
    return np.kron(frame, np.ones((scale, scale, 1))).astype(np.uint8)


def record_episode(env, model, device, greedy=True, scale=8):
    """Run one episode, return (frames, return, length, terminated)."""
    obs_rgb, _ = env.reset()
    # Unwrap TimeLimit / other wrappers to get the base SlitherEnv
    base = env.unwrapped
    game = base._state
    config = base.game_config

    lstm_state = model.get_initial_state(1, device)
    frames = [upscale(obs_rgb, scale)]
    done, ret, length = False, 0.0, 0

    while not done:
        state_obs = extract_obs(game, config)
        obs_t = torch.from_numpy(state_obs).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, value, lstm_state = model.forward(obs_t, lstm_state)

        if greedy:
            action = logits.argmax(-1).item()
        else:
            action = torch.distributions.Categorical(logits=logits).sample().item()

        obs_rgb, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ret += reward
        length += 1
        frames.append(upscale(obs_rgb, scale))

    return frames, ret, length, terminated


def main():
    p = argparse.ArgumentParser(description="Record PPO agent videos")
    p.add_argument("--checkpoint", default="runs/puffer_gpu/final.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", default="videos_ppo")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MLPLSTMPolicy(obs_dim=OBS_DIM, act_dim=6).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    env = gym.make("Slither-v0")

    for policy in ["greedy", "stochastic"]:
        greedy = policy == "greedy"
        print(f"\n--- {policy} policy ---")
        for ep in range(args.episodes):
            frames, ret, length, terminated = record_episode(
                env, model, device, greedy=greedy, scale=args.scale
            )
            outcome = "died" if terminated else "survived"
            fname = f"{policy}_ep{ep+1}_{outcome}_ret{ret:.0f}_len{length}.mp4"
            path = os.path.join(args.outdir, fname)
            iio.imwrite(path, frames, fps=args.fps)
            print(f"  {fname}  ({length} frames, return={ret:.1f})")

    env.close()
    print(f"\nVideos saved to {args.outdir}/")


if __name__ == "__main__":
    main()
