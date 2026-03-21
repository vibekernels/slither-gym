"""Record video episodes of the trained PPO agent using the original renderer."""

import argparse
import os

import numpy as np
import torch
import imageio.v3 as iio

import slither_gym  # noqa: F401 – registers Slither-v0
import gymnasium as gym

from puffer_rl.model import MLPPolicy, MLPLSTMPolicy, CNNPolicy

# Must match engine.pyx constants
K_FOOD = 16
K_NPC = 8
MAX_SEG = 500
OBS_DIM = 54
VIEWPORT = 200.0
SPATIAL_H = 32
SPATIAL_W = 32
SPATIAL_C = 5


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


def extract_spatial_obs(game_state, config):
    """Extract (5, 32, 32) spatial grid + (3,) scalar obs from game state."""
    player = game_state.player
    spatial = np.zeros((SPATIAL_C, SPATIAL_H, SPATIAL_W), dtype=np.float32)
    scalar = np.zeros(3, dtype=np.float32)

    if not player.alive:
        return spatial, scalar

    px, py = player.head_pos
    d = player.direction
    cd, sd = np.cos(d), np.sin(d)
    vp = VIEWPORT
    cell_size = (2.0 * vp) / SPATIAL_W
    inv_cell = 1.0 / cell_size
    half_h = SPATIAL_H * 0.5
    half_w = SPATIAL_W * 0.5
    ar = config.arena_radius

    # Channel 3: boundary
    for r in range(SPATIAL_H):
        for c in range(SPATIAL_W):
            ego_x = (r - half_h + 0.5) * cell_size
            ego_y = (c - half_w + 0.5) * cell_size
            wx = px + cd * ego_x - sd * ego_y
            wy = py + sd * ego_x + cd * ego_y
            if wx * wx + wy * wy > ar * ar:
                spatial[3, r, c] = 1.0

    # Channel 0: food
    food = game_state.food
    active_mask = food.active
    if np.any(active_mask):
        fpos = food.positions[active_mask]
        dx = fpos[:, 0] - px
        dy = fpos[:, 1] - py
        ego_x = cd * dx + sd * dy
        ego_y = -sd * dx + cd * dy
        gr = (ego_x * inv_cell + half_h).astype(int)
        gc = (ego_y * inv_cell + half_w).astype(int)
        mask = (gr >= 0) & (gr < SPATIAL_H) & (gc >= 0) & (gc < SPATIAL_W)
        spatial[0, gr[mask], gc[mask]] = 1.0

    # Channel 1: own body
    segs = player.active_segments()
    if len(segs) > 0:
        dx = segs[:, 0] - px
        dy = segs[:, 1] - py
        ego_x = cd * dx + sd * dy
        ego_y = -sd * dx + cd * dy
        gr = (ego_x * inv_cell + half_h).astype(int)
        gc = (ego_y * inv_cell + half_w).astype(int)
        mask = (gr >= 0) & (gr < SPATIAL_H) & (gc >= 0) & (gc < SPATIAL_W)
        spatial[1, gr[mask], gc[mask]] = 1.0

    # Channel 4: own head
    spatial[4, SPATIAL_H // 2, SPATIAL_W // 2] = 1.0

    # Channel 2: NPC body
    for snake in game_state.snakes[1:]:
        if not snake.alive:
            continue
        segs = snake.active_segments()
        if len(segs) == 0:
            continue
        dx = segs[:, 0] - px
        dy = segs[:, 1] - py
        ego_x = cd * dx + sd * dy
        ego_y = -sd * dx + cd * dy
        gr = (ego_x * inv_cell + half_h).astype(int)
        gc = (ego_y * inv_cell + half_w).astype(int)
        mask = (gr >= 0) & (gr < SPATIAL_H) & (gc >= 0) & (gc < SPATIAL_W)
        spatial[2, gr[mask], gc[mask]] = 1.0

    # Scalars
    scalar[0] = player.length / MAX_SEG
    scalar[1] = float(player.boosting)
    scalar[2] = np.sqrt(px * px + py * py) / ar

    return spatial, scalar


def upscale(frame, scale=8):
    return np.kron(frame, np.ones((scale, scale, 1))).astype(np.uint8)


def record_episode(env, model, device, greedy=True, scale=8,
                    use_lstm=False, use_cnn=False):
    """Run one episode, return (frames, return, length, terminated, slen)."""
    obs_rgb, _ = env.reset()
    base = env.unwrapped
    game = base._state
    config = base.game_config

    lstm_state = model.get_initial_state(1, device) if use_lstm else None
    frames = [upscale(obs_rgb, scale)]
    done, ret, length = False, 0.0, 0
    max_frames = 800

    while not done and length < max_frames:
        with torch.no_grad():
            if use_cnn:
                sp, sc = extract_spatial_obs(game, config)
                sp_t = torch.from_numpy(sp).unsqueeze(0).to(device)
                sc_t = torch.from_numpy(sc).unsqueeze(0).to(device)
                logits, value = model.forward(sp_t, sc_t)
            elif use_lstm:
                state_obs = extract_obs(game, config)
                obs_t = torch.from_numpy(state_obs).unsqueeze(0).to(device)
                logits, value, lstm_state = model.forward(obs_t, lstm_state)
            else:
                state_obs = extract_obs(game, config)
                obs_t = torch.from_numpy(state_obs).unsqueeze(0).to(device)
                logits, value = model.forward(obs_t)

        if greedy:
            action = logits.argmax(-1).item()
        else:
            action = torch.distributions.Categorical(logits=logits).sample().item()

        obs_rgb, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ret += reward
        length += 1
        frames.append(upscale(obs_rgb, scale))

    slen = game.player.length if game.player.alive else 0
    return frames, ret, length, terminated, slen


def main():
    p = argparse.ArgumentParser(description="Record PPO agent videos")
    p.add_argument("--checkpoint", default="runs/puffer_90M/final.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", default="videos_ppo")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--lstm", action="store_true")
    p.add_argument("--cnn", action="store_true")
    p.add_argument("--hidden_dim", type=int, default=256)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.cnn:
        model = CNNPolicy(spatial_channels=SPATIAL_C, spatial_h=SPATIAL_H,
                          spatial_w=SPATIAL_W, scalar_dim=3, act_dim=6,
                          hidden_dim=args.hidden_dim).to(device)
    elif args.lstm:
        model = MLPLSTMPolicy(obs_dim=OBS_DIM, act_dim=6).to(device)
    else:
        model = MLPPolicy(obs_dim=OBS_DIM, act_dim=6).to(device)
    # Strip _orig_mod. prefix from torch.compile checkpoints
    sd = ckpt["model"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    kind = "CNN" if args.cnn else ("LSTM" if args.lstm else "MLP")
    print(f"Loaded: {args.checkpoint} ({kind})")

    env = gym.make("Slither-v0")

    for policy in ["greedy", "stochastic"]:
        greedy = policy == "greedy"
        print(f"\n--- {policy} policy ---")
        for ep in range(args.episodes):
            frames, ret, length, terminated, slen = record_episode(
                env, model, device, greedy=greedy, scale=args.scale,
                use_lstm=args.lstm, use_cnn=args.cnn,
            )
            outcome = "died" if terminated else "survived"
            fname = f"{policy}_ep{ep+1}_{outcome}_slen{slen}_ret{ret:.0f}_len{length}.mp4"
            path = os.path.join(args.outdir, fname)
            iio.imwrite(path, frames, fps=args.fps)
            print(f"  {fname}  ({length} frames, slen={slen}, return={ret:.1f})")

    env.close()
    print(f"\nVideos saved to {args.outdir}/")


if __name__ == "__main__":
    main()
