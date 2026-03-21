#!/usr/bin/env python3
"""Record video episodes of the trained DreamerV3 agent and a random baseline."""

import argparse
import os
import numpy as np
import torch
import imageio.v3 as iio
import slither_gym
import gymnasium as gym
from slither_gym.dreamer.agent import DreamerV3Agent


def upscale(frame: np.ndarray, scale: int = 8) -> np.ndarray:
    """Nearest-neighbor upscale a small frame for visibility."""
    return np.kron(frame, np.ones((scale, scale, 1))).astype(np.uint8)


def record_episode(env, agent, training: bool, scale: int = 8):
    """Run one episode, return (frames, return, length, terminated)."""
    obs, _ = env.reset()
    if agent is not None:
        agent.init_state(1)
    frames = [upscale(obs, scale)]
    done, ret, length = False, 0.0, 0

    while not done:
        if agent is not None:
            action = agent.act(obs, training=training)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ret += reward
        length += 1
        frames.append(upscale(obs, scale))

    return frames, ret, length, terminated


def main():
    parser = argparse.ArgumentParser(description="Record agent play videos")
    parser.add_argument("--checkpoint", default="runs/slither/checkpoint_final.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outdir", default="videos")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per policy type")
    parser.add_argument("--scale", type=int, default=8,
                        help="Upscale factor (64px * scale)")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    env = gym.make("Slither-v0")

    # Load trained agent
    agent = DreamerV3Agent(action_dim=env.action_space.n, device=args.device, compile_models=False)
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}\n")

    policies = [
        ("greedy", agent, False),
        ("stochastic", agent, True),
        ("random", None, False),
    ]

    for policy_name, ag, training in policies:
        print(f"--- Recording {args.episodes} episodes: {policy_name} policy ---")
        for ep in range(args.episodes):
            frames, ret, length, terminated = record_episode(
                env, ag, training, args.scale
            )
            outcome = "died" if terminated else "survived"
            fname = f"{policy_name}_ep{ep+1}_{outcome}_ret{ret:.0f}_len{length}.mp4"
            path = os.path.join(args.outdir, fname)
            iio.imwrite(path, frames, fps=args.fps)
            print(f"  {fname}  ({length} frames)")

    env.close()
    print(f"\nAll videos saved to {args.outdir}/")


if __name__ == "__main__":
    main()
