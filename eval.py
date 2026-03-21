#!/usr/bin/env python3
"""Evaluate a trained DreamerV3 agent on Slither-v0."""

import argparse
import numpy as np
import torch
import slither_gym
import gymnasium as gym
from slither_gym.dreamer.agent import DreamerV3Agent


def evaluate(checkpoint_path: str, num_episodes: int = 50, device: str = "cuda",
             frame_stack: int = 1):
    env = gym.make("Slither-v0")
    agent = DreamerV3Agent(action_dim=env.action_space.n, device=device, compile_models=False,
                           frame_stack=frame_stack)
    agent.load(checkpoint_path)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Running {num_episodes} evaluation episodes (greedy policy)...\n")

    returns, lengths, peak_sizes, kills, deaths = [], [], [], [], []

    for ep in range(num_episodes):
        obs, info = env.reset()
        agent.init_state(1)
        done = False
        ep_return = 0.0
        ep_len = 0
        ep_kills = 0
        ep_death = False
        max_size = info.get("length", 10)

        while not done:
            action = agent.act(obs, training=False)  # greedy
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_len += 1
            max_size = max(max_size, info.get("length", 0))

            if reward >= 4.5:
                ep_kills += 1
            if terminated:
                ep_death = True

        final_size = info.get("length", 0)
        returns.append(ep_return)
        lengths.append(ep_len)
        peak_sizes.append(max_size)
        kills.append(ep_kills)
        deaths.append(1 if ep_death else 0)

        print(f"  Episode {ep+1:3d}: return={ep_return:7.2f}  length={ep_len:5d}  "
              f"size={final_size:3d} (peak {max_size:3d})  kills={ep_kills:2d}  "
              f"{'DIED' if ep_death else 'SURVIVED'}")

    # Summary statistics
    returns = np.array(returns)
    lengths = np.array(lengths)
    peak_sizes = np.array(peak_sizes)
    kills = np.array(kills)
    deaths = np.array(deaths)

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY ({num_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Return:     mean={returns.mean():7.2f}  std={returns.std():7.2f}  "
          f"min={returns.min():7.2f}  max={returns.max():7.2f}")
    print(f"Length:     mean={lengths.mean():7.1f}  std={lengths.std():7.1f}  "
          f"min={lengths.min():5d}    max={lengths.max():5d}")
    print(f"Peak size:  mean={peak_sizes.mean():5.1f}  max={peak_sizes.max():3d}")
    print(f"Kills/ep:   mean={kills.mean():5.2f}")
    print(f"Death rate:  {deaths.mean()*100:.1f}%")
    print(f"Survival rate: {(1-deaths.mean())*100:.1f}%  "
          f"(survived to truncation at 4000 steps)")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/slither/checkpoint_final.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame_stack", type=int, default=1,
                        help="Must match training config")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.episodes, args.device, args.frame_stack)
