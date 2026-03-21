#!/usr/bin/env python3
"""Train DreamerV3 on Slither-v0.

Usage:
    python train.py                          # CPU (slow, for testing)
    python train.py --device cuda            # GPU (recommended)
    python train.py --device cuda --steps 1000000 --logdir runs/slither

Logs to TensorBoard. View with: tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

import slither_gym  # noqa: F401
import gymnasium as gym

from slither_gym.dreamer.agent import DreamerV3Agent
from slither_gym.dreamer.replay_buffer import ReplayBuffer
from slither_gym.env.rewards import RewardConfig


def make_env(reward_config: RewardConfig):
    """Factory that returns a function creating a Slither env (for AsyncVectorEnv)."""
    def _make():
        return gym.make("Slither-v0", reward_config=reward_config)
    return _make


class AsyncCollector:
    """Collects episodes from N parallel envs in a background thread.

    Puts completed episodes into a queue that the training loop consumes.
    Uses gymnasium.vector.AsyncVectorEnv for subprocess-based parallelism:
    env stepping happens in separate processes, overlapping with GPU training.
    """

    def __init__(
        self,
        agent: DreamerV3Agent,
        reward_config: RewardConfig,
        num_envs: int = 4,
        action_dim: int = 6,
        max_queue_size: int = 16,
    ):
        self.agent = agent
        self.action_dim = action_dim
        self.num_envs = num_envs
        self.episode_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop = threading.Event()
        self._total_steps = 0
        self._lock = threading.Lock()  # protects agent.batch_act() vs training

        # Each env tracks its own episode buffers
        self._obs_bufs: list[list] = [[] for _ in range(num_envs)]
        self._act_bufs: list[list] = [[] for _ in range(num_envs)]
        self._rew_bufs: list[list] = [[] for _ in range(num_envs)]
        self._cont_bufs: list[list] = [[] for _ in range(num_envs)]

        self.vec_env = gym.vector.AsyncVectorEnv(
            [make_env(reward_config) for _ in range(num_envs)]
        )

        self._thread: threading.Thread | None = None

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def lock(self):
        return self._lock

    def start(self):
        """Start the background collection thread."""
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the collector to stop and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def get_episodes(self, timeout: float = 0.01) -> list[dict]:
        """Drain all available completed episodes from the queue."""
        episodes = []
        while True:
            try:
                episodes.append(self.episode_queue.get_nowait())
            except queue.Empty:
                break
        return episodes

    def _collect_loop(self):
        """Background loop: step all envs, detect episode boundaries, queue results."""
        obs_all, infos = self.vec_env.reset()
        # Initialize agent state for N envs
        with self._lock:
            self.agent.init_state(self.num_envs)

        while not self._stop.is_set():
            # Get batched actions from agent (quick GPU inference)
            with self._lock:
                actions = self.agent.batch_act(obs_all, training=True)

            # One-hot encode actions for each env
            for i in range(self.num_envs):
                action_oh = np.zeros(self.action_dim, dtype=np.float32)
                action_oh[actions[i]] = 1.0
                obs_t = np.transpose(obs_all[i], (2, 0, 1))
                self._obs_bufs[i].append(obs_t)
                self._act_bufs[i].append(action_oh)

            # Step all envs (this runs in subprocesses — CPU work overlaps with GPU)
            obs_all, rewards, terminateds, truncateds, infos = self.vec_env.step(actions)

            for i in range(self.num_envs):
                self._rew_bufs[i].append(np.float32(rewards[i]))
                self._cont_bufs[i].append(np.float32(0.0 if terminateds[i] else 1.0))

                # Episode done?
                if terminateds[i] or truncateds[i]:
                    ep = {
                        "obs": np.stack(self._obs_bufs[i]),
                        "action": np.stack(self._act_bufs[i]),
                        "reward": np.array(self._rew_bufs[i], dtype=np.float32),
                        "cont": np.array(self._cont_bufs[i], dtype=np.float32),
                    }
                    self._total_steps += len(ep["reward"])
                    # Clear buffers
                    self._obs_bufs[i] = []
                    self._act_bufs[i] = []
                    self._rew_bufs[i] = []
                    self._cont_bufs[i] = []
                    # Reset agent state for this env index
                    with self._lock:
                        self.agent.reset_state_at(i)

                    # Queue episode (block if queue is full — backpressure)
                    try:
                        self.episode_queue.put(ep, timeout=5.0)
                    except queue.Full:
                        pass  # drop episode if training can't keep up

    def close(self):
        self.stop()
        self.vec_env.close()


def collect_random_episodes(reward_config: RewardConfig, action_dim: int, target_steps: int):
    """Collect random episodes for prefill. Returns list of episode dicts."""
    env = gym.make("Slither-v0", reward_config=reward_config)
    episodes = []
    total = 0
    while total < target_steps:
        obs_list, act_list, rew_list, cont_list = [], [], [], []
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            action_oh = np.zeros(action_dim, dtype=np.float32)
            action_oh[action] = 1.0
            obs_list.append(np.transpose(obs, (2, 0, 1)))
            act_list.append(action_oh)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rew_list.append(np.float32(reward))
            cont_list.append(np.float32(0.0 if terminated else 1.0))
            total += 1
        episodes.append({
            "obs": np.stack(obs_list),
            "action": np.stack(act_list),
            "reward": np.array(rew_list, dtype=np.float32),
            "cont": np.array(cont_list, dtype=np.float32),
        })
    env.close()
    return episodes, total


def collect_episode(env: gym.Env, agent: DreamerV3Agent, action_dim: int) -> dict:
    """Collect one full episode (single env, for fallback/eval)."""
    obs_list, act_list, rew_list, cont_list = [], [], [], []
    obs, info = env.reset()
    agent.init_state(1)
    done = False
    while not done:
        action = agent.act(obs, training=True)
        action_oh = np.zeros(action_dim, dtype=np.float32)
        action_oh[action] = 1.0
        obs_list.append(np.transpose(obs, (2, 0, 1)))
        act_list.append(action_oh)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rew_list.append(np.float32(reward))
        cont_list.append(np.float32(0.0 if terminated else 1.0))
    return {
        "obs": np.stack(obs_list),
        "action": np.stack(act_list),
        "reward": np.array(rew_list, dtype=np.float32),
        "cont": np.array(cont_list, dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description="Train DreamerV3 on Slither-v0")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--steps", type=int, default=500_000, help="Total env steps")
    parser.add_argument("--logdir", type=str, default="runs/slither", help="TensorBoard log dir")
    parser.add_argument("--save_every", type=int, default=50_000, help="Save checkpoint every N steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length for training")
    parser.add_argument("--train_ratio", type=int, default=512, help="Train steps per env step ratio (as in DreamerV3)")
    parser.add_argument("--prefill", type=int, default=5000, help="Random steps before training starts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--food_reward", type=float, default=1.0, help="Reward per food eaten")
    parser.add_argument("--kill_reward", type=float, default=5.0, help="Reward per kill")
    parser.add_argument("--death_scale", type=float, default=0.1, help="Death penalty = -death_scale * length")
    parser.add_argument("--survival_bonus", type=float, default=0.0, help="Per-step survival reward")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs for collection")
    parser.add_argument("--no_async", action="store_true", help="Disable async collection (use single env)")
    parser.add_argument("--rssm_type", type=str, default="gru", choices=["gru", "mamba"],
                        help="RSSM type: gru (default) or mamba (selective SSM)")
    args = parser.parse_args()

    # Setup
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(logdir))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    reward_config = RewardConfig(
        food_eaten=args.food_reward,
        kill_opponent=args.kill_reward,
        death_scale=args.death_scale,
        survival_per_step=args.survival_bonus,
    )

    action_dim = 6  # Discrete(6) — read from env below

    agent = DreamerV3Agent(
        action_dim=action_dim,
        device=args.device,
        use_amp=not args.no_amp,
        compile_models=False,  # disabled: incompatible with multi-thread inference
        rssm_type=args.rssm_type,
    )

    resumed_env_steps = 0
    resumed_train_steps = 0
    if args.resume:
        resumed_env_steps, resumed_train_steps = agent.load(args.resume)
        print(f"Resumed from {args.resume} (env_steps={resumed_env_steps}, train_steps={resumed_train_steps})")

    buffer = ReplayBuffer(capacity=1_000_000, seq_len=args.seq_len)

    # --- Prefill with random actions ---
    print(f"Prefilling buffer with {args.prefill} steps of random actions...")
    prefill_eps, prefill_steps = collect_random_episodes(reward_config, action_dim, args.prefill)
    for ep in prefill_eps:
        buffer.add_episode(ep)
    print(f"Prefilled {buffer.total_steps} steps across {len(buffer._episodes)} episodes")

    # --- Main training loop ---
    total_env_steps = resumed_env_steps + buffer.total_steps
    train_steps = resumed_train_steps
    episode_count = 0
    last_save = 0
    pending_env_steps = 0  # steps collected but not yet trained on
    start_time = time.time()

    use_async = not args.no_async and args.device != "cpu"
    amp_status = "ON (bf16)" if agent.use_amp else "OFF"
    print(f"\nStarting training on {args.device} for {args.steps} env steps...")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Batch size: {args.batch_size}, Sequence length: {args.seq_len}")
    print(f"Mixed precision: {amp_status}")
    print(f"Collection: {'async (' + str(args.num_envs) + ' envs)' if use_async else 'sync (1 env)'}")
    print(f"Logging to: {logdir}\n")

    if use_async:
        collector = AsyncCollector(
            agent=agent,
            reward_config=reward_config,
            num_envs=args.num_envs,
            action_dim=action_dim,
        )
        collector.start()

        while total_env_steps < args.steps:
            # Grab any completed episodes
            new_episodes = collector.get_episodes()

            if not new_episodes and pending_env_steps == 0:
                # No episodes yet and nothing to train on — wait a bit
                time.sleep(0.05)
                continue

            # Process new episodes
            for ep in new_episodes:
                ep_len = len(ep["reward"])
                ep_return = ep["reward"].sum()
                episode_count += 1
                total_env_steps += ep_len
                pending_env_steps += ep_len
                buffer.add_episode(ep)

                elapsed = time.time() - start_time
                sps = total_env_steps / elapsed if elapsed > 0 else 0
                print(f"Episode {episode_count:>5d} | "
                      f"Steps: {total_env_steps:>8d}/{args.steps} | "
                      f"Return: {ep_return:>7.2f} | "
                      f"Length: {ep_len:>4d} | "
                      f"SPS: {sps:.0f}")

                writer.add_scalar("episode/return", ep_return, total_env_steps)
                writer.add_scalar("episode/length", ep_len, total_env_steps)
                writer.add_scalar("performance/sps", sps, total_env_steps)

            # Train: proportional to accumulated env steps
            if pending_env_steps > 0:
                n_train = max(1, pending_env_steps * args.train_ratio // (args.batch_size * args.seq_len))
                # Cap per iteration to keep collection responsive
                n_train = min(n_train, 64)

                with collector.lock:
                    for _ in range(n_train):
                        batch = buffer.sample(args.batch_size)
                        metrics = agent.train_step(batch)
                        train_steps += 1

                # Deduct trained portion
                trained_steps = n_train * args.batch_size * args.seq_len // args.train_ratio
                pending_env_steps = max(0, pending_env_steps - trained_steps)

                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", v, total_env_steps)
                writer.add_scalar("train/train_steps", train_steps, total_env_steps)

            # Save checkpoint
            if total_env_steps - last_save >= args.save_every:
                ckpt_path = logdir / f"checkpoint_{total_env_steps}.pt"
                with collector.lock:
                    agent.save(str(ckpt_path), env_steps=total_env_steps, train_steps=train_steps)
                print(f"  -> Saved checkpoint: {ckpt_path}")
                last_save = total_env_steps

        collector.close()

    else:
        # Synchronous single-env fallback
        env = gym.make("Slither-v0", reward_config=reward_config)
        while total_env_steps < args.steps:
            episode = collect_episode(env, agent, action_dim)
            ep_len = len(episode["reward"])
            ep_return = episode["reward"].sum()
            episode_count += 1
            total_env_steps += ep_len
            buffer.add_episode(episode)

            elapsed = time.time() - start_time
            sps = total_env_steps / elapsed if elapsed > 0 else 0
            print(f"Episode {episode_count:>5d} | "
                  f"Steps: {total_env_steps:>8d}/{args.steps} | "
                  f"Return: {ep_return:>7.2f} | "
                  f"Length: {ep_len:>4d} | "
                  f"SPS: {sps:.0f}")

            writer.add_scalar("episode/return", ep_return, total_env_steps)
            writer.add_scalar("episode/length", ep_len, total_env_steps)
            writer.add_scalar("performance/sps", sps, total_env_steps)

            n_train = max(1, ep_len * args.train_ratio // (args.batch_size * args.seq_len))
            for _ in range(n_train):
                batch = buffer.sample(args.batch_size)
                metrics = agent.train_step(batch)
                train_steps += 1

            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", v, total_env_steps)
            writer.add_scalar("train/train_steps", train_steps, total_env_steps)

            if total_env_steps - last_save >= args.save_every:
                ckpt_path = logdir / f"checkpoint_{total_env_steps}.pt"
                agent.save(str(ckpt_path), env_steps=total_env_steps, train_steps=train_steps)
                print(f"  -> Saved checkpoint: {ckpt_path}")
                last_save = total_env_steps
        env.close()

    # Final save
    agent.save(str(logdir / "checkpoint_final.pt"), env_steps=total_env_steps, train_steps=train_steps)
    print(f"\nTraining complete! {total_env_steps} env steps, {train_steps} train steps")
    print(f"Final checkpoint: {logdir / 'checkpoint_final.pt'}")
    writer.close()


if __name__ == "__main__":
    main()
