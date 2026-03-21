"""Simple replay buffer that stores episodes and samples fixed-length sequences."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Stores complete episodes and samples contiguous subsequences for training.

    Observations are stored and sampled as uint8 (4x less memory than float32).
    Sampled batches are placed in pinned memory for async CPU→GPU DMA transfers.
    """

    def __init__(self, capacity: int = 1_000_000, seq_len: int = 50,
                 pin_memory: bool = True, frame_stack: int = 1):
        self.capacity = capacity
        self.seq_len = seq_len
        self.frame_stack = frame_stack
        self.pin_memory = pin_memory and torch.cuda.is_available()

        self._episodes: list[dict[str, np.ndarray]] = []
        self._total_steps = 0
        self._ep_lengths: list[int] = []  # cached for fast sampling

        # Pre-allocated pinned-memory tensors, lazily initialized on first sample
        self._pinned_obs: torch.Tensor | None = None
        self._pinned_actions: torch.Tensor | None = None
        self._pinned_rewards: torch.Tensor | None = None
        self._pinned_conts: torch.Tensor | None = None
        self._pinned_batch_size: int = 0

    def add_episode(self, episode: dict[str, np.ndarray]):
        """Add a completed episode. Keys: obs, action, reward, cont (continue flag)."""
        ep_len = len(episode["reward"])
        if ep_len < self.seq_len + self.frame_stack - 1:
            return  # too short (need extra frames for stacking history)

        # Ensure obs is stored as uint8
        if episode["obs"].dtype != np.uint8:
            episode["obs"] = (episode["obs"] * 255).clip(0, 255).astype(np.uint8) \
                if episode["obs"].max() <= 1.0 else episode["obs"].astype(np.uint8)

        self._episodes.append(episode)
        self._ep_lengths.append(ep_len)
        self._total_steps += ep_len

        # Evict old episodes if over capacity
        while self._total_steps > self.capacity and len(self._episodes) > 1:
            removed = self._episodes.pop(0)
            self._ep_lengths.pop(0)
            self._total_steps -= len(removed["reward"])

    @property
    def total_steps(self):
        return self._total_steps

    def _ensure_pinned(self, batch_size: int, stacked_obs_shape, act_dim: int):
        """Allocate pinned-memory tensors if needed."""
        if not self.pin_memory or self._pinned_batch_size == batch_size:
            return
        T = self.seq_len
        self._pinned_obs = torch.empty(
            (batch_size, T, *stacked_obs_shape), dtype=torch.uint8
        ).pin_memory()
        self._pinned_actions = torch.empty(
            (batch_size, T, act_dim), dtype=torch.float32
        ).pin_memory()
        self._pinned_rewards = torch.empty(
            (batch_size, T), dtype=torch.float32
        ).pin_memory()
        self._pinned_conts = torch.empty(
            (batch_size, T), dtype=torch.float32
        ).pin_memory()
        self._pinned_batch_size = batch_size

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of sequences of length seq_len (vectorized).

        Returns numpy arrays backed by pinned memory when available.
        Obs are uint8; the agent handles GPU-side float32 conversion.
        When frame_stack > 1, obs channels are stacked: (B, T, 3*K, 64, 64).
        """
        n_eps = len(self._episodes)
        K = self.frame_stack

        # Pick random episodes and start indices in bulk
        # Need start >= K-1 so we have history frames for stacking
        ep_indices = np.random.randint(0, n_eps, size=batch_size)
        max_starts = np.array([self._ep_lengths[i] - self.seq_len for i in ep_indices])
        min_starts = np.full(batch_size, K - 1)
        max_starts = np.maximum(max_starts, min_starts)
        starts = min_starts + (np.random.random(batch_size) * (max_starts - min_starts + 1)).astype(np.intp)

        sample_ep = self._episodes[ep_indices[0]]
        C, H, W = sample_ep["obs"].shape[1:]   # (3, 64, 64)
        act_dim = sample_ep["action"].shape[1]
        stacked_shape = (C * K, H, W)

        if self.pin_memory:
            self._ensure_pinned(batch_size, stacked_shape, act_dim)
            obs = self._pinned_obs.numpy()
            actions = self._pinned_actions.numpy()
            rewards = self._pinned_rewards.numpy()
            conts = self._pinned_conts.numpy()
        else:
            obs = np.empty((batch_size, self.seq_len, *stacked_shape), dtype=np.uint8)
            actions = np.empty((batch_size, self.seq_len, act_dim), dtype=np.float32)
            rewards = np.empty((batch_size, self.seq_len), dtype=np.float32)
            conts = np.empty((batch_size, self.seq_len), dtype=np.float32)

        for i in range(batch_size):
            ep = self._episodes[ep_indices[i]]
            s = starts[i]
            e = s + self.seq_len
            if K == 1:
                obs[i] = ep["obs"][s:e]
            else:
                # Stack K frames along channel dim for each timestep
                for t_idx, t in enumerate(range(s, e)):
                    # Gather frames [t-(K-1), ..., t] and concatenate channels
                    frames = ep["obs"][t - K + 1:t + 1]  # (K, C, H, W)
                    obs[i, t_idx] = frames.reshape(C * K, H, W)
            actions[i] = ep["action"][s:e]
            rewards[i] = ep["reward"][s:e]
            conts[i] = ep["cont"][s:e]

        return {
            "obs": obs,           # (B, T, 3*K, 64, 64) uint8
            "action": actions,    # (B, T, action_dim)
            "reward": rewards,    # (B, T)
            "cont": conts,        # (B, T)
        }
