"""MLP-LSTM policy/value model for PPO."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPLSTMPolicy(nn.Module):
    """MLP encoder -> LSTM -> policy + value heads.

    Uses a fast path when no episode boundaries exist in the sequence,
    calling the LSTM on the full sequence at once (cuDNN fused kernel).
    Falls back to step-by-step processing only when resets are needed.
    """

    def __init__(self, obs_dim=54, act_dim=6, hidden_dim=128, lstm_dim=128):
        super().__init__()
        self.lstm_dim = lstm_dim

        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, lstm_dim)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.policy_head = layer_init(nn.Linear(lstm_dim, act_dim), std=0.01)
        self.value_head = layer_init(nn.Linear(lstm_dim, 1), std=1.0)

    def get_initial_state(self, batch_size, device="cpu"):
        return (
            torch.zeros(1, batch_size, self.lstm_dim, device=device),
            torch.zeros(1, batch_size, self.lstm_dim, device=device),
        )

    def forward(self, obs, lstm_state, done=None):
        """
        Args:
            obs: (T, N, obs_dim) or (N, obs_dim)
            lstm_state: tuple (h, c) each (1, N, lstm_dim)
            done: (T, N) float tensor or None
        Returns:
            logits: (T, N, act_dim) or (N, act_dim)
            values: (T, N) or (N,)
            new_lstm_state: tuple (h, c)
        """
        single = obs.dim() == 2
        if single:
            obs = obs.unsqueeze(0)
            if done is not None:
                done = done.unsqueeze(0)

        T, N = obs.shape[:2]
        x = self.encoder(obs.reshape(T * N, -1)).reshape(T, N, -1)

        h, c = lstm_state

        # Fast path: no episode boundaries → single fused LSTM call
        if done is None or done.sum() == 0:
            x, (h, c) = self.lstm(x, (h, c))
        else:
            # Slow path: step-by-step with LSTM state resets (rare)
            outputs = []
            for t in range(T):
                if t > 0:
                    mask = (1.0 - done[t - 1]).unsqueeze(0).unsqueeze(-1)
                    h = h * mask
                    c = c * mask
                out, (h, c) = self.lstm(x[t : t + 1], (h, c))
                outputs.append(out)
            x = torch.cat(outputs, dim=0)

        logits = self.policy_head(x)
        values = self.value_head(x).squeeze(-1)

        if single:
            logits = logits.squeeze(0)
            values = values.squeeze(0)

        return logits, values, (h, c)

    def get_action_and_value(self, obs, lstm_state, action=None, done=None):
        """Convenience: returns (action, log_prob, entropy, value, new_state)."""
        logits, values, new_state = self.forward(obs, lstm_state, done)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), values, new_state
