"""Policy/value models for PPO: pure MLP and MLP-LSTM variants."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ──────────────────────────────────────────────────────────────────────
#  Pure MLP (no recurrence — fully parallel training)
# ──────────────────────────────────────────────────────────────────────

class MLPPolicy(nn.Module):
    """Pure MLP policy/value network. No hidden state, fully parallel."""

    def __init__(self, obs_dim=54, act_dim=6, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.policy_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.value_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, obs):
        """obs: any shape (..., obs_dim). Returns logits, values."""
        x = self.encoder(obs)
        return self.policy_head(x), self.value_head(x).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        """Returns (action, log_prob, entropy, value)."""
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), values


# ──────────────────────────────────────────────────────────────────────
#  MLP-LSTM (kept for comparison / optional use)
# ──────────────────────────────────────────────────────────────────────

class MLPLSTMPolicy(nn.Module):
    """MLP encoder -> LSTM -> policy + value heads."""

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
        single = obs.dim() == 2
        if single:
            obs = obs.unsqueeze(0)
            if done is not None:
                done = done.unsqueeze(0)

        T, N = obs.shape[:2]
        x = self.encoder(obs.reshape(T * N, -1)).reshape(T, N, -1)

        h, c = lstm_state

        if done is None or done.sum() == 0:
            x, (h, c) = self.lstm(x, (h, c))
        else:
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
        logits, values, new_state = self.forward(obs, lstm_state, done)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), values, new_state
