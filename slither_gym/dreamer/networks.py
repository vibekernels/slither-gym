"""DreamerV3 neural network components.

Implements: CNN encoder/decoder, RSSM world model, MLP actor & critic,
with symlog transforms and unimix categoricals per the DreamerV3 paper.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, OneHotCategorical
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class RSSMState(NamedTuple):
    """RSSM state as a NamedTuple for torch.compile compatibility."""
    deter: torch.Tensor  # (B, deter_dim)
    stoch: torch.Tensor  # (B, stoch_dim, class_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def twohot_encode(x: torch.Tensor, num_bins: int = 255, low: float = -20.0, high: float = 20.0) -> torch.Tensor:
    """Encode scalar values as two-hot vectors over a uniform grid."""
    x = x.clamp(low, high)
    # Direct bin index computation (uniform grid: no search needed)
    step = (high - low) / (num_bins - 1)
    below = ((x - low) / step).long().clamp(0, num_bins - 2)
    above = below + 1
    # Interpolation weights
    weight_above = (x - (low + below * step)) / step
    weight_above = weight_above.clamp(0, 1)
    weight_below = 1 - weight_above
    # Two-hot
    result = torch.zeros(*x.shape, num_bins, device=x.device)
    result.scatter_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
    result.scatter_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
    return result


_twohot_bins_cache: dict[tuple, torch.Tensor] = {}

def twohot_decode(logits: torch.Tensor, num_bins: int = 255, low: float = -20.0, high: float = 20.0) -> torch.Tensor:
    """Decode two-hot logits to scalar values."""
    key = (num_bins, low, high, logits.device)
    bins = _twohot_bins_cache.get(key)
    if bins is None:
        bins = torch.linspace(low, high, num_bins, device=logits.device)
        _twohot_bins_cache[key] = bins
    probs = F.softmax(logits, dim=-1)
    return (probs * bins).sum(-1)


class LayerNormSiLU(nn.Module):
    """LayerNorm + SiLU activation (DreamerV3 default)."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return F.silu(self.norm(x))


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods = []
    for i in range(layers):
        d_in = in_dim if i == 0 else hidden
        mods.append(nn.Linear(d_in, hidden))
        mods.append(LayerNormSiLU(hidden))
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)


# ---------------------------------------------------------------------------
# CNN Encoder / Decoder (for 64x64 images)
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """Encodes 64x64xC images to a flat embedding vector."""

    def __init__(self, depth: int = 32, out_dim: int = 512, in_channels: int = 3):
        super().__init__()
        # 64->32->16->8->4 with channels depth*[1,2,4,8]
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, depth, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(depth, depth * 2, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(depth * 2, depth * 4, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(depth * 4, depth * 8, 4, 2, 1), nn.SiLU(),
        )
        self.flat_dim = depth * 8 * 4 * 4
        self.fc = nn.Linear(self.flat_dim, out_dim)
        self.norm = LayerNormSiLU(out_dim)
        self.use_checkpointing = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, 3, 64, 64) float
        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            x = grad_checkpoint(self.convs, obs, use_reentrant=False)
        else:
            x = self.convs(obs)
        x = x.reshape(x.shape[0], -1)
        return self.norm(self.fc(x))


class CNNDecoder(nn.Module):
    """Decodes latent state to 64x64x3 image reconstruction."""

    def __init__(self, in_dim: int, depth: int = 32):
        super().__init__()
        self.depth = depth
        self.fc = nn.Linear(in_dim, depth * 8 * 4 * 4)
        self.norm = LayerNormSiLU(depth * 8 * 4 * 4)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(depth * 8, depth * 4, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(depth * 2, depth, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(depth, 3, 4, 2, 1),  # -> 64x64x3, no activation (MSE loss)
        )
        self.use_checkpointing = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.fc(features))
        x = x.reshape(-1, self.depth * 8, 4, 4)
        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            return grad_checkpoint(self.deconvs, x, use_reentrant=False)
        return self.deconvs(x)  # (B, 3, 64, 64)


# ---------------------------------------------------------------------------
# RSSM World Model
# ---------------------------------------------------------------------------

class RSSM(nn.Module):
    """Recurrent State-Space Model with discrete latents (DreamerV3 style).

    State = (deterministic h, stochastic z).
    z is represented as `num_classes` categorical variables each with `class_size` classes.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        action_dim: int = 3,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        class_size: int = 32,
        hidden_dim: int = 512,
        unimix: float = 0.01,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.class_size = class_size
        self.unimix = unimix

        stoch_flat = stoch_dim * class_size

        # Prior: h -> z_hat (imagination)
        self.prior_net = mlp(deter_dim, hidden_dim, stoch_dim * class_size, layers=1)

        # Posterior: h, embed -> z (training)
        self.post_net = mlp(deter_dim + embed_dim, hidden_dim, stoch_dim * class_size, layers=1)

        # Sequence model: z_prev, a -> input for GRU
        self.pre_gru = nn.Sequential(
            nn.Linear(stoch_flat + action_dim, hidden_dim),
            LayerNormSiLU(hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

    @property
    def state_dim(self) -> int:
        """Total feature dimension: deter + stoch_flat."""
        return self.deter_dim + self.stoch_dim * self.class_size

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        return RSSMState(
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, self.class_size, device=device),
        )

    def get_features(self, state: RSSMState) -> torch.Tensor:
        stoch_flat = state.stoch.reshape(state.stoch.shape[0], -1)
        return torch.cat([state.deter, stoch_flat], dim=-1)

    def _categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical with straight-through and unimix.

        Uses the Gumbel-max trick instead of OneHotCategorical.sample() to avoid
        aten::multinomial, which is incompatible with CUDA graph capture.
        Produces identical samples from the same distribution.
        """
        logits = logits.reshape(-1, self.stoch_dim, self.class_size)
        # Uniform mix for exploration
        if self.unimix > 0:
            probs = F.softmax(logits, dim=-1)
            probs = (1 - self.unimix) * probs + self.unimix / self.class_size
            logits = torch.log(probs + 1e-8)
        # Gumbel-max trick: u ~ Uniform(0,1), gumbel = -log(-log(u))
        u = torch.zeros_like(logits).uniform_().clamp_(1e-20, 1 - 1e-7)
        gumbel = -(-u.log()).log()
        sample = F.one_hot((logits + gumbel).argmax(-1), self.class_size).to(logits.dtype)
        # Straight-through estimator
        probs = F.softmax(logits, dim=-1)
        return sample + probs - probs.detach()

    def observe_step(self, prev_state: RSSMState, action: torch.Tensor, embed: torch.Tensor):
        """One step of posterior (training). Returns prior and posterior states."""
        prior_state = self.imagine_step(prev_state, action)
        h = prior_state.deter

        # Posterior
        post_logits = self.post_net(torch.cat([h, embed], dim=-1))
        post_stoch = self._categorical(post_logits)
        post_state = RSSMState(deter=h, stoch=post_stoch)

        return post_state, prior_state, post_logits.reshape(-1, self.stoch_dim, self.class_size)

    def imagine_step(self, prev_state: RSSMState, action: torch.Tensor) -> RSSMState:
        """One step of prior (imagination). Returns prior state."""
        stoch_flat = prev_state.stoch.reshape(prev_state.stoch.shape[0], -1)
        x = self.pre_gru(torch.cat([stoch_flat, action], dim=-1))
        h = self.gru(x, prev_state.deter)

        prior_logits = self.prior_net(h)
        prior_stoch = self._categorical(prior_logits)
        return RSSMState(deter=h, stoch=prior_stoch)

    def get_prior_logits(self, deter: torch.Tensor) -> torch.Tensor:
        return self.prior_net(deter).reshape(-1, self.stoch_dim, self.class_size)


# ---------------------------------------------------------------------------
# Parallel associative scan
# ---------------------------------------------------------------------------

def associative_scan(gates: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Parallel prefix scan (Hillis-Steele) for linear recurrence.

    Computes h[0] = v[0], h[t] = g[t] * h[t-1] + v[t] for t > 0.
    All operations are element-wise. O(T log T) work, O(log T) depth.

    Args:
        gates: (T, *batch_dims) — multiplicative coefficients
        values: (T, *batch_dims) — additive terms
    Returns:
        (T, *batch_dims) — prefix scan results (the h values)
    """
    T = gates.shape[0]
    for d in range(int(math.ceil(math.log2(max(T, 2))))):
        stride = 1 << d
        if stride >= T:
            break
        # Combine position i with position i-stride for all i >= stride
        new_g = gates[stride:] * gates[:T - stride]
        new_v = gates[stride:] * values[:T - stride] + values[stride:]
        gates = torch.cat([gates[:stride], new_g], dim=0)
        values = torch.cat([values[:stride], new_v], dim=0)
    return values


# ---------------------------------------------------------------------------
# Mamba RSSM (Selective State Space replacement for GRU)
# ---------------------------------------------------------------------------

class MambaRSSM(nn.Module):
    """RSSM with Mamba-style selective SSM replacing the GRU.

    Key differences from GRU RSSM:
    1. Linear recurrence enables parallel scan during training (O(log T) depth)
    2. Input-dependent A, B, C (selective mechanism) provides content-based filtering
    3. No stochastic feedback into sequence model (enables parallel training)

    The SSM internal state (ssm_h) is packed into RSSMState.deter to maintain
    the same interface as the GRU RSSM.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        action_dim: int = 3,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        class_size: int = 32,
        hidden_dim: int = 512,
        d_state: int = 16,
        unimix: float = 0.01,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.class_size = class_size
        self.unimix = unimix
        self.d_state = d_state

        stoch_flat = stoch_dim * class_size

        # Input projection (action for imagine, action+embed for observe)
        self.pre_ssm = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            LayerNormSiLU(hidden_dim),
        )
        self.embed_proj = nn.Linear(embed_dim, hidden_dim, bias=False)

        # Project to SSM input + gate (Mamba-style gated output)
        self.in_proj = nn.Linear(hidden_dim, deter_dim * 2)

        # SSM parameters
        # A initialized with log-spaced values (like S4D-Lin / Mamba)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            torch.log(A).unsqueeze(0).expand(deter_dim, -1).clone()
        )
        self.D = nn.Parameter(torch.ones(deter_dim))

        # Input-dependent SSM parameter projections (from x_ssm)
        self.proj_dt = nn.Linear(deter_dim, deter_dim)
        self.proj_B = nn.Linear(deter_dim, d_state, bias=False)
        self.proj_C = nn.Linear(deter_dim, d_state, bias=False)

        # Initialize dt bias so softplus(bias) ∈ [0.001, 0.1]
        dt_min, dt_max = 0.001, 0.1
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(deter_dim) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
            self.proj_dt.bias.copy_(inv_dt)

        # Output normalization
        self.out_norm = nn.LayerNorm(deter_dim)

        # Prior/posterior networks (same as GRU RSSM)
        self.prior_net = mlp(deter_dim, hidden_dim, stoch_dim * class_size, layers=1)
        self.post_net = mlp(deter_dim + embed_dim, hidden_dim, stoch_dim * class_size, layers=1)

    @property
    def state_dim(self) -> int:
        """Feature dimension: deter + stoch_flat (not including packed ssm_h)."""
        return self.deter_dim + self.stoch_dim * self.class_size

    def _pack_state(self, deter: torch.Tensor, stoch: torch.Tensor,
                    ssm_h: torch.Tensor) -> RSSMState:
        """Pack ssm_h into deter for interface compatibility."""
        packed = torch.cat([deter, ssm_h.reshape(deter.shape[0], -1)], dim=-1)
        return RSSMState(deter=packed, stoch=stoch)

    def _unpack_state(self, state: RSSMState):
        """Extract deter and ssm_h from packed state."""
        deter = state.deter[:, :self.deter_dim]
        ssm_h = state.deter[:, self.deter_dim:].reshape(
            -1, self.deter_dim, self.d_state
        )
        return deter, state.stoch, ssm_h

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        packed_dim = self.deter_dim + self.deter_dim * self.d_state
        return RSSMState(
            deter=torch.zeros(batch_size, packed_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, self.class_size, device=device),
        )

    def get_features(self, state: RSSMState) -> torch.Tensor:
        """Extract features (deter + stoch_flat), excluding packed ssm_h."""
        deter = state.deter[:, :self.deter_dim]
        stoch_flat = state.stoch.reshape(state.stoch.shape[0], -1)
        return torch.cat([deter, stoch_flat], dim=-1)

    def _categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical with straight-through and unimix (Gumbel-max)."""
        logits = logits.reshape(-1, self.stoch_dim, self.class_size)
        if self.unimix > 0:
            probs = F.softmax(logits, dim=-1)
            probs = (1 - self.unimix) * probs + self.unimix / self.class_size
            logits = torch.log(probs + 1e-8)
        u = torch.zeros_like(logits).uniform_().clamp_(1e-20, 1 - 1e-7)
        gumbel = -(-u.log()).log()
        sample = F.one_hot((logits + gumbel).argmax(-1), self.class_size).to(logits.dtype)
        probs = F.softmax(logits, dim=-1)
        return sample + probs - probs.detach()

    def _ssm_step(self, x: torch.Tensor, ssm_h: torch.Tensor):
        """Single selective SSM step.

        Args:
            x: (B, hidden_dim) preprocessed input
            ssm_h: (B, deter_dim, d_state) SSM internal state
        Returns:
            deter: (B, deter_dim) output
            new_ssm_h: (B, deter_dim, d_state) new SSM state
        """
        xz = self.in_proj(x)
        x_ssm, x_gate = xz.chunk(2, dim=-1)  # each (B, deter_dim)

        dt = F.softplus(self.proj_dt(x_ssm))   # (B, deter_dim)
        B_param = self.proj_B(x_ssm)            # (B, d_state)
        C_param = self.proj_C(x_ssm)            # (B, d_state)

        A = -torch.exp(self.A_log)                          # (deter_dim, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A)                # (B, deter_dim, d_state)
        dBx = (dt * x_ssm).unsqueeze(-1) * B_param.unsqueeze(1)  # (B, deter_dim, d_state)

        new_ssm_h = dA * ssm_h + dBx

        y = torch.einsum('bdn,bn->bd', new_ssm_h, C_param)  # (B, deter_dim)
        y = y + self.D * x_ssm
        y = y * F.silu(x_gate)
        return self.out_norm(y), new_ssm_h

    def observe_step(self, prev_state: RSSMState, action: torch.Tensor,
                     embed: torch.Tensor):
        """One step of posterior (training/inference). Same interface as GRU RSSM."""
        _, _, ssm_h = self._unpack_state(prev_state)
        x = self.pre_ssm(action) + self.embed_proj(embed)
        new_deter, new_ssm_h = self._ssm_step(x, ssm_h)

        post_logits = self.post_net(torch.cat([new_deter, embed], dim=-1))
        post_stoch = self._categorical(post_logits)
        prior_logits = self.prior_net(new_deter).reshape(-1, self.stoch_dim, self.class_size)

        post_state = self._pack_state(new_deter, post_stoch, new_ssm_h)
        return post_state, None, prior_logits

    def imagine_step(self, prev_state: RSSMState, action: torch.Tensor) -> RSSMState:
        """One step of prior (imagination). Same interface as GRU RSSM."""
        _, _, ssm_h = self._unpack_state(prev_state)
        x = self.pre_ssm(action)
        new_deter, new_ssm_h = self._ssm_step(x, ssm_h)

        prior_logits = self.prior_net(new_deter)
        prior_stoch = self._categorical(prior_logits)
        return self._pack_state(new_deter, prior_stoch, new_ssm_h)

    def observe_sequence(self, actions: torch.Tensor, embeds: torch.Tensor):
        """Process all T steps in parallel using associative scan.

        No stochastic feedback — actions and embeddings are the only inputs,
        enabling parallel computation of all deterministic states.

        Args:
            actions: (B, T, action_dim)
            embeds: (B, T, embed_dim)
        Returns:
            features: (T, B, state_dim)
            post_stochs: (T, B, stoch_dim, class_size)
            prior_logits: (T, B, stoch_dim, class_size)
            final_deter: (B, deter_dim) raw (not packed)
            final_stoch: (B, stoch_dim, class_size)
            final_ssm_h: (B, deter_dim, d_state)
        """
        B, T = actions.shape[:2]

        # Pre-process all inputs at once
        x = self.pre_ssm(actions.reshape(B * T, -1))
        x = x + self.embed_proj(embeds.reshape(B * T, -1))
        x = x.reshape(B, T, -1).permute(1, 0, 2)  # (T, B, hidden_dim)

        # Project to SSM input + gate
        xz = self.in_proj(x)
        x_ssm, x_gate = xz.chunk(2, dim=-1)  # each (T, B, deter_dim)

        # Input-dependent SSM parameters
        dt = F.softplus(self.proj_dt(x_ssm))   # (T, B, deter_dim)
        B_param = self.proj_B(x_ssm)            # (T, B, d_state)
        C_param = self.proj_C(x_ssm)            # (T, B, d_state)

        # Discretize A and compute scan inputs
        A = -torch.exp(self.A_log)                           # (deter_dim, d_state)
        gates = torch.exp(dt.unsqueeze(-1) * A)              # (T, B, D, N)
        values = (dt * x_ssm).unsqueeze(-1) * B_param.unsqueeze(2)  # (T, B, D, N)

        # Parallel associative scan (starts from zero SSM state)
        h_all = associative_scan(gates, values)  # (T, B, D, N)

        # Compute SSM output
        y = torch.einsum('tbdn,tbn->tbd', h_all, C_param)
        y = y + self.D * x_ssm
        y = y * F.silu(x_gate)
        deter_all = self.out_norm(y)  # (T, B, deter_dim)

        # Compute priors and posteriors for all timesteps (batched)
        TB = T * B
        deter_flat = deter_all.reshape(TB, -1)
        embeds_flat = embeds.permute(1, 0, 2).reshape(TB, -1)

        prior_logits = self.prior_net(deter_flat).reshape(
            T, B, self.stoch_dim, self.class_size
        )
        post_input = torch.cat([deter_flat, embeds_flat], dim=-1)
        post_stochs = self._categorical(self.post_net(post_input)).reshape(
            T, B, self.stoch_dim, self.class_size
        )

        # Features = deter + stoch_flat
        stoch_flat = post_stochs.reshape(T, B, -1)
        features = torch.cat([deter_all, stoch_flat], dim=-1)

        return (features, post_stochs, prior_logits,
                deter_all[-1], post_stochs[-1], h_all[-1],
                deter_all, h_all)

    def get_prior_logits(self, deter: torch.Tensor) -> torch.Tensor:
        return self.prior_net(deter).reshape(-1, self.stoch_dim, self.class_size)


# ---------------------------------------------------------------------------
# Actor & Critic
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Discrete action policy head."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 512, layers: int = 3):
        super().__init__()
        self.net = mlp(state_dim, hidden, action_dim, layers=layers)

    def forward(self, features: torch.Tensor) -> torch.distributions.Distribution:
        logits = self.net(features)
        return OneHotCategorical(logits=logits)

    def log_prob(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self.forward(features)
        return dist.log_prob(actions)


class Critic(nn.Module):
    """Critic predicting twohot-encoded value (DreamerV3 style)."""

    def __init__(self, state_dim: int, hidden: int = 512, layers: int = 3, num_bins: int = 255):
        super().__init__()
        self.net = mlp(state_dim, hidden, num_bins, layers=layers)
        self.num_bins = num_bins

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns twohot logits."""
        return self.net(features)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        """Returns scalar value estimate."""
        logits = self.forward(features)
        return twohot_decode(logits, self.num_bins)


# ---------------------------------------------------------------------------
# Full World Model
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    def __init__(
        self,
        action_dim: int = 3,
        embed_dim: int = 512,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        class_size: int = 32,
        hidden_dim: int = 512,
        cnn_depth: int = 32,
        reward_bins: int = 255,
        rssm_type: str = "gru",
        in_channels: int = 3,
    ):
        super().__init__()
        self.encoder = CNNEncoder(depth=cnn_depth, out_dim=embed_dim, in_channels=in_channels)
        if rssm_type == "mamba":
            self.rssm = MambaRSSM(
                embed_dim=embed_dim,
                action_dim=action_dim,
                deter_dim=deter_dim,
                stoch_dim=stoch_dim,
                class_size=class_size,
                hidden_dim=hidden_dim,
            )
        else:
            self.rssm = RSSM(
                embed_dim=embed_dim,
                action_dim=action_dim,
                deter_dim=deter_dim,
                stoch_dim=stoch_dim,
                class_size=class_size,
                hidden_dim=hidden_dim,
            )
        state_dim = self.rssm.state_dim
        self.decoder = CNNDecoder(in_dim=state_dim, depth=cnn_depth)
        self.reward_head = mlp(state_dim, hidden_dim, reward_bins, layers=2)
        self.continue_head = mlp(state_dim, hidden_dim, 1, layers=2)

        self.reward_bins = reward_bins

    @property
    def state_dim(self):
        return self.rssm.state_dim
