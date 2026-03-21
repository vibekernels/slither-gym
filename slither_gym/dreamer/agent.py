"""DreamerV3 agent: world model training, imagination, actor-critic."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import OneHotCategorical

from .networks import (
    WorldModel, Actor, Critic, RSSMState, MambaRSSM,
    symlog, symexp, twohot_encode, twohot_decode,
)


def _imagine_sequence(rssm, actor_net, init_deter, init_stoch, horizon, action_dim):
    """Run H steps of imagination (prior + actor). Compiled as a single graph.

    Uses Gumbel-max sampling for actions (compile-friendly). Log-prob and entropy
    are recomputed outside this function for the actor loss.
    Works for both GRU and Mamba RSSM via rssm.get_features/imagine_step.
    """
    B = init_deter.shape[0]
    D = rssm.state_dim

    all_features = torch.empty(horizon + 1, B, D, device=init_deter.device)
    all_actions = torch.empty(horizon, B, action_dim, device=init_deter.device)

    state = RSSMState(deter=init_deter, stoch=init_stoch)
    all_features[0] = rssm.get_features(state)

    for t in range(horizon):
        features = all_features[t]
        action_logits = actor_net(features)
        u = torch.zeros_like(action_logits).uniform_().clamp_(1e-20, 1 - 1e-7)
        gumbel = -(-u.log()).log()
        action = F.one_hot(
            (action_logits + gumbel).argmax(-1), action_dim
        ).to(action_logits.dtype)
        all_actions[t] = action

        state = rssm.imagine_step(state, action)
        all_features[t + 1] = rssm.get_features(state)

    return all_features, all_actions


def _critic_forward(critic_net, features, lambda_returns, reward_bins):
    """Critic forward + loss, compiled as single graph."""
    critic_logits = critic_net(features)
    target = symlog(lambda_returns).reshape(-1)
    target_twohot = twohot_encode(target, reward_bins)
    critic_loss = -torch.sum(target_twohot * F.log_softmax(critic_logits, dim=-1), dim=-1).mean()
    return critic_loss


def _actor_forward(actor_net, features, actions, advantages, entropy_scale, action_dim):
    """Actor forward + loss, compiled as single graph."""
    logits = actor_net(features)
    dist = OneHotCategorical(logits=logits)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy()
    actor_loss = -(log_probs * advantages).mean()
    actor_loss = actor_loss - entropy_scale * entropy.mean()
    return actor_loss, entropy.mean()


def _mamba_observe_wrapper(rssm, actions, embed):
    """Wrapper for MambaRSSM.observe_sequence (standalone for torch.compile)."""
    return rssm.observe_sequence(actions, embed)


def _mamba_wm_forward(encoder, rssm, decoder, reward_head, continue_head,
                       obs, actions, rewards, conts, B, T, reward_bins,
                       free_nats, kl_scale):
    """Full Mamba world model forward pass + loss computation.

    Compiled as a single torch.compile graph so backward kernels get fused.
    """
    # Encode
    obs_flat = obs.reshape(B * T, *obs.shape[2:])
    embed_flat = encoder(obs_flat)
    embed = embed_flat.reshape(B, T, -1)

    # RSSM observe (parallel scan)
    all_features, all_post_stochs, all_prior_logits, \
        final_deter, final_stoch, final_ssm_h, \
        deter_all, h_all = rssm.observe_sequence(actions, embed)

    features = all_features.permute(1, 0, 2)  # (T, B, D) -> (B, T, D)
    feat_flat = features.reshape(B * T, -1)

    # Decode (reconstructs 3-channel image; target is last 3 channels of stacked obs)
    recon = decoder(feat_flat)
    recon_target = obs_flat[:, -3:]  # last 3 channels = most recent frame
    recon_loss = F.mse_loss(recon, recon_target)

    # Reward prediction
    reward_logits = reward_head(feat_flat).reshape(B, T, -1)
    reward_target = twohot_encode(symlog(rewards), reward_bins)
    reward_loss = -torch.sum(
        reward_target * F.log_softmax(reward_logits, dim=-1), dim=-1
    ).mean()

    # Continue prediction
    cont_logits = continue_head(feat_flat).reshape(B, T)
    cont_loss = F.binary_cross_entropy_with_logits(cont_logits, conts)

    # KL loss (vectorized)
    post_probs = all_post_stochs.clamp(1e-8, 1.0)
    prior_probs = F.softmax(all_prior_logits, dim=-1).clamp(1e-8, 1.0)
    kl_per_step = (post_probs * (post_probs.log() - prior_probs.log())).sum(dim=-1).mean(dim=(1, 2))
    kl_per_step = torch.clamp(kl_per_step, min=free_nats)
    kl_loss = kl_per_step.mean()

    loss = recon_loss + reward_loss + cont_loss + kl_scale * kl_loss

    return loss, recon_loss, reward_loss, cont_loss, kl_loss, \
        final_deter, final_stoch, final_ssm_h, \
        deter_all, h_all, all_post_stochs


def _observe_sequence(rssm, init_deter, init_stoch, actions, embed, T):
    """Run T steps of RSSM observe (posterior). Compiled as a single graph.

    Returns flat tensors instead of lists to avoid torch.compile graph breaks
    from Python container mutations.
    """
    B = init_deter.shape[0]
    S, C = rssm.stoch_dim, rssm.class_size
    D = rssm.state_dim

    # Pre-allocate output tensors
    all_features = torch.empty(T, B, D, device=init_deter.device)
    all_post_stochs = torch.empty(T, B, S, C, device=init_deter.device)
    all_prior_logits = torch.empty(T, B, S, C, device=init_deter.device)

    deter, stoch = init_deter, init_stoch
    for t in range(T):
        prev = RSSMState(deter=deter, stoch=stoch)
        post_state, prior_state, prior_logits = rssm.observe_step(
            prev, actions[:, t], embed[:, t]
        )
        stoch_flat = post_state.stoch.reshape(B, -1)
        all_features[t] = torch.cat([post_state.deter, stoch_flat], dim=-1)
        all_post_stochs[t] = post_state.stoch
        all_prior_logits[t] = prior_logits
        deter, stoch = post_state.deter, post_state.stoch

    return all_features, all_post_stochs, all_prior_logits, deter, stoch



class DreamerV3Agent:
    """Full DreamerV3 agent with world model, actor, and critic."""

    def __init__(
        self,
        action_dim: int = 3,
        device: str = "cpu",
        # Architecture
        embed_dim: int = 512,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        class_size: int = 32,
        hidden_dim: int = 512,
        cnn_depth: int = 32,
        # Training
        lr_model: float = 1e-4,
        lr_actor: float = 3e-5,
        lr_critic: float = 3e-5,
        imagine_horizon: int = 15,
        gamma: float = 0.997,
        lambda_: float = 0.95,
        entropy_scale: float = 3e-3,
        reward_bins: int = 255,
        free_nats: float = 1.0,
        kl_scale: float = 0.5,
        use_amp: bool = True,
        compile_models: bool = True,
        rssm_type: str = "gru",
        grad_checkpoint: bool = False,
        imagine_starts: int = 1,
        frame_stack: int = 1,
    ):
        # TF32 gives ~1.5x matmul speedup on Ampere+ GPUs with negligible precision loss
        torch.set_float32_matmul_precision("high")
        # Auto-tune convolution algorithms for the current hardware
        torch.backends.cudnn.benchmark = True

        self.device = torch.device(device)
        self.action_dim = action_dim
        self.rssm_type = rssm_type
        self.imagine_horizon = imagine_horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_scale = entropy_scale
        self.reward_bins = reward_bins
        self.free_nats = free_nats
        self.kl_scale = kl_scale
        self.imagine_starts = imagine_starts
        self.frame_stack = frame_stack

        # Mixed precision
        self.use_amp = use_amp and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16

        # Networks
        # Gradient checkpointing: trades ~10% compute for ~40% less VRAM,
        # enabling B=512+ on GPUs that would otherwise OOM at B=384.
        self._grad_checkpoint = grad_checkpoint and self.device.type == "cuda"

        self.world_model = WorldModel(
            action_dim=action_dim,
            embed_dim=embed_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            class_size=class_size,
            hidden_dim=hidden_dim,
            cnn_depth=cnn_depth,
            reward_bins=reward_bins,
            rssm_type=rssm_type,
            in_channels=3 * frame_stack,
        ).to(self.device)

        if self._grad_checkpoint:
            self.world_model.encoder.use_checkpointing = True
            self.world_model.decoder.use_checkpointing = True

        state_dim = self.world_model.state_dim
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, hidden_dim, num_bins=reward_bins).to(self.device)
        self.slow_critic = Critic(state_dim, hidden_dim, num_bins=reward_bins).to(self.device)
        self.slow_critic.load_state_dict(self.critic.state_dict())

        # torch.compile for key modules
        if compile_models and self.device.type == "cuda":
            try:
                self.world_model.encoder = torch.compile(self.world_model.encoder)
                self.world_model.decoder = torch.compile(self.world_model.decoder)
            except Exception:
                pass  # fallback gracefully if compile fails

        # Compiled RSSM sequence functions.
        # Compiling the entire loop as one graph eliminates Python loop overhead
        # between steps (~2.2x speedup on RSSM forward vs per-step compilation).
        self._use_mamba = rssm_type == "mamba"
        _compile_mode = "max-autotune-no-cudagraphs" if self._use_mamba else "default"
        if self.device.type == "cuda":
            try:
                if self._use_mamba:
                    self._compiled_observe_seq = torch.compile(
                        _mamba_observe_wrapper, mode=_compile_mode
                    )
                else:
                    self._compiled_observe_seq = torch.compile(
                        _observe_sequence, mode="default"
                    )
                self._compiled_imagine_seq = torch.compile(
                    _imagine_sequence, mode=_compile_mode
                )
            except Exception:
                if self._use_mamba:
                    self._compiled_observe_seq = _mamba_observe_wrapper
                else:
                    self._compiled_observe_seq = _observe_sequence
                self._compiled_imagine_seq = _imagine_sequence
        else:
            if self._use_mamba:
                self._compiled_observe_seq = _mamba_observe_wrapper
            else:
                self._compiled_observe_seq = _observe_sequence
            self._compiled_imagine_seq = _imagine_sequence

        # Compile decoder for training (fuses forward + backward convolution kernels).
        if self.device.type == "cuda":
            try:
                self._compiled_decoder = torch.compile(self.world_model.decoder, mode="default")
            except Exception:
                self._compiled_decoder = self.world_model.decoder
        else:
            self._compiled_decoder = self.world_model.decoder

        # Compiled full world model forward (encoder→RSSM→decoder→heads→losses).
        # Fuses forward and backward kernels for significant backward pass speedup.
        # max-autotune-no-cudagraphs: autotuned Triton kernels without CUDA graph
        # capture (CUDA graphs conflict with sequential RNN/SSM loops).
        if self._use_mamba and self.device.type == "cuda":
            try:
                self._compiled_wm_forward = torch.compile(
                    _mamba_wm_forward, mode="max-autotune-no-cudagraphs", dynamic=False
                )
            except Exception:
                self._compiled_wm_forward = _mamba_wm_forward
        else:
            self._compiled_wm_forward = None  # GRU uses separate compilation

        # Compiled actor-critic forward+loss functions
        if self.device.type == "cuda":
            try:
                self._compiled_critic_fwd = torch.compile(
                    _critic_forward, mode="default"
                )
                self._compiled_actor_fwd = torch.compile(
                    _actor_forward, mode="default"
                )
            except Exception:
                self._compiled_critic_fwd = _critic_forward
                self._compiled_actor_fwd = _actor_forward
        else:
            self._compiled_critic_fwd = _critic_forward
            self._compiled_actor_fwd = _actor_forward

        # Final RSSM state from world-model training, reused as actor-critic starting state
        self._train_state_cache: RSSMState | None = None

        # Secondary CUDA stream for overlapping WM optimizer with AC imagination
        self._opt_stream = torch.cuda.Stream() if self.device.type == "cuda" else None

        # Optimizers (fused=True on CUDA: single kernel per step instead of per-parameter)
        _fused = self.device.type == "cuda"
        self.model_opt = torch.optim.Adam(self.world_model.parameters(), lr=lr_model, eps=1e-8, fused=_fused)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-8, fused=_fused)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-8, fused=_fused)

        # Running state for acting
        self._prev_state = None
        self._prev_action = None
        self._frame_history = None  # (N, K, 3, H, W) ring buffer for frame stacking

    def init_state(self, batch_size: int = 1):
        """Reset agent state for acting."""
        self._prev_state = self.world_model.rssm.initial_state(batch_size, self.device)
        self._prev_action = torch.zeros(batch_size, self.action_dim, device=self.device)
        if self.frame_stack > 1:
            self._frame_history = None  # lazily initialized on first obs

    def _stack_frames_single(self, obs_chw: torch.Tensor) -> torch.Tensor:
        """Stack K frames for single-env inference. obs_chw: (1, 3, H, W)."""
        K = self.frame_stack
        if K == 1:
            return obs_chw
        if self._frame_history is None:
            # Initialize with K copies of the first frame
            self._frame_history = obs_chw.repeat(1, K, 1, 1)  # (1, 3*K, H, W)
        else:
            # Shift left by 3 channels and append new frame
            self._frame_history = torch.cat([
                self._frame_history[:, 3:], obs_chw
            ], dim=1)
        return self._frame_history

    def _stack_frames_batch(self, obs_nchw: torch.Tensor) -> torch.Tensor:
        """Stack K frames for batch inference. obs_nchw: (N, 3, H, W)."""
        K = self.frame_stack
        if K == 1:
            return obs_nchw
        N = obs_nchw.shape[0]
        if self._frame_history is None:
            self._frame_history = obs_nchw.repeat(1, K, 1, 1)  # (N, 3*K, H, W)
        else:
            self._frame_history = torch.cat([
                self._frame_history[:, 3:], obs_nchw
            ], dim=1)
        return self._frame_history

    @torch.no_grad()
    def act(self, obs: np.ndarray, training: bool = True) -> int:
        """Select action given a single observation."""
        if self._prev_state is None:
            self.init_state(1)

        # Preprocess obs: (H, W, 3) uint8 -> (1, 3, H, W) float
        obs_t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        obs_t = self._stack_frames_single(obs_t)  # (1, 3*K, H, W)

        # Encode
        with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            embed = self.world_model.encoder(obs_t)

        # RSSM step (keep in fp32 for recurrent stability)
        post_state, _, _ = self.world_model.rssm.observe_step(
            self._prev_state, self._prev_action, embed.float()
        )

        # Get features and sample action
        features = self.world_model.rssm.get_features(post_state)
        dist = self.actor(features)

        if training:
            action = dist.sample()
        else:
            # Greedy
            action = F.one_hot(dist.logits.argmax(-1), self.action_dim).float()

        self._prev_state = post_state
        self._prev_action = action

        return action.argmax(-1).item()

    @torch.no_grad()
    def batch_act(self, obs_batch: np.ndarray, training: bool = True) -> np.ndarray:
        """Select actions for a batch of observations. obs_batch: (N, H, W, 3) uint8.
        Returns: (N,) int actions."""
        N = obs_batch.shape[0]
        if self._prev_state is None:
            self.init_state(N)

        obs_t = torch.from_numpy(obs_batch).float().permute(0, 3, 1, 2).to(self.device) / 255.0
        obs_t = self._stack_frames_batch(obs_t)  # (N, 3*K, H, W)

        with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            embed = self.world_model.encoder(obs_t)

        post_state, _, _ = self.world_model.rssm.observe_step(
            self._prev_state, self._prev_action, embed.float()
        )

        features = self.world_model.rssm.get_features(post_state)
        dist = self.actor(features)

        if training:
            action = dist.sample()
        else:
            action = F.one_hot(dist.logits.argmax(-1), self.action_dim).float()

        self._prev_state = post_state
        self._prev_action = action

        return action.argmax(-1).cpu().numpy()

    @torch.no_grad()
    def reset_state_at(self, idx: int):
        """Reset the RSSM state for a single env index (after episode boundary)."""
        if self._prev_state is None:
            return
        init = self.world_model.rssm.initial_state(1, self.device)
        # NamedTuple is immutable, but the underlying tensors are mutable in-place
        self._prev_state.deter[idx] = init.deter[0]
        self._prev_state.stoch[idx] = init.stoch[0]
        self._prev_action[idx] = 0.0
        # Reset frame history for this env
        if self.frame_stack > 1 and self._frame_history is not None:
            self._frame_history[idx] = 0.0

    def transfer_batch(self, batch: dict[str, np.ndarray]) -> tuple:
        """Transfer a CPU batch to GPU tensors. Called from pre-fetch thread.

        Transfers obs as uint8 (4x less PCIe bandwidth) and converts to float on GPU.
        """
        nb = self.device.type == "cuda"
        # Transfer as uint8 — 4x less data over PCIe — then convert on GPU
        obs = torch.from_numpy(batch["obs"]).to(self.device, non_blocking=nb)
        obs = obs.float().div_(255.0)
        actions = torch.from_numpy(batch["action"]).to(self.device, non_blocking=nb)
        rewards = torch.from_numpy(batch["reward"]).to(self.device, non_blocking=nb)
        conts = torch.from_numpy(batch["cont"]).to(self.device, non_blocking=nb)
        return obs, actions, rewards, conts

    def train_step(self, batch: dict[str, np.ndarray] | tuple) -> dict[str, float]:
        """One training step on a batch of sequences. Returns loss metrics.

        Accepts either a dict (numpy arrays, will be transferred to GPU) or
        a tuple of pre-transferred GPU tensors from transfer_batch().
        """
        if isinstance(batch, dict):
            obs, actions, rewards, conts = self.transfer_batch(batch)
        else:
            obs, actions, rewards, conts = batch

        B, T = obs.shape[:2]

        # --- World Model Training ---
        model_metrics = self._train_world_model(obs, actions, rewards, conts, B, T)

        # --- Actor-Critic Training (imagination) ---
        ac_metrics = self._train_actor_critic(obs, actions, B, T)

        # Slow critic EMA update (vectorized: single fused kernel instead of per-param loop)
        with torch.no_grad():
            slow_params = list(self.slow_critic.parameters())
            critic_params = list(self.critic.parameters())
            torch._foreach_lerp_(slow_params, critic_params, 0.02)

        return {**model_metrics, **ac_metrics}

    def _train_world_model(self, obs, actions, rewards, conts, B, T):
        if self._compiled_wm_forward is not None:
            # Mamba: compiled full forward pass (encoder→RSSM→decoder→heads→losses)
            # Fuses forward and backward kernels for ~20% backward speedup.
            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                loss, recon_loss, reward_loss, cont_loss, kl_loss, \
                    final_deter, final_stoch, final_ssm_h, \
                    deter_all, h_all, all_post_stochs = \
                    self._compiled_wm_forward(
                        self.world_model.encoder, self.world_model.rssm,
                        self.world_model.decoder, self.world_model.reward_head,
                        self.world_model.continue_head,
                        obs, actions, rewards, conts, B, T,
                        self.reward_bins, self.free_nats, self.kl_scale,
                    )
            if self.imagine_starts > 1:
                # Cache all T states for multi-start imagination
                # deter_all: (T, B, D), h_all: (T, B, D, N), all_post_stochs: (T, B, S, C)
                self._train_all_states = (
                    deter_all.detach(), h_all.detach(), all_post_stochs.detach()
                )
            # Pack ssm_h into deter for imagination starting state (final state)
            packed_deter = torch.cat(
                [final_deter, final_ssm_h.reshape(B, -1)], dim=-1
            )
            self._train_state_cache = RSSMState(
                deter=packed_deter.detach(), stoch=final_stoch.detach()
            )
        else:
            # GRU: separate compilation (sequential ops are memory-bandwidth bound)
            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                obs_flat = obs.reshape(B * T, *obs.shape[2:])
                embed_flat = self.world_model.encoder(obs_flat)
                embed = embed_flat.reshape(B, T, -1)

            embed_f32 = embed.float()
            rssm = self.world_model.rssm
            init_state = rssm.initial_state(B, self.device)
            all_features, all_post_stochs, all_prior_logits, \
                final_deter, final_stoch = \
                self._compiled_observe_seq(
                    rssm, init_state.deter, init_state.stoch,
                    actions, embed_f32, T
                )
            self._train_state_cache = RSSMState(
                deter=final_deter.detach(), stoch=final_stoch.detach()
            )

            features = all_features.permute(1, 0, 2)

            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                feat_flat = features.reshape(B * T, -1)
                recon = self._compiled_decoder(feat_flat)
                obs_target = obs.reshape(B * T, *obs.shape[2:])[:, -3:]
                recon_loss = F.mse_loss(recon, obs_target)

                reward_logits = self.world_model.reward_head(feat_flat).reshape(B, T, -1)
                reward_target = twohot_encode(symlog(rewards), self.reward_bins)
                reward_loss = -torch.sum(reward_target * F.log_softmax(reward_logits, dim=-1), dim=-1).mean()

                cont_logits = self.world_model.continue_head(feat_flat).reshape(B, T)
                cont_loss = F.binary_cross_entropy_with_logits(cont_logits, conts)

            kl_loss = self._kl_loss(all_post_stochs, all_prior_logits)
            loss = recon_loss + reward_loss + cont_loss + self.kl_scale * kl_loss

        self.model_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.model_opt.step()

        return {
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "cont_loss": cont_loss.item(),
            "kl_loss": kl_loss.item(),
            "model_loss": loss.item(),
        }

    def _kl_loss(self, post_stochs, prior_logits):
        """Compute KL divergence with free nats (vectorized over T).

        Args:
            post_stochs: (T, B, S, C) posterior stochastic states
            prior_logits: (T, B, S, C) prior logits
        """
        post_probs = post_stochs.clamp(1e-8, 1.0)
        prior_probs = F.softmax(prior_logits, dim=-1).clamp(1e-8, 1.0)

        # KL(post || prior) per (T, B, stoch_dim) → mean over B and stoch_dim
        kl_per_step = (post_probs * (post_probs.log() - prior_probs.log())).sum(dim=-1).mean(dim=(1, 2))  # (T,)
        kl_per_step = torch.clamp(kl_per_step, min=self.free_nats)
        return kl_per_step.mean()

    def _get_imagination_starts(self, B, T):
        """Get starting states for imagination rollouts.

        With imagine_starts=1 (default), uses only the final posterior state.
        With imagine_starts=K, samples K random timesteps from the WM posterior,
        giving K*B starting states for more diverse imagination (DreamerV3 paper).
        """
        K = self.imagine_starts

        if K <= 1 or not hasattr(self, '_train_all_states') or self._train_all_states is None:
            # Single start from final state
            return self._train_state_cache, B

        deter_all, h_all, stochs_all = self._train_all_states  # (T, B, ...)

        # Sample K random timesteps (without replacement if K <= T)
        indices = torch.randperm(T, device=deter_all.device)[:K]

        # Gather selected timesteps: (K, B, ...)
        sel_deter = deter_all[indices]     # (K, B, D)
        sel_h = h_all[indices]             # (K, B, D, N)
        sel_stoch = stochs_all[indices]    # (K, B, S, C)

        # Reshape to (K*B, ...)
        KB = K * B
        deter_flat = sel_deter.reshape(KB, -1)
        h_flat = sel_h.reshape(KB, sel_h.shape[-2], sel_h.shape[-1])
        stoch_flat = sel_stoch.reshape(KB, sel_stoch.shape[-2], sel_stoch.shape[-1])

        # Pack deter + ssm_h (same format as single-start)
        packed_deter = torch.cat([deter_flat, h_flat.reshape(KB, -1)], dim=-1)
        state = RSSMState(deter=packed_deter, stoch=stoch_flat)
        return state, KB

    def _train_actor_critic(self, obs, actions, B, T):
        """Train actor and critic through imagination in the world model."""
        # Get starting states (single or multi-start)
        if self._train_state_cache is not None:
            state, N_starts = self._get_imagination_starts(B, T)
        else:
            with torch.no_grad():
                with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    obs_flat = obs.reshape(B * T, *obs.shape[2:])
                    embed_flat = self.world_model.encoder(obs_flat)
                embed = embed_flat.float().reshape(B, T, -1)

                rssm = self.world_model.rssm
                if self._use_mamba:
                    _, _, _, final_d, final_s, final_h, _, _ = self._compiled_observe_seq(
                        rssm, actions, embed
                    )
                    packed = torch.cat([final_d, final_h.reshape(B, -1)], dim=-1)
                    state = RSSMState(deter=packed.detach(), stoch=final_s.detach())
                else:
                    init_state = rssm.initial_state(B, self.device)
                    _, _, _, final_d, final_s = self._compiled_observe_seq(
                        rssm, init_state.deter, init_state.stoch,
                        actions, embed, T
                    )
                    state = RSSMState(deter=final_d.detach(), stoch=final_s.detach())
                N_starts = B

        # Imagine forward — actor learns through differentiable world model dynamics.
        # Entire H-step loop compiled as single graph (same technique as observe).
        features_stack, actions_stack = self._compiled_imagine_seq(
            self.world_model.rssm, self.actor.net,
            state.deter, state.stoch,
            self.imagine_horizon, self.action_dim,
        )
        # (H+1, N, D) -> (N, H+1, D) and (H, N, A) -> (N, H, A)
        features_stack = features_stack.permute(1, 0, 2)
        actions_stack = actions_stack.permute(1, 0, 2)

        # Predict rewards and continues in imagination (no grad needed, used for targets only)
        with torch.no_grad():
            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                feat_flat = features_stack[:, 1:].reshape(-1, features_stack.shape[-1])
                reward_logits = self.world_model.reward_head(feat_flat)
                imagined_rewards = symexp(twohot_decode(reward_logits, self.reward_bins))
                imagined_rewards = imagined_rewards.reshape(N_starts, self.imagine_horizon)

                cont_logits = self.world_model.continue_head(feat_flat).reshape(N_starts, self.imagine_horizon)
                imagined_conts = torch.sigmoid(cont_logits)

        # Critic values (detach: values are only used for advantages, not differentiated)
        with torch.no_grad():
            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                values = self.critic.value(features_stack.reshape(-1, features_stack.shape[-1]))
                values = values.reshape(N_starts, self.imagine_horizon + 1)

        with torch.no_grad():
            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                slow_values = self.slow_critic.value(features_stack.reshape(-1, features_stack.shape[-1]))
                slow_values = slow_values.reshape(N_starts, self.imagine_horizon + 1)

        # Compute lambda-returns
        lambda_returns = self._compute_lambda_returns(
            imagined_rewards, imagined_conts, slow_values
        )

        # --- Critic loss ---
        with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            critic_features = features_stack[:, :-1].reshape(-1, features_stack.shape[-1]).detach()
            critic_loss = self._compiled_critic_fwd(
                self.critic.net, critic_features, lambda_returns.detach(), self.reward_bins
            )

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_opt.step()

        # --- Actor loss ---
        # Advantages (with return normalization per DreamerV3)
        with torch.no_grad():
            advantages = lambda_returns - values[:, :-1]
            # Per-batch percentile normalization
            hi = torch.quantile(lambda_returns, 0.95)
            lo = torch.quantile(lambda_returns, 0.05)
            scale = max(hi - lo, 1.0)
            advantages = advantages / scale

        # Policy gradient with entropy bonus
        with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            actor_features = features_stack[:, :-1].reshape(-1, features_stack.shape[-1])
            actor_loss, entropy = self._compiled_actor_fwd(
                self.actor.net, actor_features,
                actions_stack.reshape(-1, self.action_dim),
                advantages.reshape(-1).detach(),
                self.entropy_scale, self.action_dim,
            )

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_opt.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "imagined_reward_mean": imagined_rewards.mean().item(),
            "entropy": entropy.item(),
        }

    def _compute_lambda_returns(self, rewards, conts, values):
        """Compute lambda-returns per DreamerV3 (eq. 4).

        V_t^λ = r_t + γ c_t ((1-λ) v_{t+1} + λ V_{t+1}^λ)
        with base case V_H^λ = v_H.
        """
        H = rewards.shape[1]
        returns = torch.zeros_like(rewards)
        last = values[:, -1]

        for t in reversed(range(H)):
            returns[:, t] = rewards[:, t] + self.gamma * conts[:, t] * (
                (1 - self.lambda_) * values[:, t + 1] + self.lambda_ * last
            )
            last = returns[:, t]

        return returns

    def save(self, path: str, env_steps: int = 0, train_steps: int = 0):
        torch.save({
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "slow_critic": self.slow_critic.state_dict(),
            "model_opt": self.model_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "env_steps": env_steps,
            "train_steps": train_steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.world_model.load_state_dict(ckpt["world_model"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.slow_critic.load_state_dict(ckpt["slow_critic"])
        self.model_opt.load_state_dict(ckpt["model_opt"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        return ckpt.get("env_steps", 0), ckpt.get("train_steps", 0)
