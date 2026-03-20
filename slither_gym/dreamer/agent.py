"""DreamerV3 agent: world model training, imagination, actor-critic."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .networks import (
    WorldModel, Actor, Critic, RSSMState,
    symlog, symexp, twohot_encode, twohot_decode,
)


def _imagine_sequence(rssm, actor_net, init_deter, init_stoch, horizon, action_dim):
    """Run H steps of imagination (prior + actor). Compiled as a single graph.

    Uses Gumbel-max sampling for actions (compile-friendly). Log-prob and entropy
    are recomputed outside this function for the actor loss.
    """
    B = init_deter.shape[0]
    D = rssm.state_dim

    all_features = torch.empty(horizon + 1, B, D, device=init_deter.device)
    all_actions = torch.empty(horizon, B, action_dim, device=init_deter.device)

    deter, stoch = init_deter, init_stoch
    stoch_flat = stoch.reshape(B, -1)
    all_features[0] = torch.cat([deter, stoch_flat], dim=-1)

    for t in range(horizon):
        features = all_features[t]
        # Actor forward (just the MLP, no distribution object)
        action_logits = actor_net(features)
        # Gumbel-max sampling (equivalent to OneHotCategorical.sample())
        u = torch.zeros_like(action_logits).uniform_().clamp_(1e-20, 1 - 1e-7)
        gumbel = -(-u.log()).log()
        action = F.one_hot(
            (action_logits + gumbel).argmax(-1), action_dim
        ).to(action_logits.dtype)
        all_actions[t] = action

        state = RSSMState(deter=deter, stoch=stoch)
        next_state = rssm.imagine_step(state, action)
        deter, stoch = next_state.deter, next_state.stoch
        stoch_flat = stoch.reshape(B, -1)
        all_features[t + 1] = torch.cat([deter, stoch_flat], dim=-1)

    return all_features, all_actions


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
        entropy_scale: float = 3e-4,
        reward_bins: int = 255,
        free_nats: float = 1.0,
        kl_scale: float = 0.5,
        use_amp: bool = True,
        compile_models: bool = True,
    ):
        # TF32 gives ~1.5x matmul speedup on Ampere+ GPUs with negligible precision loss
        torch.set_float32_matmul_precision("high")

        self.device = torch.device(device)
        self.action_dim = action_dim
        self.imagine_horizon = imagine_horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_scale = entropy_scale
        self.reward_bins = reward_bins
        self.free_nats = free_nats
        self.kl_scale = kl_scale

        # Mixed precision
        self.use_amp = use_amp and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16

        # Networks
        self.world_model = WorldModel(
            action_dim=action_dim,
            embed_dim=embed_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            class_size=class_size,
            hidden_dim=hidden_dim,
            cnn_depth=cnn_depth,
            reward_bins=reward_bins,
        ).to(self.device)

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
        if self.device.type == "cuda":
            try:
                self._compiled_observe_seq = torch.compile(
                    _observe_sequence, mode="default"
                )
                self._compiled_imagine_seq = torch.compile(
                    _imagine_sequence, mode="default"
                )
            except Exception:
                self._compiled_observe_seq = _observe_sequence
                self._compiled_imagine_seq = _imagine_sequence
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

        # Final RSSM state from world-model training, reused as actor-critic starting state
        self._train_state_cache: RSSMState | None = None

        # Optimizers
        self.model_opt = torch.optim.Adam(self.world_model.parameters(), lr=lr_model, eps=1e-8)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-8)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-8)

        # Running state for acting
        self._prev_state = None
        self._prev_action = None

    def init_state(self, batch_size: int = 1):
        """Reset agent state for acting."""
        self._prev_state = self.world_model.rssm.initial_state(batch_size, self.device)
        self._prev_action = torch.zeros(batch_size, self.action_dim, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray, training: bool = True) -> int:
        """Select action given a single observation."""
        if self._prev_state is None:
            self.init_state(1)

        # Preprocess obs: (H, W, 3) uint8 -> (1, 3, H, W) float
        obs_t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

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

    def reset_state_at(self, idx: int):
        """Reset the RSSM state for a single env index (after episode boundary)."""
        if self._prev_state is None:
            return
        init = self.world_model.rssm.initial_state(1, self.device)
        # NamedTuple is immutable, but the underlying tensors are mutable in-place
        self._prev_state.deter[idx] = init.deter[0]
        self._prev_state.stoch[idx] = init.stoch[0]
        self._prev_action[idx] = 0.0

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """One training step on a batch of sequences. Returns loss metrics."""
        # Transfer obs as uint8 (4x smaller than float32) and cast on GPU.
        # Remaining tensors are small enough that transfer cost is negligible.
        obs = torch.from_numpy(batch["obs"]).to(self.device).float() / 255.0   # (B, T, 3, 64, 64)
        actions = torch.from_numpy(batch["action"]).to(self.device)             # (B, T, A)
        rewards = torch.from_numpy(batch["reward"]).to(self.device)             # (B, T)
        conts = torch.from_numpy(batch["cont"]).to(self.device)                 # (B, T)

        B, T = obs.shape[:2]

        # --- World Model Training ---
        model_metrics = self._train_world_model(obs, actions, rewards, conts, B, T)

        # --- Actor-Critic Training (imagination) ---
        ac_metrics = self._train_actor_critic(obs, actions, B, T)

        # Slow critic EMA update
        with torch.no_grad():
            for sp, tp in zip(self.slow_critic.parameters(), self.critic.parameters()):
                sp.data.lerp_(tp.data, 0.02)

        return {**model_metrics, **ac_metrics}

    def _train_world_model(self, obs, actions, rewards, conts, B, T):
        with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            # Encode all observations
            obs_flat = obs.reshape(B * T, *obs.shape[2:])
            embed_flat = self.world_model.encoder(obs_flat)
            embed = embed_flat.reshape(B, T, -1)

        # Run RSSM forward (fp32 — at B=32, dim=512 ops are memory-bandwidth bound,
        # so bfloat16 tensor cores don't help and the manual GRU adds kernel overhead).
        # Entire T-step loop compiled as single graph to eliminate Python overhead.
        embed_f32 = embed.float()
        init_state = self.world_model.rssm.initial_state(B, self.device)

        all_features, all_post_stochs, all_prior_logits, final_deter, final_stoch = \
            self._compiled_observe_seq(
                self.world_model.rssm, init_state.deter, init_state.stoch,
                actions, embed_f32, T
            )

        # Cache final posterior state (detached) for actor-critic imagination starting state.
        self._train_state_cache = RSSMState(
            deter=final_deter.detach(), stoch=final_stoch.detach()
        )
        features = all_features.permute(1, 0, 2)  # (T, B, D) -> (B, T, D)

        with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            # Decode
            feat_flat = features.reshape(B * T, -1)
            recon = self._compiled_decoder(feat_flat)
            obs_target = obs.reshape(B * T, *obs.shape[2:])

            # Image reconstruction loss (MSE on symlog-scaled pixels)
            recon_loss = F.mse_loss(recon, obs_target)

            # Reward prediction loss (twohot cross-entropy)
            reward_logits = self.world_model.reward_head(feat_flat).reshape(B, T, -1)
            reward_target = twohot_encode(
                symlog(rewards), self.reward_bins
            )
            reward_loss = -torch.sum(reward_target * F.log_softmax(reward_logits, dim=-1), dim=-1).mean()

            # Continue prediction loss (binary cross-entropy)
            cont_logits = self.world_model.continue_head(feat_flat).reshape(B, T)
            cont_loss = F.binary_cross_entropy_with_logits(cont_logits, conts)

        # KL loss — tensors already in (T, B, S, C) format from compiled sequence
        kl_loss = self._kl_loss(all_post_stochs, all_prior_logits)

        # Total
        loss = recon_loss + reward_loss + cont_loss + self.kl_scale * kl_loss

        self.model_opt.zero_grad()
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

    def _train_actor_critic(self, obs, actions, B, T):
        """Train actor and critic through imagination in the world model."""
        # Reuse the final posterior state from world-model training instead of
        # re-running the entire 50-step RSSM loop.  Falls back to full recomputation
        # if the cache is unavailable.
        if self._train_state_cache is not None:
            state = self._train_state_cache
        else:
            with torch.no_grad():
                with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    obs_flat = obs.reshape(B * T, *obs.shape[2:])
                    embed_flat = self.world_model.encoder(obs_flat)
                embed = embed_flat.float().reshape(B, T, -1)

                init_state = self.world_model.rssm.initial_state(B, self.device)
                _, _, _, final_d, final_s = self._compiled_observe_seq(
                    self.world_model.rssm, init_state.deter, init_state.stoch,
                    actions, embed, T
                )
                state = RSSMState(deter=final_d.detach(), stoch=final_s.detach())

        # Imagine forward — actor learns through differentiable world model dynamics.
        # Entire H-step loop compiled as single graph (same technique as observe).
        features_stack, actions_stack = self._compiled_imagine_seq(
            self.world_model.rssm, self.actor.net,
            state.deter, state.stoch,
            self.imagine_horizon, self.action_dim,
        )
        # (H+1, B, D) -> (B, H+1, D) and (H, B, A) -> (B, H, A)
        features_stack = features_stack.permute(1, 0, 2)
        actions_stack = actions_stack.permute(1, 0, 2)

        # Predict rewards and continues in imagination (no grad needed, used for targets only)
        with torch.no_grad():
            feat_flat = features_stack[:, 1:].reshape(-1, features_stack.shape[-1])
            reward_logits = self.world_model.reward_head(feat_flat)
            imagined_rewards = symexp(twohot_decode(reward_logits, self.reward_bins))
            imagined_rewards = imagined_rewards.reshape(B, self.imagine_horizon)

            cont_logits = self.world_model.continue_head(feat_flat).reshape(B, self.imagine_horizon)
            imagined_conts = torch.sigmoid(cont_logits)

        # Critic values (detach: values are only used for advantages, not differentiated)
        with torch.no_grad():
            values = self.critic.value(features_stack.reshape(-1, features_stack.shape[-1]))
            values = values.reshape(B, self.imagine_horizon + 1)

        slow_values = self.slow_critic.value(features_stack.reshape(-1, features_stack.shape[-1]).detach())
        slow_values = slow_values.reshape(B, self.imagine_horizon + 1)

        # Compute lambda-returns
        lambda_returns = self._compute_lambda_returns(
            imagined_rewards, imagined_conts, slow_values
        )

        # --- Critic loss ---
        critic_features = features_stack[:, :-1].reshape(-1, features_stack.shape[-1]).detach()
        critic_logits = self.critic(critic_features)
        target = symlog(lambda_returns.detach()).reshape(-1)
        target_twohot = twohot_encode(target, self.reward_bins)
        critic_loss = -torch.sum(target_twohot * F.log_softmax(critic_logits, dim=-1), dim=-1).mean()

        self.critic_opt.zero_grad()
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
        actor_features = features_stack[:, :-1].reshape(-1, features_stack.shape[-1])
        action_dist = self.actor(actor_features)
        log_probs = action_dist.log_prob(actions_stack.reshape(-1, self.action_dim))
        entropy = action_dist.entropy()

        actor_loss = -(log_probs * advantages.reshape(-1).detach()).mean()
        actor_loss -= self.entropy_scale * entropy.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_opt.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "imagined_reward_mean": imagined_rewards.mean().item(),
            "entropy": entropy.mean().item(),
        }

    def _compute_lambda_returns(self, rewards, conts, values):
        """Compute GAE-lambda returns."""
        H = rewards.shape[1]
        returns = torch.zeros_like(rewards)
        last = values[:, -1]

        for t in reversed(range(H)):
            delta = rewards[:, t] + self.gamma * conts[:, t] * last - values[:, t]
            last = values[:, t] + delta
            returns[:, t] = (1 - self.lambda_) * values[:, t] + self.lambda_ * (
                rewards[:, t] + self.gamma * conts[:, t] * last
            )
            last = returns[:, t]

        return returns

    def save(self, path: str):
        torch.save({
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "slow_critic": self.slow_critic.state_dict(),
            "model_opt": self.model_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
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
