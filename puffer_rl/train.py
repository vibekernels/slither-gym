"""PPO training with Cython vectorized environments.

Supports both pure-MLP (default, fully parallel) and MLP-LSTM models.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Prevent CPU thread over-subscription on small models.
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

from puffer_rl.engine import VecSlither
from puffer_rl.model import MLPPolicy, MLPLSTMPolicy


# --------------------------------------------------------------------------- #
#  Rollout collection                                                          #
# --------------------------------------------------------------------------- #

def collect_rollout_mlp(engine, model, obs, device, rollout_len,
                        num_envs, obs_dim):
    """Collect rollout with a pure-MLP model (no hidden state)."""
    buf_obs  = np.zeros((rollout_len, num_envs, obs_dim), dtype=np.float32)
    buf_act  = np.zeros((rollout_len, num_envs), dtype=np.int64)
    buf_logp = np.zeros((rollout_len, num_envs), dtype=np.float32)
    buf_val  = np.zeros((rollout_len, num_envs), dtype=np.float32)
    buf_rew  = np.zeros((rollout_len, num_envs), dtype=np.float32)
    buf_done = np.zeros((rollout_len, num_envs), dtype=np.float32)

    ep_returns, ep_lengths, ep_snake_lengths = [], [], []

    with torch.no_grad():
        for t in range(rollout_len):
            buf_obs[t] = obs
            obs_t = torch.from_numpy(obs).to(device)
            action, logprob, _, value = model.get_action_and_value(obs_t)
            act_np = action.cpu().numpy()
            buf_act[t]  = act_np
            buf_logp[t] = logprob.cpu().numpy()
            buf_val[t]  = value.cpu().numpy()

            obs, rewards, dones, ep_ret, ep_len, ep_slen = engine.step(
                act_np.astype(np.intc)
            )
            buf_rew[t]  = rewards
            buf_done[t] = dones.astype(np.float32)

            for i in range(num_envs):
                if dones[i]:
                    ep_returns.append(float(ep_ret[i]))
                    ep_lengths.append(int(ep_len[i]))
                    ep_snake_lengths.append(int(ep_slen[i]))

        # Bootstrap
        obs_t = torch.from_numpy(obs).to(device)
        _, _, _, bootstrap_val = model.get_action_and_value(obs_t)
        bootstrap_val = bootstrap_val.cpu().numpy()

    return {
        "obs": buf_obs, "actions": buf_act, "logprobs": buf_logp,
        "values": buf_val, "rewards": buf_rew, "dones": buf_done,
        "bootstrap_value": bootstrap_val, "final_obs": obs,
        "ep_returns": ep_returns, "ep_lengths": ep_lengths,
        "ep_snake_lengths": ep_snake_lengths,
    }


def collect_rollout_lstm(engine, model, obs, lstm_state, device,
                         rollout_len, num_envs, obs_dim):
    """Collect rollout with LSTM model (maintains hidden state)."""
    buf_obs  = np.zeros((rollout_len, num_envs, obs_dim), dtype=np.float32)
    buf_act  = np.zeros((rollout_len, num_envs), dtype=np.int64)
    buf_logp = np.zeros((rollout_len, num_envs), dtype=np.float32)
    buf_val  = np.zeros((rollout_len, num_envs), dtype=np.float32)
    buf_rew  = np.zeros((rollout_len, num_envs), dtype=np.float32)
    buf_done = np.zeros((rollout_len, num_envs), dtype=np.float32)

    ep_returns, ep_lengths, ep_snake_lengths = [], [], []

    with torch.no_grad():
        for t in range(rollout_len):
            buf_obs[t] = obs
            obs_t = torch.from_numpy(obs).to(device)
            action, logprob, _, value, lstm_state = model.get_action_and_value(
                obs_t, lstm_state
            )
            act_np = action.cpu().numpy()
            buf_act[t]  = act_np
            buf_logp[t] = logprob.cpu().numpy()
            buf_val[t]  = value.cpu().numpy()

            obs, rewards, dones, ep_ret, ep_len, ep_slen = engine.step(
                act_np.astype(np.intc)
            )
            buf_rew[t]  = rewards
            buf_done[t] = dones.astype(np.float32)

            done_mask = torch.from_numpy(dones).bool().to(device)
            if done_mask.any():
                lstm_state = (
                    lstm_state[0] * (~done_mask).float().unsqueeze(0).unsqueeze(-1),
                    lstm_state[1] * (~done_mask).float().unsqueeze(0).unsqueeze(-1),
                )

            for i in range(num_envs):
                if dones[i]:
                    ep_returns.append(float(ep_ret[i]))
                    ep_lengths.append(int(ep_len[i]))
                    ep_snake_lengths.append(int(ep_slen[i]))

        obs_t = torch.from_numpy(obs).to(device)
        _, _, _, bootstrap_val, _ = model.get_action_and_value(obs_t, lstm_state)
        bootstrap_val = bootstrap_val.cpu().numpy()

    return {
        "obs": buf_obs, "actions": buf_act, "logprobs": buf_logp,
        "values": buf_val, "rewards": buf_rew, "dones": buf_done,
        "bootstrap_value": bootstrap_val, "final_obs": obs,
        "final_lstm_state": lstm_state,
        "ep_returns": ep_returns, "ep_lengths": ep_lengths,
        "ep_snake_lengths": ep_snake_lengths,
    }


# --------------------------------------------------------------------------- #
#  GAE                                                                         #
# --------------------------------------------------------------------------- #

def compute_gae(rewards, values, dones, bootstrap_value, gamma, gae_lambda):
    T, N = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_val = bootstrap_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


# --------------------------------------------------------------------------- #
#  PPO update — MLP version (fully random minibatches, no sequential deps)     #
# --------------------------------------------------------------------------- #

def ppo_update_mlp(model, optimizer, rollout, clip_ratio, epochs,
                   num_minibatches, ent_coef, vf_coef, max_grad_norm,
                   device, amp_dtype=torch.float32):
    T, N = rollout["obs"].shape[:2]
    total = T * N
    mb_size = total // num_minibatches

    # Flatten (T, N, ...) → (T*N, ...)
    obs_t      = torch.from_numpy(rollout["obs"].reshape(total, -1)).to(device)
    act_t      = torch.from_numpy(rollout["actions"].reshape(total)).to(device)
    old_logp_t = torch.from_numpy(rollout["logprobs"].reshape(total)).to(device)
    adv_t      = torch.from_numpy(rollout["advantages"].reshape(total)).to(device)
    ret_t      = torch.from_numpy(rollout["returns"].reshape(total)).to(device)

    metrics = {"pg_loss": 0, "vf_loss": 0, "entropy": 0, "clipfrac": 0}
    num_updates = 0

    for _ in range(epochs):
        perm = torch.randperm(total, device=device)
        for start in range(0, total, mb_size):
            end = min(start + mb_size, total)
            idx = perm[start:end]

            mb_obs   = obs_t[idx]
            mb_act   = act_t[idx]
            mb_oldlp = old_logp_t[idx]
            mb_adv   = adv_t[idx]
            mb_ret   = ret_t[idx]

            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            with torch.amp.autocast("cuda", dtype=amp_dtype,
                                    enabled=amp_dtype != torch.float32):
                _, new_logp, entropy, new_val = model.get_action_and_value(
                    mb_obs, action=mb_act
                )
                ratio = torch.exp(new_logp - mb_oldlp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_ratio,
                                    1 + clip_ratio) * mb_adv
                pg_loss  = -torch.min(surr1, surr2).mean()
                vf_loss  = 0.5 * ((new_val - mb_ret) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss + vf_coef * vf_loss - ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            metrics["pg_loss"]  += pg_loss.item()
            metrics["vf_loss"]  += vf_loss.item()
            metrics["entropy"]  += ent_loss.item()
            metrics["clipfrac"] += (
                ((ratio - 1.0).abs() > clip_ratio).float().mean().item()
            )
            num_updates += 1

    return {k: v / max(num_updates, 1) for k, v in metrics.items()}


# --------------------------------------------------------------------------- #
#  PPO update — LSTM version (env-grouped minibatches)                         #
# --------------------------------------------------------------------------- #

def ppo_update_lstm(model, optimizer, rollout, initial_lstm_state,
                    clip_ratio, epochs, num_minibatches, ent_coef, vf_coef,
                    max_grad_norm, device, amp_dtype=torch.float32):
    T, N = rollout["obs"].shape[:2]
    mb_size = N // num_minibatches

    obs_t      = torch.from_numpy(rollout["obs"]).to(device)
    act_t      = torch.from_numpy(rollout["actions"]).to(device)
    old_logp_t = torch.from_numpy(rollout["logprobs"]).to(device)
    adv_t      = torch.from_numpy(rollout["advantages"]).to(device)
    ret_t      = torch.from_numpy(rollout["returns"]).to(device)
    done_t     = torch.from_numpy(rollout["dones"]).to(device)

    metrics = {"pg_loss": 0, "vf_loss": 0, "entropy": 0, "clipfrac": 0}
    num_updates = 0

    for _ in range(epochs):
        perm = np.random.permutation(N)
        for start in range(0, N, mb_size):
            end = start + mb_size
            if end > N:
                break
            idx = perm[start:end]
            mb_h0 = initial_lstm_state[0][:, idx].contiguous()
            mb_c0 = initial_lstm_state[1][:, idx].contiguous()
            mb_adv = adv_t[:, idx]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            with torch.amp.autocast("cuda", dtype=amp_dtype,
                                    enabled=amp_dtype != torch.float32):
                _, new_logp, entropy, new_val, _ = model.get_action_and_value(
                    obs_t[:, idx], (mb_h0, mb_c0),
                    action=act_t[:, idx], done=done_t[:, idx],
                )
                ratio = torch.exp(new_logp - old_logp_t[:, idx])
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_ratio,
                                    1 + clip_ratio) * mb_adv
                pg_loss  = -torch.min(surr1, surr2).mean()
                vf_loss  = 0.5 * ((new_val - ret_t[:, idx]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss + vf_coef * vf_loss - ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            metrics["pg_loss"]  += pg_loss.item()
            metrics["vf_loss"]  += vf_loss.item()
            metrics["entropy"]  += ent_loss.item()
            metrics["clipfrac"] += (
                ((ratio - 1.0).abs() > clip_ratio).float().mean().item()
            )
            num_updates += 1

    return {k: v / max(num_updates, 1) for k, v in metrics.items()}


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="PPO + Cython Slither")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_steps", type=int, default=5_000_000)
    p.add_argument("--rollout_len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--num_minibatches", type=int, default=4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--lstm", action="store_true",
                   help="Use MLP-LSTM instead of pure MLP")
    p.add_argument("--lstm_dim", type=int, default=128)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--logdir", type=str, default="runs/puffer_ppo")
    p.add_argument("--save_every", type=int, default=100_000)
    p.add_argument("--checkpoint", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_lstm = args.lstm

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- engine ----
    engine = VecSlither(n_envs=args.num_envs, seed=args.seed)
    obs_dim = engine.obs_dim
    act_dim = 6
    print(f"Engine: {args.num_envs} envs, obs_dim={obs_dim}")

    # ---- model ----
    if use_lstm:
        model = MLPLSTMPolicy(obs_dim=obs_dim, act_dim=act_dim,
                              hidden_dim=args.hidden_dim,
                              lstm_dim=args.lstm_dim).to(device)
    else:
        model = MLPPolicy(obs_dim=obs_dim, act_dim=act_dim,
                          hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    total_params = sum(p.numel() for p in model.parameters())
    kind = "MLP-LSTM" if use_lstm else "MLP"
    print(f"Model: {kind}, {total_params:,} parameters")

    if args.compile and device.type == "cuda":
        model = torch.compile(model)
        print("torch.compile enabled")

    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    if use_amp:
        print(f"Mixed precision: {amp_dtype}")

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {args.checkpoint} at step {start_step}")
    else:
        start_step = 0

    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(args.logdir)

    # ---- initial state ----
    obs = engine.reset_all()
    lstm_state = None
    init_lstm = None
    if use_lstm:
        lstm_state = model.get_initial_state(args.num_envs, device)
        init_lstm = (lstm_state[0].clone(), lstm_state[1].clone())

    steps_per_update = args.num_envs * args.rollout_len
    num_updates = (args.total_steps - start_step) // steps_per_update
    global_step = start_step

    print(f"Training for {num_updates} updates "
          f"({steps_per_update} steps/update, total {args.total_steps} steps)")
    print(f"  PPO: epochs={args.ppo_epochs}, mb={args.num_minibatches}, "
          f"clip={args.clip_ratio}, ent={args.ent_coef}")

    t_start = time.time()
    recent_returns = []
    recent_lengths = []
    recent_snake_lengths = []

    for update in range(num_updates):
        t_update = time.time()

        # ---- collect ----
        if use_lstm:
            init_lstm = (lstm_state[0].clone(), lstm_state[1].clone())
            rollout = collect_rollout_lstm(
                engine, model, obs, lstm_state, device,
                args.rollout_len, args.num_envs, obs_dim,
            )
            obs = rollout["final_obs"]
            lstm_state = rollout["final_lstm_state"]
        else:
            rollout = collect_rollout_mlp(
                engine, model, obs, device,
                args.rollout_len, args.num_envs, obs_dim,
            )
            obs = rollout["final_obs"]

        recent_returns.extend(rollout["ep_returns"])
        recent_lengths.extend(rollout["ep_lengths"])
        recent_snake_lengths.extend(rollout["ep_snake_lengths"])

        # ---- GAE ----
        adv, ret = compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"],
            rollout["bootstrap_value"], args.gamma, args.gae_lambda,
        )
        rollout["advantages"] = adv.astype(np.float32)
        rollout["returns"] = ret.astype(np.float32)

        # ---- PPO ----
        if use_lstm:
            metrics = ppo_update_lstm(
                model, optimizer, rollout,
                (init_lstm[0].to(device), init_lstm[1].to(device)),
                args.clip_ratio, args.ppo_epochs, args.num_minibatches,
                args.ent_coef, args.vf_coef, args.max_grad_norm,
                device, amp_dtype,
            )
        else:
            metrics = ppo_update_mlp(
                model, optimizer, rollout,
                args.clip_ratio, args.ppo_epochs, args.num_minibatches,
                args.ent_coef, args.vf_coef, args.max_grad_norm,
                device, amp_dtype,
            )

        global_step += steps_per_update
        dt = time.time() - t_update
        sps = steps_per_update / dt

        # ---- log ----
        if len(recent_returns) > 0 and (update + 1) % 10 == 0:
            mean_ret  = np.mean(recent_returns[-100:])
            mean_len  = np.mean(recent_lengths[-100:])
            mean_slen = np.mean(recent_snake_lengths[-100:])
            writer.add_scalar("charts/episode_return", mean_ret, global_step)
            writer.add_scalar("charts/episode_length", mean_len, global_step)
            writer.add_scalar("charts/snake_length", mean_slen, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/pg_loss", metrics["pg_loss"], global_step)
            writer.add_scalar("losses/vf_loss", metrics["vf_loss"], global_step)
            writer.add_scalar("losses/entropy", metrics["entropy"], global_step)
            writer.add_scalar("losses/clipfrac", metrics["clipfrac"], global_step)

            elapsed = time.time() - t_start
            print(f"[{global_step:>8d}] "
                  f"ret={mean_ret:+.2f}  len={mean_len:.0f}  "
                  f"slen={mean_slen:.1f}  sps={sps:.0f}  "
                  f"pg={metrics['pg_loss']:.4f}  vf={metrics['vf_loss']:.4f}  "
                  f"ent={metrics['entropy']:.4f}  "
                  f"clip={metrics['clipfrac']:.3f}  "
                  f"elapsed={elapsed:.0f}s")

        if global_step % args.save_every < steps_per_update:
            ckpt_path = os.path.join(args.logdir, f"ckpt_{global_step}.pt")
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step, "args": vars(args)}, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(args.logdir, "final.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": global_step, "args": vars(args)}, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")
    writer.close()


if __name__ == "__main__":
    main()
