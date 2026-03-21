"""PPO training with Cython vectorized environments and async data collection."""

import argparse
import copy
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Prevent CPU thread over-subscription which causes severe slowdowns
# on small models. Scale up for larger models / GPU training.
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

from puffer_rl.engine import VecSlither
from puffer_rl.model import MLPLSTMPolicy


# --------------------------------------------------------------------------- #
#  Rollout collection                                                          #
# --------------------------------------------------------------------------- #

def collect_rollout(engine, model, obs, lstm_state, device, rollout_len,
                    num_envs, obs_dim):
    """Collect a rollout of (rollout_len, num_envs) transitions.

    Returns dict of numpy arrays + final obs/lstm_state for bootstrapping.
    """
    buf_obs     = np.zeros((rollout_len, num_envs, obs_dim), dtype=np.float32)
    buf_act     = np.zeros((rollout_len, num_envs), dtype=np.int64)
    buf_logp    = np.zeros((rollout_len, num_envs), dtype=np.float32)
    buf_val     = np.zeros((rollout_len, num_envs), dtype=np.float32)
    buf_rew     = np.zeros((rollout_len, num_envs), dtype=np.float32)
    buf_done    = np.zeros((rollout_len, num_envs), dtype=np.float32)

    ep_returns = []
    ep_lengths = []
    ep_snake_lengths = []

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

            # Reset LSTM for done envs
            done_mask = torch.from_numpy(dones).bool().to(device)
            if done_mask.any():
                lstm_state = (
                    lstm_state[0] * (~done_mask).float().unsqueeze(0).unsqueeze(-1),
                    lstm_state[1] * (~done_mask).float().unsqueeze(0).unsqueeze(-1),
                )

            # Log completed episodes
            for i in range(num_envs):
                if dones[i]:
                    ep_returns.append(float(ep_ret[i]))
                    ep_lengths.append(int(ep_len[i]))
                    ep_snake_lengths.append(int(ep_slen[i]))

        # Bootstrap value
        obs_t = torch.from_numpy(obs).to(device)
        _, _, _, bootstrap_val, _ = model.get_action_and_value(obs_t, lstm_state)
        bootstrap_val = bootstrap_val.cpu().numpy()

    return {
        "obs": buf_obs,
        "actions": buf_act,
        "logprobs": buf_logp,
        "values": buf_val,
        "rewards": buf_rew,
        "dones": buf_done,
        "bootstrap_value": bootstrap_val,
        "final_obs": obs,
        "final_lstm_state": lstm_state,
        "ep_returns": ep_returns,
        "ep_lengths": ep_lengths,
        "ep_snake_lengths": ep_snake_lengths,
    }


# --------------------------------------------------------------------------- #
#  Async collector (double-buffered, overlaps collection with GPU training)    #
# --------------------------------------------------------------------------- #

class AsyncCollector:
    """Runs rollout collection in a background thread using a CPU model copy."""

    def __init__(self, engine, gpu_model, obs, lstm_state, device,
                 rollout_len, num_envs, obs_dim):
        self.engine = engine
        self.device = device
        self.rollout_len = rollout_len
        self.num_envs = num_envs
        self.obs_dim = obs_dim

        # CPU copy of the model for inference during collection
        self.cpu_model = copy.deepcopy(gpu_model).cpu().eval()
        self.gpu_model = gpu_model

        self.obs = obs
        # Move LSTM state to CPU for collection
        self.lstm_state = (lstm_state[0].cpu(), lstm_state[1].cpu())

        self._result_queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._weights_updated = threading.Event()
        self._thread = None

    def start(self):
        """Start background collection thread."""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            rollout = collect_rollout(
                self.engine, self.cpu_model, self.obs, self.lstm_state,
                "cpu", self.rollout_len, self.num_envs, self.obs_dim,
            )
            self.obs = rollout["final_obs"]
            self.lstm_state = rollout["final_lstm_state"]
            self._result_queue.put(rollout)

            # Wait for weights to be updated before next collection
            self._weights_updated.wait()
            self._weights_updated.clear()

    def get_rollout(self):
        """Block until next rollout is ready."""
        return self._result_queue.get()

    def update_weights(self):
        """Copy latest GPU model weights to the CPU model."""
        cpu_sd = {k: v.cpu() for k, v in self.gpu_model.state_dict().items()}
        self.cpu_model.load_state_dict(cpu_sd)
        self._weights_updated.set()

    def stop(self):
        self._stop.set()
        self._weights_updated.set()  # unblock if waiting
        if self._thread:
            self._thread.join(timeout=5)


# --------------------------------------------------------------------------- #
#  GAE computation                                                             #
# --------------------------------------------------------------------------- #

def compute_gae(rewards, values, dones, bootstrap_value, gamma, gae_lambda):
    """Compute GAE advantages and returns.

    All inputs are numpy arrays of shape (T, N).
    Returns (advantages, returns) of same shape.
    """
    T, N = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = bootstrap_value
        else:
            next_val = values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# --------------------------------------------------------------------------- #
#  PPO update                                                                  #
# --------------------------------------------------------------------------- #

def ppo_update(model, optimizer, rollout, initial_lstm_state,
               clip_ratio, epochs, num_minibatches, ent_coef, vf_coef,
               max_grad_norm, device, amp_dtype=torch.float32):
    """PPO clipped objective update with LSTM.

    Minibatches split across environments (not time) to preserve sequences.
    """
    T = rollout["obs"].shape[0]
    N = rollout["obs"].shape[1]
    mb_size = N // num_minibatches

    # Move to GPU
    obs_t      = torch.from_numpy(rollout["obs"]).to(device)       # (T,N,D)
    act_t      = torch.from_numpy(rollout["actions"]).to(device)   # (T,N)
    old_logp_t = torch.from_numpy(rollout["logprobs"]).to(device)  # (T,N)
    adv_t      = torch.from_numpy(rollout["advantages"]).to(device)
    ret_t      = torch.from_numpy(rollout["returns"]).to(device)
    done_t     = torch.from_numpy(rollout["dones"]).to(device)     # (T,N)

    metrics = {"pg_loss": 0, "vf_loss": 0, "entropy": 0, "clipfrac": 0}
    num_updates = 0

    for _ in range(epochs):
        perm = np.random.permutation(N)
        for start in range(0, N, mb_size):
            end = start + mb_size
            if end > N:
                break
            idx = perm[start:end]

            mb_obs  = obs_t[:, idx]
            mb_act  = act_t[:, idx]
            mb_oldlp = old_logp_t[:, idx]
            mb_adv  = adv_t[:, idx]
            mb_ret  = ret_t[:, idx]
            mb_done = done_t[:, idx]
            mb_h0   = initial_lstm_state[0][:, idx].contiguous()
            mb_c0   = initial_lstm_state[1][:, idx].contiguous()

            # Normalise advantages per minibatch
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            with torch.amp.autocast("cuda", dtype=amp_dtype,
                                    enabled=amp_dtype != torch.float32):
                _, new_logp, entropy, new_val, _ = model.get_action_and_value(
                    mb_obs, (mb_h0, mb_c0), action=mb_act, done=mb_done
                )

                ratio = torch.exp(new_logp - mb_oldlp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio,
                                    1.0 + clip_ratio) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()
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
#  Main training loop                                                          #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="PPO + Cython Slither")
    # Environment
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    # Training
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
    # Model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--lstm_dim", type=int, default=128)
    # Rewards
    p.add_argument("--food_reward", type=float, default=1.5)
    p.add_argument("--kill_reward", type=float, default=10.0)
    p.add_argument("--death_scale", type=float, default=0.1)
    p.add_argument("--survival_bonus", type=float, default=-0.005)
    p.add_argument("--boost_cost", type=float, default=-0.01)
    # System
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model for faster training")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable mixed-precision (bf16) training")
    p.add_argument("--async_collect", action="store_true",
                   help="Overlap collection (CPU) with training (GPU)")
    p.add_argument("--logdir", type=str, default="runs/puffer_ppo")
    p.add_argument("--save_every", type=int, default=100_000)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Resume from checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- build engine ----
    engine = VecSlither(
        n_envs=args.num_envs, seed=args.seed,
        food_reward=args.food_reward, kill_reward=args.kill_reward,
        death_scale=args.death_scale, survival_bonus=args.survival_bonus,
        boost_cost=args.boost_cost,
    )
    obs_dim = engine.obs_dim
    act_dim = 6
    print(f"Engine: {args.num_envs} envs, obs_dim={obs_dim}, act_dim={act_dim}")

    # ---- build model ----
    model = MLPLSTMPolicy(
        obs_dim=obs_dim, act_dim=act_dim,
        hidden_dim=args.hidden_dim, lstm_dim=args.lstm_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    # torch.compile for kernel fusion
    if args.compile and device.type == "cuda":
        model = torch.compile(model)
        print("torch.compile enabled")

    # Mixed precision
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    if use_amp:
        print(f"Mixed precision: {amp_dtype}")

    # Resume checkpoint
    start_step = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {args.checkpoint} at step {start_step}")

    # ---- logging ----
    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(args.logdir)

    # ---- initial state ----
    obs = engine.reset_all()
    lstm_state = model.get_initial_state(args.num_envs, device)
    initial_lstm_h = lstm_state[0].clone()
    initial_lstm_c = lstm_state[1].clone()

    steps_per_update = args.num_envs * args.rollout_len
    num_updates = (args.total_steps - start_step) // steps_per_update
    global_step = start_step

    # ---- async collector ----
    collector = None
    if args.async_collect and device.type == "cuda":
        collector = AsyncCollector(
            engine, model, obs, lstm_state, device,
            args.rollout_len, args.num_envs, obs_dim,
        )
        collector.start()
        print("Async collection enabled (CPU inference + GPU training)")

    print(f"Training for {num_updates} updates "
          f"({steps_per_update} steps/update, total {args.total_steps} steps)")
    print(f"  PPO: epochs={args.ppo_epochs}, minibatches={args.num_minibatches}, "
          f"clip={args.clip_ratio}, ent={args.ent_coef}")

    t_start = time.time()
    recent_returns = []
    recent_lengths = []
    recent_snake_lengths = []

    for update in range(num_updates):
        t_update = time.time()

        # ---- collect rollout ----
        if collector is not None:
            rollout = collector.get_rollout()
        else:
            # Store initial LSTM state for training
            initial_lstm_h = lstm_state[0].clone()
            initial_lstm_c = lstm_state[1].clone()
            rollout = collect_rollout(
                engine, model, obs, lstm_state, device,
                args.rollout_len, args.num_envs, obs_dim,
            )
            obs = rollout["final_obs"]
            lstm_state = rollout["final_lstm_state"]

        # Track episode stats
        recent_returns.extend(rollout["ep_returns"])
        recent_lengths.extend(rollout["ep_lengths"])
        recent_snake_lengths.extend(rollout["ep_snake_lengths"])

        # ---- compute GAE ----
        advantages, returns = compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"],
            rollout["bootstrap_value"], args.gamma, args.gae_lambda,
        )
        rollout["advantages"] = advantages.astype(np.float32)
        rollout["returns"] = returns.astype(np.float32)

        # ---- PPO update ----
        if collector is not None:
            # For async, initial LSTM state was from before this rollout
            # We need to store it from the collection side
            # Use zeros as approximation (rollout is long enough)
            init_lstm = model.get_initial_state(args.num_envs, device)
        else:
            init_lstm = (initial_lstm_h.to(device), initial_lstm_c.to(device))

        metrics = ppo_update(
            model, optimizer, rollout, init_lstm,
            args.clip_ratio, args.ppo_epochs, args.num_minibatches,
            args.ent_coef, args.vf_coef, args.max_grad_norm, device,
            amp_dtype,
        )

        # Update async collector weights
        if collector is not None:
            collector.update_weights()

        global_step += steps_per_update
        dt = time.time() - t_update
        sps = steps_per_update / dt

        # ---- logging ----
        if len(recent_returns) > 0 and (update + 1) % 10 == 0:
            mean_ret = np.mean(recent_returns[-100:])
            mean_len = np.mean(recent_lengths[-100:])
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

        # ---- checkpoint ----
        if global_step % args.save_every < steps_per_update:
            ckpt_path = os.path.join(args.logdir, f"ckpt_{global_step}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # ---- final save ----
    final_path = os.path.join(args.logdir, "final.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": global_step,
        "args": vars(args),
    }, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")

    if collector is not None:
        collector.stop()
    writer.close()


if __name__ == "__main__":
    main()
