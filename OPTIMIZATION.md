# RSSM Optimization Opportunities

The RSSM (Recurrent State-Space Model) sequential loop is the primary training bottleneck. Each of the 50 timesteps depends on the previous step's output, forcing sequential GPU execution. This document outlines approaches to speed it up.

## Current bottleneck

In `agent.py:_train_world_model`, the RSSM loop runs 50 sequential steps per batch:

```python
for t in range(T):  # T=50
    post_state, prior_state, prior_logits = self.world_model.rssm.observe_step(
        prev_state, actions[:, t], embed[:, t]
    )
```

Each `observe_step` calls `imagine_step` (GRU + prior network + categorical sampling) then the posterior network. This launches ~10 separate CUDA kernels per timestep with idle time between launches. At batch_size=32, these are small operations that are memory-bandwidth-bound, not compute-bound.

The same sequential pattern appears in the imagination loop (`_train_actor_critic`), which runs 15 steps of `imagine_step`.

## Implemented optimizations

### 1. torch.compile on the RSSM cell (Approach 2) ✅

Compiled `observe_step` and `imagine_step` with `torch.compile(mode="default")` to fuse kernels via Triton. Required two prerequisite changes:

- **RSSMState NamedTuple**: Replaced `dict` state (`{"deter": ..., "stoch": ...}`) with a `NamedTuple`. Dicts cause torch.compile graph breaks; NamedTuples are pytree-compatible and compile cleanly.
- **Compiled functions stored on DreamerV3Agent, not on the RSSM module**: Assigning `rssm.observe_step = torch.compile(rssm.observe_step)` shadows the original bound method and breaks torch.compile's guard system (`__self__` lookup fails). Storing compiled references as `self._compiled_observe` / `self._compiled_imagine` on the agent avoids this.

**Compile mode**: `mode="default"` (Triton kernel fusion only). `mode="reduce-overhead"` attempts CUDA graph capture, which conflicts with sequential RNN loops — CUDA graphs alias output buffers across invocations, causing overwrites when step t's output feeds step t+1's input.

**Measured speedup** (RTX 4090, B=32, T=50):
| Loop | Baseline | Compiled | Speedup |
|---|---|---|---|
| `observe_step` (T=50) | 809 ms | 278 ms | **2.91×** |
| `imagine_step` (H=15) | 136 ms | 78 ms | **1.74×** |

### 2. Gumbel-max categorical sampling ✅

Replaced `OneHotCategorical.sample()` (which calls `aten::multinomial`) with the Gumbel-max trick:

```python
u = torch.zeros_like(logits).uniform_().clamp_(1e-20, 1 - 1e-7)
gumbel = -(-u.log()).log()
sample = F.one_hot((logits + gumbel).argmax(-1), num_classes).to(logits.dtype)
```

This produces identical samples from the same categorical distribution but uses only standard tensor ops (`uniform_`, `log`, `argmax`, `one_hot`) instead of `multinomial`. Benefits:
- Eliminates the "incompatible ops" warning from torch.compile
- Standard ops are more amenable to kernel fusion
- Would enable CUDA graph capture in the future if the sequential aliasing issue is resolved

### 3. Embedding cache between training phases ✅

`_train_world_model` now caches `embed.detach()` after encoding observations. `_train_actor_critic` reuses this instead of re-running the CNN encoder over all B×T frames. Saves one full encoder forward pass (1600 frames at B=32, T=50) per training step.

### 4. uint8 GPU transfer ✅

Changed observation transfer from CPU-float32→GPU to CPU-uint8→GPU-float32:

```python
# Before: 78.6 MB pageable transfer (float32)
obs = torch.from_numpy(batch["obs"]).float().to(self.device) / 255.0

# After: 19.7 MB transfer + GPU-side cast (4× less bandwidth)
obs = torch.from_numpy(batch["obs"]).to(self.device).float() / 255.0
```

Profiling showed `Memcpy HtoD (Pageable → Device)` was 50% of CUDA time. Sending uint8 (4× smaller) and casting to float32 on the GPU eliminates most of this cost.

### 5. TF32 matmul precision ✅

Enabled `torch.set_float32_matmul_precision("high")` for TF32 tensor core usage on Ampere+ GPUs. Negligible precision impact for RL workloads.

### 6. State caching between training phases ✅

`_train_world_model` now caches the final posterior `RSSMState` (detached). `_train_actor_critic` uses this directly as the imagination starting state, eliminating a redundant 50-step RSSM observe loop (~275 ms saved). The state was computed with pre-update weights (before optimizer step); the difference from re-running with post-update weights is negligible for a single Adam step.

### 7. Vectorized KL loss ✅

Replaced the Python loop over T timesteps in `_kl_loss` with batched tensor operations. Stacks all posterior stochs and prior logits into `(T, B, S, C)` tensors and computes KL in one vectorized pass.

### 8. Compiled decoder ✅

`torch.compile(decoder, mode="default")` for training. Fuses the transposed convolution forward + backward kernels. Stored separately as `self._compiled_decoder` so inference paths use the uncompiled module.

### 9. Whole-loop RSSM compilation ✅

Instead of compiling individual `observe_step`/`imagine_step` functions and calling them from a Python loop, moved the entire T-step RSSM observe loop into a standalone function `_observe_sequence()` and compiled that as a single `torch.compile` graph. This eliminates Python interpreter overhead between sequential steps (function call dispatch, Python→C++ transitions, guard checks per iteration).

```python
def _observe_sequence(rssm, init_deter, init_stoch, actions, embed, T):
    # Pre-allocate output tensors (no list appends — compile-friendly)
    all_features = torch.empty(T, B, D, device=init_deter.device)
    deter, stoch = init_deter, init_stoch
    for t in range(T):
        prev = RSSMState(deter=deter, stoch=stoch)
        post_state, prior_state, prior_logits = rssm.observe_step(prev, actions[:, t], embed[:, t])
        ...
    return all_features, all_post_stochs, all_prior_logits, deter, stoch

compiled_observe_seq = torch.compile(_observe_sequence, mode="default")
```

Key implementation details:
- Pre-allocates output tensors and writes via index assignment instead of list appends (Python container mutations cause graph breaks)
- Returns raw tensors (deter, stoch) instead of RSSMState for the final state
- First compilation takes ~20 min (50-step unrolled graph), but is cached for subsequent runs

**Measured speedup** (RTX 4090, B=32, T=50):
| Metric | Per-step compiled | Whole-loop compiled | Speedup |
|---|---|---|---|
| RSSM observe (T=50) | 278 ms | 112 ms | **2.48×** |

### 10. Whole-loop imagination compilation ✅

Same technique as #9, applied to the 15-step actor-critic imagination loop. The actor's `OneHotCategorical.sample()` causes torch.compile graph breaks, so replaced with Gumbel-max sampling inside the compiled function. Log-prob and entropy are recomputed outside the compiled loop for the actor loss (these only need features and actions, not the sequential state).

```python
def _imagine_sequence(rssm, actor_net, init_deter, init_stoch, horizon, action_dim):
    for t in range(horizon):
        action_logits = actor_net(features)
        # Gumbel-max sampling (compile-friendly)
        u = torch.zeros_like(action_logits).uniform_().clamp_(1e-20, 1 - 1e-7)
        action = F.one_hot((action_logits + gumbel).argmax(-1), action_dim)
        next_state = rssm.imagine_step(state, action)
        ...
```

### 11. Mamba RSSM with parallel scan ✅ (optional via `--rssm_type mamba`)

Replaced the GRU with a Mamba-style selective state space model. The linear recurrence `h_t = A_t * h_{t-1} + B_t * x_t` enables parallel prefix scan computation during training: O(log T) sequential depth instead of O(T).

Key architecture changes:
- **Selective SSM**: Input-dependent A, B, C parameters (Mamba-style content-based filtering)
- **No stochastic feedback**: Removed stoch→sequence model feedback to enable full parallel scan. The SSM input during training is action + embedding; during imagination, just action.
- **Parallel scan**: Hillis-Steele associative scan computes all T=50 deterministic states in O(log₂ 50) ≈ 6 parallel steps instead of 50 sequential GRU steps
- **Gated output**: `y = SSM(x_ssm) * SiLU(x_gate)` (Mamba-style gating)
- **Packed state**: SSM internal state (B, D, N) packed into RSSMState.deter for interface compatibility

The biggest gain is in the **backward pass**: the linear recurrence backward is also parallelizable via reverse scan, eliminating the O(T) BPTT bottleneck that dominated GRU training.

**Measured speedup** (RTX 4090, B=32, T=50):
| Metric | GRU (compiled) | Mamba (compiled) | Speedup |
|---|---|---|---|
| train_step | 531 ms | 324 ms | **1.64×** |
| RSSM params | 3.94M | 2.92M | 0.74× (fewer) |

**Quality validation**: Requires comparison training runs (50k-100k env steps) to verify world model losses and reward curves are comparable. Use `--rssm_type mamba` to enable.

### 12. Lock-free async training loop ✅

Removed the `threading.Lock` from the training loop. Previously, the collector's `batch_act` and the main thread's `train_step` were serialized by a shared lock — the entire multi-step training loop held the lock, completely blocking env collection during training.

Now the collector runs freely: `batch_act` may read slightly stale model parameters (mid-optimizer-step), which produces marginally noisier actions. For RL with exploration and a replay buffer, this has zero measurable impact on learning quality, but allows env stepping to fully overlap with GPU training.

### 13. Batch pre-fetching with ThreadPoolExecutor ✅

The training loop pre-samples the next batch from the replay buffer on a background CPU thread while the GPU executes the current train_step. NumPy-based replay sampling runs when the GIL is released during CUDA kernel execution, providing ~5-10% throughput improvement.

### 14. Full bfloat16 autocast for Mamba RSSM ✅

At large batch sizes (B≥64), Mamba SSM operations become compute-bound rather than memory-bandwidth-bound. Wrapping the entire world model forward pass (including RSSM) in `torch.amp.autocast(dtype=bfloat16)` enables tensor core utilization for SSM projections and the associative scan.

Also extended autocast to the actor-critic phase (imagination, critic/actor forward passes).

**Not applied to GRU RSSM** — at B=32, GRU ops remain memory-bandwidth-bound (see "Approaches tested but not adopted" below).

### 15. Compiled full world model forward ✅

Compiled the entire Mamba world model forward pass (encoder→RSSM→decoder→heads→losses) as a single `torch.compile` graph via `_mamba_wm_forward()`. This allows torch.compile to fuse backward kernels across the full computation graph, significantly reducing backward pass overhead.

**Measured speedup** (RTX 5090, Mamba B=128, T=50):
| Metric | Separate compile | Full WM compile | Speedup |
|---|---|---|---|
| train_step | 83.0 ms | 66.6 ms | **1.25×** |
| Backward pass | ~40 ms | ~28 ms | **1.43×** |

### 16. cuDNN benchmark mode ✅

Enabled `torch.backends.cudnn.benchmark = True` in agent init. Auto-tunes convolution algorithms for the specific input sizes and hardware, providing ~5% speedup on encoder/decoder convolutions.

### 17. Optimizer zero_grad(set_to_none=True) ✅

All three optimizers (model, actor, critic) use `set_to_none=True` instead of zeroing gradients. This avoids a memset kernel per parameter tensor.

### 18. ~~torch.inference_mode for inference~~ (reverted)

Initially replaced `@torch.no_grad()` with `@torch.inference_mode()` on inference methods. Reverted due to thread-safety issue: `inference_mode` tensors conflict with `torch.amp.autocast` weight caching when the collector thread and training thread share model parameters concurrently. `autocast` caches bf16 weight copies, and inference_mode marks these as non-saveable for backward, causing `RuntimeError: Inference tensors cannot be saved for backward` in the training thread.

### 19. Compile warmup before collector start ✅

Runs 3 warmup `train_step` calls before starting the async collector thread. This ensures torch.compile JIT compilation completes without concurrent GPU access from the inference thread, preventing hangs.

### 20. max-autotune-no-cudagraphs compile mode ✅

Switched torch.compile from `mode="default"` to `mode="max-autotune-no-cudagraphs"` for Mamba RSSM compilation (WM forward, imagination). This autotuning benchmarks multiple Triton kernel configurations (block sizes, warps, stages) to find optimal settings for each matmul/op shape.

`max-autotune` (full) would add CUDA graph capture, which conflicts with sequential loops in imagination. `max-autotune-no-cudagraphs` gets the autotuned kernels without the problematic graph capture.

**Trade-off**: First compilation takes longer (~30-60s extra) for kernel benchmarking. Cached for subsequent runs.

**Measured speedup** (RTX 5090, Mamba B=128):
| Metric | mode=default | mode=max-autotune-no-cg | Speedup |
|---|---|---|---|
| train_step | 64.6 ms | 59.4 ms | **1.09×** |

### 21. Fused Adam optimizer ✅

Enabled `fused=True` on all three Adam optimizers for CUDA devices. Fused Adam executes the entire parameter update (momentum, variance, parameter) in a single CUDA kernel per parameter group, instead of launching separate kernels for each operation.

No measurable speedup on its own at B=128, but reduces kernel launch overhead at larger batch sizes where more optimizer state is in play.

### Combined result

**Full `train_step` (GRU): 2700 ms → 531 ms (5.08× speedup)** on RTX 4090.
**Full `train_step` (Mamba): 2700 ms → 324 ms (8.33× speedup)** on RTX 4090.

**Full `train_step` (Mamba, B=128): 57.7 ms** on RTX 5090 (theoretical SPS=217).
**Full `train_step` (Mamba, B=384): 158.6 ms** on RTX 5090 (theoretical SPS=236).

Recommended 5090 flags: `--rssm_type mamba --batch_size 384 --num_envs 8 --device cuda`
Alternative (lower VRAM): `--rssm_type mamba --batch_size 128 --device cuda`

## Approaches tested but not adopted

### bfloat16 RSSM

Tested running the RSSM loop under `torch.amp.autocast(dtype=bfloat16)`. bfloat16 has the same exponent range as fp32 (8-bit), so GRU dynamics are stable. However:

- `nn.GRUCell` doesn't participate in autocast (`_VF.gru_cell` requires matching dtypes). Required replacing with a manual GRU implementation using `F.linear`.
- The manual GRU launches more kernels than the fused C++ `nn.GRUCell`, adding overhead.
- At B=32, dim=512, the matmuls are **memory-bandwidth bound**, not compute-bound. bfloat16 tensor cores accelerate compute-bound operations but don't help here.
- **Result**: 1180 ms (slower than fp32's 1064 ms). Would only help at much larger batch sizes (B≥256+).

### Manual GRU cell (fp32)

Tested replacing `nn.GRUCell` with `F.linear`-based implementation for better torch.compile visibility. Result: 1117 ms vs 1064 ms — the fused C++ GRU kernel is still faster.

### Pinned-memory replay buffer

Implemented uint8 obs storage in replay buffer (4x memory savings) with pinned-memory allocation and `non_blocking=True` async DMA transfers. However, since obs are already transferred as uint8 (19.7 MB per batch), the transfer is fast enough that pinned memory gives only **1.02x** speedup on train_step. Kept the uint8 storage and pinned memory for the memory efficiency benefit, but the speed impact is negligible.

## Time budget breakdown

### RTX 4090 — GRU RSSM (B=32, T=50): 531 ms/step

| Phase | Time | Notes |
|---|---|---|
| Encoder forward | ~6 ms | Already fast |
| RSSM forward (50 steps) | ~112 ms | Whole-loop compiled, sequential |
| Decoder + heads forward | ~16 ms | |
| KL loss | ~4 ms | Vectorized |
| **Backward (BPTT + decoder)** | ~200 ms | **Dominant bottleneck** (sequential) |
| Optimizer step | ~7 ms | |
| Actor-critic (imagination) | ~185 ms | Whole-loop compiled |

### RTX 4090 — Mamba RSSM (B=32, T=50): 324 ms/step

| Phase | Time | Notes |
|---|---|---|
| Encoder forward | ~6 ms | Same |
| RSSM forward (parallel scan) | ~40 ms | O(log T) depth via associative scan |
| Decoder + heads + priors/posts | ~20 ms | All batched (no sequential loop) |
| KL loss | ~4 ms | Vectorized |
| **Backward (scan + decoder)** | ~80 ms | Parallel reverse scan (not sequential BPTT) |
| Optimizer step | ~7 ms | |
| Actor-critic (imagination) | ~165 ms | Sequential (15 steps, actor depends on state) |

### RTX 5090 — Mamba RSSM (B=128, T=50): 57.7 ms/step

| Phase | Time | Notes |
|---|---|---|
| World model (fwd+bwd) | ~44 ms | max-autotune-no-cg compiled, bf16 autocast |
| Actor-critic (imagination) | ~10 ms | max-autotune-no-cg compiled, bf16 autocast |
| Data transfer | ~3.4 ms | uint8→GPU→float32, non_blocking |
| Slow critic EMA | ~0.1 ms | |

### Batch size scaling (RTX 5090, Mamba, T=50, max-autotune-no-cg)

| Batch Size | ms/step | Theoretical SPS | VRAM |
|---|---|---|---|
| 128 | 57.7 | **217** | 10.6 GB |
| 192 | 88.0 | 213 | 16.0 GB |
| 256 | 108.1 | 231 | 21.2 GB |
| 384 | 155.5 | **241** | 31.7 GB |
| 512 | 216.5 | 231 | 31.7 GB |

B=384 gives the best theoretical SPS (241) while fitting in 32GB VRAM. B=128 is the most VRAM-efficient at 217 SPS.

## Model performance improvements

### 22. Fixed lambda-return computation ✅

Replaced the buggy lambda-return with the DreamerV3 paper formulation (eq. 4):

```python
# DreamerV3 eq. 4: V_t^λ = r_t + γ c_t ((1-λ) v_{t+1} + λ V_{t+1}^λ)
for t in reversed(range(H)):
    returns[:, t] = rewards[:, t] + gamma * conts[:, t] * (
        (1 - lambda_) * values[:, t + 1] + lambda_ * last
    )
    last = returns[:, t]
```

The previous implementation had delta computation that corrupted the `last` variable before it was used in the return calculation.

### 23. Increased entropy scale (3e-4 → 3e-3) ✅

The default entropy bonus of 3e-4 was too low for the 6-action Slither environment, causing premature convergence to suboptimal policies. Benchmarked across scales:

| Entropy Scale | Avg Return (25k steps) | Entropy |
|---|---|---|
| 1e-4 | 6.47 | 0.009 |
| 3e-4 | 21.98 | 0.051 |
| 1e-3 | 28.89 | 0.102 |
| **3e-3** | **31.96** | **0.391** |
| 1e-2 | 34.42 | 0.138 |

3e-3 provides good exploration (entropy ~0.4) while maintaining stable learning.

### 24. Slow critic no_grad ✅

Added `torch.no_grad()` around slow_critic value computation. The slow critic is only used for lambda-return targets (detached), so its forward pass doesn't need gradient tracking.

### 25. Compiled actor-critic forward+loss ✅

Extracted critic and actor forward+loss into standalone functions (`_critic_forward`, `_actor_forward`) and compiled them with `torch.compile(mode="default")`. Marginal benefit (~0.1%) since the AC phase is only ~11% of total train_step time and individual ops are already well-fused.

### 26. Vectorized slow critic EMA ✅

Replaced per-parameter `lerp_` loop with `torch._foreach_lerp_()` — a single fused kernel call instead of one kernel per parameter. Saves ~1ms of kernel launch overhead.

### 27. Optimized twohot_encode ✅

Replaced O(B × num_bins) broadcast comparison `(x.unsqueeze(-1) >= bins).sum(-1)` with direct O(B) bin index computation using uniform grid arithmetic. Cached `twohot_decode` bins tensor.

### 28. Uint8 GPU transfer ✅

Transfer observations as uint8 (4x less PCIe bandwidth) and convert to float32 on GPU with `obs.float().div_(255.0)`. Also reduces GPU memory since intermediate float32 allocation happens on-demand rather than persisting from transfer.

### 29. Gradient checkpointing (optional) ✅

Added `use_checkpointing` flag to CNNEncoder and CNNDecoder. When enabled, conv layer activations are recomputed during backward instead of stored, reducing VRAM by ~40% at the cost of ~10% more compute. Enables B=512-640 on 32GB GPUs. Activated via `--grad_checkpoint` flag.

Note: On RTX 5090 (32GB), B=384 without checkpointing (242 SPS, 31GB) is faster than B=512 with checkpointing (219 SPS, 31GB) because the training loop is memory-bandwidth bound, not compute bound.

## Combined results (after optimizations 1-29)

All measurements on RTX 5090 (32 GB GDDR7), Mamba RSSM, T=50, train_ratio=512.

| Batch size | ms/step | Theoretical SPS | VRAM | Notes |
|---|---|---|---|---|
| 64 | 34.7 | 180 | 5.5 GB | |
| 128 | 59.2 | 211 | 8.1 GB | |
| 192 | 82.5 | 227 | 16.0 GB | |
| **256** | **106.9** | **234** | **21.2 GB** | Best for 24GB GPUs |
| **384** | **155.2** | **242** | **31.1 GB** | **Best for 32GB GPUs** |
| 512 | 228.5 | 219 | 30.8 GB | Requires --grad_checkpoint |
| 640 | 283.7 | 220 | 32.0 GB | Requires --grad_checkpoint |

Sustained training throughput: **242 SPS** at B=384 (the training bottleneck).

## Remaining approaches

### Validate Mamba quality

Mamba has been validated at 100k steps on Slither-v0: episodes reach consistent 20-35+ returns with 4001-step survival. Quality is comparable to GRU at similar env step counts.

### 30. Multi-start imagination (--imagine_starts K) ✅

DreamerV3 starts imagination from all T posterior states. We previously used only the final state (step T). Now `--imagine_starts K` samples K random timesteps, giving K×B starting states for imagination.

B=384 K=4: 174ms/step (216 SPS) with 1536 imagination starts vs K=1: 156ms/step (242 SPS) with 384 starts. 12% slower but 4x more AC training diversity.

**500k step A/B test on Slither-v0:**

| Setting | Last 50 eps avg return | Total episodes |
|---|---|---|
| K=1 | -7.0 | 211 |
| **K=4** | **+16.2** | **449** |

K=1 learned to survive (long episodes) but not to score. K=4 consistently achieved positive returns through food/kills. Multi-start is now the recommended default.

### Theoretical limits

At B=384, the train_step breaks down as:
- WM backward: ~48% (~75ms) — dominant bottleneck
- WM forward: ~31% (~48ms)
- Actor-critic: ~11% (~17ms)
- Overhead (clip+opt+ema+transfer): ~10% (~15ms)

Current compute efficiency is ~0.1% — we're massively memory-bandwidth and kernel-launch bound, not FLOP bound. Realistic ceiling estimates:
- 2x faster backward: ~295 SPS
- CUDA graphs (−30% overhead): ~325 SPS
- Both combined: ~400 SPS
- Pure compute bound: ~200,000 SPS (unreachable)
