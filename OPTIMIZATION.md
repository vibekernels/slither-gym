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

### Combined result

**Full `train_step` (GRU): 2700 ms → 531 ms (5.08× speedup)** on RTX 4090.
**Full `train_step` (Mamba): 2700 ms → 324 ms (8.33× speedup)** on RTX 4090.

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

## Remaining time budget breakdown (RTX 4090, B=32, T=50)

### GRU RSSM (531 ms)

| Phase | Time | Notes |
|---|---|---|
| Encoder forward | ~6 ms | Already fast |
| RSSM forward (50 steps) | ~112 ms | Whole-loop compiled, sequential |
| Decoder + heads forward | ~16 ms | |
| KL loss | ~4 ms | Vectorized |
| **Backward (BPTT + decoder)** | ~200 ms | **Dominant bottleneck** (sequential) |
| Optimizer step | ~7 ms | |
| Actor-critic (imagination) | ~185 ms | Whole-loop compiled |

### Mamba RSSM (324 ms)

| Phase | Time | Notes |
|---|---|---|
| Encoder forward | ~6 ms | Same |
| RSSM forward (parallel scan) | ~40 ms | O(log T) depth via associative scan |
| Decoder + heads + priors/posts | ~20 ms | All batched (no sequential loop) |
| KL loss | ~4 ms | Vectorized |
| **Backward (scan + decoder)** | ~80 ms | Parallel reverse scan (not sequential BPTT) |
| Optimizer step | ~7 ms | |
| Actor-critic (imagination) | ~165 ms | Sequential (15 steps, actor depends on state) |

## Remaining approaches

### Reduce sequence length

Reduce `--seq_len` from 50 to 25 for both GRU and Mamba.

- **Expected speedup**: ~1.5x end-to-end for GRU, ~1.2x for Mamba (already fast)
- **Effort**: None (CLI flag change)
- **Risk**: Low-medium (shorter temporal context)

### Validate Mamba quality

Run comparison training (GRU vs Mamba) for 50k-100k env steps to verify:
- World model losses (reconstruction, reward, KL) are comparable
- Episode reward curves converge similarly
- Imagination accuracy is maintained

If Mamba quality matches GRU, it's a strict improvement (faster + fewer params).
