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

### Combined result

**Full `train_step`: 2700 ms → 531 ms (5.08× speedup)** on RTX 4090.

## Approaches tested but not adopted

### bfloat16 RSSM

Tested running the RSSM loop under `torch.amp.autocast(dtype=bfloat16)`. bfloat16 has the same exponent range as fp32 (8-bit), so GRU dynamics are stable. However:

- `nn.GRUCell` doesn't participate in autocast (`_VF.gru_cell` requires matching dtypes). Required replacing with a manual GRU implementation using `F.linear`.
- The manual GRU launches more kernels than the fused C++ `nn.GRUCell`, adding overhead.
- At B=32, dim=512, the matmuls are **memory-bandwidth bound**, not compute-bound. bfloat16 tensor cores accelerate compute-bound operations but don't help here.
- **Result**: 1180 ms (slower than fp32's 1064 ms). Would only help at much larger batch sizes (B≥256+).

### Manual GRU cell (fp32)

Tested replacing `nn.GRUCell` with `F.linear`-based implementation for better torch.compile visibility. Result: 1117 ms vs 1064 ms — the fused C++ GRU kernel is still faster.

## Remaining time budget breakdown (RTX 4090, B=32, T=50)

| Phase | Time | Notes |
|---|---|---|
| Encoder forward | ~6 ms | Already fast |
| RSSM forward (50 steps) | ~112 ms | Whole-loop compiled |
| Decoder + heads forward | ~16 ms | |
| KL loss | ~4 ms | Vectorized |
| **Backward (BPTT + decoder)** | ~200 ms | **Dominant bottleneck** |
| Optimizer step | ~7 ms | |
| Actor-critic (imagination) | ~185 ms | Whole-loop compiled (15 imagine + actor + critic) |

The backward pass through 50 RSSM steps (BPTT) is the main remaining target. It's fundamentally limited by the sequential reverse traversal of the computation graph.

## Remaining approaches

### Approach 1: Custom Triton kernel for RSSM step

Fuse the GRU cell + linear layers + LayerNorm + SiLU + categorical sampling into a single hand-written Triton kernel per RSSM step. The whole-loop compilation (optimization #9) already achieves much of this via torch.compile's Triton codegen, so the remaining gain from a hand-tuned kernel would be smaller than originally estimated.

- **Expected speedup**: ~1.3-1.5x on the RSSM forward (diminishing returns after whole-loop compile)
- **Effort**: High (custom Triton kernel with backward pass)
- **Risk**: Low (no algorithmic change)

### Approach 2: Linear recurrence with parallel scan

Replace the GRU with a linear recurrence (e.g., S4, S5, Mamba-style) that supports parallel prefix scan computation: O(log T) sequential depth instead of O(T).

- **Expected speedup**: 10-20x on the sequential dimension
- **Effort**: High (architecture change, revalidation needed)
- **Risk**: High (different inductive bias, may hurt world model quality)
- **References**:
  - S5: "Simplified State Space Layers for Sequence Modeling"
  - Mamba: "Linear-Time Sequence Modeling with Selective State Spaces"

The GRU's nonlinear gates are what prevent parallelization. A linear recurrence `h_t = A * h_{t-1} + B * x_t` can be computed for all T in parallel using an associative scan. The tradeoff is that linear recurrences have weaker expressivity per step, which may require wider hidden states to compensate.

### Approach 3: Reduce sequence length

The simplest option: reduce `--seq_len` from 50 to 25.

- **Expected speedup**: ~2x on the RSSM loop (~1.5x end-to-end)
- **Effort**: None (CLI flag change)
- **Risk**: Low-medium (shorter temporal context may hurt world model predictions for long-horizon dependencies)

Can be combined with any of the above approaches.

### Approach 4: Pinned-memory replay buffer

The profiler showed pageable HtoD transfers are still significant. Pre-allocating replay buffer storage in pinned (page-locked) memory would enable true async DMA transfers via `non_blocking=True`, overlapping data transfer with computation.

- **Expected speedup**: ~1.1-1.2x end-to-end
- **Effort**: Medium (replay buffer refactor)
- **Risk**: Low (increases host memory usage, pinned memory is a limited resource)

## Recommended next steps

1. **Pinned-memory replay buffer** — straightforward engineering, eliminates remaining HtoD bottleneck
2. **Reduce seq_len** — quick experiment to validate the speedup is worth the context tradeoff
3. **Linear recurrence** — biggest potential gain but requires architecture research
