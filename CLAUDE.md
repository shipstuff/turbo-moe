# TurboMoE — Combining TurboQuant KV-Cache Compression with Flash-MoE

**Combining KV-cache compression (TurboQuant) with 4-bit MoE inference (Flash-MoE) on Apple Silicon.**

## What

TurboMoE combines two orthogonal techniques for maximum inference efficiency on Apple Silicon Macs:

| Technique | Source | What it does |
|---|---|---|
| **TurboQuant** | [Google/MLX](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) | Random rotation + Lloyd-Max 2-bit MSE quantization + QJL residual for KV-cache compression |
| **Flash-MoE** | [danveloper/github.com/danveloper/flash-moe](https://github.com/danveloper/flash-moe) | Pure C/Metal inference with FMA-optimized 4-bit dequant kernels + SSD expert streaming for MoE models |

The goal is to run MoE models with both compressed weights and compressed KV-cache on Apple Silicon.

## Why

MoE models like Qwen3.5-397B have massive weight footprints that must be streamed from SSD. Compressing the KV-cache frees unified memory bandwidth and OS page cache for expert streaming, which dominates MoE inference time.

On dense models, TurboQuant alone gives about 5.6x KV-cache compression on K/V. In this repo the two techniques stack: compress the KV-cache and the model weights.

## Status

Early development, but the core TurboQuant path is working end-to-end and integrated into the Flash-MoE inference engine.

**Primary test machine:** Apple Silicon Mac mini M4
**Development workflow:** Develop locally, validate on Apple Silicon hardware, push to GitHub.

This file is the canonical project document. `AGENTS.md` and `README.md` point here.

## Quick Start

```bash
git clone git@github.com:shipstuff/turbo-moe.git
cd turbo-moe
git submodule update --init --recursive
cd flash_moe/metal_infer
make

# Run inference
./infer --prompt "Hello world" --tokens 50

# With TurboQuant KV-cache enabled
TQ_KV=1 ./infer --prompt "Hello world" --tokens 50 -T
```

## Repository Structure

```
turbomoe/
├── CLAUDE.md              ← You are here
├── README.md              ← Public overview
├── PLAN.md                ← Full technical plan
├── flash_moe/             ← Forked flash_moe submodule (danveloper/flash-moe)
│   ├── metal_infer/
│   │   ├── infer.m        ← Complete inference engine (~7000 lines)
│   │   ├── shaders.metal  ← Metal compute kernels (base + TurboQuant)
│   │   ├── packed_experts/ ← 4-bit quantized expert weights
│   │   └── results.tsv    ← 60+ experiments documented
│   └── ...
├── turboquant/            ← TurboQuant source (ported to Metal/C)
│   ├── turboquant.metal
│   └── INTEGRATION.md
└── scripts/
    ├── prepare_model.sh
    └── benchmark.sh
```

## Model

**Qwen3.5-397B-A17B** — 397B parameter MoE, 60 layers, 512 experts, K=4 active + 1 shared expert.
- **4-bit experts** (209GB on disk): 4.36 tok/s on M3 Max, tool calling OK — production config
- **2-bit experts** (120GB on disk): 5.74 tok/s, breaks JSON/tool calling output
- **Current mini (M4 Pro) results (2026-04-12, all optimizations applied):**

Short context (1-token prompt, 64 gen tokens, warm page cache):

| Config | tok/s (best) | tok/s (typical) | cmd1_wait | cmd2_wait | expert_io | total_layer |
|--------|-------------|-----------------|-----------|-----------|-----------|-------------|
| Baseline | 8.68 | 8.5-8.7 | 1.056ms | 0.061ms | 0.656ms | 1.850ms |
| **TQ_KV=1** | **8.80** | 8.7-8.8 | 1.018ms | 0.072ms | 0.656ms | 1.825ms |
| Cold (1st run) | 5.4 | — | 1.304ms | 0.063ms | 1.545ms | 2.990ms |

Long context (258-token prompt, 2048 gen tokens):

| Context | Baseline tok/s | TQ_KV=1 tok/s | cmd1_wait | expert_io |
|---|---|---|---|---|
| ~260 tok | 4.95 | 5.27 | 1.574ms | 1.634ms |
| ~500 tok | 5.69 | — | — | — |
| ~1k tok | 5.58 | — | — | — |
| ~2k tok | 4.77 | — | — | — |
| ~2.7k tok | 4.67 | — | 1.826ms | 1.590ms |

TQ at 258-tok prompt: +6.5% gen speed, -14% TTFT, -10% cmd1_wait.
TQ advantage compounds at longer context (7.5× KV compression: 33.4 MB vs 251 MB).
At short warm context, TQ is now slightly FASTER than baseline due to CMD1 fusion
eliminating the TQ dispatch overhead.

**Optimization history (2026-04-12 session):**
- Pre-optimization baseline: 5.91 tok/s (128 tok) / 8.35 tok/s (64 tok warm)
- +CMD1+CMD2 fusion (linear + full-attn layers): +4.7%
- +Fused flash-attention with fp16 KV cache: +1-2%
- +Compute encoder consolidation: +0.5%
- +TQ CMD1 fusion: cmd2_wait 0.42→0.06ms (86% reduction)
- +Block-tiled flash-attention: bandwidth savings at long context
- **Combined: 8.35 → 8.7 tok/s (+4-7% at short context)**
- At 200k context: fused tiled attention + fp16 KV halves bandwidth vs float32

**⚠️ The older "5.86 / 5.99 / 14.72 / 14.80 tok/s" numbers that used to live in
this section were bogus** — two independent bugs were corrupting inference:
(1) a bad `expert_index.json` (Apr 10 10:35) regenerated the packed_experts
files with bytes shifted by exactly `data_start = 47245` (the safetensors file
header size), so every expert block contained bytes from neighboring tensors;
(2) every stack-allocated `InferPreadTask tasks[MAX_K]` array across 7 code
sites was not zeroing the `lz4_comp_buf`/`lz4_comp_size` fields, so
`io_pool_worker` took the LZ4 decompress branch on garbage values, pread
returned -1, and the malloc cache populated with zero-filled buffers. See
`flash_moe/benchmarks/2026-04-10-post-repack-e2e.md` for the full forensic
write-up.

**TurboQuant fixed end-to-end on 2026-04-11.** Eight separate bugs needed
to be fixed before TQ produced coherent output:
1. `tq_pack_update` was binding the per-step CPU scratch buffers to the
   kernel's `k_norms_out`/`v_norms_out` slots instead of the persistent
   per-position cache that `tq_fused_attention` reads. Norms cache stayed
   all zeros forever — the *primary* bug.
2. Gram-Schmidt rotation matrix was not orthonormal (mixed row/col indices).
3. `tq_fused_attention` Phase 3 didn't broadcast `global_max`/`global_sum`
   across the threadgroup.
4. Phase 4 inverse rotation was a scalar × column-sum, not a matmul.
5. 2-bit centroids `{-1.5,-0.5,0.5,1.5}` assume unit-variance inputs but
   `(Pi·k)/||k||` has per-dim std `1/sqrt(HEAD_DIM)`. Needed `sqrt(256)=16`
   scale at encode/decode.
6. Sigmoid gate was applied twice (once in shader Phase 5, once in
   `cmd_fused`'s `sigmoid_gate` kernel).
7. `tq_pack_update`'s `kv_head=1` norm write was guarded by `tgid==0` which
   only fires for `kv_head=0`.
8. Q was passed to `tq_fused_attention` unrotated, breaking
   `dot(Pi·Q, Pi·K) = dot(Q, K)`. Need to pre-rotate Q on CPU before the
   kernel sees `buf_attn_q`.

See `flash_moe/benchmarks/2026-04-10-post-repack-e2e.md` for the full
debugging log and how each bug was found.

## Quantization Formats

| Format | Disk | Quality | Tool Calling | Notes |
|--------|------|---------|--------------|-------|
| 4-bit experts | 209GB | Excellent | Yes | Current best |
| 2-bit experts | 120GB | Good* | No (breaks JSON) | Faster but unreliable |
| TQ KV-cache | — | ✅ Working | ✅ | 7.5x KV cache compression. Now FASTER than baseline at short ctx (8.80 vs 8.68) due to CMD1 fusion. +6.5% at medium ctx. |
| FP16 KV-cache | — | ✅ Working | ✅ | 2x KV bandwidth reduction for non-TQ path. Fused flash-attention with online softmax. |

## Performance Profile (M4 Pro, warm cache, 64 tokens, 2026-04-12)

Per-layer breakdown (after all CMD1+CMD2 fusion + encoder consolidation):
```
expert_io:      0.656ms  (35.5%) — loading expert weights from SSD via OS page cache
cmd1_wait:      1.056ms  (57.1%) — GPU projections + fused attention + routing
cmd2_wait:      0.061ms  ( 3.3%) — nearly eliminated by CMD1+CMD2 fusion
tq_kernel_time: 0.000ms  ( 0.0%) — TQ fused into CMD1
total_layer:    1.850ms
```

**Key: cmd2_wait was reduced from 0.426ms to 0.061ms** by fusing all CMD2 work
(o_proj, residual, norm, routing) into CMD1 for both linear and full-attention layers.
cmd1_wait is now the dominant GPU cost and scales with context length (attention).

**Remaining optimization targets:**
1. **Fused flash-attention tuning** — at 200k context, attention is 93% of cmd1_wait.
   Block-tiled kernel written (activates at seq_len ≥ 512) but needs perf tuning
   for longer contexts. Reference: FlashAttention-2 tiling approach.
2. **expert_io** (35.5%) — persistent expert hot-set in GPU memory. Cross-prompt
   generalization ~42%. Expert I/O overlap is architecturally impossible (routing
   depends on full attention chain — investigated and confirmed 2026-04-12).
3. **Prefill speed** — sequential prefill at ~193ms/token is the wall for long
   prompts (2.7k tokens = 8.9 min). Hybrid batch/serial switchover would help.

## What We Tried (Highlights)

### Kept
- **CMD1+CMD2 fusion** (2026-04-12): Encodes all CMD2 work (o_proj, residual, norm, routing)
  into CMD1 for both linear and full-attention layers. cmd2_wait 0.426→0.061ms (86% reduction).
  GPU RoPE kernel `fullattn_norm_rope_kv` for full-attn layers. Combined +4.7%.
- **Fused flash-attention with fp16 KV cache** (2026-04-12): Single kernel replaces
  scores+softmax+values+sigmoid_gate using online softmax. KV cache 33.6→16.8 MB/layer.
  Eliminates 25MB intermediate scores buffer at 200k context.
- **Block-tiled flash-attention** (2026-04-12): `fused_flash_attention_fp16_tiled` loads
  KV blocks into threadgroup shared memory. 8× bandwidth reduction at long context.
  Context-adaptive: tiled at seq_len >= 512, untiled below.
- **TQ CMD1 fusion** (2026-04-12): GPU Q rotation kernel + tq_pack_update + tq_fused_attention
  all encoded into CMD1. TQ now matches or exceeds baseline speed at short context.
- **Compute encoder consolidation** (2026-04-12): Share Metal compute encoders across batch
  matvec dispatches and sequential dependent kernels. cmd1_submit 0.038→0.025ms (-34%).
- **Malloc cache release after prefill** (2026-04-12): Free 18GB cache before generation
  so OS page cache reclaims memory. 3× cold TTFT + generation recovery.
- **`repack_experts_v2.py`** (2026-04-10): rebuilds `packed_experts/layer_XX.bin` using the safetensors library's own header offsets instead of the hand-maintained `expert_index.json`, which had `data_offsets[0]` stored as raw (without the `8+hdr_len` adjustment). This is the ONLY way to correctly rebuild packed experts.
- **lz4_comp_buf stack-init fix** (2026-04-10): `memset(tasks, 0, sizeof(tasks))` at every `InferPreadTask tasks[MAX_K]` declaration — eliminates the silent malloc-cache pread failure and its bogus "+87% speedup" phantom benchmark.
- FMA-optimized dequant kernel: +12% from rearrange `(nibble*scale+bias)*x` → `fma(nibble, scale*x, bias*x)`
- "Trust the OS" page cache: OS manages expert caching better than any custom cache

### Discarded (with reasoning)
- **Temporal hedge on tq_fused_attention (-H flag)**: regressed performance — dispatch overhead exceeds hedge benefit on M4 Pro, no periodic refresh stalls to hedge
- **All custom expert caching**: Metal LRU, malloc cache, LZ4 compressed cache — all slower due to GPU memory pressure or decompression overhead
- **F_RDADVISE prefetch**: net 0% — unified memory means SSD DMA stalls GPU memory controller
- **Expert file clustering**: 0% — NVMe ignores scatter at 7MB granularity
- **Temporal expert prediction**: -18% — 25% hit rate wastes SSD bandwidth
- **MLP routing predictor**: 31% accuracy — worse than temporal baseline
- **LZ4 expert compression**: -13% — decompression overhead > warm cache savings
- **GPU LUT dequant kernel**: -2% — indirect register access serializes execution
- **GPU private buffer compression**: -20% pipeline — blit cost 4×7MB > matvec savings
- **Spin-poll GPU wait**: -23% — CPU thermal competition with GPU
- **mmap expert files**: -5x — per-page fault overhead on cold data
- **Speculative early routing**: -38% — cache pollution + overhead
- **dispatch_io**: -70% — dispatch_data management overhead
- **MTP speculative decoding**: break-even — MoE I/O scales per-token unlike dense models
- **CMD3+CMD1 command buffer merge**: -5% — GPU gets 0.32ms free overlap during CPU inter-layer work from early CMD3 commit; delaying commit loses more overlap than the 0.15ms commit savings
- **Expert I/O overlap (MTLSharedEvent)**: impossible — routing is last in CMD1, no GPU work after it to overlap with. Investigated and confirmed 2026-04-12
- **ANE layer offload**: -43% — serial dependency (ANE attention → GPU experts → ANE attention) kills pipeline. Reconfirmed with mini-02 measurements 2026-04-12

## Tech Stack

- **Language:** C + Objective-C + Metal compute shaders
- **Build:** Make (Xcode toolchain on Mac)
- **Test:** Metal GPU tests + C unit tests
- **No Python** in the inference path

## Quick Start (on mini)

```bash
cd ~/projects/turbomoe/flash_moe/metal_infer
make
./infer --prompt "Explain quantum computing" --tokens 100
TQ_KV=1 ./infer --prompt "Explain quantum computing" --tokens 100 -T
```

## Upstream Dependencies

- **Flash-MoE**: github.com/danveloper/flash-moe — base C/Metal inference engine
- **TurboQuant**: Google/MLX — KV-cache compression (rotation + Lloyd-Max 2-bit + QJL)
- **Apple LLM in a Flash**: inspiration for SSD expert streaming + "Trust the OS" page cache principle

## Open Research Questions

1. **Fused flash-attention tuning**: Block-tiled kernel written (`fused_flash_attention_fp16_tiled`)
   but needs perf tuning for 200k context. Currently activates at seq_len >= 512. At 200k, KV cache
   is 200MB/layer — tiling gives 8× bandwidth reduction by sharing K/V blocks across simdgroups.
   *Status: kernel functional, correctness verified, context-adaptive dispatch in place.*
2. **Persistent expert hot-set**: Pin top-N frequent experts in GPU memory. Cross-prompt overlap ~42%.
   Expected 10-15% expert_io reduction. Routing analysis data exists.
   *Status: scoped, not implemented.*
3. **Prefill speed**: Sequential at ~193ms/token. Batch prefill (T=4) gives 10× cold-cache speedup
   but +57% warm-cache regression. Hybrid batch/serial switchover would give best of both.
   *Status: batch prefill shipped, hybrid switchover not implemented.*
4. **ANE parallel helper model**: Run 0.8B model on ANE alongside flash_moe on GPU. Proven on
   mini-02: 45 tok/s ANE, 10% GPU overhead. Product-level improvement, not tok/s.
   *Status: validated architecture, not integrated.*

## Closed Research Questions (2026-04-12)

- **Expert I/O overlap**: Architecturally impossible. Routing depends on full attention chain.
  MTLSharedEvent investigated — no GPU work after routing to overlap with. Confirmed.
- **ANE layer offload**: 43% wall-clock regression from serial dependency (ANE attention →
  GPU experts → ANE attention). Reconfirmed with mini-02 data. Blocked permanently for MoE.
- **CMD3+CMD1 merge**: GPU overlap (0.32ms) > commit savings (0.15ms). Deferred pipeline's
  early CMD3 commit is a feature. Attempted and reverted.
- **Attention kernel bf16**: At short context, attention is only 0.5% of compute (projections
  dominate). Apple Silicon accelerates fp16 not bf16. KV cache already uses fp16.
- **Speculative early routing**: -38% from cache pollution. Re-evaluated for ANE: 6.6ms
  dispatch overhead >> 1ms overlap target.
- **cmd2_wait reduction**: Already at 0.061ms (was 0.426ms). Tapped out.

## Batch prefill (cold-cache only — 2026-04-11)

`--batch-prefill T` is opt-in and cold-cache-only. Warm-cache real prompts regress at long context
(+57% at 138 tok) because per-layer batching breaks the intra-token CMD3↔CMD1 pipelining. See
`flash_moe/docs/2026-04-11-batch-prefill-scoping.md` afternoon update for the full loop-inversion
trace and the two refactor approaches investigated (multi-buffered deferred state, MoE cross-token
decoupling). Both concluded the warm-cache ceiling is GPU-saturation-bound at ~150 ms/token.
