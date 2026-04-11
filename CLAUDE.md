# TurboMoE — Combining TurboQuant KV-Cache Compression with Flash-MoE

## Project Overview

Combines two orthogonal techniques:
- **TurboQuant** (Google/MLX): Random rotation + Lloyd-Max 2-bit MSE quantization + QJL residual for KV-cache compression (5.6x on K/V)
- **Flash-MoE** (danveloper/github.com/danveloper/flash-moe): Pure C/Metal inference with FMA-optimized 4-bit dequant kernels + SSD expert streaming for MoE models

The goal is to run MoE models with both compressed weights AND compressed KV-cache on Apple Silicon.

**Primary test machine:** Mac mini M4 (carl@192.168.0.61)
**Development workflow:** Develop on WSL, push to mini over SSH, test there.

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
│   ├── INTEGRATION.md
│   └── infer.m.patch
└── scripts/
    ├── prepare_model.sh
    └── benchmark.sh
```

## Model

**Qwen3.5-397B-A17B** — 397B parameter MoE, 60 layers, 512 experts, K=4 active + 1 shared expert.
- **4-bit experts** (209GB on disk): 4.36 tok/s on M3 Max, tool calling OK — production config
- **2-bit experts** (120GB on disk): 5.74 tok/s, breaks JSON/tool calling output
- **Current mini (M4 Pro) results (2026-04-11 morning session, TurboQuant fixed!):**

| Config | Tokens | tok/s | Quality | Notes |
|--------|--------|-------|---------|-------|
| Baseline (no cache) | 128 | 5.91 | ✅ Good | OS page cache; 104-token coherent story |
| **TQ_KV=1 (fixed)** | 128 | 5.31 | ✅ Good | KV cache 33.4 MB (7.5x compression) |
| **TQ_KV=1 (fixed)** | 256 | 5.15 | ✅ Good | 256 coherent tokens, no drift |
| malloc-cache-64 | 128 | 6.13 | ✅ Good | 0% hit (thrashing — 64 slots vs 240 active/tok) |
| malloc-cache-512 | 96 | 6.07 | ✅ Good | 32.2% hit rate (8569/26640) |
| `--predict` | 128 | 2.51 | ✅ Good | 26% hit rate, -58% speed net regression |

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
| TQ KV-cache | — | ✅ Working | ✅ | 7.5x KV cache compression, ~12% gen slowdown at 128 tok |

## Performance Profile (M4 Pro, baseline no-cache, 128 tokens)

Per-layer breakdown:
```
expert_io:      1.337ms  (49.5%) — loading expert weights from SSD via OS page cache
cmd1_wait:      0.858ms  (31.8%) — GPU attention kernel
cmd2_wait:      0.426ms  (15.8%) — residual/normalization kernel
tq_kernel_time: 0.000ms  ( 0.0%) — TurboQuant disabled
total_layer:    2.703ms
```

**Research targets for further optimization:**
1. **expert_io** (49.5%) — persistent expert working set in GPU memory, expert clustering by co-activation patterns, mixed-bit compression per expert based on sensitivity analysis
2. **cmd1_wait** (31.8%) — reduced-precision attention inner loops (fp16/bf16 vs fp32), IO-aware attention kernel tiling for Apple Silicon GPU shared memory
3. **Actually-working TurboQuant** — once the CPU-cache-corruption bug is root-caused, the 4 shader-level fixes already in place should bring TQ to approximate correctness

## What We Tried (Highlights)

### Kept
- **`repack_experts_v2.py`** (2026-04-10): rebuilds `packed_experts/layer_XX.bin` using the safetensors library's own header offsets instead of the hand-maintained `expert_index.json`, which had `data_offsets[0]` stored as raw (without the `8+hdr_len` adjustment). This is the ONLY way to correctly rebuild packed experts.
- **lz4_comp_buf stack-init fix** (2026-04-10): `memset(tasks, 0, sizeof(tasks))` at every `InferPreadTask tasks[MAX_K]` declaration — eliminates the silent malloc-cache pread failure and its bogus "+87% speedup" phantom benchmark.
- TQ latency histogram instrumentation (-T flag): zero stalls detected, p99.99=0.424ms — TQ dispatch path is stall-free (even if it produces wrong answers)
- MODEL_PATH_DEFAULT fixed to /Users/carl/models/Qwen3.5-397B-A17B-4bit
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

1. **expert_io optimization**: Can a small persistent "expert working set" stay hot in GPU memory across
   layer transitions without being evicted by memory pressure?
3. **Mixed-bit experts**: Some experts may be more quantization-sensitive than others. Per-expert
   bit-width optimization could preserve quality while reducing memory bandwidth.
4. **Attention kernel register tiling**: Apple Silicon GPU has fixed shared memory. Better tiling
   could reduce cmd1_wait (currently 33.8% of layer time).
