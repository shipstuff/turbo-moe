# TurboMoE — Context Dump

**Generated:** 2026-04-06
**Session:** Discord #llm / TurboQuant + Flash-MoE combination planning

---

## What We Combined

### TurboQuant MLX (`sharpner/turboquant-mlx`)
- **What:** KV-cache quantization from Google's TurboQuant paper (2025) on Apple Silicon via MLX
- **Repo:** https://github.com/sharpner/turboquant-mlx
- **Key files:**
  - `turboquant/cache.py` — TurboQuantKVCache, drop-in replacement for mlx-lm KVCache
  - `turboquant/attention.py` — attention patching into mlx-lm
  - `turboquant/rotation.py` — random QR rotation matrix generation
  - `turboquant/qjl.py` — QJL (Johnson-Lindenstrauss) residual encoding
  - `turboquant/codebook.py` — Lloyd-Max MSE quantizer codebook
  - `turboquant/fused_qjl.py` — custom Metal kernel for QJL sign-bit scoring
- **Compression:** 5.6x on KV-cache (92 bytes vs 512 bytes fp16 per token at head_dim=128)
- **Benchmark results (from README):**
  - V2 3-bit rot+QJL: Llama 3.2 3B +5.3%, Llama 3.1 8B +7.8%, Mistral 7B +5.1%, Gemma 3 4B -1.1%
  - V2 4-bit rotated: Llama 3.2 3B -0.8%, Llama 3.1 8B +1.4%, Mistral 7B +1.4%, Gemma 3 4B +2.9%
- **Tech:** Python/MLX, Metal custom kernels for QJL

### Flash-MoE (`danveloper/flash-moe`)
- **What:** Pure C/Metal inference engine running Qwen3.5-397B-A17B (397B param MoE) on MacBook Pro M3 Max 48GB
- **Repo:** https://github.com/danveloper/flash-moe (3.4k stars)
- **Key files:**
  - `metal_infer/infer.m` — ~7000 line inference engine (C/ObjC)
  - `metal_infer/shaders.metal` — Metal compute shaders (1200+ lines)
  - `metal_infer/chat.m` — interactive chat TUI with tool calling
  - `metal_infer/Makefile` — build system
  - `metal_infer/tokenizer.h` — C BPE tokenizer (449 lines, single-header)
- **Architecture:** 60 layers: 45 GatedDeltaNet (linear attention) + 15 standard full attention
- **Key techniques:**
  1. SSD expert streaming — 512 experts per layer, K=4 activated + 1 shared, ~6.75MB each
  2. FMA-optimized dequant kernel — `(nibble * scale + bias) * x` → `fma(nibble, scale*x, bias*x)`, +12%
  3. Hand-written Metal kernels: dequant matvec, SwiGLU fused, RMS norm, GPU RoPE, attention
  4. "Trust the OS" page cache — no custom cache, 71% hit rate naturally
  5. Deferred CMD3 execution — GPU expert forward runs while CPU prepares next layer
  6. Accelerate BLAS for delta-net linear attention
- **Results:** 4.36 tok/s on M3 Max 48GB, full tool calling at 4-bit, 209GB on disk
- **Per-layer pipeline (4.28ms avg at 4-bit):**
  - CMD3(prev) → CMD1: attn proj + delta-net [1.22ms GPU]
  - CPU flush [0.01ms]
  - CMD2: o_proj + norm + routing + shared [0.55ms GPU]
  - CPU softmax + topK [0.003ms]
  - I/O: parallel pread K=4 experts [2.41ms SSD]
  - CMD3: expert fwd + combine + norm [DEFERRED]

---

## Mini 01 Scout (carl@192.168.0.61)

### Machine
- Mac mini M4 (Apple Silicon)
- `/users/shared/` is the main data volume
- Git available: `git version 2.50.1`
- CPU: Apple Silicon ARM

### Models on Mini 01

| Path | Model | Size | Notes |
|---|---|---|---|
| `/users/shared/models/carl/models/hf/hauhau-qwen35-9b/` | HauHau Qwen3.5 9B | 5.2GB | Dense, good for dev |
| `/users/shared/models/carl/models/hf/jackrong-qwen35-27b/` | JackRong Qwen3.5 27B | 15GB | Dense, larger |
| `/users/shared/models/huggingface/hub/models--google--gemma-3-12b-it/` | Gemma 3 12B | ~24GB | Sparse (HF format, incomplete) |
| `/users/shared/models/huggingface/hub/models--mlx-community--gemma-3-12b-it-4bit/` | Gemma 3 12B 4bit MLX | ~6GB | MLX format |
| `/users/shared/models/huggingface/hub/models--prince-canuma--LTX-2.3-distilled/` | LTX 2.3 Distilled | ~2GB | Video model |

### Local (WSL/skynet) Models

| Path | Model | Size |
|---|---|---|
| `~/models/hf/hauhau-qwen35-9b/` | Qwen3.5 9B | 5.2GB |
| `~/models/hf/jackrong-qwen35-27b/` | Qwen3.5 27B | 15GB |
| `/mnt/models/carl/models/hf/*` | Same as mini | — |
| `~/models/ltx23/` | LTX 2.3 22B dev/distilled | ~107GB total |
| `/mnt/models/exo/mlx-community--Qwen3-Coder-Next-6bit/` | Qwen3 Coder Next 6bit | 8 files |
| `/mnt/models/exo/mlx-community--Qwen3-Coder-Next-8bit/` | Qwen3 Coder Next 8bit | 17 files |

### NAS (SMB mount, `/mnt/models/`)

- `//192.168.0.99/models` — 21TB volume, 11TB used
- Full model library including everything above plus:
  - `Step-3.5-Flash/` — Step 3.5 Flash model
  - `step3p5_flash_Q4_K_S.gguf*` — Multi-part GGUF (10+ parts)
  - `huggingface/hub/` — HF cache

### Selected Dev Model: `hauhau-qwen35-9b`

Reasoning: Dense model, 5.2GB, same model on both mini and local. TurboQuant KV-cache works on attention layers (which this has), so we can test the integration without needing an MoE model. The MoE expert streaming from Flash-MoE doesn't apply, but the Metal compute pipeline and TurboQuant integration can be validated.

**Note:** No MoE model is currently available on the mini. If Qwen3.5-397B becomes available, it can be added later — the architecture is designed to be model-agnostic.

---

## Network / SSH Access

- **Mini 01:** `carl@192.168.0.61` ✓ (passwordless SSH confirmed)
- **Mini 02:** `192.168.0.62`
- **Mini 03:** `192.168.0.63`
- **No SSH key configured** in `~/.ssh/config` — using default key auth
- **WSL mirror networking:** `192.168.0.61` accessible directly from WSL

---

## Repo Plan

```
~/projects/turbomoe/          ← Created on mini
├── .git/                     ← Git repo initialized
├── CLAUDE.md                 ← Project overview + status
├── README.md                 ← Public overview
├── PLAN.md                   ← Full technical plan
└── [flash_moe subtree]       ← Fork → carl-shipstuff/flash_moe

Development workflow:
1. Edit files in WSL: ~/projects/turbomoe/
2. Push to mini: rsync -avz ~/projects/turbomoe/ carl@192.168.0.61:~/turbomoe/
   OR git push from mini (but files live in WSL)
```

**Decision: Use git push from mini.** Initialize git on mini, add a remote that pushes to a bare repo accessible from WSL, or simply work on the mini directly. Given that the mini is the test machine, the cleanest approach is:

1. Repo lives on mini at `~/turbomoe`
2. From WSL, we SSH in to edit/test
3. For now: use rsync for file transfer until git remote is set up

---

## Implementation Plan (from PLAN.md)

1. Fork `danveloper/flash_moe` → `carl-shipstuff/flash_moe`
2. Add as git subtree in `~/turbomoe/flash_moe`
3. Port TurboQuant codebook → C constants
4. Write `turboquant_encode.metal` (CPU rotation + Metal quantize)
5. Write `turboquant_decode.metal` (Metal decode kernel)
6. Write encode/decode round-trip tests
7. Integrate into `infer.m` — replace KV-cache storage with TurboQuant
8. Fused decode into attention kernel
9. Benchmark: quality vs compression ratio
10. Add QJL residual path (optional v2)
