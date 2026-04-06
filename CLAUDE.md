# TurboMoE — Combining TurboQuant KV-Cache Compression with Flash-MoE

## Project Overview

Combines two orthogonal techniques:
- **TurboQuant** (Google/MLX): Random rotation + Lloyd-Max 2-bit MSE quantization + QJL residual for KV-cache compression (5.6x on K/V)
- **Flash-MoE** (danveloper): Pure C/Metal inference with FMA-optimized 4-bit dequant kernels + SSD expert streaming for MoE models

The goal is to run MoE models with both compressed weights AND compressed KV-cache on Apple Silicon.

**Primary test machine:** Mac mini M4 (mini-01, `carl@192.168.0.61`)
**Development workflow:** Clone locally on WSL, push to mini over SSH, test there.

## Repository Structure

```
turbomoe/
├── CLAUDE.md              ← You are here
├── README.md              ← Public overview
├── PLAN.md                ← Full technical plan
├── flash_moe/             ← Forked flash_moe submodule
│   ├── metal_infer/
│   │   ├── infer.m
│   │   ├── shaders.metal
│   │   └── Makefile
│   └── ...
├── turboquant/            ← TurboQuant source (ported to Metal/C)
│   ├── turboquant_encode.metal
│   ├── turboquant_decode.metal
│   └── turboquant_decode.m
├── models/                ← Model weights (symlink to mini's model dir)
├── tests/
│   ├── test_turboquant_encode.m
│   ├── test_turboquant_decode.m
│   └── test_flashmoe_attn.m
└── scripts/
    ├── prepare_model.sh
    └── benchmark.sh
```

## Model

Currently targeting `hauhau-qwen35-9b` (5.2GB, dense) on mini for development.

The integration architecture is **model-agnostic** — the approach applies to any transformer with KV-cache, with the most dramatic gains on MoE models where attention layers also benefit from KV compression on top of expert weight streaming.

## Tech Stack

- **Language:** C + Objective-C + Metal compute shaders
- **Build:** Make (Xcode toolchain on Mac)
- **Test:** Metal GPU tests + C unit tests
- **No Python** in the inference path

## Quick Start (on mini)

```bash
cd ~/turbomoe
git pull origin main
make
./infer --prompt "Hello world" --tokens 50
```

## Status

- [x] Repo initialized, plan documented
- [ ] Clone flash_moe fork
- [ ] Port TurboQuant encode kernel to Metal
- [ ] Port TurboQuant decode kernel to Metal
- [ ] Integrate into flash_moe attention pipeline
- [ ] Build and test on mini M4
- [ ] Benchmark: KV-cache compression ratio vs quality
- [ ] (Stretch) Support Qwen3.5-397B-A17B if model becomes available
