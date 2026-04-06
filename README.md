# TurboMoE

**Combining KV-cache compression (TurboQuant) with 4-bit MoE inference (Flash-MoE) on Apple Silicon.**

---

## What

TurboMoE combines two techniques for maximum inference efficiency on Apple Silicon Macs:

| Technique | Source | What it does |
|---|---|---|
| **TurboQuant** | Google/MLX | 5.6x KV-cache compression via rotation + Lloyd-Max quantization + QJL residual |
| **Flash-MoE** | danveloper | 4-bit dequantization with FMA kernels + SSD expert streaming for MoE models |

Both techniques are **orthogonal** and stack — compress the KV-cache *and* the model weights.

## Why

MoE models like Qwen3.5-397B have massive weight footprints that must be streamed from SSD. Compressing the KV-cache frees up unified memory bandwidth and OS page cache for the expert streaming that dominates MoE inference time.

On dense models, TurboQuant alone gives 5.6x KV-cache compression (92 bytes vs 512 bytes per token at head_dim=128).

## Status

Early development. See [CLAUDE.md](CLAUDE.md) for full context and [PLAN.md](PLAN.md) for the technical roadmap.

## Quick Start

```bash
# On your Mac mini
git clone ~/turbomoe  # or pull latest
cd turbomoe
make

# Run inference
./infer --prompt "Hello world" --tokens 50

# With TurboQuant KV-cache
./infer --prompt "Hello world" --tokens 50 --turboquant
```

## Models

Development currently targets `hauhau-qwen35-9b` (5.2GB, dense) for fast iteration. The architecture is model-agnostic and applies to any transformer with KV-cache.

## Build

```bash
make        # full build
make test   # unit tests
make clean  # nuke build artifacts
```
