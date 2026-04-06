# TurboMoE — Technical Plan

## Goal

Combine TurboQuant's KV-cache compression (Google/MLX, 5.6x) with Flash-MoE's FMA-optimized 4-bit dequantization and SSD expert streaming, targeting Apple Silicon for maximum inference efficiency.

---

## Part 0: Fork and Set Up

```bash
# On GitHub: fork danveloper/flash_moe → carl-shipstuff/flash_moe
# Clone locally
git clone https://github.com/carl-shipstuff/flash_moe.git flash_moe
cd flash_moe && git remote rename origin upstream

# Add turboquant source
mkdir -p turboquant
```

**Subtree vs submodule decision:** Use a **subtree** (not submodule) for flash_moe so we can make modifications to the Metal shaders freely and track our changes. Submodules are too rigid for this project since we need to modify the attention pipeline.

---

## Part 1: TurboQuant KV-Cache Encoding (Pre-Forward)

The TurboQuant pipeline from `turboquant-mlx` compresses each K/V vector as:

```
Input: (D,) fp32 vector after RoPE
Step 1: Rotation      x_rot = R @ x          (D, D) random orthonormal matrix
Step 2: MSE quantize  idx, norm              2-bit Lloyd-Max codebook lookup
Step 3: Residual      r = x_rot - cent[idx]  QJL sketch of residual
Step 4: Pack          2-bit indices → uint32 (16 indices per word)
```

**Memory layout per token (head_dim=128, 2-bit):**
```
Key packed indices:     128/16 * 4 = 32 bytes (uint32, 2-bit packed)
Key sign bits (QJL):     128/32 * 4 = 16 bytes (uint32)
Key norm:                 4 bytes (float32)
Key residual norm:        4 bytes (float32)
Value packed indices:    32 bytes
Value norm:               4 bytes
Total:                   92 bytes vs 512 bytes fp16 = 5.6x compression
```

### Metal Kernel: `turboquant_encode`

The encode kernel runs at **KV-cache update time** (pre-forward), not at decode time.

**Input:** `(D,)` fp32 key/value after RoPE
**Output:** packed 2-bit indices + norms + (optional) QJL sign bits

**Challenge:** The rotation matmul `(D,D) @ (D,)` naively is O(D²). But we can pre-compute the combined rotation+JL matrix and do it in one pass, or use the fact that R is orthonormal (so we only need to apply it, not compute R⁻¹).

For the initial implementation, do the rotation on CPU (it's O(D²) once per token which is fine for pre-forward) and focus Metal effort on the decode path which runs at every attention lookup.

### What to Port First

1. **`codebook.py`** → `codebook.c` — Pre-compute Lloyd-Max centroids offline, embed as constant arrays. Do this once, store in a header.
2. **`rotation.py`** → `rotation.c` — Generate random orthonormal rotation matrix, store as constant `float R[D*D]`.
3. **`fused_qjl.py`** → `qjl.c` — QJL encode (Hadamard transform + sign accumulation). The Metal kernel for this is the most custom part.

**Simplify for v1:** Skip QJL residual. Use only MSE quantization (rotation + Lloyd-Max 2-bit). This gives **~3x compression** instead of 5.6x, but with a much simpler kernel. The QJL path can be added once the basics work.

### v1 Target Format

```
Per token, head_dim=128, 2-bit:
  Key packed indices:  32 bytes  (uint32, 2-bit packed)
  Key norm:             4 bytes   (float32)
  Value packed indices: 32 bytes  (uint32, 2-bit packed)
  Value norm:           4 bytes   (float32)
  Total:               72 bytes vs 512 fp16 = 7.1x compression
```

---

## Part 2: TurboQuant KV-Cache Decoding (Attention Forward)

This is the **hot path** — runs at every attention layer, every token. Needs to be fast.

```
Input:  packed 2-bit indices + norms
Output: (D,) fp32 key/value for SDPA
Step 1: Unpack indices, lookup centroids  → cent[idx]
Step 2: Multiply by norm                  → cent[idx] * norm
```

**Metal Kernel: `turboquant_decode`**

```metal
kernel void turboquant_decode(
    device const uint32_t* packed_indices,  // (D/16,) uint32 — 16 x 2-bit per word
    device const float* k_norms,            // () float32 per token
    device const uint32_t* centroids,       // (16, D) pre-computed Lloyd-Max centroids
    device float* k_out,                    // (D,) reconstructed fp32
    constant uint& D,
    uint tid [[thread_position_in_grid]]
) {
    uint packed_idx = tid / 16;
    uint within_packed = tid % 16;
    uint nibble = (packed_indices[packed_idx] >> (within_packed * 2)) & 0x3;
    
    // Lookup centroid: centroids[nibble * D + tid]
    // Multiply by norm
    k_out[tid] = centroids[nibble * D + tid] * k_norms[0];
}
```

This is a simple lookup + multiply. SIMD group reduction to sum across threads for the final output vector. Very fast.

---

## Part 3: Integrate into Flash-MoE Attention Pipeline

Flash-MoE's attention path (from `infer.m`):

```
For each attention layer:
  1. Attention projection: Q, K, V from input
     - gate_proj / up_proj → SwiGLU
     - wq/wk/wv projections (4-bit dequant matvec via dequant_matvec_4bit_fma)
  2. K/V → store in KV-cache
  3. Attention: batched_gpu_attention(Q, K_cache, V_cache)
  4. Output: o_proj + residual + RMS norm
```

**Where TurboQuant plugs in:**

- **Step 2:** Instead of storing fp16 K/V, store TurboQuant-encoded (packed 2-bit + norms).
- **Step 3:** Before batched GPU attention, run `turboquant_decode` to reconstruct fp32 K/V on-the-fly.
  - Alternatively: fuse `turboquant_decode` directly into the attention kernel so K never materializes as full fp32 in HBM — decode directly into registers.
- **Step 1:** KV-cache for Q doesn't need compression (only one Q vector per token), skip.

**Key insight:** Flash-MoE already streams expert weights from SSD. The KV-cache lives in unified memory and is accessed at every layer. Compressing it gives:
- More room for the OS page cache (which Flash-MoE relies on for expert streaming)
- Less memory bandwidth for KV-cache reads during attention
- On 397B MoE models: 15 attention layers × 60 total layers = 25% of layers benefit immediately

---

## Part 4: FMA-Optimized TurboQuant Decode

Flash-MoE discovered that `(nibble * scale + bias) * x` → `fma(nibble, scale*x, bias*x)` gives 12% speedup. TurboQuant decode has a similar structure: `centroid[idx] * norm`. The centroid lookup is essentially the "scale" and `norm` is the multiplier.

We can fuse the centroid lookup + norm multiply into a single FMA-style kernel:

```metal
// Instead of:
//   float c = centroids[nibble * D + tid];
//   k_out[tid] = c * norm;

// Fuse: pre-compute centroids already scaled
// centroids_scaled[nibble * D + tid] = centroids[nibble * D + tid] * norm;
// Then just lookup:
//   k_out[tid] = centroids_scaled[nibble * D + tid];

// Or better: since norm is per-token (not per-head), do it outside the inner loop
```

Actually the norm multiply is just one scalar × vector multiply, not worth FMA-izing. The bigger win is fusing the decode directly into the attention kernel.

---

## Part 5: Testing Plan

### Unit Tests

```bash
# test_turboquant_encode: generate random vectors, encode, verify round-trip
./test_turboquant_encode

# test_turboquant_decode: decode packed → fp32, compare to original (within quant error)
./test_turboquant_decode

# test_flashmoe_attn: run attention with TurboQuant cache vs fp16 cache, measure quality
./test_flashmoe_attn
```

### Integration Test

```bash
# Run inference with TurboQuant KV-cache
./infer --prompt "Explain quantum entanglement" --tokens 100 --turboquant

# Compare output quality to baseline (no TurboQuant)
./infer --prompt "Explain quantum entanglement" --tokens 100

# Benchmark
./infer --prompt "Hello" --tokens 50 --timing --turboquant
```

---

## Memory Math

For `hauhau-qwen35-9b` (dense, 9B params, ~18GB at fp16):
- KV-cache at 4096 ctx: 2 (K+V) × 4096 ctx × 36 layers × 8 kv_heads × 128 head_dim × 2 bytes = ~72 MB fp16
- With TurboQuant 2-bit: ~10 MB
- Savings: 62 MB per 4096 ctx

For `Qwen3.5-397B-A17B` (MoE, 397GB at 4-bit):
- KV-cache at 4096 ctx: ~72 MB fp16 (same structure, fewer KV heads in MoE)
- With TurboQuant 2-bit: ~10 MB
- Plus 4-bit expert weights: ~209 GB on disk, ~6.75 MB per active expert streamed from SSD
- Combined: KV-cache compression + weight compression = massive memory headroom

---

## Open Questions

1. **QJL worth it?** The QJL residual adds 5.6x vs 3x compression but adds significant kernel complexity. Start without it.
2. **Decode fusion:** Should turboquant_decode run as a separate kernel or be fused into the attention kernel? Fused = fewer memory round-trips but more complex kernel.
3. **Rotation on encode:** The random rotation matrix R is currently generated at cache init time. Should it be fixed (seeded) or regenerated per-sequence? Fixed is simpler and deterministic.
4. **Model compatibility:** The 9B model is dense, not MoE. We test the KV-cache path first; the MoE expert streaming stays unchanged.

---

## Implementation Order

1. [ ] Fork flash_moe, add as subtree
2. [ ] Extract TurboQuant codebook → C constants
3. [ ] Write `turboquant_encode.metal` (CPU rotation + Metal quantize)
4. [ ] Write `turboquant_decode.metal` (Metal decode kernel)
5. [ ] Write tests for encode/decode round-trip
6. [ ] Integrate into `infer.m` — replace KV-cache storage with TurboQuant
7. [ ] Fused decode into attention kernel
8. [ ] Benchmark: quality vs compression ratio
9. [ ] Add QJL residual path (optional v2)
10. [ ] If/when Qwen3.5-397B is available: test on MoE model
