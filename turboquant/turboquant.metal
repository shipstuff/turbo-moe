// turboquant.metal — TurboQuant KV-cache compression for flash-moE
//
// Compresses K/V from fp32 → 2-bit Lloyd-Max quantized + random rotation.
// Attention runs on compressed data without full decompression.
//
// Storage format (per KV head, per token):
//   uint32 k_packed[TQ_WORDS_PER_HEAD=16]: 2-bit centroid indices, 16 per word
//   float32 k_norm:  L2 norm
//   Total: 68 bytes vs 512 bytes fp16 = 7.5x compression
//
// Constants must match infer.m:
//   HEAD_DIM=256, NUM_ATTN_HEADS=32, NUM_KV_HEADS=2, TQ_WORDS_PER_HEAD=16

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================
constant int HEAD_DIM           = 256;
constant int NUM_ATTN_HEADS     = 32;
constant int NUM_KV_HEADS       = 2;
constant int TQ_WORDS_PER_HEAD   = HEAD_DIM / 16;    // 16
constant int LANES_PER_SIMD     = 8;
constant int SIMD_SIZE         = 32;
constant int THREADS_PER_HEAD  = 256;
constant int NUM_CENTROIDS      = 4;
constant float RMS_EPS          = 1e-6f;

// 2-bit Lloyd-Max centroids: {-1.5, -0.5, +0.5, +1.5}
// Actual value = centroid[idx] * L2_norm
constant float CENTROIDS[NUM_CENTROIDS] = { -1.5f, -0.5f, 0.5f, 1.5f };

// ============================================================================
// Kernel 0: tq_encode_packed
// ============================================================================
//
// Grid: (TQ_WORDS_PER_HEAD * NUM_KV_HEADS * T_new, 1, 1) = (32 * T_new, 1, 1)
// Threadgroup: (16, 1, 1)
//
// Inputs:
//   k_new, v_new:        [T_new, NUM_KV_HEADS, HEAD_DIM] flat fp32
//   rotation_matrix:     [HEAD_DIM, HEAD_DIM] row-major Pi
//   k_norms_in, v_norms_in: [T_new, NUM_KV_HEADS] precomputed L2 norms
//
// Outputs:
//   k_packed_out, v_packed_out: [T_new, NUM_KV_HEADS, TQ_WORDS_PER_HEAD] uint32

kernel void tq_encode_packed(
    device const float  *k_new,
    device const float  *v_new,
    device const float  *rotation_matrix,
    device const float  *k_norms_in,
    device const float  *v_norms_in,
    device       uint32_t *k_packed_out,
    device       uint32_t *v_packed_out,
    constant     uint32_t &T_new,
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]]
) {
    uint item = tgid;
    uint items_per_token = NUM_KV_HEADS * TQ_WORDS_PER_HEAD;
    uint token_idx = item / items_per_token;
    uint rem = item % items_per_token;
    uint kv_head = rem / TQ_WORDS_PER_HEAD;
    uint word_idx = rem % TQ_WORDS_PER_HEAD;

    if (token_idx >= T_new) return;

    uint k_base = (token_idx * NUM_KV_HEADS + kv_head) * HEAD_DIM;
    uint v_base = (token_idx * NUM_KV_HEADS + kv_head) * HEAD_DIM;

    float inv_k_norm = 1.0f / (k_norms_in[token_idx * NUM_KV_HEADS + kv_head] + 1e-8f);
    float inv_v_norm = 1.0f / (v_norms_in[token_idx * NUM_KV_HEADS + kv_head] + 1e-8f);

    uint32_t k_packed = 0;
    uint32_t v_packed = 0;

    for (int i = 0; i < 16; i++) {
        uint d = word_idx * 16 + i;
        if (d >= HEAD_DIM) break;

        float k_dot = 0.0f;
        float v_dot = 0.0f;
        uint mat_base = d * HEAD_DIM;
        for (int j = 0; j < HEAD_DIM; j++) {
            k_dot += k_new[k_base + j] * rotation_matrix[mat_base + j];
            v_dot += v_new[v_base + j] * rotation_matrix[mat_base + j];
        }
        float k_rot = k_dot * inv_k_norm;
        float v_rot = v_dot * inv_v_norm;

        uint k_idx = (k_rot >= 1.0f) ? 3u : (k_rot >= 0.0f) ? 2u : (k_rot >= -1.0f) ? 1u : 0u;
        uint v_idx = (v_rot >= 1.0f) ? 3u : (v_rot >= 0.0f) ? 2u : (v_rot >= -1.0f) ? 1u : 0u;

        k_packed |= (k_idx & 0x3u) << (i * 2);
        v_packed |= (v_idx & 0x3u) << (i * 2);
    }

    uint out_base = (token_idx * NUM_KV_HEADS + kv_head) * TQ_WORDS_PER_HEAD;
    k_packed_out[out_base + word_idx] = k_packed;
    v_packed_out[out_base + word_idx] = v_packed;
}


// ============================================================================
// Kernel 1: tq_fused_attention
// ============================================================================
//
// Fused single-token generation attention with TurboQuant KV cache.
// One dispatch does: unpack → dot → softmax → value acc → inv rotate → gate.
//
// Grid: (NUM_ATTN_HEADS * THREADS_PER_HEAD, 1, 1) = (8192, 1, 1)
// Threadgroup: (256, 1, 1) = 8 simdgroups × 32 lanes
//
// 8 simdgroups (down from 32) to fit within 32KB threadgroup memory.
// Each simdgroup processes 32 keys in parallel (stride 8).
// Phase 2: cross-simdgroup softmax uses 16-element reductions (fits in 4KB).
//
// Buffer indices:
//   0: q_rot     [NUM_ATTN_HEADS * HEAD_DIM]
//   1: q_gate    [NUM_ATTN_HEADS * HEAD_DIM]
//   2: k_packed  [GPU_TQ_SEQ, NUM_KV_HEADS, TQ_WORDS_PER_HEAD] uint32
//   3: k_norms   [GPU_TQ_SEQ, NUM_KV_HEADS]
//   4: v_packed  [GPU_TQ_SEQ, NUM_KV_HEADS, TQ_WORDS_PER_HEAD] uint32
//   5: v_norms   [GPU_TQ_SEQ, NUM_KV_HEADS]
//   6: rot_matrix   [HEAD_DIM, HEAD_DIM]
//   7: inv_rot_matrix [HEAD_DIM, HEAD_DIM]
//   8: attn_out  [NUM_ATTN_HEADS * HEAD_DIM] OUTPUT
//   9:  T_kv
//   10: GPU_TQ_SEQ
//   11: causal_offset
//
// Threadgroup memory (fits under 32KB):
//   tg_max[8]:   8 × 4 = 32 bytes
//   tg_sum[8]:   8 × 4 = 32 bytes
//   tg_acc[8 * HEAD_DIM]: 8 × 256 × 4 = 8192 bytes
//   total: 8256 bytes << 32768

kernel void tq_fused_attention(
    device const float  *q_rot,
    device const float   *q_gate,
    device const uint32_t *k_packed,
    device const float   *k_norms,
    device const uint32_t *v_packed,
    device const float   *v_norms,
    device const float   *rot_matrix,
    device const float   *inv_rot_matrix,
    device       float   *attn_out,
    constant     uint32_t &T_kv,
    constant     uint32_t &GPU_TQ_SEQ,
    constant     uint32_t &causal_offset,
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint head = tid / THREADS_PER_HEAD;
    if (head >= NUM_ATTN_HEADS) return;

    int heads_per_kv = NUM_ATTN_HEADS / NUM_KV_HEADS;
    uint kv_head = head / heads_per_kv;
    uint q_off = head * HEAD_DIM;

    // 8 simdgroups, each handles keys at stride 8: sid, sid+8, sid+16, ...
    // Keys per simdgroup = ceil(T_kv / 8) but we loop with stride 8
    int num_sg = 8;

    float local_max = -1e10f;
    float local_sum = 0.0f;
    float local_acc[LANES_PER_SIMD];
    for (int i = 0; i < LANES_PER_SIMD; i++) local_acc[i] = 0.0f;

    // ---- Phase 1: Parallel scoring + online softmax + value accumulation ----
    // Each simdgroup s (0..7) processes keys: s, s+8, s+16, ...
    for (uint k = sid; k < T_kv; k += num_sg) {
        uint k_packed_base = k * NUM_KV_HEADS * TQ_WORDS_PER_HEAD + kv_head * TQ_WORDS_PER_HEAD;
        float k_norm = k_norms[k * NUM_KV_HEADS + kv_head];
        uint v_packed_base = k * NUM_KV_HEADS * TQ_WORDS_PER_HEAD + kv_head * TQ_WORDS_PER_HEAD;
        float v_norm = v_norms[k * NUM_KV_HEADS + kv_head];

        float partial_qk[LANES_PER_SIMD];
        for (int i = 0; i < LANES_PER_SIMD; i++) {
            uint d = lane + i * SIMD_SIZE;
            if (d >= HEAD_DIM) { partial_qk[i] = 0.0f; continue; }
            uint word_idx_d = d / 16;
            uint bit_off_d = (d % 16) * 2;
            uint32_t k_word = k_packed[k_packed_base + word_idx_d];
            uint k_idx = (k_word >> bit_off_d) & 0x3u;
            float k_val = CENTROIDS[k_idx] * k_norm;
            partial_qk[i] = q_rot[q_off + d] * k_val;
        }
        float partial_dot = 0.0f;
        for (int i = 0; i < LANES_PER_SIMD; i++) partial_dot += partial_qk[i];
        float dot = simd_sum(partial_dot);
        float score = dot * rsqrt((float)HEAD_DIM + 1e-6f);

        if (k > (T_kv - 1 + causal_offset)) score = -1e10f;

        float new_max = max(local_max, score);
        float exp_old = metal::fast::exp(local_max - new_max);
        float exp_new = metal::fast::exp(score - new_max);
        float factor = exp_old;
        float w = exp_new;

        for (int i = 0; i < LANES_PER_SIMD; i++) {
            uint d = lane + i * SIMD_SIZE;
            if (d >= HEAD_DIM) continue;
            uint word_idx_d = d / 16;
            uint bit_off_d = (d % 16) * 2;
            uint32_t v_word = v_packed[v_packed_base + word_idx_d];
            uint v_idx = (v_word >> bit_off_d) & 0x3u;
            float v_val = CENTROIDS[v_idx] * v_norm;
            local_acc[i] = local_acc[i] * factor + w * v_val;
        }

        local_max = new_max;
        local_sum = local_sum * factor + w;
    }

    // ---- Phase 2: Cross-simdgroup reduction via threadgroup memory ----
    // Reduced from 32 simdgroups → 8 simdgroups to fit 32KB limit
    threadgroup float tg_max[8];
    threadgroup float tg_sum[8];
    threadgroup float tg_acc[8 * HEAD_DIM];

    if (lane == 0) {
        tg_max[sid] = local_max;
        tg_sum[sid] = local_sum;
    }
    for (int i = 0; i < LANES_PER_SIMD; i++) {
        uint d = lane + i * SIMD_SIZE;
        if (d < HEAD_DIM) {
            tg_acc[sid * HEAD_DIM + d] = local_acc[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 3: Global softmax normalization (lane 0 of simdgroup 0) ----
    float global_max = -1e10f;
    float global_sum = 0.0f;
    if (lid == 0) {
        for (uint s = 0; s < num_sg; s++) {
            global_max = max(global_max, tg_max[s]);
        }
        for (uint s = 0; s < num_sg; s++) {
            global_sum += tg_sum[s] * metal::fast::exp(tg_max[s] - global_max);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = (global_sum > 1e-8f) ? (1.0f / global_sum) : 0.0f;

    // ---- Phase 4: Reconstruct + inverse rotation ----
    float acc_rot[LANES_PER_SIMD];
    for (int i = 0; i < LANES_PER_SIMD; i++) {
        uint d = lane + i * SIMD_SIZE;
        if (d >= HEAD_DIM) { acc_rot[i] = 0.0f; continue; }
        float accum = 0.0f;
        for (uint s = 0; s < num_sg; s++) {
            float factor = metal::fast::exp(tg_max[s] - global_max);
            accum += tg_acc[s * HEAD_DIM + d] * factor;
        }
        acc_rot[i] = (global_sum > 1e-8f) ? (accum * inv_sum) : 0.0f;
    }

    float out[LANES_PER_SIMD];
    for (int i = 0; i < LANES_PER_SIMD; i++) {
        uint d = lane + i * SIMD_SIZE;
        if (d >= HEAD_DIM) { out[i] = 0.0f; continue; }
        float sum = 0.0f;
        for (uint j = 0; j < HEAD_DIM; j++) {
            sum += acc_rot[i] * inv_rot_matrix[j * HEAD_DIM + d];
        }
        out[i] = sum;
    }

    // ---- Phase 5: Apply sigmoid gate ----
    for (int i = 0; i < LANES_PER_SIMD; i++) {
        uint d = lane + i * SIMD_SIZE;
        if (d >= HEAD_DIM) continue;
        float g = q_gate[q_off + d];
        float gate = 1.0f / (1.0f + metal::fast::exp(-g));
        out[i] *= gate;
    }

    // ---- Write output ----
    for (int i = 0; i < LANES_PER_SIMD; i++) {
        uint d = lane + i * SIMD_SIZE;
        if (d < HEAD_DIM) {
            attn_out[q_off + d] = out[i];
        }
    }
}


// ============================================================================
// Kernel 2: tq_pack_update
// ============================================================================
//
// Appends one new token's K/V to the TurboQuant cache at position `pos`.
//
// Grid: (NUM_KV_HEADS * TQ_WORDS_PER_HEAD, 1, 1) = (32, 1, 1)
// Threadgroup: (32, 1, 1)
//
// Constants:
//   pos: cache write position
//   GPU_TQ_SEQ: pre-allocated buffer size

kernel void tq_pack_update(
    device const float  *k_new,
    device const float  *v_new,
    device const float  *rotation_matrix,
    device       uint32_t *k_packed_out,
    device       float   *k_norms_out,
    device       uint32_t *v_packed_out,
    device       float   *v_norms_out,
    constant     uint32_t &pos,
    constant     uint32_t &GPU_TQ_SEQ,
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint kv_head  = tgid / TQ_WORDS_PER_HEAD;
    uint word_idx = tgid % TQ_WORDS_PER_HEAD;

    uint k_off = kv_head * HEAD_DIM;
    uint v_off = kv_head * HEAD_DIM;

    // L2 norm via simd_sum across all 32 lanes
    float k_local_sq = 0.0f;
    float v_local_sq = 0.0f;
    for (int i = 0; i < LANES_PER_SIMD; i++) {
        uint d = lane + i * SIMD_SIZE;
        if (d >= HEAD_DIM) continue;
        float kd = k_new[k_off + d];
        float vd = v_new[v_off + d];
        k_local_sq += kd * kd;
        v_local_sq += vd * vd;
    }
    float k_norm = metal::fast::sqrt(simd_sum(k_local_sq) + RMS_EPS);
    float v_norm = metal::fast::sqrt(simd_sum(v_local_sq) + RMS_EPS);

    // Lane 0 writes norms
    if (tgid == 0) {
        k_norms_out[pos * NUM_KV_HEADS + kv_head] = k_norm;
        v_norms_out[pos * NUM_KV_HEADS + kv_head] = v_norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_k_norm = 1.0f / (k_norm + 1e-8f);
    float inv_v_norm = 1.0f / (v_norm + 1e-8f);

    uint32_t k_packed = 0;
    uint32_t v_packed = 0;

    for (int i = 0; i < 16; i++) {
        uint d = word_idx * 16 + i;
        if (d >= HEAD_DIM) break;
        float k_dot = 0.0f;
        float v_dot = 0.0f;
        uint mat_base = d * HEAD_DIM;
        for (int j = 0; j < HEAD_DIM; j++) {
            k_dot += k_new[k_off + j] * rotation_matrix[mat_base + j];
            v_dot += v_new[v_off + j] * rotation_matrix[mat_base + j];
        }
        float k_rot = k_dot * inv_k_norm;
        float v_rot = v_dot * inv_v_norm;
        uint k_idx = (k_rot >= 1.0f) ? 3u : (k_rot >= 0.0f) ? 2u : (k_rot >= -1.0f) ? 1u : 0u;
        uint v_idx = (v_rot >= 1.0f) ? 3u : (v_rot >= 0.0f) ? 2u : (v_rot >= -1.0f) ? 1u : 0u;
        k_packed |= (k_idx & 0x3u) << (i * 2);
        v_packed |= (v_idx & 0x3u) << (i * 2);
    }

    uint out_base = pos * NUM_KV_HEADS * TQ_WORDS_PER_HEAD + kv_head * TQ_WORDS_PER_HEAD;
    k_packed_out[out_base + word_idx] = k_packed;
    v_packed_out[out_base + word_idx] = v_packed;
}


// ============================================================================
// Kernel 3: tq_dequant_all
// ============================================================================
//
// Prefill fallback: decode entire TurboQuant KV cache to fp32.
//
// Grid: (HEAD_DIM, NUM_KV_HEADS, 1) = (256, 2, 1)
// Threadgroup: (256, 1, 1)

kernel void tq_dequant_all(
    device const uint32_t *k_packed,
    device const float    *k_norms,
    device       float   *k_out,
    device const uint32_t *v_packed,
    device const float   *v_norms,
    device       float   *v_out,
    constant     uint32_t &T_kv,
    constant     uint32_t &GPU_TQ_SEQ,
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint d = tgid;
    if (d >= HEAD_DIM) return;

    uint word_idx = d / 16;
    uint bit_off = (d % 16) * 2;

    for (uint k = 0; k < T_kv; k++) {
        for (uint kv_h = 0; kv_h < NUM_KV_HEADS; kv_h++) {
            uint packed_base = k * NUM_KV_HEADS * TQ_WORDS_PER_HEAD + kv_h * TQ_WORDS_PER_HEAD;
            uint norms_idx = k * NUM_KV_HEADS + kv_h;
            uint out_base = k * NUM_KV_HEADS * HEAD_DIM + kv_h * HEAD_DIM;

            uint32_t k_word = k_packed[packed_base + word_idx];
            uint k_idx = (k_word >> bit_off) & 0x3u;
            k_out[out_base + d] = CENTROIDS[k_idx] * k_norms[norms_idx];

            uint32_t v_word = v_packed[packed_base + word_idx];
            uint v_idx = (v_word >> bit_off) & 0x3u;
            v_out[out_base + d] = CENTROIDS[v_idx] * v_norms[norms_idx];
        }
    }
}
