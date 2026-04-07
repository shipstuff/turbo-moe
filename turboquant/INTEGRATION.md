# TurboQuant Integration Guide — flash-moe

This file contains all the changes needed to integrate TurboQuant KV-cache
compression into danveloper/flash-moe's metal_infer/infer.m.

Files changed:
  - metal_infer/infer.m   — add TurboQuant buffers, pipelines, dispatch logic
  - metal_infer/shaders.metal — append turboquant.metal to end of file

## CHANGE 1: Constants
Location: after "#define RMS_NORM_EPS 1e-6f" (around line 79)

Add:
```objc
// TurboQuant KV-cache compression (TurboMoE project)
#define GPU_TQ_SEQ           GPU_KV_SEQ
#define TQ_WORDS_PER_HEAD    (HEAD_DIM / 16)   // 16 for 2-bit (16 indices per uint32)
```

## CHANGE 2: MetalCtx struct members
Location: in MetalCtx struct, after the existing GPU attention buffer declarations
(around line 959, after "id<MTLBuffer> buf_attn_gate;")

Add:
```objc
    // TurboQuant KV-cache compression (TurboMoE)
    // TurboQuant KV storage: per token per KV head = 16*4 + 4 = 68 bytes
    id<MTLBuffer> buf_tq_k_packed[NUM_FULL_ATTN_LAYERS]; // [GPU_TQ_SEQ, NUM_KV_HEADS, TQ_WORDS_PER_HEAD] uint32
    id<MTLBuffer> buf_tq_k_norms[NUM_FULL_ATTN_LAYERS];  // [GPU_TQ_SEQ, NUM_KV_HEADS] float32
    id<MTLBuffer> buf_tq_v_packed[NUM_FULL_ATTN_LAYERS]; // [GPU_TQ_SEQ, NUM_KV_HEADS, TQ_WORDS_PER_HEAD] uint32
    id<MTLBuffer> buf_tq_v_norms[NUM_FULL_ATTN_LAYERS];  // [GPU_TQ_SEQ, NUM_KV_HEADS] float32
    id<MTLBuffer> buf_tq_rot;        // [HEAD_DIM * HEAD_DIM] float32 — orthogonal Pi
    id<MTLBuffer> buf_tq_inv_rot;    // [HEAD_DIM * HEAD_DIM] float32 — Pi^T
    // TurboQuant encode scratch buffers
    id<MTLBuffer> buf_tq_encode_k;           // [NUM_KV_HEADS * HEAD_DIM] fp32
    id<MTLBuffer> buf_tq_encode_v;           // [NUM_KV_HEADS * HEAD_DIM] fp32
    id<MTLBuffer> buf_tq_encode_k_norms;     // [NUM_KV_HEADS] fp32
    id<MTLBuffer> buf_tq_encode_v_norms;     // [NUM_KV_HEADS] fp32
    // TurboQuant attention pipelines
    id<MTLComputePipelineState> tq_fused_attn_pipe;   // 1-dispatch fused attention
    id<MTLComputePipelineState> tq_encode_pipe;         // tq_encode_packed kernel
    id<MTLComputePipelineState> tq_pack_update_pipe;  // append new token to cache
    id<MTLComputePipelineState> tq_dequant_pipe;       // prefill: decode all to fp32
    int use_tq_kv;  // 1 = compressed KV, 0 = float KV (default)
```

## CHANGE 3: Pipeline creation
Location: in metal_setup(), after "ctx->sigmoid_gate_pipe = makePipe(@"sigmoid_gate");"
(around line 1057)

Add:
```objc
    ctx->tq_fused_attn_pipe  = makePipe(@"tq_fused_attention");
    ctx->tq_encode_pipe      = makePipe(@"tq_encode_packed");
    ctx->tq_pack_update_pipe  = makePipe(@"tq_pack_update");
    ctx->tq_dequant_pipe     = makePipe(@"tq_dequant_all");
    if (!ctx->tq_fused_attn_pipe) {
        fprintf(stderr, "[metal] WARNING: tq_fused_attention not available — TurboQuant disabled\n");
    } else {
        printf("[metal] TurboQuant pipelines ready (use_tq_kv=%d)\n", ctx->use_tq_kv);
    }
```

## CHANGE 4: Buffer allocation
Location: in metal_setup(), after the existing "GPU attention buffers" section
(after the buf_attn_gate allocation, around line 1190)

Add:
```objc
    // TurboQuant KV-cache buffers (compressed)
    {
        size_t tq_packed_size = (size_t)GPU_TQ_SEQ * NUM_KV_HEADS * TQ_WORDS_PER_HEAD * sizeof(uint32_t);
        size_t tq_norm_size   = (size_t)GPU_TQ_SEQ * NUM_KV_HEADS * sizeof(float);
        size_t rot_size       = (size_t)HEAD_DIM * HEAD_DIM * sizeof(float);

        for (int i = 0; i < NUM_FULL_ATTN_LAYERS; i++) {
            ctx->buf_tq_k_packed[i] = [ctx->device newBufferWithLength:tq_packed_size
                                                             options:MTLResourceStorageModeShared];
            ctx->buf_tq_k_norms[i]  = [ctx->device newBufferWithLength:tq_norm_size
                                                            options:MTLResourceStorageModeShared];
            ctx->buf_tq_v_packed[i] = [ctx->device newBufferWithLength:tq_packed_size
                                                             options:MTLResourceStorageModeShared];
            ctx->buf_tq_v_norms[i]  = [ctx->device newBufferWithLength:tq_norm_size
                                                            options:MTLResourceStorageModeShared];
        }

        ctx->buf_tq_rot    = [ctx->device newBufferWithLength:rot_size options:MTLResourceStorageModeShared];
        ctx->buf_tq_inv_rot = [ctx->device newBufferWithLength:rot_size options:MTLResourceStorageModeShared];

        // Encode scratch buffers
        size_t encode_kv_size = NUM_KV_HEADS * HEAD_DIM * sizeof(float);
        size_t encode_norms_size = NUM_KV_HEADS * sizeof(float);
        size_t encode_packed_size = NUM_KV_HEADS * TQ_WORDS_PER_HEAD * sizeof(uint32_t);
        ctx->buf_tq_encode_k          = [ctx->device newBufferWithLength:encode_kv_size options:MTLResourceStorageModeShared];
        ctx->buf_tq_encode_v          = [ctx->device newBufferWithLength:encode_kv_size options:MTLResourceStorageModeShared];
        ctx->buf_tq_encode_k_norms    = [ctx->device newBufferWithLength:encode_norms_size options:MTLResourceStorageModeShared];
        ctx->buf_tq_encode_v_norms    = [ctx->device newBufferWithLength:encode_norms_size options:MTLResourceStorageModeShared];
        ctx->buf_tq_encode_k_packed   = [ctx->device newBufferWithLength:encode_packed_size options:MTLResourceStorageModeShared];
        ctx->buf_tq_encode_v_packed   = [ctx->device newBufferWithLength:encode_packed_size options:MTLResourceStorageModeShared];

        // Generate orthogonal rotation matrix Pi via QR decomposition
        float *rot_raw = (float *)malloc(HEAD_DIM * HEAD_DIM * sizeof(float));
        float *Q = (float *)malloc(HEAD_DIM * HEAD_DIM * sizeof(float));
        srand(42);
        for (int i = 0; i < HEAD_DIM * HEAD_DIM; i++) rot_raw[i] = (float)(rand() % 1000 - 500) / 500.0f;
        // Gram-Schmidt orthonormalization
        for (int j = 0; j < HEAD_DIM; j++) {
            for (int i = 0; i < HEAD_DIM; i++) Q[j * HEAD_DIM + i] = rot_raw[j * HEAD_DIM + i];
            for (int i = 0; i < j; i++) {
                float dot = 0.0f;
                for (int k = 0; k < HEAD_DIM; k++) dot += Q[i * HEAD_DIM + k] * Q[j * HEAD_DIM + k];
                for (int k = 0; k < HEAD_DIM; k++) Q[j * HEAD_DIM + k] -= dot * Q[i * HEAD_DIM + k];
            }
            float norm = 0.0f;
            for (int k = 0; k < HEAD_DIM; k++) norm += Q[j * HEAD_DIM + k] * Q[j * HEAD_DIM + k];
            norm = sqrtf(norm + 1e-8f);
            for (int k = 0; k < HEAD_DIM; k++) Q[j * HEAD_DIM + k] /= norm;
        }
        float *rot_ptr = (float *)[ctx->buf_tq_rot contents];
        float *inv_ptr = (float *)[ctx->buf_tq_inv_rot contents];
        memcpy(rot_ptr, Q, HEAD_DIM * HEAD_DIM * sizeof(float));
        // Pi^(-1) = Pi^T for orthogonal matrices
        for (int i = 0; i < HEAD_DIM; i++)
            for (int j = 0; j < HEAD_DIM; j++)
                inv_ptr[i * HEAD_DIM + j] = rot_ptr[j * HEAD_DIM + i];
        free(rot_raw); free(Q);

        size_t per_layer_bytes = tq_packed_size * 2 + tq_norm_size * 2;
        printf("[metal] TurboQuant: %d layers × %.1f KB/layer KV, rot matrix %.1f KB\n",
               NUM_FULL_ATTN_LAYERS,
               (double)per_layer_bytes / 1e3,
               (double)rot_size * 2 / 1e3);

        // Enable if TQ_KV=1 env var or CLI flag
        ctx->use_tq_kv = (getenv("TQ_KV") && atoi(getenv("TQ_KV")) == 1) ? 1 : 0;
        if (ctx->use_tq_kv && !ctx->tq_fused_attn_pipe) {
            fprintf(stderr, "[metal] TQ_KV=1 but TurboQuant pipelines unavailable — falling back to float KV\n");
            ctx->use_tq_kv = 0;
        }
        if (ctx->use_tq_kv) printf("[metal] TurboQuant KV compression ENABLED\n");
    }
```

## CHANGE 5: KV cache update — append new token
Location: in full_attention_forward(), after the existing float KV cache update
(after "kv->len++;" around line 2270 in the original file)

The existing code does:
```objc
    int cache_pos = kv->len;
    memcpy(kv->k_cache + cache_pos * kv_dim, k, kv_dim * sizeof(float));
    memcpy(kv->v_cache + cache_pos * kv_dim, v, kv_dim * sizeof(float));
    kv->len++;
```

After those lines, add:
```objc
    // ---- TurboQuant KV cache update ----
    if (g_metal && g_metal->use_tq_kv && g_metal->tq_pack_update_pipe) {
        int fa_idx = (layer_idx + 1) / FULL_ATTN_INTERVAL - 1;
        uint32_t pos32 = (uint32_t)cache_pos;
        uint32_t GPU_TQ_SEQ32 = GPU_TQ_SEQ;
        float *encode_k = (float *)[g_metal->buf_tq_encode_k contents];
        float *encode_v = (float *)[g_metal->buf_tq_encode_v contents];
        float *encode_k_norms = (float *)[g_metal->buf_tq_encode_k_norms contents];
        float *encode_v_norms = (float *)[g_metal->buf_tq_encode_v_norms contents];

        // Copy K and V to encode scratch buffers
        memcpy(encode_k, k, NUM_KV_HEADS * HEAD_DIM * sizeof(float));
        memcpy(encode_v, v, NUM_KV_HEADS * HEAD_DIM * sizeof(float));

        // CPU-side L2 norm computation
        for (int kv_h = 0; kv_h < NUM_KV_HEADS; kv_h++) {
            float k_sum_sq = 0.0f, v_sum_sq = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                float kd = encode_k[kv_h * HEAD_DIM + d];
                float vd = encode_v[kv_h * HEAD_DIM + d];
                k_sum_sq += kd * kd;
                v_sum_sq += vd * vd;
            }
            encode_k_norms[kv_h] = sqrtf(k_sum_sq + RMS_NORM_EPS);
            encode_v_norms[kv_h] = sqrtf(v_sum_sq + RMS_NORM_EPS);
        }

        // GPU encode (rotate + quantize + pack)
        id<MTLCommandBuffer> cmd = [g_metal->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_metal->tq_pack_update_pipe];
        [enc setBuffer:encode_k          offset:0 atIndex:0];
        [enc setBuffer:encode_v          offset:0 atIndex:1];
        [enc setBuffer:(float *)[g_metal->buf_tq_rot contents]    offset:0 atIndex:2];
        [enc setBuffer:(uint32_t *)[g_metal->buf_tq_k_packed[fa_idx] contents] offset:0 atIndex:3];
        [enc setBuffer:encode_k_norms    offset:0 atIndex:4];
        [enc setBuffer:(uint32_t *)[g_metal->buf_tq_v_packed[fa_idx] contents] offset:0 atIndex:5];
        [enc setBuffer:encode_v_norms    offset:0 atIndex:6];
        [enc setBytes:&pos32          length:4 atIndex:7];
        [enc setBytes:&GPU_TQ_SEQ32   length:4 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(NUM_KV_HEADS * TQ_WORDS_PER_HEAD, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];  // ensure KV is written before attention reads it
        // Note: for async pipeline, encode on CPU and dispatch without wait,
        // then use a GPU event to order attention after encode
    }
```

## CHANGE 6: Attention dispatch replacement
Location: in fused_layer_forward(), in the "FULLY FUSED CMD2" section
(around line 4820, where gpu_attn_fuse is checked)

The existing code has:
```objc
    int gpu_attn_fuse = (is_full && !attn_out_for_oproj && g_metal && g_metal->attn_scores_pipe
                         && kv && kv->len >= 32 && kv->len < GPU_KV_SEQ);
```

Add after:
```objc
    int use_tq_attn = (gpu_attn_fuse && g_metal->use_tq_kv
                        && g_metal->tq_fused_attn_pipe
                        && kv && kv->len < GPU_TQ_SEQ);
```

Then, in the attention dispatch section (where the 3 encoders are issued for attn_scores,
attn_softmax, attn_values), replace those 3 encoders with:

```objc
        if (use_tq_attn) {
            // ---- TurboQuant fused attention: 1 dispatch instead of 3 ----
            uint32_t T_kv32 = (uint32_t)kv->len;
            uint32_t causal_offset32 = 0;
            uint32_t GPU_TQ_SEQ32 = GPU_TQ_SEQ;

            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            [enc setComputePipelineState:g_metal->tq_fused_attn_pipe];

            [enc setBuffer:g_metal->buf_attn_q                              offset:0 atIndex:0];
            [enc setBuffer:g_metal->buf_attn_gate                           offset:0 atIndex:1];
            [enc setBuffer:(uint32_t *)[g_metal->buf_tq_k_packed[fa_idx] contents] offset:0 atIndex:2];
            [enc setBuffer:(float *)[g_metal->buf_tq_k_norms[fa_idx] contents]  offset:0 atIndex:3];
            [enc setBuffer:(uint32_t *)[g_metal->buf_tq_v_packed[fa_idx] contents] offset:0 atIndex:4];
            [enc setBuffer:(float *)[g_metal->buf_tq_v_norms[fa_idx] contents]  offset:0 atIndex:5];
            [enc setBuffer:(float *)[g_metal->buf_tq_rot contents]         offset:0 atIndex:6];
            [enc setBuffer:(float *)[g_metal->buf_tq_inv_rot contents]     offset:0 atIndex:7];
            [enc setBuffer:g_metal->buf_attn_out                           offset:0 atIndex:8];
            [enc setBytes:&T_kv32            length:4 atIndex:9];
            [enc setBytes:&GPU_TQ_SEQ32      length:4 atIndex:10];
            [enc setBytes:&causal_offset32   length:4 atIndex:11];

            [enc dispatchThreadgroups:MTLSizeMake(NUM_ATTN_HEADS * 256, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        } else {
            // Existing 3-dispatch float KV path
            // Enc A1: attn_scores_batched
            // Enc A2: attn_softmax_batched
            // Enc A3: attn_values_batched
            ... [existing code unchanged] ...
        }
```

## CHANGE 7: Concatenate Metal shader
Location: at the end of metal_infer/shaders.metal

Run:
```bash
cat turboquant/turboquant.metal >> metal_infer/shaders.metal
```

## VERIFICATION CHECKLIST

After applying all changes and rebuilding:

1. Baseline (no TurboQuant):
   ./infer --prompt "Hello world" --tokens 50
   → Should produce same output as before (use_tq_kv=0 by default)

2. TurboQuant enabled:
   TQ_KV=1 ./infer --prompt "Hello world" --tokens 50
   → Should work identically; check printf "TurboQuant KV compression ENABLED"

3. Quality check:
   → Compare outputs with and without TQ_KV=1 — should be identical for short prompts
   → Perplexity benchmark at longer context should show <5% degradation

4. Memory check:
   → Instruments / Activity Monitor should show reduced memory for KV cache buffers

## BUILD COMMANDS

```bash
# On the Mac mini (mini-01):
cd ~/turbomoe/flash_moe/metal_infer
make clean && make

# Run with TurboQuant disabled (default):
./infer --prompt "Explain quantum mechanics" --tokens 100

# Run with TurboQuant enabled:
TQ_KV=1 ./infer --prompt "Explain quantum mechanics" --tokens 100
```
