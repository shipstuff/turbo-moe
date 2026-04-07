#!/usr/bin/env python3
with open("/Users/carl/projects/turbomoe/flash_moe/metal_infer/infer.m") as f:
    src = f.read()

fixes = [
    ("[tq_enc setBuffer:(float *)[g_metal->buf_tq_rot contents] offset:0 atIndex:2]",
     "[tq_enc setBuffer:g_metal->buf_tq_rot offset:0 atIndex:2]"),
    ("[tq_enc setBuffer:(uint32_t *)[g_metal->buf_tq_k_packed[tq_fa_idx] contents] offset:0 atIndex:3]",
     "[tq_enc setBuffer:g_metal->buf_tq_k_packed[tq_fa_idx] offset:0 atIndex:3]"),
    ("[tq_enc setBuffer:(uint32_t *)[g_metal->buf_tq_v_packed[tq_fa_idx] contents] offset:0 atIndex:5]",
     "[tq_enc setBuffer:g_metal->buf_tq_v_packed[tq_fa_idx] offset:0 atIndex:5]"),
    ("[tq_enc setBytes:&pos32 offset:0 atIndex:7]",
     "[tq_enc setBytes:&pos32 length:4 atIndex:7]"),
    ("[tq_enc setBytes:&GPU_TQ_SEQ32 offset:0 atIndex:8]",
     "[tq_enc setBytes:&GPU_TQ_SEQ32 length:4 atIndex:8]"),
    ("[enc setBuffer:(uint32_t *)[g_metal->buf_tq_k_packed[fa_idx] contents]  offset:0 atIndex:2]",
     "[enc setBuffer:g_metal->buf_tq_k_packed[fa_idx]  offset:0 atIndex:2]"),
    ("[enc setBuffer:(float *)[g_metal->buf_tq_k_norms[fa_idx] contents]     offset:0 atIndex:3]",
     "[enc setBuffer:g_metal->buf_tq_k_norms[fa_idx]     offset:0 atIndex:3]"),
    ("[enc setBuffer:(uint32_t *)[g_metal->buf_tq_v_packed[fa_idx] contents] offset:0 atIndex:4]",
     "[enc setBuffer:g_metal->buf_tq_v_packed[fa_idx] offset:0 atIndex:4]"),
    ("[enc setBuffer:(float *)[g_metal->buf_tq_v_norms[fa_idx] contents]     offset:0 atIndex:5]",
     "[enc setBuffer:g_metal->buf_tq_v_norms[fa_idx]     offset:0 atIndex:5]"),
    ("[enc setBuffer:(float *)[g_metal->buf_tq_rot contents]                 offset:0 atIndex:6]",
     "[enc setBuffer:g_metal->buf_tq_rot                 offset:0 atIndex:6]"),
    ("[enc setBuffer:(float *)[g_metal->buf_tq_inv_rot contents]              offset:0 atIndex:7]",
     "[enc setBuffer:g_metal->buf_tq_inv_rot              offset:0 atIndex:7]"),
]

for old, new in fixes:
    if old in src:
        src = src.replace(old, new)
        print("fixed: " + old[:60])
    else:
        print("NOT FOUND: " + old[:60])

with open("/Users/carl/projects/turbomoe/flash_moe/metal_infer/infer.m", "w") as f:
    f.write(src)
print("done")
