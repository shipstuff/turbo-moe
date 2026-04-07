#!/usr/bin/env python3
with open("/Users/carl/projects/turbomoe/flash_moe/metal_infer/infer.m") as f:
    src = f.read()

fixes = [
    ("[tq_enc setBuffer:encode_k    offset:0 atIndex:0]",
     "[tq_enc setBuffer:g_metal->buf_tq_encode_k offset:0 atIndex:0]"),
    ("[tq_enc setBuffer:encode_v    offset:0 atIndex:1]",
     "[tq_enc setBuffer:g_metal->buf_tq_encode_v offset:0 atIndex:1]"),
    ("[tq_enc setBuffer:encode_k_norms offset:0 atIndex:4]",
     "[tq_enc setBuffer:g_metal->buf_tq_encode_k_norms offset:0 atIndex:4]"),
    ("[tq_enc setBuffer:encode_v_norms offset:0 atIndex:6]",
     "[tq_enc setBuffer:g_metal->buf_tq_encode_v_norms offset:0 atIndex:6]"),
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
