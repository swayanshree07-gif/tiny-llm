# Int8 weight-only quantization

Per-output-channel symmetric int8 for every Linear (incl. LM head; embeddings/LayerNorm stay fp32), `tinyllm/quant.py`. Val loss on 200×32 fixed windows (seed 1234), bf16 autocast. Size = persistent weights in memory. tok/s = cached greedy, bf16 autocast, 240 new tokens. Regenerate with `scripts/eval_int8.py`.

| model | step | weights | val loss | ppl | Δ loss | size MB | tok/s B=1 | tok/s B=8 |
|---|---|---|---|---|---|---|---|---|
| base-29M | 40000 | fp32 | 1.2410 | 3.459 | – | 112 | 468 | 3635 |
| base-29M | 40000 | int8 | 1.2412 | 3.460 | +0.0001 | 44 | 366 | 2845 |
| large-91M | 18000 | fp32 | 1.2192 | 3.384 | – | 348 | 306 | 2430 |
| large-91M | 18000 | int8 | 1.2193 | 3.385 | +0.0002 | 111 | 241 | 1765 |

Notes: Day 25 check (Δ val loss ≤ +0.02) passes by a wide margin (+0.0001 / +0.0002). Weights shrink 2.5–3.1x
(the fp32 token embedding is kept; it is most of the base model's remaining 44 MB). Int8 is ~20–27% **slower**:
there is no int8 matmul kernel here, so each Linear dequantizes its weight to bf16 on every call. The win is
memory/disk, not speed; a fused int8 GEMM (or caching the dequantized weights, which defeats the point) would be
needed for speed.
