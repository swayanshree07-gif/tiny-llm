# KV cache benchmark

Model: checkpoint step 14030, device cuda, fp32, greedy, prompt 8 tokens + 240 new (stays within block_size).

| batch | new tokens | no cache tok/s | cache tok/s | speedup | identical |
|---|---|---|---|---|---|
| 1 | 240 | 380 | 600 | 1.58x | True (1.0000) |
| 8 | 240 | 532 | 4801 | 9.02x | True (1.0000) |
| 32 | 240 | 526 | 11414 | 21.69x | True (1.0000) |

Notes: at batch 1 the 29M model is bound by per-step kernel-launch / Python overhead, not compute, so
the cache only gives ~1.6x; with larger batches the uncached path's O(T^2) recompute dominates and
the cache wins 9–22x. Measured on step 14030 of the main run while nothing else used the GPU.
