# Inference speed benchmark

Device cuda (NVIDIA GeForce RTX 5050 Laptop GPU); greedy; prompt 8 random tokens + up to 240 new; median of 3 runs. bf16 = autocast (weights stay fp32). Regenerate with `scripts/infer_bench.py`.

| model | precision | batch | new tokens | no cache tok/s | cache tok/s | speedup | greedy identical |
|---|---|---|---|---|---|---|---|
| base-29M (step 40000) | fp32 | 1 | 240 | 354 | 559 | 1.58x | True (1.0000) |
| base-29M (step 40000) | fp32 | 8 | 240 | 525 | 4478 | 8.52x | True (1.0000) |
| base-29M (step 40000) | fp32 | 32 | 240 | 527 | 11425 | 21.70x | True (1.0000) |
| base-29M (step 40000) | bf16 | 1 | 240 | 464 | 426 | 0.92x | False (0.8024) |
| base-29M (step 40000) | bf16 | 8 | 240 | 1620 | 3281 | 2.03x | False (0.7182) |
| base-29M (step 40000) | bf16 | 32 | 240 | 1731 | 13034 | 7.53x | False (0.6550) |
| large-91M (step 18000) | fp32 | 1 | 240 | 122 | 375 | 3.07x | True (1.0000) |
| large-91M (step 18000) | fp32 | 8 | 240 | 179 | 2453 | 13.67x | True (1.0000) |
| large-91M (step 18000) | fp32 | 32 | 240 | 179 | 4458 | 24.84x | True (1.0000) |
| large-91M (step 18000) | bf16 | 1 | 240 | 301 | 285 | 0.95x | True (1.0000) |
| large-91M (step 18000) | bf16 | 8 | 240 | 610 | 2263 | 3.71x | True (1.0000) |
| large-91M (step 18000) | bf16 | 32 | 240 | 617 | 6948 | 11.25x | False (0.7169) |
| chat-v3-91M (step 1000) | fp32 | 1 | 240 | 122 | 373 | 3.04x | True (1.0000) |
| chat-v3-91M (step 1000) | fp32 | 8 | 240 | 179 | 2454 | 13.69x | True (1.0000) |
| chat-v3-91M (step 1000) | fp32 | 32 | 240 | 180 | 4461 | 24.83x | True (1.0000) |
| chat-v3-91M (step 1000) | bf16 | 1 | 240 | 301 | 282 | 0.94x | False (0.2177) |
| chat-v3-91M (step 1000) | bf16 | 8 | 240 | 610 | 2264 | 3.71x | True (1.0000) |
| chat-v3-91M (step 1000) | bf16 | 32 | 240 | 617 | 6939 | 11.24x | False (0.7723) |

Notes:
- **fp32: cached greedy == uncached greedy for every model and batch** (the exactness check; also
  `tests/test_kv_cache.py`, `tests/test_infer_bench.py`).
- bf16 rows are not bit-identical: the cached path runs different matmul shapes (1 query vs the whole
  prefix), so bf16 rounding differs, a near-tie argmax flips once, and the rest of the sequence diverges.
  This is expected and harmless for sampling; use fp32 when exact reproducibility matters.
- Batch 1 is launch/Python-bound: the cache helps 1.6x (base) / 3x (large) in fp32, and bf16 autocast gives
  no gain there (cast overhead ≈ the matmul savings). The uncached path is O(T²) and the cache wins
  8–25x at batch 8–32. Best throughput: base bf16+cache B=32 ≈ 13k tok/s; large fp32+cache B=32 ≈ 4.5k
  tok/s, bf16 ≈ 6.9k tok/s. chat-v3 has the large architecture, so its speed matches large.
