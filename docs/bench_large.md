# Larger-model benchmark (RTX 5050 Laptop, bf16, torch.compile, T=256)

| model | micro-batch | params (M) | tokens/s | peak VRAM (GB) | fits < 6.5 GB | min / 300M tok |
|---|---|---|---|---|---|---|
| 29M d512 L8 (base) | 32 | 29.4 | 88.7k | 2.12 | yes | 56 |
| 54M d640 L10 | 32 | 54.4 | 50.1k | 3.21 | yes | 100 |
| 91M d768 L12 | 16 | 91.3 | 29.3k | 2.92 | yes | 171 |
| 91M d768 L12 | 32 | 91.3 | 30.8k | 4.61 | yes | 163 |
| 91M d768 L12 +ckpt | 32 | 91.3 | 26.1k | 1.97 | yes | 191 |
| 91M d768 L12 +ckpt | 64 | 91.3 | 26.6k | 2.78 | yes | 188 |
| 120M d768 L16 | 32 | 119.6 | 23.4k | 5.98 | yes | 214 |
| 120M d768 L16 +ckpt | 32 | 119.6 | 20.0k | 2.40 | yes | 250 |

## Decision (Day 15, 2026-11-09)
Chosen: **91M d768 L12, micro-batch 32 × grad_accum 2 (16,384 tok/step), no gradient checkpointing**
(`configs/large.json`). It fits with margin (4.6 GB in the bench, 4.96 GB peak in the real run, budget 6.5 GB) and
checkpointing would cost ~15% speed for VRAM we don't need. The 120M L16 model fits too (5.98 GB) but is ~25% slower,
so it would see fewer tokens in the 4 planned sessions. Schedule: 18,000 steps ≈ 295M tokens (~2.6 h at 31k tok/s,
4 sessions of ≤ 42 min). Gradient checkpointing stays available (`model.grad_checkpoint`), tested in `tests/test_grad_ckpt.py`.
Checkpoints are 1.1 GB each (weights + AdamW), 2 kept = 2.2 GB, so the older base checkpoint was deleted for disk room.
