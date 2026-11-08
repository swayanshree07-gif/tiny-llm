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
