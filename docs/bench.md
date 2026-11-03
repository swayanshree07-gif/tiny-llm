# Training throughput benchmark (RTX 5050 Laptop, bf16)

| model | micro-batch | compile | params (M) | tokens/s | peak VRAM (GB) | fits < 6.5 GB |
|---|---|---|---|---|---|---|
| 29M d512 L8 T256 | 16 | False | 29.4 | 69.4k | 1.51 | yes |
| 29M d512 L8 T256 | 32 | False | 29.4 | 66.3k | 2.63 | yes |
| 29M d512 L8 T256 | 64 | False | 29.4 | 63.1k | 4.83 | yes |
| 29M d512 L8 T256 | 32 | True | 29.4 | 87.0k | 2.12 | yes |
| 29M d512 L8 T512 | 16 | False | 29.4 | 64.8k | 2.63 | yes |
| 29M d512 L8 T512 | 32 | False | 29.4 | 61.7k | 4.83 | yes |
| 51M d640 L10 T256 | 32 | False | 54.4 | 39.0k | 3.71 | yes |
