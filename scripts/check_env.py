"""Environment sanity check: torch build, GPU, bf16 matmul, SDPA, VRAM budget.

Run: .venv/bin/python scripts/check_env.py
Exits non-zero if any check fails.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from tinyllm.device import VRAM_BUDGET_BYTES, best_autocast_dtype, get_device


def main() -> int:
    ok = True
    print(f"python      : {sys.version.split()[0]}")
    print(f"torch       : {torch.__version__} (cuda build {torch.version.cuda})")
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1
    dev = get_device()
    props = torch.cuda.get_device_properties(dev)
    cc = f"sm_{props.major}{props.minor}"
    print(f"gpu         : {props.name} ({cc}), {props.total_memory / 1024**3:.2f} GiB")
    print(f"arch list   : {torch.cuda.get_arch_list()}")
    if cc not in torch.cuda.get_arch_list():
        print(f"WARN: {cc} not in compiled arch list (PTX JIT may be used)")
    dtype = best_autocast_dtype(dev)
    print(f"autocast    : {dtype}")

    # bf16 matmul correctness vs fp32 + throughput
    a = torch.randn(4096, 4096, device=dev)
    b = torch.randn(4096, 4096, device=dev)
    ref = a @ b
    with torch.autocast("cuda", dtype=dtype):
        out = a @ b
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    print(f"matmul err  : {rel:.2e} (bf16 vs fp32)")
    ok &= rel < 1e-2
    torch.cuda.synchronize()
    a16, b16 = a.to(dtype), b.to(dtype)
    for _ in range(3):
        a16 @ b16
    torch.cuda.synchronize()
    t = time.perf_counter()
    n = 20
    for _ in range(n):
        a16 @ b16
    torch.cuda.synchronize()
    tflops = 2 * 4096**3 * n / (time.perf_counter() - t) / 1e12
    print(f"bf16 matmul : {tflops:.1f} TFLOP/s")

    # Fused attention kernel works with a causal mask
    q = torch.randn(8, 8, 256, 64, device=dev, dtype=dtype)
    o = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    ok &= bool(torch.isfinite(o).all())
    print(f"sdpa causal : ok, out {tuple(o.shape)}")

    # Backward pass through a tiny layer
    lin = torch.nn.Linear(64, 64).to(dev)
    with torch.autocast("cuda", dtype=dtype):
        loss = lin(torch.randn(32, 64, device=dev)).pow(2).mean()
    loss.backward()
    ok &= lin.weight.grad is not None and bool(torch.isfinite(lin.weight.grad).all())
    print("backward    : ok")

    peak = torch.cuda.max_memory_allocated()
    print(f"peak alloc  : {peak / 1024**2:.0f} MiB (budget {VRAM_BUDGET_BYTES / 1024**3:.1f} GiB)")
    ok &= peak < VRAM_BUDGET_BYTES
    print("RESULT      :", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
