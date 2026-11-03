import pytest
import torch

from tinyllm.device import best_autocast_dtype, get_device


def test_cpu_fallback():
    assert get_device("cpu").type == "cpu"
    assert best_autocast_dtype(torch.device("cpu")) == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_bf16():
    dev = get_device()
    assert dev.type == "cuda"
    assert best_autocast_dtype(dev) == torch.bfloat16
    x = torch.randn(64, 64, device=dev, dtype=torch.bfloat16)
    assert torch.isfinite(x @ x).all()
