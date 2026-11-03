import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tinyllm.data import TokenDataset  # noqa: E402


@pytest.fixture
def tokfile(tmp_path):
    p = tmp_path / "t.bin"
    np.arange(1000, dtype=np.uint16).tofile(p)
    return p


def test_shift_shapes_dtypes(tokfile):
    ds = TokenDataset(tokfile, block_size=16, seed=0)
    x, y = ds.get_batch(8)
    assert x.shape == y.shape == (8, 16) and x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y, x + 1)  # arange data: next token = token + 1


def test_in_range(tokfile):
    ds = TokenDataset(tokfile, block_size=50, seed=1)
    for _ in range(200):
        x, y = ds.get_batch(16)
        assert x.min() >= 0 and y.max() <= 999


def test_rng_state_resume(tokfile):
    a = TokenDataset(tokfile, 16, seed=3)
    a.get_batch(4)
    st = a.state_dict()
    xa, _ = a.get_batch(4)
    b = TokenDataset(tokfile, 16, seed=99)
    b.load_state_dict(st)
    xb, _ = b.get_batch(4)
    assert torch.equal(xa, xb)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_batch(tokfile):
    x, y = TokenDataset(tokfile, 16).get_batch(4, "cuda")
    assert x.is_cuda and torch.equal(x.cpu() + 1, y.cpu())
