import pytest

torch = pytest.importorskip("torch")
from tinyllm.model import CausalSelfAttention  # noqa: E402


def _pair(d=64, h=4):
    torch.manual_seed(0)
    a = CausalSelfAttention(d, h, impl="manual").eval()
    b = CausalSelfAttention(d, h, impl="sdpa").eval()
    b.load_state_dict(a.state_dict())
    return a, b


def test_manual_matches_sdpa():
    a, b = _pair()
    x = torch.randn(3, 17, 64)
    torch.testing.assert_close(a(x), b(x), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_causal(impl):
    torch.manual_seed(1)
    m = CausalSelfAttention(32, 4, impl=impl).eval()
    x = torch.randn(2, 12, 32)
    x2 = x.clone()
    x2[:, 7:] = torch.randn(2, 5, 32)  # change only the future
    y, y2 = m(x), m(x2)
    torch.testing.assert_close(y[:, :7], y2[:, :7])
    assert not torch.allclose(y[:, 7:], y2[:, 7:])


def test_gradient_flows():
    m = CausalSelfAttention(32, 2)
    m(torch.randn(2, 5, 32)).sum().backward()
    assert all(p.grad is not None for p in m.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_bf16_paths_agree():
    a, b = _pair()
    a, b = a.cuda(), b.cuda()
    x = torch.randn(2, 128, 64, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ya, yb = a(x), b(x)
    torch.testing.assert_close(ya.float(), yb.float(), atol=2e-2, rtol=2e-2)
