import pytest

torch = pytest.importorskip("torch")
from tinyllm.model import apply_rope, rope_cache  # noqa: E402


def test_relative_position_property():
    # <rope(q, m), rope(k, n)> depends only on m - n
    torch.manual_seed(0)
    D = 16
    cos, sin = rope_cache(64, D)
    q, k = torch.randn(D), torch.randn(D)

    def score(m, n):
        qm = apply_rope(q.expand(64, D), cos, sin)[m]
        kn = apply_rope(k.expand(64, D), cos, sin)[n]
        return (qm * kn).sum()

    for off in (0, 3, 10):
        vals = torch.stack([score(m + off, m) for m in (0, 5, 20, 40)])
        torch.testing.assert_close(vals, vals[0].expand(4), atol=1e-4, rtol=1e-4)
    assert not torch.isclose(score(10, 0), score(13, 0))


def test_rope_preserves_norm_and_pos0_identity():
    cos, sin = rope_cache(8, 8)
    x = torch.randn(2, 3, 8, 8)
    y = apply_rope(x, cos, sin)
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))
    torch.testing.assert_close(y[..., 0, :], x[..., 0, :])
