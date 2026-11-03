"""PLAN Day 24: KV cache gives the same results as recomputing the full context."""
import pytest

torch = pytest.importorskip("torch")
from tinyllm.generate import generate  # noqa: E402
from tinyllm.model import GPT, GPTConfig  # noqa: E402


def make(pos="rope", impl="sdpa"):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=64, block_size=24, n_layers=2, n_heads=4, d_model=32,
                         pos=pos, attn_impl=impl)).eval()


@pytest.mark.parametrize("pos", ["rope", "learned"])
@pytest.mark.parametrize("impl", ["sdpa", "manual"])
def test_incremental_logits_match_full(pos, impl):
    m = make(pos, impl)
    idx = torch.randint(0, 64, (2, 20))
    full, _ = m(idx)
    cache = m.new_kv_cache(2)
    # prefill 7, then a chunk of 5 (exercises the start>0, T>1 mask), then one token at a time
    parts = [m(idx[:, :7], kv_cache=cache)[0], m(idx[:, 7:12], kv_cache=cache)[0]]
    parts += [m(idx[:, t:t + 1], kv_cache=cache)[0] for t in range(12, 20)]
    torch.testing.assert_close(torch.cat(parts, dim=1), full, atol=1e-5, rtol=1e-4)
    assert cache[0].pos == 20


def test_cache_overflow_raises():
    m = make()
    cache = m.new_kv_cache(1)
    m(torch.zeros(1, 24, dtype=torch.long), kv_cache=cache)
    with pytest.raises(AssertionError):
        m(torch.zeros(1, 1, dtype=torch.long), kv_cache=cache)


@pytest.mark.parametrize("pos", ["rope", "learned"])
def test_greedy_cached_equals_uncached_within_block(pos):
    m = make(pos)
    idx = torch.randint(0, 64, (3, 5))
    a = generate(m, idx, 19, temperature=0)                    # ends exactly at block_size 24
    b = generate(m, idx, 19, temperature=0, use_cache=True)
    assert torch.equal(a, b)


def test_sampling_cached_equals_uncached_within_block():
    m = make()
    idx = torch.randint(0, 64, (2, 4))
    kw = dict(temperature=1.0, top_k=20, top_p=0.9)
    a = generate(m, idx, 20, generator=torch.Generator().manual_seed(3), **kw)
    b = generate(m, idx, 20, generator=torch.Generator().manual_seed(3), use_cache=True, **kw)
    assert torch.equal(a, b)


def test_cached_runs_past_block_size_and_long_prompt():
    m = make()
    out = generate(m, torch.randint(0, 64, (2, 3)), 60, temperature=0, use_cache=True)
    assert out.shape == (2, 63)
    out = generate(m, torch.randint(0, 64, (1, 40)), 5, temperature=0, use_cache=True)  # prompt > block
    assert out.shape == (1, 45)
    # first token after a too-long prompt: cache path sees last 12 tokens, so compare with that
    p = torch.randint(0, 64, (1, 40))
    first = generate(m, p, 1, temperature=0, use_cache=True)[0, -1]
    assert int(first) == int(m(p[:, -12:])[0][0, -1].argmax())


def test_eot_with_cache():
    m = make()
    idx = torch.zeros(1, 2, dtype=torch.long)
    nxt = int(generate(m, idx, 1, temperature=0)[0, -1])
    assert generate(m, idx, 50, temperature=0, eot_id=nxt, use_cache=True).shape == (1, 3)
