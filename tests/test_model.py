import math

import pytest

torch = pytest.importorskip("torch")
from tinyllm.model import GPT, GPTConfig  # noqa: E402

SMALL = GPTConfig(vocab_size=512, block_size=32, n_layers=2, n_heads=4, d_model=64, pos="learned")
ROPE = GPTConfig(vocab_size=512, block_size=32, n_layers=2, n_heads=4, d_model=64, pos="rope")


def test_forward_shapes_and_init_loss():
    torch.manual_seed(0)
    m = GPT(SMALL)
    idx = torch.randint(0, 512, (4, 32))
    tgt = torch.randint(0, 512, (4, 32))  # independent targets (idx itself would be "easy" with tied weights)
    logits, loss = m(idx, tgt)
    assert logits.shape == (4, 32, 512)
    assert abs(loss.item() - math.log(512)) < 0.3  # ~uniform at init


def test_weight_tying_and_param_count():
    m = GPT(SMALL)
    assert m.head.weight is m.tok_emb.weight
    d, L, V, T = 64, 2, 512, 32
    per_block = 4 * d * d + 8 * d * d + 4 * d  # attn + mlp + 2 LN (w+b)
    expected = V * d + T * d + L * per_block + 2 * d
    assert m.num_params(non_embedding=False) == expected


@pytest.mark.parametrize("cfg", [SMALL, ROPE], ids=["learned", "rope"])
def test_whole_model_causal(cfg):
    torch.manual_seed(0)
    m = GPT(cfg).eval()
    idx = torch.randint(0, 512, (2, 20))
    idx2 = idx.clone()
    idx2[:, 10:] = torch.randint(0, 512, (2, 10))
    a, _ = m(idx)
    b, _ = m(idx2)
    torch.testing.assert_close(a[:, :10], b[:, :10])


def test_block_size_enforced():
    with pytest.raises(AssertionError):
        GPT(SMALL)(torch.zeros(1, 33, dtype=torch.long))


def test_default_config_size():
    n = GPT(GPTConfig()).num_params()
    assert 20e6 < n < 35e6, n  # plan: ~20-30M params


def test_rope_model_has_no_pos_params():
    m = GPT(ROPE)
    assert m.pos_emb is None
    assert m.num_params(False) == GPT(SMALL).num_params(False) - 32 * 64
    _, loss = m(torch.randint(0, 512, (2, 32)), torch.randint(0, 512, (2, 32)))
    assert abs(loss.item() - math.log(512)) < 0.3
