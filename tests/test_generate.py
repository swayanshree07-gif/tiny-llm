import pytest

torch = pytest.importorskip("torch")
from tinyllm.checkpoint import save_checkpoint  # noqa: E402
from tinyllm.generate import filter_logits, generate, load_model  # noqa: E402
from tinyllm.model import GPT, GPTConfig  # noqa: E402

CFG = GPTConfig(vocab_size=50, block_size=16, n_layers=2, n_heads=2, d_model=32)


def finite(t):
    return torch.isfinite(t[0]).nonzero().flatten().tolist()


def test_top_k_toy():
    lg = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
    assert finite(filter_logits(lg, top_k=2)) == [1, 3]
    assert finite(filter_logits(lg, top_k=1)) == [1]
    assert finite(filter_logits(lg, top_k=10)) == [0, 1, 2, 3, 4]


def test_top_p_toy():
    probs = torch.tensor([[0.1, 0.5, 0.25, 0.15]])
    lg = probs.log()
    assert finite(filter_logits(lg, top_p=0.5)) == [1]          # 0.5 alone reaches 0.5
    assert finite(filter_logits(lg, top_p=0.6)) == [1, 2]       # 0.5 < 0.6 -> add 0.25
    assert finite(filter_logits(lg, top_p=0.76)) == [1, 2, 3]   # 0.75 < 0.76 -> add 0.15
    assert finite(filter_logits(lg, top_p=0.01)) == [1]         # top token always kept
    assert finite(filter_logits(lg, top_p=1.0)) == [0, 1, 2, 3]
    # renormalised kept mass
    p = filter_logits(lg, top_p=0.6).softmax(-1)
    torch.testing.assert_close(p, torch.tensor([[0.0, 0.5 / 0.75, 0.25 / 0.75, 0.0]]))


def test_top_p_batched_rows_independent():
    lg = torch.tensor([[0.0, 10.0, 0.0], [10.0, 0.0, 9.9]])
    out = filter_logits(lg, top_p=0.9)
    assert torch.isfinite(out[0]).tolist() == [False, True, False]
    assert torch.isfinite(out[1]).tolist() == [True, False, True]


def test_top_k1_equals_greedy_and_is_deterministic():
    torch.manual_seed(0)
    m = GPT(CFG)
    idx = torch.randint(0, 50, (3, 4))
    greedy = generate(m, idx, 20, temperature=0)
    for seed in (1, 2):
        g = torch.Generator().manual_seed(seed)
        assert torch.equal(generate(m, idx, 20, temperature=1.0, top_k=1, generator=g), greedy)
    assert greedy.shape == (3, 24) and torch.equal(greedy[:, :4], idx)


def test_sampling_seeded_and_crops_context():
    torch.manual_seed(0)
    m = GPT(CFG)
    idx = torch.randint(0, 50, (2, 3))
    a = generate(m, idx, 30, 1.0, top_p=0.9, generator=torch.Generator().manual_seed(7))
    b = generate(m, idx, 30, 1.0, top_p=0.9, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b) and a.shape == (2, 33)  # 33 > block_size 16: context cropping works


def test_eot_stops_early():
    torch.manual_seed(0)
    m = GPT(CFG)
    idx = torch.zeros(1, 2, dtype=torch.long)
    nxt = int(generate(m, idx, 1, temperature=0)[0, -1])
    out = generate(m, idx, 50, temperature=0, eot_id=nxt)
    assert out.shape == (1, 3)


def test_load_model_roundtrip(tmp_path):
    m = GPT(CFG)
    save_checkpoint({"step": 5, "model": m.state_dict(), "model_config": CFG.to_dict()}, tmp_path, 5)
    m2, st = load_model(tmp_path)
    assert st["step"] == 5
    idx = torch.randint(0, 50, (1, 8))
    torch.testing.assert_close(m2(idx)[0], m.eval()(idx)[0])
