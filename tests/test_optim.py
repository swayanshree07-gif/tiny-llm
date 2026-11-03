import math

import pytest

torch = pytest.importorskip("torch")
from tinyllm.model import GPT, GPTConfig  # noqa: E402
from tinyllm.optim import lr_at, make_optimizer, param_groups  # noqa: E402


def test_schedule_values():
    kw = dict(max_lr=1.0, min_lr=0.1, warmup=10, total=110)
    assert lr_at(0, **kw) == pytest.approx(0.1)          # (0+1)/10
    assert lr_at(4, **kw) == pytest.approx(0.5)
    assert lr_at(9, **kw) == pytest.approx(1.0)          # end of warmup
    assert lr_at(10, **kw) == pytest.approx(1.0)         # cosine start
    assert lr_at(60, **kw) == pytest.approx(0.55)        # half way: (1+0.1)/2
    assert lr_at(110, **kw) == pytest.approx(0.1)
    assert lr_at(500, **kw) == pytest.approx(0.1)        # stays at min after total
    vals = [lr_at(s, **kw) for s in range(10, 111)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))  # monotone decay


def test_schedule_no_warmup():
    assert lr_at(0, 1.0, 0.0, 0, 100) == pytest.approx(1.0)
    assert lr_at(100, 1.0, 0.0, 0, 100) == pytest.approx(0.0)
    assert lr_at(50, 1.0, 0.0, 0, 100) == pytest.approx(0.5)
    assert math.isfinite(lr_at(0, 1.0, 0.0, 0, 0))


def test_decay_groups():
    m = GPT(GPTConfig(vocab_size=128, block_size=16, n_layers=2, n_heads=2, d_model=32))
    decay, no_decay = param_groups(m, 0.1)
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    ids = [id(p) for g in (decay, no_decay) for p in g["params"]]
    assert len(ids) == len(set(ids))                     # tied weight appears once
    assert sum(p.numel() for g in (decay, no_decay) for p in g["params"]) == m.num_params(False)
    assert any(p is m.tok_emb.weight for p in decay["params"])
    # every LayerNorm param is undecayed
    ln = {id(p) for n, p in m.named_parameters() if "ln" in n}
    assert ln <= {id(p) for p in no_decay["params"]}
    opt = make_optimizer(m, 1e-3)
    assert len(opt.param_groups) == 2
