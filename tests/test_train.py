"""Training-loop checks on a tiny CPU model and a synthetic, learnable token stream."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tinyllm.model import GPT, GPTConfig  # noqa: E402
from tinyllm.train import TrainConfig, accumulate_grads, config_from_dict, train  # noqa: E402

V = 64
MCFG = GPTConfig(vocab_size=V, block_size=16, n_layers=2, n_heads=2, d_model=32)


@pytest.fixture
def data_dir(tmp_path):
    # learnable stream: token t+1 = (t*5 + 3) % V with occasional random resets
    rng = np.random.default_rng(0)
    n = 20000
    toks = np.empty(n, dtype=np.uint16)
    t = 0
    for i in range(n):
        t = int(rng.integers(V)) if rng.random() < 0.05 else (t * 5 + 3) % V
        toks[i] = t
    toks[: n * 9 // 10].tofile(tmp_path / "train.bin")
    toks[n * 9 // 10:].tofile(tmp_path / "valid.bin")
    return tmp_path


def tcfg(data_dir, out, **kw):
    base = dict(data_dir=str(data_dir), out_dir=str(out), batch_size=8, max_steps=20, warmup=5,
                lr=3e-3, min_lr=3e-4, eval_interval=10, eval_iters=4, log_interval=1,
                ckpt_interval=10, device="cpu", dtype="fp32")
    base.update(kw)
    return TrainConfig(**base)


def test_grad_accum_matches_full_batch():
    torch.manual_seed(0)
    m = GPT(MCFG)
    x = torch.randint(0, V, (8, 16))
    y = torch.randint(0, V, (8, 16))
    from contextlib import nullcontext
    accumulate_grads(m, [(x, y)], nullcontext)
    full = [p.grad.clone() for p in m.parameters()]
    m.zero_grad(set_to_none=True)
    accumulate_grads(m, [(x[i:i + 2], y[i:i + 2]) for i in range(0, 8, 2)], nullcontext)
    for g_full, p in zip(full, m.parameters()):
        torch.testing.assert_close(p.grad, g_full, rtol=1e-4, atol=1e-6)


def test_overfit_one_batch(data_dir, tmp_path):
    torch.manual_seed(0)
    cfg = tcfg(data_dir, tmp_path / "o", max_steps=300, warmup=10, lr=3e-3, min_lr=3e-3,
               overfit_one_batch=True, ckpt_interval=0, log_interval=50, batch_size=4)
    res = train(MCFG, cfg, verbose=False)
    assert res["last_train_loss"] < 0.1


def test_resume_is_exact(data_dir, tmp_path):
    full = train(MCFG, tcfg(data_dir, tmp_path / "a"), verbose=False)
    part = train(MCFG, tcfg(data_dir, tmp_path / "b", stop_at_step=10), verbose=False)
    assert part["step"] == 10
    rest = train(MCFG, tcfg(data_dir, tmp_path / "b"), verbose=False)
    assert rest["step"] == 20
    la = [h["loss"] for h in full["history"] if h["type"] == "train"]
    lb = [h["loss"] for h in part["history"] + rest["history"] if h["type"] == "train"]
    assert la == lb  # bit-identical losses on CPU fp32
    # metrics log on disk is continuous and has no duplicated steps
    recs = [json.loads(ln) for ln in (tmp_path / "b" / "metrics.jsonl").read_text().splitlines()]
    steps = [r["step"] for r in recs if r["type"] == "train"]
    assert steps == list(range(1, 21))
    # final weights identical
    from tinyllm.checkpoint import latest_checkpoint, load_checkpoint
    sa = load_checkpoint(latest_checkpoint(tmp_path / "a"))["model"]
    sb = load_checkpoint(latest_checkpoint(tmp_path / "b"))["model"]
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_metrics_and_val_decreases(data_dir, tmp_path):
    out = tmp_path / "m"
    res = train(MCFG, tcfg(data_dir, out, max_steps=60, eval_interval=20, ckpt_interval=20,
                           grad_accum=2), verbose=False)
    vals = [h for h in res["history"] if h["type"] == "val"]
    assert [v["step"] for v in vals] == [0, 20, 40, 60]
    assert vals[-1]["loss"] < vals[0]["loss"] - 1.0
    assert vals[-1]["ppl"] < vals[0]["ppl"]
    lrs = [h["lr"] for h in res["history"] if h["type"] == "train"]
    assert max(lrs) == pytest.approx(3e-3) and lrs[-1] < lrs[10]
    ck = sorted(p.name for p in out.glob("ckpt_*.pt"))
    assert ck == ["ckpt_0000040.pt", "ckpt_0000060.pt"]  # keep latest 2


def test_resume_rejects_different_model(data_dir, tmp_path):
    train(MCFG, tcfg(data_dir, tmp_path / "r", max_steps=10), verbose=False)
    other = GPTConfig(**{**MCFG.to_dict(), "n_layers": 3})
    with pytest.raises(ValueError):
        train(other, tcfg(data_dir, tmp_path / "r", max_steps=20), verbose=False)


def test_config_from_dict_and_cli_overrides(data_dir, tmp_path):
    with pytest.raises(ValueError):
        config_from_dict({"train": {"no_such_key": 1}})
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train as train_cli
    cfg = {"model": MCFG.to_dict(), "train": {"data_dir": str(data_dir), "device": "cpu",
                                              "dtype": "fp32", "max_steps": 4, "eval_interval": 2,
                                              "eval_iters": 2, "batch_size": 4, "log_interval": 1}}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    out = tmp_path / "cli"
    assert train_cli.main(["--config", str(p), "--set", f"train.out_dir={out}", "train.max_steps=6",
                           "train.ckpt_interval=3"]) == 0
    assert (out / "config.json").exists()
    assert json.loads((out / "config.json").read_text())["train"]["max_steps"] == 6
    assert sorted(x.name for x in out.glob("ckpt_*.pt")) == ["ckpt_0000003.pt", "ckpt_0000006.pt"]


def _instruct_dir(tmp_path):
    # rows: prompt tokens (no loss) then a learnable sequence; targets -100 on the prompt
    rng = np.random.default_rng(0)
    d = tmp_path / "inst"
    d.mkdir()
    for name, n in (("train.npz", 64), ("valid.npz", 16)):
        x = np.empty((n, 16), dtype=np.uint16)
        y = np.full((n, 16), -100, dtype=np.int16)
        for r in range(n):
            seq = [int(rng.integers(V))] * 4 + [0] * 13
            for i in range(4, 17):
                seq[i] = (seq[i - 1] * 5 + 3) % V
            x[r] = seq[:16]
            y[r, 3:] = seq[4:17]
        np.savez(d / name, x=x, y=y)
    return d


def test_instruct_finetune_from_checkpoint(data_dir, tmp_path):
    base = train(MCFG, tcfg(data_dir, tmp_path / "base", max_steps=10), verbose=False)
    assert base["step"] == 10
    from tinyllm.checkpoint import latest_checkpoint, load_checkpoint
    w0 = load_checkpoint(latest_checkpoint(tmp_path / "base"))["model"]["tok_emb.weight"]
    inst = _instruct_dir(tmp_path)
    ft_cfg = tcfg(inst, tmp_path / "ft", data_kind="instruct", train_file="train.npz",
                  val_file="valid.npz", init_from=str(tmp_path / "base"), max_steps=40, eval_interval=40)
    out = train(MCFG, ft_cfg, verbose=False)
    vals = [h["loss"] for h in out["history"] if h["type"] == "val"]
    assert vals[-1] < vals[0] * 0.7
    st = load_checkpoint(latest_checkpoint(tmp_path / "ft"))
    assert st["train_config"]["init_from"] == str(tmp_path / "base")
    assert not torch.equal(st["model"]["tok_emb.weight"], w0)  # weights moved on from the base


def test_init_from_loads_weights(data_dir, tmp_path):
    train(MCFG, tcfg(data_dir, tmp_path / "base", max_steps=10), verbose=False)
    a = train(MCFG, tcfg(data_dir, tmp_path / "a", max_steps=1, init_from=str(tmp_path / "base"),
                         lr=0.0, min_lr=0.0, eval_interval=1), verbose=False)
    b = train(MCFG, tcfg(data_dir, tmp_path / "b", max_steps=1, eval_interval=1), verbose=False)
    va = [h["loss"] for h in a["history"] if h["type"] == "val"][0]
    vb = [h["loss"] for h in b["history"] if h["type"] == "val"][0]
    assert va < vb - 0.3  # trained weights start far below a fresh init
