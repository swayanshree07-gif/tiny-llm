"""Gradient checkpointing must give the same loss and gradients as the plain forward."""
import torch

from tinyllm.model import GPT, GPTConfig


def _grads(gc: bool):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=256, block_size=32, n_layers=3, n_heads=4, d_model=64, grad_checkpoint=gc)
    m = GPT(cfg)
    x = torch.randint(0, 256, (2, 32), generator=torch.Generator().manual_seed(1))
    _, loss = m(x, x)
    loss.backward()
    return loss.item(), {n: p.grad.clone() for n, p in m.named_parameters()}


def test_grad_checkpoint_matches():
    l0, g0 = _grads(False)
    l1, g1 = _grads(True)
    assert abs(l0 - l1) < 1e-6
    for n in g0:
        torch.testing.assert_close(g0[n], g1[n], rtol=1e-5, atol=1e-7)


def test_grad_checkpoint_off_in_eval():
    cfg = GPTConfig(vocab_size=256, block_size=16, n_layers=2, n_heads=2, d_model=32, grad_checkpoint=True)
    m = GPT(cfg).eval()
    with torch.no_grad():
        logits, _ = m(torch.zeros(1, 4, dtype=torch.long))
    assert logits.shape == (1, 4, 256)
