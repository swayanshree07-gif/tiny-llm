import torch

from tinyllm.model import GPT, GPTConfig
from tinyllm.quant import Int8Linear, model_bytes, quantize_model, quantize_per_channel


def test_per_channel_roundtrip_error_bounded():
    torch.manual_seed(0)
    w = torch.randn(64, 32) * torch.linspace(0.01, 3, 64)[:, None]
    q, s = quantize_per_channel(w)
    assert q.dtype == torch.int8 and s.shape == (64,)
    assert q.abs().max() == 127
    err = (q.float() * s[:, None] - w).abs()
    assert (err <= s[:, None] / 2 + 1e-6).all()  # at most half a step per row


def test_quantized_model_close_and_smaller():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=256, block_size=32, n_layers=2, d_model=64, n_heads=4)
    m = GPT(cfg).eval()
    x = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        ref, loss_ref = m(x, x)
        before = model_bytes(m)
        quantize_model(m)
        out, loss_q = m(x, x)
    assert sum(isinstance(mod, Int8Linear) for mod in m.modules()) == 2 * 4 + 1
    assert not any(type(mod) is torch.nn.Linear for mod in m.modules())
    assert abs(loss_q.item() - loss_ref.item()) < 0.01
    assert torch.allclose(out, ref, atol=0.05)
    assert model_bytes(m) < before
