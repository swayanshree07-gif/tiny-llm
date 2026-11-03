import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_plot_writes_png(tmp_path):
    import plot_metrics
    m = tmp_path / "metrics.jsonl"
    recs = [{"type": "val", "step": 0, "loss": 9.0, "ppl": 8103.0}]
    recs += [{"type": "train", "step": s, "loss": 9 - s / 20, "lr": 1e-3 * min(1, s / 10)} for s in range(1, 101)]
    recs += [{"type": "val", "step": 100, "loss": 4.1, "ppl": 60.3}]
    m.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = plot_metrics.plot(m, tmp_path / "f" / "loss.png")
    assert out.exists() and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
