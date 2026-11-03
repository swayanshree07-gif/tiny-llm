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


def test_compare_runs_dedupes_and_tables(tmp_path):
    import compare_runs
    runs = []
    for name, off in (("a", 0.0), ("b", 0.3)):
        d = tmp_path / name
        d.mkdir()
        recs = [{"type": "val", "step": 0, "loss": 9.0, "ppl": 8103.0}]
        recs += [{"type": "train", "step": s, "loss": 5.0, "lr": 1e-3, "tok_per_s": 1000.0} for s in (150, 200)]
        recs += [{"type": "val", "step": 100, "loss": 3.0 + off},
                 {"type": "val", "step": 200, "loss": 9.9},          # superseded after a resume
                 {"type": "val", "step": 200, "loss": 2.0 + off}]
        (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        runs.append(d)
    assert compare_runs.load_run(runs[0])["val"] == [(0, 9.0), (100, 3.0), (200, 2.0)]
    out, table = compare_runs.compare(runs, ["A", "B"], tmp_path / "c.png", "t")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "| A | 200 | 2.0000 |" in table and "| B | 200 | 2.3000 |" in table and "1.0k" in table
