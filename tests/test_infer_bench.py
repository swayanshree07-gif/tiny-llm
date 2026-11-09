import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import infer_bench  # noqa: E402


def test_infer_bench_random_cpu(tmp_path):
    out = tmp_path / "b.md"
    assert infer_bench.main(["--device", "cpu", "--new", "12", "--reps", "1", "--batch", "1", "2",
                             "--prec", "fp32", "--out", str(out)]) == 0
    text = out.read_text()
    assert text.count("| random-tiny |") == 2
    assert "True (1.0000)" in text  # cached greedy == uncached greedy
