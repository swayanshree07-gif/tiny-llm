import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_lint", ROOT / "scripts" / "check_lint.py")
check_lint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_lint)


def test_repo_is_lint_clean():
    assert check_lint.lint(ROOT) == []


def test_lint_catches_planted_problems(tmp_path):
    for d in ("tinyllm", "scripts", "tests"):
        (tmp_path / d).mkdir()
    (tmp_path / "tinyllm" / "m.py").write_text('"""doc"""\nimport os\n\ndef f(x):\n    return x\n')
    (tmp_path / "scripts" / "s.py").write_text("x = 1\n")
    probs = "\n".join(check_lint.lint(tmp_path))
    assert "unused import 'os'" in probs
    assert "'f' has no return annotation" in probs and "arg 'x' not annotated" in probs
    assert "scripts/s.py:1: no module docstring" in probs
