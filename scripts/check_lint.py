"""Stdlib-only lint check (no ruff/pyflakes in the venv).

Checks every .py in tinyllm/, scripts/, tests/:
  - unused imports (name imported but never referenced; `__init__.py` and `# noqa` lines skipped)
  - a module docstring in tinyllm/ and scripts/
  - public functions in tinyllm/ have a return annotation and annotated arguments
Exit code 1 if anything is found. Usage: python scripts/check_lint.py [--root DIR]
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


def unused_imports(tree: ast.Module, lines: list[str]) -> list[tuple[int, str]]:
    imported: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            if "noqa" in lines[node.lineno - 1]:
                continue
            for a in node.names:
                name = (a.asname or a.name).split(".")[0]
                imported[name] = node.lineno
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.value.id for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    # names exported via __all__ or used in string annotations
    src = "\n".join(lines)
    return [(ln, f"unused import '{nm}'") for nm, ln in imported.items()
            if nm not in used and f"'{nm}" not in src and f'"{nm}' not in src]


def missing_hints(tree: ast.Module) -> list[tuple[int, str]]:
    out = []
    for node in tree.body:
        fns = [node] if isinstance(node, ast.FunctionDef) else (
            [n for n in node.body if isinstance(n, ast.FunctionDef)] if isinstance(node, ast.ClassDef) else [])
        for fn in fns:
            if fn.name.startswith("_") and fn.name != "__init__":
                continue
            args = [a for a in fn.args.args + fn.args.kwonlyargs if a.arg not in ("self", "cls")]
            if fn.returns is None and fn.name != "__init__":
                out.append((fn.lineno, f"'{fn.name}' has no return annotation"))
            for a in args:
                if a.annotation is None:
                    out.append((fn.lineno, f"'{fn.name}' arg '{a.arg}' not annotated"))
    return out


def lint(root: Path) -> list[str]:
    problems = []
    for sub in ("tinyllm", "scripts", "tests"):
        for p in sorted((root / sub).glob("*.py")):
            src = p.read_text()
            tree = ast.parse(src)
            lines = src.splitlines()
            found = [] if p.name == "__init__.py" else unused_imports(tree, lines)
            if sub != "tests" and p.name != "__init__.py" and ast.get_docstring(tree) is None:
                found.append((1, "no module docstring"))
            if sub == "tinyllm":
                found += missing_hints(tree)
            problems += [f"{p.relative_to(root)}:{ln}: {msg}" for ln, msg in found]
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    probs = lint(Path(ap.parse_args().root))
    for p in probs:
        print(p)
    print(f"RESULT {'OK' if not probs else 'FAIL'} ({len(probs)} problems)")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
