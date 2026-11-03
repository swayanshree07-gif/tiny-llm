"""Train (or resume) a GPT from a JSON config.

  .venv/bin/python scripts/train.py --config configs/base.json
  .venv/bin/python scripts/train.py --config configs/base.json --set train.max_minutes=40 model.n_layers=6

Always run long sessions under `timeout` (project rule: <= 45 min), e.g.
  timeout 45m .venv/bin/python scripts/train.py --config configs/base.json --set train.max_minutes=42
SIGTERM from `timeout` is caught: the current step finishes and a checkpoint is saved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm.train import config_from_dict, train  # noqa: E402


def parse_value(s: str):
    try:
        return json.loads(s)  # numbers, true/false/null, lists
    except json.JSONDecodeError:
        return s  # plain string


def apply_overrides(cfg: dict, items: list[str]) -> dict:
    for item in items:
        key, _, val = item.partition("=")
        section, _, name = key.partition(".")
        if section not in ("model", "train") or not name or not _:
            raise SystemExit(f"bad override {item!r}; use model.KEY=VAL or train.KEY=VAL")
        cfg.setdefault(section, {})[name] = parse_value(val)
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VAL")
    args = ap.parse_args(argv)
    cfg = apply_overrides(json.loads(Path(args.config).read_text()), args.set)
    mcfg, tcfg = config_from_dict(cfg)
    out = Path(tcfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    res = train(mcfg, tcfg)
    print(f"[done] step {res['step']} train loss {res['last_train_loss']:.4f} "
          f"val loss {res['last_val_loss']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
