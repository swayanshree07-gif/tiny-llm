#!/usr/bin/env bash
# Day 21 ablation: RoPE vs learned positions, same model/budget (8x512, 3000 steps, 49M tokens each).
# Each run is capped by `timeout` (and max_minutes=20 in its config). ~10 min each at ~88k tok/s.
# Afterwards: docs/figures/abl_pos.png + docs/abl_pos_table.md. Re-running resumes/finishes cleanly.
set -u
cd "$(dirname "$0")/.."
mkdir -p .logs
for p in rope learned; do
  timeout 22m .venv/bin/python scripts/train.py --config configs/abl_pos_$p.json \
    > .logs/abl_pos_$p.log 2>&1
  echo "exit=$?" >> .logs/abl_pos_$p.log
done
.venv/bin/python scripts/compare_runs.py ckpt/abl_pos_rope ckpt/abl_pos_learned \
  --labels RoPE learned --title "Positions ablation: val loss (8x512, 3000 steps)" \
  --out docs/figures/abl_pos.png --table docs/abl_pos_table.md
