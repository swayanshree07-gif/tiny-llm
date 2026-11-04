#!/usr/bin/env bash
# Day 23 sweep: learning rate and batch size at an equal token budget (49M tokens, 8x512 model).
#   baseline   = ckpt/abl_pos_rope (lr 6e-4, 16k tok/step, 3000 steps) from scripts/run_abl_pos.sh
#   lr 3e-4, lr 1.2e-3 (16k tok/step, 3000 steps); bs32k (lr 6e-4, 32k tok/step, 1500 steps)
# Each run: `timeout 22m` (+ max_minutes=20 in config), ~10 min at ~88k tok/s. A run whose
# metrics already contain the final val step is skipped. After the plot/table, the .pt files of
# finished sweep runs are deleted (metrics.jsonl stays) to keep the folder under the disk limit.
# Usage: scripts/run_sweep.sh [run names...]   (default: all three)
set -u
cd "$(dirname "$0")/.."
mkdir -p .logs
runs=("$@"); [ ${#runs[@]} -eq 0 ] && runs=(sweep_lr3e-4 sweep_lr1.2e-3 sweep_bs32k)
finished() {  # $1 = run name: true if metrics has a val record at max_steps
  .venv/bin/python - "$1" <<'PY'
import json, sys, pathlib
n = sys.argv[1]; steps = json.load(open(f"configs/{n}.json"))["train"]["max_steps"]
m = pathlib.Path(f"ckpt/{n}/metrics.jsonl")
ok = m.exists() and any(json.loads(l).get("type") == "val" and json.loads(l)["step"] == steps
                        for l in m.read_text().splitlines() if l.strip())
sys.exit(0 if ok else 1)
PY
}
for n in "${runs[@]}"; do
  if finished "$n"; then echo "$n: already finished, skipping"; continue; fi
  timeout 22m .venv/bin/python scripts/train.py --config "configs/$n.json" > ".logs/$n.log" 2>&1
  echo "exit=$?" >> ".logs/$n.log"
done
dirs=(); labels=()
[ -f ckpt/abl_pos_rope/metrics.jsonl ] && { dirs+=(ckpt/abl_pos_rope); labels+=("lr6e-4_bs16k"); }
for n in sweep_lr3e-4 sweep_lr1.2e-3 sweep_bs32k; do
  [ -f "ckpt/$n/metrics.jsonl" ] && { dirs+=("ckpt/$n"); labels+=("${n#sweep_}"); }
done
.venv/bin/python scripts/compare_runs.py "${dirs[@]}" --labels "${labels[@]}" \
  --title "LR / batch sweep: val loss (8x512, 49M tokens)" \
  --x tokens --out docs/figures/sweep.png --table docs/sweep_table.md || exit 1
for n in sweep_lr3e-4 sweep_lr1.2e-3 sweep_bs32k; do
  finished "$n" && rm -f "ckpt/$n"/ckpt_*.pt
done
