#!/usr/bin/env bash
# Final demo (PLAN Day 30): load the trained checkpoints and show what they do.
#   1. env check (GPU or CPU)   2. base model stories   3. large model story
#   4. instruction-following story   5. a scripted chat with the chat model
# Every step runs under `timeout`. Usage: bash scripts/demo.sh [--device cpu|cuda] [--quick]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DEVICE=cuda; QUICK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
if [ "$DEVICE" = cuda ] && ! $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "[demo] CUDA not available, falling back to CPU"; DEVICE=cpu
fi
NEW=200; [ "$QUICK" = 1 ] && NEW=60
T=300
need() { ls "$1"/ckpt_*.pt >/dev/null 2>&1 || { echo "[demo] missing checkpoint in $1" >&2; exit 3; }; }

echo "=== 1. base model (8x512, ~30M params): 2 stories ==="
need ckpt/base
timeout $T $PY scripts/generate.py --ckpt ckpt/base --prompt "Once upon a time" -n 2 \
  --max-new-tokens $NEW --temperature 0.8 --top-p 0.95 --seed 1 --device $DEVICE

if [ "$QUICK" = 0 ]; then
  echo; echo "=== 2. large model (91M params): 1 story ==="
  need ckpt/large
  timeout $T $PY scripts/generate.py --ckpt ckpt/large --prompt "Lily found a tiny door" -n 1 \
    --max-new-tokens $NEW --temperature 0.8 --top-p 0.95 --seed 1 --device $DEVICE

  echo; echo "=== 3. instruction fine-tune (v4) ==="
  need ckpt/instruct_v4
  timeout $T $PY scripts/generate.py --ckpt ckpt/instruct_v4 \
    --prompt $'Prompt: Write a story that uses the words dog, ball, park.\nStory:' -n 1 \
    --max-new-tokens $NEW --temperature 0.8 --top-p 0.95 --seed 1 --device $DEVICE
fi

echo; echo "=== 4. scripted chat (chat v3) ==="
need ckpt/chat_large_v3
printf '%s\n' "Tell me a story about a cat who wants to fly." "What happens next?" "/quit" | \
  timeout $T $PY scripts/chat.py --ckpt ckpt/chat_large_v3 --max-new $NEW --seed 1 --device $DEVICE
echo; echo "[demo] DONE"
