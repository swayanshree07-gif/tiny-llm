# tiny-llm — a small GPT language model from scratch

A GPT-style language model written from scratch in PyTorch (no `transformers`):
own byte-level BPE tokenizer, decoder-only transformer, training loop with
mixed precision / gradient accumulation / cosine LR / checkpoint-resume,
validation perplexity, and sampling with temperature / top-k / top-p.
It's trained on a subset of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(CDLA-Sharing-1.0) and should end up writing short coherent stories.

See `PLAN.md` for the 30-day plan and `PROGRESS.md` for the daily log.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 5050 = sm_120
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/check_env.py      # GPU / bf16 / SDPA sanity check
.venv/bin/python -m pytest -q              # tests
```

## Data
```bash
.venv/bin/python scripts/download_data.py --train-mb 700   # -> data/raw/{train,valid}.txt
```
This downloads the full validation file and the first 700 MiB of the V2 train file,
cut at a story boundary. `data/`, `ckpt/` and `.venv/` are git-ignored.

```bash
.venv/bin/python scripts/train_tokenizer.py --mb 200 --vocab 8192   # -> data/tok/tokenizer.json (~25 s)
.venv/bin/python scripts/tokenize_data.py                           # -> data/tok/{valid,train}.bin + meta.json (~1 min)
```
Tokens are stored as uint16; id 8191 is `<|endoftext|>` (story separator).
Current numbers: 4.14 bytes/token, 176.7M train tokens, 5.4M val tokens.

## Layout
```
tinyllm/        library code: device.py, bpe_ref.py, bpe.py, tokenizer.py, data.py, model.py
scripts/        check_env.py, download_data.py, train_tokenizer.py, tokenize_data.py
tests/          pytest suite
docs/           report and figures (later)
```

## Limits (from CLAUDE.md)
VRAM < ~6.5 GB, each training session ≤ 45 min under `timeout`, whole folder ≤ 12 GB,
data ≤ 1.5 GB, keep only the latest 2 checkpoints.

## Status
Done (Days 1–10 of `PLAN.md`, all in the Day 1 session): environment check, data download,
reference + fast byte-level BPE, final 8192-token tokenizer, tokenised train/val
`.bin` files, random-window data loader, causal self-attention (manual and SDPA paths),
and the GPT model (pre-LN, tied embeddings, RoPE or learned positions; default 8 layers ×
512 wide, 29.4M params). 36 tests pass.
Not yet: training loop, checkpointing, evaluation, generation.
