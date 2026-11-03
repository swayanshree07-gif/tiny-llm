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

## Train
```bash
# main run (29.4M params, 16k tokens/step, torch.compile, 42-min budget per session; resumes automatically)
timeout 45m .venv/bin/python scripts/train.py --config configs/base.json > .logs/base.log 2>&1
# quick experiments: override any key
.venv/bin/python scripts/train.py --config configs/smoke.json --set train.max_steps=200 model.pos=learned
```
Each run directory (`train.out_dir`, e.g. `ckpt/base/`) holds `config.json`, `metrics.jsonl`
(train loss / LR / grad norm / tok/s / peak memory, and val loss / perplexity) and the
latest 2 `ckpt_<step>.pt` files (~350 MB each with AdamW state). Rerunning the same command
resumes from the latest checkpoint, with identical results (tested). `timeout`'s SIGTERM is caught: the
current step finishes, then a val eval runs and a checkpoint is written. Send output to a
log file, not `| tail`.

```bash
.venv/bin/python scripts/plot_metrics.py ckpt/base/metrics.jsonl --out docs/figures/loss.png
.venv/bin/python scripts/generate.py --ckpt ckpt/base --prompt "Once upon a time" -n 3 \
    --temperature 0.8 --top-p 0.95          # --top-k K, --temperature 0 for greedy
.venv/bin/python scripts/bench.py --out docs/bench.md   # throughput / VRAM table
```

## Layout
```
tinyllm/        device.py, bpe_ref.py, bpe.py, tokenizer.py, data.py, model.py,
                optim.py (AdamW groups, warmup+cosine), checkpoint.py, train.py, generate.py
scripts/        check_env.py, download_data.py, train_tokenizer.py, tokenize_data.py,
                train.py, generate.py, plot_metrics.py, bench.py
configs/        base.json (main run), smoke.json (small model, quick checks)
tests/          pytest suite
docs/           bench.md, figures/ (loss curves), report later
```

## Limits (from CLAUDE.md)
VRAM < ~6.5 GB, each training session ≤ 45 min under `timeout`, whole folder ≤ 12 GB,
data ≤ 1.5 GB, keep only the latest 2 checkpoints.

## Status
Days 1–17 of `PLAN.md` are done, plus the Day 19 plot script; the main training run has started.
- Data/tokenizer: TinyStories subset, own 8192-token byte-level BPE, 176.7M train tokens.
- Model: GPT (pre-LN, RoPE or learned positions, tied embeddings), 29.4M params.
- Training: bf16 autocast, gradient accumulation, clipping, AdamW decay groups, warmup + cosine
  LR, JSON configs + CLI overrides, val loss/perplexity, JSONL metrics, checkpoints (keep 2),
  exact resume, time budget and SIGTERM handling, optional `torch.compile` (+31% throughput).
- Generation: temperature / top-k / top-p, stops at `<|endoftext|>`.
- Main run `ckpt/base`: step 651 / 40,000, val loss 2.219 (ppl 9.2); samples are already
  readable short stories. Curve: `docs/figures/loss.png`. Later sessions resume it.
- Not yet: KV cache, ablations, final report.
