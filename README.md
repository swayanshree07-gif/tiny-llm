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
```bash
.venv/bin/python scripts/check_tokenizer.py            # full-val round-trip, .bin matches, bytes/token
.venv/bin/python scripts/check_tokenizer.py --split train --mb 50   # same on a train prefix
.venv/bin/python scripts/check_tokens.py               # .bin sizes vs meta, id range, random slices readable
.venv/bin/python scripts/check_loader.py               # batches: y == x shifted, ids in range, == memmap, resume, GPU speed
.venv/bin/python scripts/check_attention.py            # trained model: manual == sdpa attention, causality, cached chunk
.venv/bin/python scripts/check_rope.py                 # RoPE: relative-position property, shift invariance of the trained model
.venv/bin/python scripts/check_overfit.py              # training loop: overfit one batch (<0.1) at full size, ~15 s
.venv/bin/python scripts/check_accum.py                # grad accum k×(B/k) == 1×B (fp32), clipping, decay groups, bf16, ~10 s
.venv/bin/python scripts/check_schedule.py             # warmup+cosine values, short CLI run logs the LR curve (docs/figures/lr_*.png)
.venv/bin/python scripts/check_resume.py               # 20 steps == 10 + save + resume + 10 (bit-exact, full size), ~20 s
```
The loader (`tinyllm/data.py`) draws random contiguous 257-token windows from the uint16 memmap
(x = first 256, y = last 256), pins them and copies them to the GPU without blocking.
It moves about 50–70M tok/s to the GPU, around 600× what training consumes.

## Train
```bash
# main run (29.4M params, 16k tokens/step, torch.compile, 42-min budget per session; resumes automatically)
timeout 45m .venv/bin/python scripts/train.py --config configs/base.json > .logs/base.log 2>&1
# larger model (91.3M params, 12x768, 16k tokens/step, 4.95 GB peak; 18,000 steps over ~4 sessions)
timeout 45m .venv/bin/python scripts/train.py --config configs/large.json > .logs/large.log 2>&1
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
.venv/bin/python scripts/bench_kv.py --ckpt ckpt/base --out docs/kv_cache.md   # KV cache speedup
.venv/bin/python scripts/sample_suite.py --ckpt ckpt/base --out samples/suite.md  # prompts x settings + stats
.venv/bin/python scripts/compare_runs.py ckpt/runA ckpt/runB --labels A B --out docs/figures/cmp.png  # val curves + table
scripts/run_abl_pos.sh     # RoPE vs learned positions ablation (2 x ~10 min GPU) -> docs/figures/abl_pos.png
scripts/run_sweep.sh       # LR 3e-4 / 1.2e-3 and 32k-token batch (3 x ~10 min GPU) -> docs/figures/sweep.png
```
`compare_runs.py --x tokens` plots against training tokens instead of steps (needed when
runs use different batch sizes). The sweep reuses `ckpt/abl_pos_rope` as its lr 6e-4 / 16k
baseline, so run `run_abl_pos.sh` first. It deletes the `.pt` files of finished sweep runs
after plotting (the metrics stay), so the extra disk use is temporary.
### Instruction fine-tune (PLAN Day 27)
```bash
.venv/bin/python scripts/build_instruct.py        # -> data/instruct/{train,valid}.npz + *_prompts.jsonl (~5 s)
.venv/bin/python scripts/eval_instruct.py --ckpt ckpt/base --out docs/instruct_eval_base.md   # before
timeout 20m .venv/bin/python scripts/train.py --config configs/instruct.json > .logs/instruct.log 2>&1
.venv/bin/python scripts/eval_instruct.py --ckpt ckpt/instruct --out docs/instruct_eval_ft.md  # after
# v2: bigger set (132k rows), ~1 epoch, lower LR
.venv/bin/python scripts/build_instruct.py --out data/instruct_v2 --train 160000
timeout 25m .venv/bin/python scripts/train.py --config configs/instruct_v2.json > .logs/instruct_v2.log 2>&1
.venv/bin/python scripts/eval_instruct.py --ckpt ckpt/instruct_v2 --out docs/instruct_eval_ft_v2.md
.venv/bin/python scripts/prompt_sensitivity.py --ckpt ckpt/instruct_v2   # own vs swapped prompt story loss
# v3: words drawn from the story's salient (repeated) words; anyfix = v1/v2-style prompts with the fragment bug fixed
.venv/bin/python scripts/build_instruct.py --out data/instruct_v3 --train 160000 --word-mode salient
.venv/bin/python scripts/build_instruct.py --out data/instruct_anyfix --train 0      # fixed eval prompts only
timeout 25m .venv/bin/python scripts/train.py --config configs/instruct_v3.json > .logs/instruct_v3.log 2>&1
.venv/bin/python scripts/eval_instruct.py --ckpt ckpt/instruct_v3 --prompts data/instruct_v3/valid_prompts.jsonl --out docs/instruct_eval_v3_on_v3.md
.venv/bin/python scripts/prompt_sensitivity.py --ckpt ckpt/instruct_v3 --data data/instruct_v3
# v4 (best): same rows as v3, loss on requested-word tokens weighted x5 (mask from mark_words.py)
.venv/bin/python scripts/mark_words.py --data data/instruct_v3            # -> train_wordmask.npz
timeout 25m .venv/bin/python scripts/train.py --config configs/instruct_v4.json > .logs/instruct_v4.log 2>&1
.venv/bin/python scripts/eval_instruct.py --ckpt ckpt/instruct_v4 --prompts data/instruct_v3/valid_prompts.jsonl --out docs/instruct_eval_v4_on_v3.md
```
The examples are built from our own stories, so nothing extra is downloaded. Each prompt is either
`Prompt: Write a short story that uses the words "a", "b" and "c".\nStory:` or `... that begins with: "<sentence>"`.
The loss is computed on the story only (prompt and padding targets are -100). `eval_instruct.py` measures
how often the generated story actually contains the requested words or starts with the requested sentence.
`train.init_from` loads the weights only (with a fresh optimizer), and `train.data_kind=instruct` reads `.npz` rows.

Generation uses a KV cache by default (`--no-cache` turns it off); greedy output is identical
either way while the text fits in the 256-token context.

## Layout
```
tinyllm/        device.py, bpe_ref.py, bpe.py, tokenizer.py, data.py, model.py,
                optim.py (AdamW groups, warmup+cosine), checkpoint.py, train.py, generate.py,
                quality.py (repetition / length stats for samples), instruct.py (fine-tune data)
scripts/        check_env.py, download_data.py, train_tokenizer.py, tokenize_data.py,
                train.py, generate.py, plot_metrics.py, bench.py, bench_kv.py, sample_suite.py,
                compare_runs.py, check_tokenizer.py, check_tokens.py, check_loader.py, check_attention.py, check_model.py,
                check_rope.py, check_overfit.py, check_accum.py, check_schedule.py, check_resume.py,
                build_instruct.py, mark_words.py,
                eval_instruct.py, prompt_sensitivity.py, run_abl_pos.sh, run_sweep.sh
configs/        base.json (main run), smoke.json (small model, quick checks),
                abl_pos_{rope,learned}.json (positions ablation, 3000 steps each),
                sweep_{lr3e-4,lr1.2e-3,bs32k}.json (LR / batch sweep, 49M tokens each),
                instruct.json / instruct_v2..v4.json (Day 27 fine-tunes from ckpt/base)
tests/          pytest suite
docs/           REPORT.md (draft), bench.md, kv_cache.md, instruct_eval_*.md, figures/
samples/        sample-suite outputs (samples/tmp/ is git-ignored)
```

## Limits (from CLAUDE.md)
VRAM < ~6.5 GB, each training session ≤ 45 min under `timeout`, whole folder ≤ 12 GB,
data ≤ 1.5 GB, keep only the latest 2 checkpoints.

## Status
Days 1–26 of `PLAN.md` are done (the Day 8 attention, Day 9 model and Day 10 RoPE checks re-run as scripts on the trained model on 2026-11-07, the Day 11 overfit, Day 12 grad-accum, Day 13 schedule and Day 14 resume checks on 2026-11-08), and Day 27 (instruction fine-tune) has four passes. A draft report is in `docs/REPORT.md`.
- Data/tokenizer: TinyStories subset, own 8192-token byte-level BPE, 176.7M train tokens,
  4.14 bytes/token. `scripts/check_tokenizer.py` and `scripts/check_tokens.py` re-verify it.
- Model: GPT (pre-LN, RoPE or learned positions, tied embeddings), 29.4M params. `scripts/check_model.py`
  re-verifies it at full size (param formula, init stds, init loss 9.07–9.08 vs ln 8192 = 9.01, causality,
  gradients to all params, strict checkpoint load); `--random --cpu` is the quick pytest mode.
  `scripts/check_rope.py` checks RoPE: <R_m q, R_n k> depends only on m − n over all 256×256 pairs, and shifting
  every position by 1/37/256 leaves the trained model's loss unchanged (1.35589).
- Training: bf16 autocast, gradient accumulation, clipping, AdamW decay groups, warmup + cosine
  LR, JSON configs + CLI overrides, val loss/perplexity, JSONL metrics, checkpoints (keep 2),
  exact resume, time budget and SIGTERM handling, optional `torch.compile` (+31% throughput),
  weight-only `init_from` for fine-tunes. `scripts/check_overfit.py` re-verifies the loop: one fixed batch
  goes from 9.14 to < 0.1 at step 51 (control on fresh batches stays at 4.3). `scripts/check_accum.py` shows 2/4/8 micro-batches give
  the same fp32 grads as one batch (rel err < 1e-6) and the same AdamW update (3e-5). `scripts/check_schedule.py` checks the LR schedule against a closed form
  and that a short CLI run logs exactly that LR at every step (`docs/figures/lr_schedule.png`, `lr_run.png`).
  `scripts/check_resume.py` shows a 20-step full-size run (bf16, dropout 0.1) and a 10 + resume + 10 run give
  bit-identical losses, weights, AdamW state and RNG/sampler state.
- Generation: temperature / top-k / top-p, stops at `<|endoftext|>`, KV cache (identical greedy
  output; 1.6x faster at batch 1, 9–22x at batch 8–32, see `docs/kv_cache.md`).
- Main run `ckpt/base`: **finished, 40,000 steps (655M tokens), val loss 1.229 (ppl 3.42)**.
  Stories are coherent and end on their own (`samples/suite_step40000.md`). Curve: `docs/figures/loss.png`.
- Ablations at 49M tokens: RoPE 1.557 vs learned 1.587 (`docs/figures/abl_pos.png`). LR 1.2e-3 1.519 beats
  6e-4 1.557 and 3e-4 1.645. A 32k batch at equal tokens is worse at 1.624 (`docs/figures/sweep.png`).
- Instruction fine-tune v1 (`ckpt/instruct`): prompt-following goes from 0% to 16.5%. "Begins with" prompts
  reach 88.6%, but "uses the words" prompts only 1.2% (23.6% of the requested words appear).
- Fine-tune v2 (`ckpt/instruct_v2`, 132k rows, ~1 epoch): val loss 1.082 (better than v1's 1.103), but follow rate
  is about the same at 15.0%. `scripts/prompt_sensitivity.py` shows the models barely use the word list
  (wrong words cost +0.01 nats/token, against +0.21 for a wrong first sentence). See `docs/REPORT.md` §5.3.
- Fine-tune v3 (`ckpt/instruct_v3`): fixed a prompt bug (~9% of word prompts asked for fragments like "uddenly") and
  drew the words from the story's repeated (salient) words. Follow rate 16.5% / 20.0% on fixed / salient prompts
  (v2: 15.0% / 18.5%), words partial credit 31–37% (v2 23–27%), word-list sensitivity about doubled. Still only ~3%
  of stories use all three words.
- Fine-tune v4 (`ckpt/instruct_v4`, best): v3's rows with the loss on requested-word tokens weighted ×5. Follow rate
  **25.0% / 39.0%** on fixed / salient prompts (words-prompts 11.5% / 27.3%, partial 52–63%), "begins" unchanged
  (87–89%). The v1/v2 checkpoints were deleted to save disk (metrics and eval docs kept).
- **Extended goals (2026-11-08):** `PLAN.md` Days 15–27 were rewritten around a larger model, a chat fine-tune with
  `chat.py`, inference speed (KV-cache benchmark, int8) and an evaluation report. Day 15: optional gradient
  checkpointing (`model.grad_checkpoint`, same grads, tested), candidates benchmarked in `docs/bench_large.md`
  (`scripts/bench_large.py`), chosen `configs/large.json` (91.3M params). The large run (`ckpt/large`, ~1.1 GB per checkpoint) is finished: 18,000 steps (295M tokens), val loss 1.208 (ppl 3.35; base 29M model at the same step/tokens: 1.348); curve in `docs/figures/large_loss.png`
  (ppl 3.78) vs the 29M base model's 1.442 at the same step. Resume with
  `timeout 45m .venv/bin/python scripts/train.py --config configs/large.json --set train.max_minutes=40`.
  The fine-tune v3 checkpoint and v2 data were deleted to free disk.
- Not yet: large-vs-base comparison + samples, chat fine-tune and `chat.py`, int8, eval report, demo script.
