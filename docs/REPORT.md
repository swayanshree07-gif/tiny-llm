# tiny-llm — report (draft, started Day 6; to be finalised on PLAN Day 29)

A 29.4M-parameter GPT trained from scratch on a 700 MiB subset of TinyStories, on one
RTX 5050 Laptop GPU (8 GB) in three ~42-minute sessions. Everything (tokenizer, model,
training loop, sampling) is our own code on top of plain PyTorch.

## 1. Data and tokenizer
- TinyStories V2 (GPT-4) text, CDLA-Sharing-1.0: full validation file (27,630 stories) and the
  first 700 MiB of train (895,926 stories), cut at a story boundary.
- Own byte-level BPE (GPT-2 pre-tokenisation regex), vocab 8192: 256 bytes + 7,935 merges +
  `<|endoftext|>` (id 8191), trained on the first 200 MiB of train.
- 4.14 UTF-8 bytes per token; 176.7M train tokens and 5.4M val tokens, stored as uint16 memmaps.
- Checks: `scripts/check_tokenizer.py` (full-val round-trip, the .bin equals a fresh encoding,
  EOT count == story count) and `scripts/check_tokens.py` (file sizes vs meta, id range,
  random slices decode to readable text).

## 2. Model
| | |
|---|---|
| layers × d_model × heads | 8 × 512 × 8 (head dim 64) |
| context | 256 tokens |
| positions | RoPE (learned positions available as a switch) |
| block | pre-LayerNorm, causal self-attention (SDPA), GELU MLP 4×, residuals |
| embeddings | tied input/output |
| parameters | 29.4M |

## 3. Training
bf16 autocast, AdamW (β = 0.9/0.95, weight decay 0.1 on matrices only), grad clip 1.0,
micro-batch 32 × grad-accum 2 × 256 tokens = 16,384 tokens/step, LR 6e-4 with 500 warmup steps
and cosine decay to 6e-5 at 40,000 steps (655M tokens ≈ 3.7 epochs of the subset).
`torch.compile` raised throughput from 66k to ~88.7k tok/s; peak VRAM 2.24 GB.
Checkpoints include the optimizer, RNG and data state, so resume is exact (tested).

![main run loss](figures/loss.png)

| step | val loss | val ppl |
|---|---|---|
| 651 | 2.219 | 9.2 |
| 14,030 | 1.384 | 4.0 |
| 27,495 | 1.279 | 3.59 |
| 40,000 | 1.229 | 3.42 |

## 4. Samples
See `samples/suite_step40000.md` (6 prompts × 4 decoding settings, fixed seeds). 23/24 samples
end on their own with `<|endoftext|>`; repeated-4-gram fraction 0.010–0.049. Greedy example:

> Once upon a time, there was a little girl named Lily. She had a big, red ball that she loved to
> play with. One day, she went to the park with her mom and took her ball with her. […] After
> playing, Lily and Tim became good friends. They learned that sharing made them happy. The moral
> of the story is to always share with others and be kind.

## 5. What changed results
- **Positions**: RoPE beats learned positions, 1.557 vs 1.587 val loss at 49M tokens (section 5.1).
- **Learning rate**: at 49M tokens, 1.2e-3 > 6e-4 > 3e-4 (1.519 / 1.557 / 1.645). The batch-32k run at equal
  tokens is worse (1.624) (section 5.2).
- `torch.compile`: +31% throughput, the same loss (so more tokens per session).
- KV cache: identical greedy tokens, 1.6× faster at batch 1 and 9–22× at batch 8–32 (`kv_cache.md`).

### 5.1 Positions ablation
Same 8×512 model and 16k-token batch as the main run, 3000 steps (49.2M tokens) each with its own
warmup 300 + cosine schedule (`scripts/run_abl_pos.sh`, Day 6).

![positions ablation](figures/abl_pos.png)

| positions | final val loss | val ppl | tok/s |
|---|---|---|---|
| RoPE | **1.557** | 4.75 | 86.6k* |
| learned | 1.587 | 4.89 | 90.1k |

RoPE is ahead at every eval. The gap is largest early (0.28 nats at step 250, 0.14 at 500) and narrows to
0.030 at the end, so RoPE mostly speeds up early learning. One seed per arm, so seed-to-seed spread is not
measured. But the gap is consistent across all 12 evals (same fixed 40 val windows). *RoPE's tok/s is lower partly because
CPU jobs ran alongside it; the extra rotation costs a few % at most.

### 5.2 LR / batch sweep
Equal 49.2M-token budget, 8×512 model, each run with its own warmup + cosine (to lr/10)
(`scripts/run_sweep.sh`, Day 6). The baseline is the positions-ablation RoPE run.

![LR / batch sweep](figures/sweep.png)

| run | lr | tokens/step | steps | final val loss | ppl |
|---|---|---|---|---|---|
| baseline | 6e-4 | 16k | 3000 | 1.557 | 4.75 |
| lr3e-4 | 3e-4 | 16k | 3000 | 1.645 | 5.18 |
| lr1.2e-3 | 1.2e-3 | 16k | 3000 | **1.519** | 4.57 |
| bs32k | 6e-4 | 32k | 1500 | 1.624 | 5.07 |

- At this short budget, a higher LR is better: going 3e-4 → 6e-4 → 1.2e-3 gives 1.645 → 1.557 → 1.519. It is
  still improving at 1.2e-3, and training stayed stable (val fell at every eval; grad norm ≤ 0.53 after step 500), so the best LR is ≥ 1.2e-3.
- Doubling the batch at the same LR and token count is worse (1.624 vs 1.557): half as many optimizer steps.
  It would likely need a higher LR to catch up. With 49M tokens we are well below the critical batch size.
- Caveat: these are 3000-step runs. The main run (40k steps, lr 6e-4) might also have gained from a higher
  LR, but that wasn't tested at full length. One seed per run.

### 5.3 Instruction fine-tune (PLAN Day 27)
Prompts are made from our own stories ("uses the words …" / "begins with …"), and the loss is on the
story only. Measured with `scripts/eval_instruct.py` on 200 held-out prompts (T=0.8, top-p 0.95):

| model | follows prompt |
|---|---|
| base (step 40,000) | 0.0% (`instruct_eval_base.md`) |
| fine-tuned (1000 steps, lr 1e-4) | 16.5% overall: "begins with" 88.6%, "uses the words" 1.2% (`instruct_eval_ft.md`) |

Partial credit on "uses the words" prompts (share of requested words that appear): base 4.8%, fine-tuned 23.6%.

The fine-tuned model reliably writes a story after `Story:` and copies a given opening sentence. It
rarely uses three requested words: it tends to write a typical story and ignore them. Val loss (story
tokens of held-out rows) fell from 1.227 to 1.103 by step 100, then rose slightly to 1.118 at step 1000,
while train loss fell to 0.92. One epoch is 516 steps, so the rise starts inside the first epoch. The
quick gain is the prompt format, and after that the model fits these 33k stories. Next to try: a few
times more rows at ≤ 1 epoch and fewer steps, and judging "uses the words" by partial credit too.

## 6. Limitations and what's next
- The 256-token context covers most stories; longer generations use a rebuilt cache window.
- Only a 700 MiB subset was used (disk limit), so the main run saw each story ~3.7 times.
