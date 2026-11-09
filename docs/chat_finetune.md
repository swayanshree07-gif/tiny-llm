# Chat fine-tune (PLAN Days 21–22), 2026-11-09

## Template (`tinyllm/chat.py`)
Plain text on the existing 8192 vocab, no new tokens:

    User: <message>
    Assistant: <reply><|endoftext|>
    User: <message>
    Assistant: <reply><|endoftext|>

Loss only on assistant reply tokens and the EOT that closes each reply (everything else is -100).
Tested in `tests/test_chat.py` (template text, round trip, mask = exactly the reply tokens, row targets,
turn truncation).

## Data (`scripts/build_chat.py` -> `data/chat`, 64 MB)
Built from our own stories: "uses the word(s)" requests (1–3 words, 3 phrasings), "begins with"/"continue this"
requests, free "tell me a story" requests, and fixed small-talk turns. 1–3 turns per conversation, packed into one
256-token row; turns that don't fit are dropped from the end.

| split | rows | turns/row 1/2/3 | words / begins / free / smalltalk turns | target tokens |
|---|---|---|---|---|
| train | 34,212 | 31,618 / 2,454 / 140 | 18,023 / 9,721 / 5,141 / 4,061 | 64.2% |
| valid | 561 | 519 / 40 / 2 | 312 / 146 / 79 / 68 | 64.1% |

Multi-turn rows are rare (7.6%) because one story already fills most of the 256-token context.

## Fine-tunes of `ckpt/large` (91M, step 18000)
Eval: `eval_instruct.py --chat` on the same 200 fixed prompts (`data/instruct_anyfix`) used for v4, with each
request wrapped in the chat template.

| model | steps, lr | val loss (chat valid) | follows all | begins | words | words present |
|---|---|---|---|---|---|---|
| large, no fine-tune (instruct format) | – | 1.274 | 0.0% | 0.0% | 0.0% | 6.3% |
| chat_large | 1000, 5e-5 | 1.070 (min 1.064 @100) | 15.5% | 80.0% | 1.8% | 20.2% |
| chat_large_v2 (+word tokens ×5) | 600, 5e-5 | 1.065 | 14.0% | 65.7% | 3.0% | 26.5% |
| instruct_v4 (29M, reference) | 2000 | – | 25.0% | ~88% | 11.5% | 52% |

Result: chat formatting works (replies end with EOT 194–196/200), but the Day 22 check (≥ 25%) is **not met yet**.
Val loss plateaus by step ~100, so more steps of the same data don't help. Likely causes: only ~half the word
requests use 3 words (eval always asks for 3), words come half from "any" content words, and v4 trained 2000 steps on
132k rows. Next attempt: always-3-word requests with salient words, more rows, weight ×5, compare.

## v3: always-3-word requests (2026-11-09, Day 20 session)
`build_chat.py --mode three --out data/chat_v3`: 70% word requests, always 3 words (salient words when the story has
≥ 3, else any content words, 70/30), 22% begins, rest free; 33,778 train / 585 valid rows. `configs/chat_large_v3.json`:
1000 steps from `ckpt/large`, word tokens ×5, `save_optimizer: false` (weights-only checkpoint, 0.36 GB instead of
1.1 GB, because the disk was at 11.57/12 GB and deleting the v2 checkpoint was not allowed in this session).

| model | val loss (own valid) | follows all | begins | words | words present |
|---|---|---|---|---|---|
| chat_large_v3 | 1.311 → 1.108 (flat from step 100) | **20.5%** | 68.6% | 10.3% | 42.6% |

Best chat model so far (v1 15.5%, v2 14.0%), still below v4's 25%: the Day 22 check is **not met**. Decision: accept
v3 as the chat model and move on to `chat.py` (Day 23). Ideas left: more word-request rows (v4 had 132k), a lower
begins share, training v4-style data on the large model.
