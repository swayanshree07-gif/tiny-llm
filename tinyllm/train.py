"""Training loop: bf16 autocast, gradient accumulation, grad clipping, AdamW decay groups,
warmup + cosine LR, periodic validation (loss / perplexity), JSONL metrics,
checkpointing (keep latest N) and exact resume.

Config is a JSON file of the form {"model": {...GPTConfig...}, "train": {...TrainConfig...}};
see configs/*.json and scripts/train.py.
"""
from __future__ import annotations

import contextlib
import json
import math
import signal
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from .data import TokenDataset
from .instruct import InstructDataset
from .device import best_autocast_dtype, get_device
from .model import GPT, GPTConfig
from .optim import lr_at, make_optimizer


@dataclass
class TrainConfig:
    data_dir: str = "data/tok"
    out_dir: str = "ckpt/run"
    batch_size: int = 16          # micro-batch (sequences per forward)
    grad_accum: int = 1           # micro-batches per optimizer step
    max_steps: int = 1000         # length of the LR schedule (optimizer steps)
    stop_at_step: int | None = None  # stop early (schedule still uses max_steps); for sessions/tests
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 20
    eval_batch_size: int | None = None  # default: batch_size
    eval_seed: int = 1234         # fixed val windows -> comparable val loss across evals/sessions
    log_interval: int = 10
    ckpt_interval: int = 500
    keep_ckpts: int = 2
    seed: int = 1337
    device: str = "cuda"
    dtype: str = "auto"           # "auto" (bf16 on GPU), "bf16", "fp32"
    compile: bool = False
    max_minutes: float | None = None  # wall-clock budget; saves a checkpoint and stops cleanly
    resume: bool = True           # continue from the latest checkpoint in out_dir if present
    overfit_one_batch: bool = False
    train_file: str = "train.bin"
    val_file: str = "valid.bin"
    data_kind: str = "tokens"     # "tokens" (uint16 .bin windows) or "instruct" (.npz rows with masked targets)
    init_from: str | None = None  # fine-tune: load model weights (only) from this ckpt file/dir on a fresh start
    train_mask_file: str | None = None  # instruct: .npz with bool "m" (rows x T) of positions to up-weight
    mask_loss_weight: float = 1.0       # loss weight of masked positions (others 1); weighted mean over targets


def config_from_dict(d: dict) -> tuple[GPTConfig, TrainConfig]:
    def build(cls, sub):
        known = {f.name for f in fields(cls)}
        unknown = set(sub) - known
        if unknown:
            raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
        return cls(**sub)
    return build(GPTConfig, d.get("model", {})), build(TrainConfig, d.get("train", {}))


def autocast_ctx(device: torch.device, dtype_name: str):
    if dtype_name == "fp32":
        return contextlib.nullcontext()
    dt = torch.bfloat16 if dtype_name == "bf16" else best_autocast_dtype(device)
    if device.type != "cuda" or dt == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dt)


def accumulate_grads(model: torch.nn.Module, batches, ctx_factory) -> float:
    """Forward/backward over micro-batches, each loss scaled by 1/k so grads equal one big batch.

    Returns the mean micro-batch loss (python float). Equal-size micro-batches assumed.
    """
    k = len(batches)
    total = 0.0
    for batch in batches:
        x, y = batch[0], batch[1]
        with ctx_factory():
            if len(batch) == 3:  # per-position weights: weighted mean over non-ignored targets
                logits, _ = model(x)
                ce = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none")
                w = batch[2].flatten() * (y.flatten() != -100)
                loss = (ce * w).sum() / w.sum().clamp_min(1e-8)
            else:
                _, loss = model(x, y)
        (loss / k).backward()
        total += loss.detach().float()
    return float(total / k)


@torch.no_grad()
def evaluate(model: torch.nn.Module, ds: TokenDataset, iters: int, batch_size: int,
             device: torch.device, ctx_factory, seed: int) -> float:
    """Mean loss over `iters` fixed (seeded) random val windows. Restores ds RNG afterwards."""
    was_training = model.training
    model.eval()
    saved = ds.state_dict()
    ds.rng = np.random.default_rng(seed)
    losses = []
    for _ in range(iters):
        x, y = ds.get_batch(batch_size, device)[:2]  # val loss is always unweighted
        with ctx_factory():
            _, loss = model(x, y)
        losses.append(loss.float().item())
    ds.load_state_dict(saved)
    model.train(was_training)
    return sum(losses) / len(losses)


class _StopFlag:
    """Set on SIGTERM/SIGINT (e.g. from `timeout`) so the loop saves a checkpoint and exits."""

    def __init__(self) -> None:
        self.stop = False
        self._old = {}

    def __enter__(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._old[sig] = signal.signal(sig, self._handler)
            except ValueError:  # not in main thread
                pass
        return self

    def _handler(self, signum, frame):
        _say(f"[train] got signal {signum}, stopping after this step")
        self.stop = True

    def __exit__(self, *exc):
        for sig, h in self._old.items():
            signal.signal(sig, h)


def _say(msg: str) -> None:
    """print() that never raises: if stdout is a closed pipe (e.g. `| tail` killed by `timeout`),
    training must still reach its final checkpoint."""
    try:
        print(msg, flush=True)
    except (BrokenPipeError, OSError, ValueError):
        pass


def _log(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def train(mcfg: GPTConfig, tcfg: TrainConfig, verbose: bool = True) -> dict:
    """Run (or resume) training. Returns a summary dict (final step, last losses, history)."""
    device = get_device(tcfg.device)
    out = Path(tcfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    torch.manual_seed(tcfg.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = Path(tcfg.data_dir)
    if tcfg.data_kind == "instruct":
        def load_rows(name: str, seed: int, mask_file: str | None = None) -> InstructDataset:
            z = np.load(data_dir / name)
            w = None
            if mask_file is not None:
                m = np.load(data_dir / mask_file)["m"]
                w = np.where(m, tcfg.mask_loss_weight, 1.0).astype(np.float32)
            ds = InstructDataset(z["x"].astype(np.int64), z["y"].astype(np.int64), seed=seed, w=w)
            assert ds.block_size == mcfg.block_size, "instruct rows must be block_size long"
            return ds
        train_ds = load_rows(tcfg.train_file, tcfg.seed, tcfg.train_mask_file)
        val_ds = load_rows(tcfg.val_file, tcfg.eval_seed)
    elif tcfg.data_kind == "tokens":
        train_ds = TokenDataset(data_dir / tcfg.train_file, mcfg.block_size, seed=tcfg.seed)
        val_ds = TokenDataset(data_dir / tcfg.val_file, mcfg.block_size, seed=tcfg.eval_seed)
    else:
        raise ValueError(f"unknown data_kind {tcfg.data_kind!r}")

    model = GPT(mcfg).to(device)
    opt = make_optimizer(model, tcfg.lr, tcfg.weight_decay, (tcfg.beta1, tcfg.beta2))
    step = 0
    ck = latest_checkpoint(out) if tcfg.resume else None
    if ck is not None:
        st = load_checkpoint(ck)
        if st["model_config"] != mcfg.to_dict():
            raise ValueError(f"checkpoint {ck} model config differs from requested config")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        train_ds.load_state_dict(st["train_data"])
        torch.set_rng_state(st["rng_cpu"])
        if device.type == "cuda" and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(st["rng_cuda"])
        step = st["step"]
        if verbose:
            _say(f"[train] resumed from {ck} at step {step}")
        # drop log records written after the checkpoint (they will be re-done)
        if metrics_path.exists():
            keep = [ln for ln in metrics_path.read_text().splitlines()
                    if ln.strip() and json.loads(ln)["step"] <= step]
            metrics_path.write_text("".join(ln + "\n" for ln in keep))
    else:
        metrics_path.write_text("")  # fresh run: start a new log
        if tcfg.init_from:
            src = Path(tcfg.init_from)
            src = latest_checkpoint(src) if src.is_dir() else src
            if src is None:
                raise FileNotFoundError(f"init_from {tcfg.init_from}: no checkpoint")
            st = load_checkpoint(src)
            if st["model_config"] != mcfg.to_dict():
                raise ValueError(f"init_from {src} model config differs from requested config")
            model.load_state_dict(st["model"])
            if verbose:
                _say(f"[train] initialised weights from {src} (step {st['step']}); fresh optimizer")

    fwd_model = torch.compile(model) if tcfg.compile else model
    ctx = lambda: autocast_ctx(device, tcfg.dtype)  # noqa: E731
    eval_bs = tcfg.eval_batch_size or tcfg.batch_size
    tokens_per_step = tcfg.batch_size * tcfg.grad_accum * mcfg.block_size
    end = min(tcfg.max_steps, tcfg.stop_at_step) if tcfg.stop_at_step is not None else tcfg.max_steps

    def do_eval() -> float:
        vl = evaluate(fwd_model, val_ds, tcfg.eval_iters, eval_bs, device, ctx, tcfg.eval_seed)
        rec = {"type": "val", "step": step, "loss": vl, "ppl": math.exp(min(vl, 50))}
        _log(metrics_path, rec)
        history.append(rec)
        if verbose:
            _say(f"[val] step {step}: loss {vl:.4f} ppl {rec['ppl']:.2f}")
        return vl

    def do_save() -> Path:
        state = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "model_config": mcfg.to_dict(),
            "train_config": asdict(tcfg),
            "train_data": train_ds.state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None,
        }
        p = save_checkpoint(state, out, step, keep=tcfg.keep_ckpts)
        if verbose:
            _say(f"[ckpt] saved {p}")
        return p

    history: list[dict] = []
    fixed_batch = None
    if verbose:
        _say(f"[train] {model.num_params()/1e6:.2f}M params, device {device}, "
              f"{tokens_per_step} tokens/step, steps {step}->{end} (schedule {tcfg.max_steps})")
    if step == 0 and tcfg.eval_interval > 0 and not tcfg.overfit_one_batch:
        do_eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    t0 = time.time()
    t_last, steps_since = time.time(), 0
    last_loss = float("nan")
    saved_at = -1
    with _StopFlag() as flag:
        while step < end:
            lr = lr_at(step, tcfg.lr, tcfg.min_lr, tcfg.warmup, tcfg.max_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            if tcfg.overfit_one_batch:
                if fixed_batch is None:
                    fixed_batch = [train_ds.get_batch(tcfg.batch_size, device) for _ in range(tcfg.grad_accum)]
                batches = fixed_batch
            else:
                batches = [train_ds.get_batch(tcfg.batch_size, device) for _ in range(tcfg.grad_accum)]
            last_loss = accumulate_grads(fwd_model, batches, ctx)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   tcfg.grad_clip if tcfg.grad_clip > 0 else float("inf"))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            steps_since += 1

            if step % tcfg.log_interval == 0 or step == end:
                dt = time.time() - t_last
                rec = {"type": "train", "step": step, "loss": last_loss, "lr": lr,
                       "grad_norm": float(gnorm), "tok_per_s": tokens_per_step * steps_since / max(dt, 1e-9),
                       "elapsed_s": time.time() - t0}
                if device.type == "cuda":
                    rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3
                _log(metrics_path, rec)
                history.append(rec)
                if verbose:
                    mem = f" mem {rec['peak_mem_gb']:.2f}G" if "peak_mem_gb" in rec else ""
                    _say(f"step {step}: loss {last_loss:.4f} lr {lr:.2e} gnorm {rec['grad_norm']:.2f} "
                          f"{rec['tok_per_s']/1e3:.1f}k tok/s{mem}")
                t_last, steps_since = time.time(), 0

            if tcfg.eval_interval > 0 and step % tcfg.eval_interval == 0 and not tcfg.overfit_one_batch:
                do_eval()
            if tcfg.ckpt_interval > 0 and step % tcfg.ckpt_interval == 0:
                do_save()
                saved_at = step
            out_of_time = tcfg.max_minutes is not None and (time.time() - t0) / 60 >= tcfg.max_minutes
            if flag.stop or out_of_time:
                if verbose:
                    _say(f"[train] stopping early at step {step} "
                          f"({'signal' if flag.stop else 'time budget'})")
                break

    if tcfg.eval_interval > 0 and not tcfg.overfit_one_batch and (not history or history[-1]["type"] != "val"):
        do_eval()
    if tcfg.ckpt_interval > 0 and saved_at != step:
        do_save()
    return {"step": step, "last_train_loss": last_loss, "history": history,
            "last_val_loss": next((h["loss"] for h in reversed(history) if h["type"] == "val"), None)}
