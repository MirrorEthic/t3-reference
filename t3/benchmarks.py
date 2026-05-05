"""Public lm-eval-harness integration for T³.

`T3LM` extends `lm_eval.api.LM` with ACT-aware loglikelihood scoring and
per-stage ponder-depth instrumentation. `run_benchmark_suite` wraps it with
a one-call runner that emits results in the schema consumed by t3atlas.

Example:

    from t3.benchmarks import run_benchmark_suite

    results = run_benchmark_suite(
        checkpoint="path/to/best.pt",
        tasks=("boolq", "arc_easy", "arc_challenge", "piqa",
               "hellaswag", "winogrande", "copa", "rte"),
        batch_size=16,
        out_path="benchmarks/v36_run3_release.json",
    )

Notes:
- Continuations are tokenized with `add_special_tokens=False`. Without this,
  BoolQ pins to ~0.22 because the BOS token gets double-counted in the
  continuation logprob.
- `_last_per_stage_steps` is read from the chain after each forward to
  produce the per-stage ponder-depth distribution that's reported alongside
  task accuracy.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from t3 import T3Model
from t3.tracing import VOCAB_TO_TOKENIZER

DEFAULT_TASKS = (
    "boolq",
    "arc_easy",
    "arc_challenge",
    "piqa",
    "hellaswag",
    "winogrande",
    "copa",
    "rte",
)


# ---------------------------------------------------------------------------
# T3LM — lm-eval-harness adapter
# ---------------------------------------------------------------------------

# `lm_eval` is an optional dependency. We try to inherit from `lm_eval.api.LM`
# when it's installed (so `lm_eval.simple_evaluate(model=lm, ...)` accepts our
# instance); when it's not, we fall back to a stub so that `import t3.benchmarks`
# still succeeds. Calling `T3LM(...)` without lm_eval installed will raise.
try:
    from lm_eval.api.model import LM as _LMBase  # type: ignore[import]

    _HAS_LM_EVAL = True
except ImportError:  # pragma: no cover

    class _LMBase:  # type: ignore[no-redef]
        pass

    _HAS_LM_EVAL = False


def _autodetect_tokenizer(ckpt_meta: dict, cfg) -> str:
    """Pick a tokenizer from the checkpoint's stored name, falling back to the
    vocab-size lookup table used by the trace generator."""
    tok = ckpt_meta.get("tokenizer")
    if tok:
        return tok
    return VOCAB_TO_TOKENIZER.get(cfg.vocab_size, "gpt2")


class T3LM(_LMBase):
    """lm-eval-harness wrapper around `T3Model`. Inherits from `lm_eval.api.LM`
    when `lm_eval` is installed; otherwise inherits from a stub so that the
    module imports cleanly. Constructor raises if `lm_eval` is missing."""

    def __init__(
        self,
        pretrained: str,
        device: str = "cuda",
        batch_size: int = 16,
        max_length: Optional[int] = None,
        eval_live_primitives: bool = False,
        null_cone_strength: float = 0.0,
        tokenizer_override: Optional[str] = None,
    ):
        if not _HAS_LM_EVAL:
            raise ImportError(
                "T3LM requires `lm_eval` (the EleutherAI lm-evaluation-harness). "
                "Install it via `pip install lm-eval`."
            )
        super().__init__()

        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._batch_size = int(batch_size)
        self._max_length_override = int(max_length) if max_length is not None else None

        from transformers import AutoTokenizer

        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        ckpt_meta = {
            "step": ckpt.get("step"),
            "val_ppl": ckpt.get("val_ppl"),
            "tokenizer": ckpt.get("tokenizer"),
            "run_id": ckpt.get("run_id"),
        }

        self._t3 = T3Model.from_checkpoint(pretrained, map_location=self._device)
        self._t3.to(self._device)
        self._t3.eval()
        self.model = self._t3.chain

        cfg = self._t3.config
        self._max_length = self._max_length_override or cfg.max_seq_len

        tok_name = tokenizer_override or _autodetect_tokenizer(ckpt_meta, cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Match ecology_strength / warmup_frac to checkpoint step. Pre-warmup
        # checkpoints (early training) need the ecology bypassed; fully-trained
        # checkpoints get the default (1.0). The released v36-run3 has
        # step=831024 and warmup_steps=200, so this resolves to 1.0.
        ckpt_step = int(ckpt_meta["step"]) if ckpt_meta["step"] is not None else 0
        warmup_steps = max(int(getattr(cfg, "blockade_warmup_steps", 200)), 1)
        eco_frac = min(ckpt_step / warmup_steps, 1.0)
        for m in self.model.modules():
            m._ecology_strength = eco_frac
            m._warmup_frac = eco_frac

        if eval_live_primitives:
            object.__setattr__(cfg, "eval_live_primitives", True)
            for stage in self.model.stages:
                object.__setattr__(stage.cfg, "eval_live_primitives", True)
        if null_cone_strength > 0:
            object.__setattr__(cfg, "null_cone_strength", null_cone_strength)
            for stage in self.model.stages:
                object.__setattr__(stage.head_state.cfg, "null_cone_strength", null_cone_strength)

        # Per-stage ponder depth log: list[list[int]], appended after each forward.
        self._ponder_log: List[List[int]] = []
        self._act_enabled = cfg.act_enabled
        self._ckpt_meta = ckpt_meta

    # -------------------- LM interface (properties) --------------------

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int], **kwargs) -> str:
        return self.tokenizer.decode(tokens)

    # -------------------- LM interface (methods) --------------------

    @torch.no_grad()
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        results: List[Tuple[float, bool]] = []
        for i in range(0, len(requests), self._batch_size):
            results.extend(self._loglikelihood_batch(requests[i : i + self._batch_size]))
        return results

    def _loglikelihood_batch(self, requests) -> List[Tuple[float, bool]]:
        all_full_ids: List[List[int]] = []
        all_cont_lens: List[int] = []
        for req in requests:
            context, continuation = req.args
            ctx_ids = self.tokenizer.encode(context)
            # Critical: continuation must NOT include BOS, or BoolQ pins to ~0.22.
            cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
            full_ids = ctx_ids + cont_ids
            if len(full_ids) > self._max_length:
                full_ids = full_ids[-self._max_length :]
                cont_len = min(len(cont_ids), len(full_ids))
            else:
                cont_len = len(cont_ids)
            all_full_ids.append(full_ids)
            all_cont_lens.append(cont_len)

        max_len = max(len(ids) for ids in all_full_ids)
        pad_id = self.tokenizer.eos_token_id
        padded = [[pad_id] * (max_len - len(ids)) + ids for ids in all_full_ids]
        input_ids = torch.tensor(padded, device=self._device)

        logits = self.model(input_ids[:, :-1])
        if isinstance(logits, tuple):
            logits = logits[0]

        if self._act_enabled and hasattr(self.model, "_last_per_stage_steps"):
            self._ponder_log.append(list(self.model._last_per_stage_steps))

        log_probs = F.log_softmax(logits, dim=-1)

        results: List[Tuple[float, bool]] = []
        for i, (full_ids, cont_len) in enumerate(zip(all_full_ids, all_cont_lens)):
            seq_len = len(full_ids)
            pad_offset = max_len - seq_len
            cont_start = seq_len - cont_len
            total_logprob = 0.0
            is_greedy = True
            for j in range(cont_len):
                pos = pad_offset + cont_start + j - 1
                if pos < 0 or pos >= log_probs.shape[1]:
                    continue
                target_id = full_ids[cont_start + j]
                total_logprob += log_probs[i, pos, target_id].item()
                if log_probs[i, pos].argmax().item() != target_id:
                    is_greedy = False
            results.append((total_logprob, is_greedy))
        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests) -> List[float]:
        results: List[float] = []
        for req in requests:
            (text,) = req.args
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) == 0:
                results.append(0.0)
                continue
            total_logprob = 0.0
            for start in range(0, len(token_ids), self._max_length):
                chunk = token_ids[start : start + self._max_length]
                if len(chunk) < 2:
                    continue
                input_ids = torch.tensor([chunk], device=self._device)
                logits = self.model(input_ids[:, :-1])
                if isinstance(logits, tuple):
                    logits = logits[0]
                if self._act_enabled and hasattr(self.model, "_last_per_stage_steps"):
                    self._ponder_log.append(list(self.model._last_per_stage_steps))
                log_probs = F.log_softmax(logits, dim=-1)
                for j in range(log_probs.shape[1]):
                    target = chunk[j + 1]
                    total_logprob += log_probs[0, j, target].item()
            results.append(total_logprob)
        return results

    @torch.no_grad()
    def generate_until(self, requests) -> List[str]:
        results: List[str] = []
        for req in requests:
            context, gen_kwargs = req.args
            stop = gen_kwargs.get("until", [])
            if isinstance(stop, str):
                stop = [stop]
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)

            token_ids = self.tokenizer.encode(context)
            if len(token_ids) > self._max_length - max_gen:
                token_ids = token_ids[-(self._max_length - max_gen) :]

            input_ids = torch.tensor([token_ids], device=self._device)
            for _ in range(max_gen):
                if input_ids.shape[1] >= self._max_length:
                    break
                logits = self.model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_logits = logits[0, -1, :]
                if temperature > 0:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = next_logits.argmax().unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                generated_text = self.tokenizer.decode(
                    input_ids[0, len(token_ids) :].tolist()
                )
                if any(s in generated_text for s in stop):
                    break
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            generated = self.tokenizer.decode(input_ids[0, len(token_ids) :].tolist())
            for s in stop:
                if s in generated:
                    generated = generated[: generated.index(s)]
            results.append(generated)
        return results


# ---------------------------------------------------------------------------
# run_benchmark_suite — one-call runner with results JSON
# ---------------------------------------------------------------------------


def _checkpoint_metadata(ckpt_path) -> dict:
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) if isinstance(ck.get("config"), dict) else {}
    return {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_dir": Path(ckpt_path).parent.name,
        "checkpoint_file": Path(ckpt_path).name,
        "step": ck.get("step"),
        "val_ppl": ck.get("val_ppl"),
        "source_model": ck.get("source_model", cfg.get("source")),
        "run_id": ck.get("run_id"),
        "tokenizer": ck.get("tokenizer"),
        "d_model": cfg.get("d_model"),
        "n_heads": cfg.get("n_heads"),
        "n_stages": cfg.get("n_stages"),
        "layers_per_stage": cfg.get("layers_per_stage"),
        "vocab_size": cfg.get("vocab_size"),
        "act_enabled": cfg.get("act_enabled"),
        "eco_key_bias": cfg.get("eco_key_bias"),
    }


def run_benchmark_suite(
    checkpoint,
    tasks: Iterable[str] = DEFAULT_TASKS,
    batch_size: int = 16,
    num_fewshot: int = 0,
    limit: Optional[int] = None,
    out_path=None,
    device: str = "cuda",
    eval_live_primitives: bool = False,
    null_cone_strength: float = 0.0,
    verbose: bool = True,
) -> dict:
    """Run lm-eval-harness on a T³ checkpoint, returning aggregated results.

    If `out_path` is given, also writes a JSON file in the schema consumed by
    t3atlas. Per-task per-stage ponder-depth statistics are included.
    """
    import lm_eval

    run_meta = _checkpoint_metadata(checkpoint)
    if verbose:
        print(f"Checkpoint: {checkpoint}")
        print(
            f"  run_id={run_meta['run_id']} step={run_meta['step']} "
            f"val_ppl={run_meta['val_ppl']}"
        )
        print(f"Tasks: {list(tasks)}")
        print(
            f"Batch size: {batch_size}  Few-shot: {num_fewshot}"
            + (f"  Limit: {limit}" if limit else "")
        )

    lm = T3LM(
        pretrained=str(checkpoint),
        device=device,
        batch_size=batch_size,
        eval_live_primitives=eval_live_primitives,
        null_cone_strength=null_cone_strength,
    )

    all_results: dict = {}
    ponder_by_task: dict = {}

    for task in tasks:
        lm._ponder_log.clear()
        if verbose:
            print(f"\n--- {task} ---")
        result = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=num_fewshot,
            limit=limit,
        )
        all_results.update(result.get("results", {}))
        if lm._ponder_log:
            arr = torch.tensor(lm._ponder_log, dtype=torch.float)
            ponder_by_task[task] = {
                "n_samples": int(arr.shape[0]),
                "per_stage_mean": arr.mean(0).tolist(),
                "per_stage_std": arr.std(0).tolist(),
                "total_mean": float(arr.sum(1).mean()),
            }

    out = {
        "schema": "t3atlas-bench-v1",
        "run_meta": run_meta,
        "config": {
            "tasks": list(tasks),
            "batch_size": batch_size,
            "num_fewshot": num_fewshot,
            "limit": limit,
            "eval_live_primitives": eval_live_primitives,
            "null_cone_strength": null_cone_strength,
            "device": device,
        },
        "created": datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
        "results": all_results,
        "ponder_by_task": ponder_by_task,
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        if verbose:
            print(f"\nwrote {out_path}")

    return out


__all__ = ["T3LM", "run_benchmark_suite", "DEFAULT_TASKS"]
