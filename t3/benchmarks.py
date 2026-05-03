"""Public lm-eval-harness integration for T³.

Wraps t3._legacy_lm_eval.T3v3LM (which extends lm_eval.api.LM) with a
re-exported, documented public class and a one-call benchmark runner.

Example:

    from t3.lm_eval import T3LM, run_benchmark_suite

    results = run_benchmark_suite(
        checkpoint="path/to/best.pt",
        tasks=("boolq", "arc_easy", "arc_challenge", "piqa",
               "hellaswag", "winogrande", "copa", "rte"),
        batch_size=16,
        out_path="benchmarks/v36_run3_release.json",
    )

The eval_bos_continuation_bug fix (add_special_tokens=False on the
continuation) is baked into the underlying T3v3LM implementation. Without
it, BoolQ pins to ~0.22.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch

from t3._legacy_lm_eval import T3v3LM as T3LM


def _checkpoint_metadata(ckpt_path: str | Path) -> dict:
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
        "mix_weights": ck.get("mix_weights"),
        "act_enabled": cfg.get("act_enabled"),
        "eco_key_bias": cfg.get("eco_key_bias"),
    }


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


def run_benchmark_suite(
    checkpoint: str | Path,
    tasks: Iterable[str] = DEFAULT_TASKS,
    batch_size: int = 16,
    num_fewshot: int = 0,
    limit: int | None = None,
    out_path: str | Path | None = None,
    device: str = "cuda",
    eval_live_primitives: bool = False,
    null_cone_strength: float = 0.0,
    verbose: bool = True,
) -> dict:
    """Run lm-eval-harness on a T³ checkpoint, returning the aggregated results.

    If `out_path` is given, also writes the results JSON in the schema
    consumed by t3atlas/build_benchmarks.py.
    """
    import lm_eval

    run_meta = _checkpoint_metadata(checkpoint)
    if verbose:
        print(f"Checkpoint: {checkpoint}")
        print(f"  run_id={run_meta['run_id']} step={run_meta['step']} "
              f"val_ppl={run_meta['val_ppl']}")
        print(f"Tasks: {list(tasks)}")
        print(f"Batch size: {batch_size}  Few-shot: {num_fewshot}"
              + (f"  Limit: {limit}" if limit else ""))

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
