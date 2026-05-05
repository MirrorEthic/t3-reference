"""Generate the canonical trace library for a T³ checkpoint.

Loops over the bundled prompt library (t3/data/prompts.json), generates one
trace JSONL per prompt, and updates the trace manifest (traces/index.json)
that the t3atlas viewer reads.

Usage:
    T3_LOCAL_CKPT=path/to/best.pt python scripts/sweep_traces.py \
        --lineage t3-124m-v36 \
        --traces-dir /path/to/diagnostic_trace/traces \
        --n-tokens 32
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from t3 import T3Model
from t3.tracing import builtin_prompts, generate_trace, update_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("T3_LOCAL_CKPT"))
    ap.add_argument("--lineage", default="t3-124m-v36",
                    help="Lineage tag used in trace meta + filename")
    ap.add_argument("--ckpt-name", default="best.pt",
                    help="Checkpoint filename to record in trace meta")
    ap.add_argument("--traces-dir", default="traces",
                    help="Output directory for JSONL files + index.json")
    ap.add_argument("--n-tokens", type=int, default=32,
                    help="Tokens to autoregressively generate per trace")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N prompts (smoke testing)")
    args = ap.parse_args()

    if not args.checkpoint:
        raise SystemExit("Pass --checkpoint or set T3_LOCAL_CKPT")

    traces_dir = Path(args.traces_dir).resolve()
    traces_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.checkpoint} on {args.device}...")
    model = T3Model.from_checkpoint(args.checkpoint, map_location=args.device)
    model.to(args.device)
    model.eval()
    rep = model.get_load_report()
    print(f"  step={rep['checkpoint_step']}  val_ppl={rep['checkpoint_val_ppl']:.4f}\n")

    prompts = builtin_prompts()
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"sweeping {len(prompts)} prompts → {traces_dir}\n")

    summary = []
    t_total = time.time()
    for i, p in enumerate(prompts, 1):
        pid = p.get("id") or p.get("prompt_id") or f"prompt{i}"
        text = p.get("prompt") or p.get("text") or ""
        out_path = traces_dir / f"{args.lineage}__{Path(args.ckpt_name).stem}__{pid}.jsonl"
        t0 = time.time()
        out = generate_trace(
            model,
            prompt=text,
            prompt_id=pid,
            n_tokens=args.n_tokens,
            temperature=args.temperature,
            lineage=args.lineage,
            ckpt_filename=args.ckpt_name,
            out_path=out_path,
        )
        meta = json.loads(out.open().readline())
        update_manifest(out, meta, traces_dir=traces_dir)
        size_kb = out.stat().st_size / 1024
        dt = time.time() - t0
        print(f"  [{i:2d}/{len(prompts)}] {pid:14s} {dt:5.1f}s  {size_kb:6.1f} KB  "
              f"frames={meta.get('n_frames')}  chain_states={meta.get('n_chain_states')}")
        summary.append((pid, dt, size_kb, meta.get("n_frames")))

    print(f"\nsweep complete in {time.time() - t_total:.1f}s")
    total_kb = sum(s[2] for s in summary)
    print(f"  {len(summary)} traces, {total_kb / 1024:.1f} MB total")
    print(f"  manifest: {traces_dir / 'index.json'}")


if __name__ == "__main__":
    main()
