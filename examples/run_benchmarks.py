"""Run the standard 8-task lm-eval-harness suite on a T³ checkpoint.

Default tasks match the t3atlas.dev/benchmarks headline set.

Usage:
    T3_LOCAL_CKPT=path/to/best.pt python examples/run_benchmarks.py
    T3_LOCAL_CKPT=path/to/best.pt python examples/run_benchmarks.py --limit 50
    python examples/run_benchmarks.py --checkpoint other.pt --tasks boolq,piqa
"""

from __future__ import annotations

import argparse
import os

from t3.benchmarks import DEFAULT_TASKS, run_benchmark_suite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("T3_LOCAL_CKPT"))
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap items per task (smoke testing)")
    ap.add_argument("--out", default="benchmarks/run_latest.json")
    args = ap.parse_args()

    if not args.checkpoint:
        raise SystemExit("Pass --checkpoint or set T3_LOCAL_CKPT")

    run_benchmark_suite(
        checkpoint=args.checkpoint,
        tasks=tuple(t.strip() for t in args.tasks.split(",") if t.strip()),
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
