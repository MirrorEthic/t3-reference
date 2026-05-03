"""Generate a JSONL trace conforming to schema v1 (see docs/TRACE_SCHEMA.md).

Usage:
    T3_LOCAL_CKPT=path/to/best.pt python examples/generate_trace.py "The capital of France is" factual
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from t3 import T3Model
from t3.tracing import generate_trace


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    prompt_id = sys.argv[2] if len(sys.argv) > 2 else "factual"
    n_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    ckpt = os.environ.get("T3_LOCAL_CKPT")
    if not ckpt:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download("mirrorethic/t3-124m-v36", "pytorch_model.bin")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = T3Model.from_checkpoint(ckpt, map_location=device)
    model.to(device)
    model.eval()

    out = generate_trace(
        model,
        prompt=prompt,
        prompt_id=prompt_id,
        n_tokens=n_tokens,
        out_path=Path("traces") / f"t3-124m-v36__best__{prompt_id}.jsonl",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
