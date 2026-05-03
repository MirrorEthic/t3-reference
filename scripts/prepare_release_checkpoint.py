"""Strip training-only state from a T³ checkpoint for HF release.

Drops:
    optimizer_state    (Adam moments — ~1GB of training scaffolding)
    data_loader_state  (which shard each corpus was at — irrelevant to inference)

Keeps everything T3Model.from_checkpoint needs:
    model_state, ecology_state, config, source_model, run_id, val_ppl, step,
    training_hparams (informational, tiny), mix_weights (informational, tiny).

Usage:
    python scripts/prepare_release_checkpoint.py \\
        --in /path/to/best.pt \\
        --out release/pytorch_model.bin
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

DROP_KEYS = {"optimizer_state", "data_loader_state"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = Path(args.in_path)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {src} ...")
    ck = torch.load(str(src), map_location="cpu", weights_only=False)

    in_size = os.path.getsize(src) / 1e6
    print(f"  full size: {in_size:.1f} MB")
    print(f"  keys before strip: {sorted(ck.keys())}")

    for k in DROP_KEYS:
        if k in ck:
            print(f"  dropping {k}")
            del ck[k]

    print(f"  keys after strip:  {sorted(ck.keys())}")
    print(f"saving {dst} ...")
    torch.save(ck, str(dst))

    out_size = os.path.getsize(dst) / 1e6
    print(f"  release size: {out_size:.1f} MB ({100 * (1 - out_size / in_size):.1f}% smaller)")


if __name__ == "__main__":
    main()
