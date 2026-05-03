"""Load the published T³ v3.6 checkpoint and run a single forward pass.

End-to-end smoke test for the reference implementation. After upload to HF,
the `hf_hub_download` line below works without modification. While developing
locally, set the T3_LOCAL_CKPT environment variable to a `best.pt` path.
"""

from __future__ import annotations

import os

import torch

from t3 import T3Model


def main() -> None:
    local = os.environ.get("T3_LOCAL_CKPT")
    if local:
        ckpt = local
    else:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download("mirrorethic/t3-124m-v36", "pytorch_model.bin")

    model = T3Model.from_checkpoint(ckpt)
    model.eval()
    rep = model.get_load_report()
    print(f"loaded run_id={rep['run_id']} step={rep['checkpoint_step']} "
          f"val_ppl={rep['checkpoint_val_ppl']:.4f}")
    print(f"missing={len(rep['missing'])}  unexpected={len(rep['unexpected'])}")

    torch.manual_seed(0)
    input_ids = torch.randint(0, 50257, (1, 16))
    with torch.no_grad():
        out = model(input_ids)
    logits = out[0] if isinstance(out, tuple) else out
    print(f"logits shape: {tuple(logits.shape)}  top-1 last-token: {int(logits[0, -1].argmax())}")


if __name__ == "__main__":
    main()
