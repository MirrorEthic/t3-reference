"""Reproduce val PPL on WikiText-103 — the canonical sanity check.

Mirrors the exact validation loop used by the v3.6 training script
(t3v3_continue_gpt2_act.py:validate). If this reports val_ppl close to the
checkpoint's recorded value (27.7592 for run-3), the vendored chain matches
the training-time forward pass. Used as the regression net before refactoring
the legacy modules into clean public ones.

Usage:
    T3_LOCAL_CKPT=/path/to/best.pt python examples/validate_wikitext.py
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from t3 import T3Model

SEQ = 1024
N_TOKENS = 200_000
MAX_WINDOWS = 30
BATCH_PER_WINDOW = 4


def load_val_tokens() -> list[int]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading GPT-2 tokenizer + WikiText-103 validation split...")
    tok = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                      split="validation", streaming=True)
    val_tokens: list[int] = []
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        val_tokens.extend(tok.encode(text, add_special_tokens=False))
        if len(val_tokens) >= N_TOKENS:
            break
    return val_tokens[:N_TOKENS]


@torch.no_grad()
def validate(model: T3Model, val_tokens: list[int], device: torch.device,
             use_bf16: bool) -> float:
    model.eval()
    chain = model.chain
    losses: list[float] = []
    autocast_kwargs = dict(device_type="cuda", dtype=torch.bfloat16) if use_bf16 \
        else dict(device_type=device.type, dtype=torch.float32, enabled=False)

    for i in range(0, len(val_tokens) - SEQ - 1, SEQ):
        batch_losses = []
        n_sub = min(BATCH_PER_WINDOW, (len(val_tokens) - i) // (SEQ + 1))
        for b in range(n_sub):
            start = i + b * (SEQ + 1)
            if start + SEQ + 1 > len(val_tokens):
                break
            chunk = torch.tensor(val_tokens[start:start + SEQ + 1],
                                 dtype=torch.long, device=device).unsqueeze(0)
            inp, tgt = chunk[:, :-1], chunk[:, 1:]
            with torch.amp.autocast(**autocast_kwargs):
                logits = chain(inp, update_state=False)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                tgt.reshape(-1),
            )
            batch_losses.append(loss.item())
        if batch_losses:
            losses.append(float(np.mean(batch_losses)))
        if len(losses) >= MAX_WINDOWS:
            break
        if len(losses) % 5 == 0 and losses:
            running = math.exp(min(float(np.mean(losses)), 20))
            print(f"  window {len(losses):2d}/{MAX_WINDOWS}  running ppl={running:.3f}")

    if not losses:
        return float("inf")
    return math.exp(min(float(np.mean(losses)), 20))


def main() -> None:
    ckpt_path = os.environ.get("T3_LOCAL_CKPT")
    if not ckpt_path:
        raise SystemExit("Set T3_LOCAL_CKPT to a best.pt path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8

    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    print(f"AMP: {'BF16' if use_bf16 else 'FP32'}\n")

    print(f"Loading model from {ckpt_path}...")
    t0 = time.time()
    model = T3Model.from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    rep = model.get_load_report()
    print(f"  loaded run_id={rep['run_id']} step={rep['checkpoint_step']} "
          f"recorded val_ppl={rep['checkpoint_val_ppl']:.4f}")
    print(f"  load took {time.time() - t0:.1f}s\n")

    val_tokens = load_val_tokens()
    print(f"  {len(val_tokens):,} validation tokens\n")

    print("Running validation...")
    t0 = time.time()
    ppl = validate(model, val_tokens, device, use_bf16)
    dt = time.time() - t0

    print(f"\n=== RESULT ===")
    print(f"  recorded val_ppl: {rep['checkpoint_val_ppl']:.4f}")
    print(f"  reproduced val_ppl: {ppl:.4f}")
    delta = ppl - rep["checkpoint_val_ppl"]
    pct = 100 * delta / rep["checkpoint_val_ppl"]
    print(f"  delta: {delta:+.4f} ({pct:+.2f}%)")
    print(f"  validation time: {dt:.1f}s")
    if abs(pct) < 1.0:
        print(f"  PASS: within 1% of recorded — vendored chain matches training-time forward.")
    elif abs(pct) < 5.0:
        print(f"  WARN: within 5% — likely numerical (autocast / RNG); still consistent.")
    else:
        print(f"  FAIL: divergence > 5% — the vendored chain has a behavior gap.")


if __name__ == "__main__":
    main()
