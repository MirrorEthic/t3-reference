# T³ Reference Implementation

Inference-time reference for **T³**, a Clifford-algebra-augmented transformer
architecture. This repository contains the architecture definition and trace
generation code; training code is held separately.

## What this is

- Architecture definition matching the T³ specification
- Inference code that loads published checkpoints
- Trace generator producing JSON matching the public trace library schema
- Examples and tests for verification

## What this is not

- A training framework (training infrastructure remains private)
- A complete reproduction kit (the empirical training methodology is documented
  in the accompanying paper)

## Quick start

```bash
pip install t3-reference
```

```python
from t3 import T3Model
from t3.tracing import generate_trace, load_trace
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download("mirrorethic/t3-124m-v36", "pytorch_model.bin")
model = T3Model.from_checkpoint(ckpt)

# Forward pass — delegates to T3Chain with per-stage ACT.
import torch
ids = torch.tensor([[464, 3139, 286, 4881, 318]])  # "The capital of France is"
logits = model(ids)

# Trace generation — JSONL conforming to docs/TRACE_SCHEMA.md.
trace_path = generate_trace(
    model,
    prompt="The capital of France is",
    prompt_id="factual",
    n_tokens=8,
    out_path="traces/smoke.jsonl",
)
trace = load_trace(trace_path)
print(f"frames={len(trace['frames'])} chain_states={len(trace['chain_states'])}")
```

## Module layout

- `t3.ecology` — `HeadState` (six primitives + Cl(3,3) coupling), `Blockade`, `Cosurvival`
- `t3.attention` — `EcologyAttention` (σ-modulated MHA + blockade)
- `t3.chain` — `T3Layer`, `T3Stage`, `T3Chain` (per-stage ACT lives here)
- `t3.model` — `T3Model` top-level wrapper
- `t3.tracing` — schema-v1 trace generator + loader
- `t3.benchmarks` — lm-eval-harness adapter (`T3LM`, `run_benchmark_suite`)

## Verification

Generated traces should match the published library at
[t3atlas.dev](https://t3atlas.dev). The `examples/verify_against_atlas.py`
script runs a published prompt and compares the output trace against the
canonical version.

## Architecture overview

T³ extends standard multi-head attention with a per-head ecology of six
conjugate primitives coupled through bivector composition in Cl(3,3) geometric
algebra. Heads interact through a learned blockade-and-cosurvival graph and
ponder adaptively per stage via output-entropy halt. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the technical specification.

## Results

T³ at 124M parameters demonstrates **cumulative pareto improvement over a
same-data, same-compute vanilla GPT-2 baseline** across reasoning benchmarks.
Both models are trained on the same 5B-token English mix (FineWeb-Edu, DCLM,
StackEdu, FineMath, Cosmopedia, Wikipedia) with the same training schedule
and batch shape; the only difference is the architecture.

The first table below is the *cumulative* envelope across the v2.5–v3.7+
T³ architectural lineage: each cell is the highest-performing T³-124M
recipe on that task. Different recipes excel at different tasks — there is
no single "best T³-124M checkpoint." This shows what the architecture
demonstrates is achievable at this size and budget.

| Task | GPT-2 vanilla (5B same data) | T³-124M (best of lineage) | Δ |
|---|---:|---:|---:|
| BoolQ | 57.3 | **72.0** | **+14.7** |
| HellaSwag | 30.8 | **48.5** | **+17.7** |
| WinoGrande | 50.4 | **59.0** | +8.6 |
| COPA | 64.0 | **70.0** | +6.0 |
| ARC-Challenge | 24.7 | **30.4** | +5.6 |
| ARC-Easy | 44.2 | **48.7** | +4.5 |
| PIQA | 61.8 | **65.2** | +3.4 |
| RTE | 54.5 | 54.5 | 0.0 |
| **mean** | | | **+7.6** |

The second table is the most recent architectural generation (v3.7+) at the
same 124M / 5B-token budget, single-recipe — the "what does the current
architecture do off the shelf?" view, without recipe-tuning per task.

| Task | GPT-2 vanilla (5B same data) | T³-v3.7+ best | Δ |
|---|---:|---:|---:|
| HellaSwag | 30.8 | 38.2 | +7.4 |
| BoolQ | 57.3 | 61.8 | +4.5 |
| WinoGrande | 50.4 | 53.5 | +3.1 |
| PIQA | 61.8 | 64.2 | +2.4 |
| ARC-Challenge | 24.7 | 26.3 | +1.6 |
| COPA | 64.0 | 64.0 | 0.0 |
| ARC-Easy | 44.2 | 42.8 | −1.4 |
| **mean** | | | **+2.5** |

Full per-recipe breakdowns, parameter-and-compute pareto plots, and
cross-corpus comparisons against larger baselines (SmolLM2-360M,
Qwen2.5-1.5B) are at [t3atlas.dev/benchmarks](https://t3atlas.dev/benchmarks).
On `BoolQ` specifically, the best T³-124M (72.0) is within a stderr of
Qwen2.5-1.5B (72.5) — a ~12× param-ratio gap closed on one task; the
pareto plots show this is not the case for every task.

## Released checkpoint

| Model | Parameters | Substrate | Val PPL | HF |
|-------|------------|-----------|---------|----|
| `t3-124m-v36` | 124.5M | GPT-2 Small | 27.76 | [`mirrorethic/t3-124m-v36`](https://huggingface.co/mirrorethic/t3-124m-v36) |

`v3.6-run3` is published as a stable architecture-verification reference,
not necessarily the per-task peak — see the cumulative table above and the
atlas viewer for per-recipe winners across the lineage. No instruction
tuning. Research / interpretability use.

## Citation

```bibtex
@misc{sutherland2026t3,
  author = {Sutherland, Garret},
  title  = {T³ Reference Implementation},
  year   = {2026},
  url    = {https://github.com/mirrorethic/t3-reference}
}
```

## Related artifacts

- Trace library: <https://t3atlas.dev>
- Benchmark dataset: <https://t3atlas.dev/benchmarks>
- Paper: citation pending

## License

Apache-2.0. Both code and weights.
