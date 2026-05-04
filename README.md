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
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download("mirrorethic/t3-124m-v36", "pytorch_model.bin")
model = T3Model.from_checkpoint(ckpt)
out = model.generate("The capital of France is", trace=True)
# out.text contains the generation; out.trace contains per-stage ecology state
```

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

## Released checkpoint

| Model | Parameters | Substrate | Val PPL | HF |
|-------|------------|-----------|---------|----|
| `t3-124m-v36` | 124.5M | GPT-2 Small | 27.76 | [`mirrorethic/t3-124m-v36`](https://huggingface.co/mirrorethic/t3-124m-v36) |

Trained on a 5B-token English mix (FineWeb-Edu, DCLM, StackEdu, FineMath,
Cosmopedia, Wikipedia). No instruction tuning. Research / interpretability use.

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
