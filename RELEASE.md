# T³ Open-Source Release — May 2026

**TL;DR.** T³ is a transformer architecture that augments standard
multi-head attention with a per-head ecology of six conjugate primitives
coupled through `Cl(3,3)` geometric algebra, plus an adaptive per-stage
output-entropy halt. This release ships the inference-time reference
implementation (Apache-2.0), one canonical checkpoint, one ablation sibling,
13 reproducible inference traces, and a re-runnable benchmark suite — all
matched to a vanilla GPT-2 baseline trained on the same data.

---

## The three artifacts

This release is a triangle. Each piece points at the other two.

| | Where | What it is | Size |
|---|---|---|---|
| **Code** | <https://github.com/MirrorEthic/t3-reference> | Inference-only reference implementation. Loads the published checkpoint, generates schema-v1 traces, reproduces the benchmark numbers. | Apache-2.0, ~30 files, 12 passing tests |
| **Weights** | <https://huggingface.co/mirrorethic/t3-124m-v36> | Canonical 124M checkpoint (run-3, step 2500). Plus sibling [`-pcloss`](https://huggingface.co/mirrorethic/t3-124m-v36-pcloss) for the inter-stage-PC ablation. | 499 MB each |
| **Atlas** | <https://t3atlas.dev> | Public trace library + benchmark page + schema. Reads the same `pytorch_model.bin` and `data.json` you can rebuild yourself from the code+weights. | ~340 MB hosted |

```
                    ┌────────────────────────────┐
                    │  github.com/MirrorEthic/   │
                    │       t3-reference         │
                    │  (Apache-2.0 code)         │
                    └────┬──────────────────┬────┘
                         │                  │
        loads via        │                  │   produces traces matching
        T3Model          │                  │   schema → renders in
        .from_checkpoint │                  │   t3atlas.dev/viewer
                         ▼                  ▼
   ┌──────────────────────────────┐   ┌──────────────────────────────┐
   │  huggingface.co/mirrorethic/ │   │      t3atlas.dev             │
   │       t3-124m-v36            │   │  (viewer + benchmarks +      │
   │  (canonical 124M, val 27.76) │◄──│   schema, public artifact)   │
   │  + ...-pcloss sibling        │   │                              │
   └──────────────────────────────┘   └──────────────────────────────┘
                       reproduces benchmark numbers via lm-eval-harness
                       reported on t3atlas.dev/benchmarks/
```

---

## What each artifact is for

### `mirrorethic/t3-reference` (code)

The canonical inference implementation. Loads the published checkpoint,
runs forward, generates ecology traces matching the public schema, runs the
8-task lm-eval-harness suite that produces the published benchmark numbers.
Intentionally **inference-only** — training infrastructure is private (per
the release-strategy: open architecture + closed training methodology, the
same model RWKV / Mamba / xLSTM use).

Key modules:
- `t3.config.T3Config` — clean public schema for the 125-key checkpoint config
- `t3.model.T3Model` — top-level wrapper with `from_checkpoint`, `forward`, `generate`
- `t3.tracing.generate_trace` — schema-v1 JSONL trace emitter
- `t3.benchmarks.run_benchmark_suite` — lm-eval-harness runner producing
  atlas-compatible JSON (`schema: "t3atlas-bench-v1"`)

### `mirrorethic/t3-124m-v36` (canonical weights)

The headline checkpoint. Run-3 from the v3.6 training campaign. **GPT-2
Small substrate**, 124.5M params, 5B-token training mix, val PPL 27.76
on WikiText-103. Picked over run-1 (PPL 27.06) because run-1 had a
scratchpad broadcast bug; the cleaner shorter run is the honest pick.

| Field | Value |
|---|---|
| Parameters | 124,500,000 |
| Stages × layers | 3 stages, `layers_per_stage = [4, 3, 5]` |
| `d_model` × `n_heads` × `d_ff` | 768 × 12 × 3072 |
| Vocab / max seq | 50257 (GPT-2 BPE) / 1024 |
| Training data | 5B tokens (FineWeb-Edu 40%, DCLM 20%, StackEdu 10%, FineMath 10%, Cosmopedia 10%, Wikipedia 10%) |
| Cumulative training step | 138,000 (135.5K substrate + 2,500 v3.6 increment) |
| Trivectors | off (the trivectors-on variant is queued for v3.7 Medium) |
| ACT | output-entropy halt, per-stage 4-step cap |
| `has_coupling` / `has_inter_stage_pc` / `has_scratchpad` | true / true / true |

### `mirrorethic/t3-124m-v36-pcloss` (ablation sibling)

Same architecture, same training, with **one** difference: the inter-stage
predictive-coding loss is un-detached (the `torch.no_grad` wrapper in the
PC predictor was removed). Result: PPL is slightly worse (28.53 vs 27.76),
reasoning is net-neutral, but the K-predictor learns a real cross-stage
map (S1→S2 correlation r=0.59 vs r=0.26 baseline). Pair with the canonical
checkpoint for the controlled ablation.

### `t3atlas.dev` (public artifact + viewer + benchmarks)

The site is the reader-facing surface for this work:

- **Viewer** (`/viewer/`) renders trace JSONL records — per-stage σ
  evolution, ACT halt curves, head-level Q (Cl(3,3) invariant), bivector
  Ω rotation. 247+ traces across 12 lineages, including the 13 fresh
  `t3-124m-v36__best__*` traces this release added.
- **Benchmarks** (`/benchmarks/`) shows the headline lm-eval-harness
  scores per lineage with a parameter-efficiency panel and a compute
  frontier panel. The headline t3-v3.6 row is now the verified-tier
  `v36_run3_release` numbers below.
- **Schema** (`/SCHEMA.md`) documents the trace JSONL format. Also
  vendored as [`docs/TRACE_SCHEMA.md`](docs/TRACE_SCHEMA.md) in the
  reference repo.

The site is the *citeable artifact*; the code and weights make it
reproducible.

---

## Three reproducibility paths

### Path 1 — load and predict

```bash
pip install git+https://github.com/MirrorEthic/t3-reference
```

```python
from huggingface_hub import hf_hub_download
from t3 import T3Model

ckpt = hf_hub_download("mirrorethic/t3-124m-v36", "pytorch_model.bin")
model = T3Model.from_checkpoint(ckpt)
model.eval()

import torch
out = model(torch.randint(0, 50257, (1, 16)))
```

### Path 2 — reproduce a trace from the public library

```python
from t3 import T3Model
from t3.tracing import generate_trace, load_trace
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download("mirrorethic/t3-124m-v36", "pytorch_model.bin")
model = T3Model.from_checkpoint(ckpt); model.eval()

# Generate a fresh trace for the same prompt the atlas uses
out = generate_trace(model, "The capital of France is",
                     prompt_id="factual", n_tokens=32,
                     out_path="my_factual.jsonl")

# Load it and compare against the canonical version on t3atlas.dev
trace = load_trace(out)
assert trace["meta"]["capabilities"]["has_coupling"] is True
assert trace["meta"]["primitive_names"] == ["E", "I", "F", "V", "C", "K"]
```

### Path 3 — reproduce the published benchmarks

```python
from t3.benchmarks import run_benchmark_suite

results = run_benchmark_suite(
    checkpoint=ckpt,                      # the same hf_hub_download from above
    out_path="benchmarks/my_run.json",    # atlas-compatible schema
)
# Should reproduce the val PPL + 8-task suite numbers below within
# autocast / RNG noise (~±1%).
```

---

## Published numbers (verified-tier on t3atlas)

All numbers are full lm-eval-harness 0.4.x runs on the canonical
checkpoint. No subset, no caveats. Reproduce with `examples/run_benchmarks.py`.

| Task | Metric | `t3-124m-v36` | `t3-124m-v36-pcloss` (sibling) |
|---|---|---:|---:|
| WikiText-103 (val) | perplexity | **27.76** | 28.53 |
| BoolQ | acc | **0.6046** | 0.6064 |
| ARC-Easy | acc | **0.4331** | 0.4398 |
| ARC-Challenge | acc | **0.2176** | 0.2099 |
| PIQA | acc | **0.6050** | 0.6028 |
| HellaSwag | acc | **0.3040** | 0.3029 |
| WinoGrande | acc | **0.5043** | 0.5075 |
| COPA | acc | **0.6000** | 0.6200 |
| RTE | acc | **0.5235** | 0.5271 |

The PCloss sibling shows the inter-stage PC predictor *can* learn (K-axis
S1→S2 correlation r=0.26→0.59) but at this scale (124M / 2.5K steps) the
learned predictor doesn't translate to downstream task gains.

For comparison panels (parameter-efficiency vs vanilla GPT-2 trained on the
same 5B-token mix; compute-frontier across lineages), see
<https://t3atlas.dev/benchmarks/>.

---

## What this release is NOT

- **Not a chat model.** 124M parameters, no instruction tuning, no RLHF, no
  safety tuning. Use it for research and architectural comparison, not for
  serving text generation.
- **Not a training framework.** Training scripts, σ-flow curriculum, data
  pipeline, and empirical training methodology are private. The paper /
  followups will document methodology; the source-of-truth implementation
  remains in MirrorEthic's hands.
- **Not the trivectors-on data point.** The released checkpoint was trained
  with `hamiltonian_trivectors=False` — the static bivector Ω is the full
  `Cl(3,3)` rotation in this checkpoint. The trivectors-on, larger-scale
  variant is a planned **v3.7 Medium retrain (354.8M, 5B tokens)** —
  scheduled separately, will land as `mirrorethic/t3-355m-v37` when ready.
- **Not the v3.4.2 Clifford Medium checkpoint.** That run reached PPL 24.85
  but had a known final-stage σ-collapse bug; the fix is in v3.7. We
  declined to ship the bugged Medium ckpt for the same release-hygiene
  reason we declined run-1.

---

## Architecture

The fundamentals (full spec at [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md);
operational ecology details at [`docs/ECOLOGY.md`](docs/ECOLOGY.md)):

- **Per-head ecology** — six bounded primitives `(E, I, F, V, C, K)`
  maintained as EMA state per head, per stage. Three conjugate pairs:
  E↔C (entropy/coherence), I↔K (intensity/chronos), F↔V (friction/valence).
- **`Cl(3,3)` coupling** — split-signature geometric algebra. Each pair
  rotates under a learned bivector Ω at frequency ω at every forward pass.
  The Cl(p,q) invariant `Q[h] = E²+I²+F²−V²−C²−K²` distinguishes spacelike
  (diverse, Q>0), timelike (constrained, Q<0), and null-cone behavior.
- **Ecology-modulated attention** — per-head softmax temperature scales with
  σ (the MLP output of the primitive vector). A learned `key_bias_proj`
  reads the primitive vector and biases keys. A blockade graph suppresses
  head correlations as `1/r⁶` over learned head positions on `T³`.
- **Adaptive computation** — per-stage output-entropy halt with confidence
  floor and a 65K-parameter retrospective difficulty predictor. Each stage
  ponders 0–4 times per token; chain-wide cap 8.
- **Inter-stage predictive coding** — each stage's HeadState predicts the
  next stage's primitives. The K-axis is the only one that learns a
  non-trivial cross-stage map at v3.6 scale.

---

## Roadmap (post-0.1.0)

| | What | When |
|---|---|---|
| 0.2.0 | **`mirrorethic/t3-355m-v37`** — Medium-scale retrain with σ-MLP width sweep (sh16/sh32/sh64) and trivectors-on. ~5 days × 3 H100s. The trivectors-on, scale-up data point. | Compute-budget dependent |
| — | Methodology paper covering training methodology, σ-flow curriculum, the FEP grounding, and full ablation results | In drafting |
| — | PyPI publication of `t3-reference` | When repo settles |
| — | CI on push (pytest, lm-eval smoke) — currently 13 tests pass locally | Trivial; deferred |

The public-module split (`t3.ecology / t3.attention / t3.chain`) and removal
of the vendored training-script files landed in 0.1.0.

---

## Citation

```bibtex
@misc{sutherland2026t3,
  author    = {Sutherland, Garret},
  title     = {T³: A Clifford-Algebra-Augmented Transformer Architecture},
  year      = {2026},
  publisher = {Hugging Face / MirrorEthic LLC},
  url       = {https://huggingface.co/mirrorethic/t3-124m-v36},
  note      = {Code: \url{https://github.com/MirrorEthic/t3-reference};
               Trace library + benchmarks: \url{https://t3atlas.dev}}
}
```

---

## License

Apache-2.0 throughout — code (`MirrorEthic/t3-reference`) and weights
(both HF repos). The patent grant clause means commercial use is allowed
without re-licensing.

## Contact

Garret Sutherland (MirrorEthic LLC) — `gsutherland@mirrorethic.com`.
