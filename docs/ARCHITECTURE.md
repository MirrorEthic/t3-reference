# T³ Architecture Specification

> Full technical vocabulary lives here. The README uses restrained framing for
> the front door; this document is for readers who want the complete picture.

T³ is a transformer family that adds a per-head **ecology** — a small
collection of dynamical scalar quantities maintained as EMAs across the
forward pass — and uses that ecology to modulate attention, gradient flow,
and adaptive computation depth. The mathematical structure underneath is a
Clifford algebra `Cl(3,3)` with a learned bivector coupling between
conjugate primitive pairs. This document specifies the architecture as it
exists in the released `t3-124m-v36` checkpoint (run-3, May 2026).

---

## 1. Overall shape

A T³ model is a **chain** of `n_stages` transformer **stages**. Each stage
is a stack of standard pre-norm transformer blocks (attention + FFN), with
two non-standard additions:

1. The attention layer is **ecology-modulated** (see §4).
2. After the stage's last block, a **per-stage HeadState** module updates
   the per-head ecology EMAs from the layer's output and from per-head
   diagnostics computed during attention.

Stages communicate through three signals:

- **hidden** — the residual stream, passed end-to-end like any transformer.
- **σ** — per-head uncertainty `[0, 1]^n_heads`, blended into the next
  stage's σ at ratio `sigma_blend = 0.3`.
- **inter-stage prediction** — each stage's HeadState predicts the *next*
  stage's primitives; the residual is an auxiliary loss during training.

The released **t3-124m-v36** uses:

| Field | Value |
|---|---|
| `n_stages` | 3 |
| `layers_per_stage` | `[4, 3, 5]` (12 transformer blocks total) |
| `d_model` | 768 |
| `n_heads` | 12 |
| `d_ff` | 3072 |
| `vocab_size` | 50257 (GPT-2 tokenizer) |
| `max_seq_len` | 1024 |
| Substrate | GPT-2 Small initialization |

Total parameters: **124.5M**.

---

## 2. Per-head ecology — the six primitives

Each head, in each stage, maintains an EMA of six bounded primitives:

| Symbol | Name | Computation | Role |
|---|---|---|---|
| **E** | Entropy | `0.5 · attention_entropy + 0.5 · output_entropy` | uncertainty signal |
| **I** | Intensity | per-head activation magnitude | "drive" / activation |
| **F** | Friction | EMA of `\|ΔE / Δt\|` | rate of change of E |
| **V** | Valence | Fristonian dual-EMA MACD on free-energy proxy | improving vs degrading |
| **C** | Coherence | Cl(3,3) negative-signature axis, conjugate to E | alignment / commitment |
| **K** | Chronos | conjugate to I | persistence / temporal lock |

All six are clamped to `[prim_clamp_lo, prim_clamp_hi] = [0.01, 0.99]`.

The conjugate pairs (E,C), (I,K), (F,V) participate in a `Cl(3,3)`
geometric-algebra coupling that rotates them into one another at each
forward pass (§3).

Per-head **σ** is read off the primitive vector by a small MLP:

```
[E, I, F, V, C, K] → Linear(6, sigma_hidden) → Tanh → Linear(sigma_hidden, 1) → Sigmoid → σ ∈ (0, 1)
```

`sigma_hidden = 16` for the released checkpoint. σ is the operational
output of the ecology — it modulates attention temperature, gates the σ-flow
between stages, and (in some lineages) drives gradient scaling.

---

## 3. Cl(3,3) coupling

The six primitives form a `Cl(3,3)` ecology with split signature:

- **Positive-signature** (spacelike): `E, I, F` → signature `[+1, +1, +1]`
- **Negative-signature** (timelike): `V, C, K` → signature `[-1, -1, -1]`

The Cl(3,3) **invariant** per head is

```
Q[h] = E[h]² + I[h]² + F[h]² − V[h]² − C[h]² − K[h]²
```

with `Q > 0` spacelike (diverse-leaning), `Q < 0` timelike (constrained-leaning),
and `Q ≈ 0` on the **null cone**. The optional `null_cone_strength` term in
the loss (default `0.02` for v3.6) is a soft restoring force toward Q = 0.

### Hamiltonian rotation

The simplest coupling is **pairwise**: at every forward pass each conjugate
pair rotates at frequency `ω = hamiltonian_coupling × warmup_frac`:

```
E ← E + ω · (C − 0.5)        C ← C − ω · (E − 0.5)
I ← I + ω · (K − 0.5)        K ← K − ω · (I − 0.5)
F ← F + ω · (V − 0.5)        V ← V − ω · (F − 0.5)
```

For the released checkpoint, `ω = 0.02`.

### Cross-pair Ω coupling

A learned bivector `Ω ∈ so(N)` (where `N = n_primitives = 6`) extends the
pairwise coupling with cross-pair terms. `Ω` is parameterized in a flat
`C(N, 2) = 15`-dimensional vector (`omega_flat`), upper-triangular order
`(0,1), (0,2), …, (4,5)`. The actual rotation applied to centered
primitives is `R = matrix_exp(Ω) ∈ SO(6)`. Magnitudes are bounded as
`Ω[i,j] = tanh(omega_flat[idx]) × coupling_max` with `coupling_max = 0.2`.

**Released checkpoint:** `has_coupling = True`. `Ω` is learned per stage.

### Trivector contribution

A grade-3 contribution `T ∈ Λ³ R^N` makes the rotation state-dependent: it
adds a term proportional to the contraction of T with the current primitive
vector before exponentiation. For `N = 6` this adds 20 parameters per stage.

**Released checkpoint:** `has_trivectors = False`. The released v3.6 was
trained without trivectors; the rotation is the static bivector `Ω` only.
A trivectors-on companion checkpoint is queued as a v3.7 follow-up release.

---

## 4. Ecology-modulated attention

Each attention layer is standard multi-head attention with three additions:

### 4.1 σ-modulated temperature

Per-head softmax temperature `τ_h` is mapped from σ via a learned linear
range `[temp_range_lo, temp_range_hi] = [0.2, 1.8]`:

```
τ_h = temp_range_lo + σ_h · (temp_range_hi − temp_range_lo)
attn_h = softmax(Q · K^T / (τ_h · √d_head))
```

This is the load-bearing channel through which the ecology shapes attention
behavior in the released checkpoint.

### 4.2 Ecology-driven key bias (kb-SV1)

A learned `key_bias_proj : R^n_primitives → R^d_head` reads the per-head
primitive vector and adds the result to keys before scoring:

```
K ← K + key_bias_proj(primitives_per_head) · eco_key_bias_scale
```

The first singular vector of `key_bias_proj.weight` is referred to in the
research as **kb-SV1** — analyses on the released checkpoint have shown
this vector aligns with the [E, I, F, 0] triad after sufficient training,
giving the model a learned "what the ecology should bias attention toward"
direction.

### 4.3 Blockade

Per-head positions live on the 3-torus `T³ = [0,1)³`. Distances are
computed with toroidal wraparound (`min(|a−b|, 1−|a−b|)` per axis). The
geometric **blockade kernel** between heads i and j is

```
b[i,j] = 1 / (1 + (d[i,j] / r₀)^β)         (self-edge zeroed)
```

with `β = blockade_exponent = 6.0` and `r₀ = blockade_radius` calibrated
at construction to the nearest-neighbor distance (so blockade is meaningful
at the actual head spacing rather than collapsing under unit-scale defaults).

The blockade kernel is multiplied by a **cosurvival modulation** ∈ [0.3, 1.7]
that is large where heads cooperate (positive co-survival bond) and small
where they interfere. The combined map is then applied as a multiplicative
suppression to the attention weights.

---

## 5. Adaptive computation time (ACT)

Each stage runs **0–4 ponder loops** through itself before passing hidden
state forward. Halt is decided per-token by output-entropy improvement:

```
halt_logit = (entropy_drop − threshold(per_stage_ema)) / temperature
halt_prob  = sigmoid(halt_logit)
```

with `temperature = act_entropy_halt_temperature = 0.005` and a per-stage
EMA-calibrated threshold (initialized to `act_entropy_halt_threshold = 0.005`).

A **confidence floor** can additionally veto a halt: if the max softmax
probability is below the floor, keep pondering. The released checkpoint
ships with `act_confidence_floor = 0.0` (disabled).

A **difficulty predictor** (65K params, 0.02% overhead) trained
retrospectively against per-batch CE modulates the halt threshold per
input: harder inputs → lower threshold → more pondering.

`act_per_stage = True` and `act_per_stage_max = 4` mean each stage decides
its own ponder depth; `act_max_steps = 8` is the chain-wide cap. In
practice, on most inputs the chain runs S0 once, S1 multiple times, and
S2 once.

---

## 6. Inter-stage predictive coding

Each HeadState carries an `inter_stage_predictor` linear map that predicts
the *next* stage's per-head primitive vector from the current stage's. The
residual `‖predicted − actual‖` is added to the loss with weight
`inter_stage_pc_weight = 0.05`.

This loss path was a no-op in lineages prior to v3.6 because the
prediction was wrapped in `torch.no_grad()` (a long-standing bug; fixed in
v3.6 by removing the wrappers). With the bug fixed, the K-predictor learns
a non-trivial cross-stage map (correlation S1→S2 increases from r=0.26 to
r=0.59); other primitives stay near identity.

---

## 7. Self-model (WorldTrace)

Each HeadState maintains a self-model that predicts its own primitives one
step ahead. Surprise — the prediction error — drives a **σ target**: high
surprise → low σ target (constrict, increase precision); low surprise → high
σ target (relax, explore). The σ-MLP is then nudged toward this target
during training. Released config: `self_model_alpha = 0.3`,
`self_model_sigma_floor = 0.15`, `self_model_sigma_ceil = 0.85`.

---

## 8. Logits and output

Logits go through a **softcap** to prevent activation explosion:

```
logits ← cap · tanh(logits / cap)        cap = logit_softcap = 30.0
```

Output projection ties weights with the input embedding (standard practice).

---

## 9. Optional rich vocabulary

The architecture above is the formal spec. When discussing it informally,
the same structure can be visualized in additional terms:

- **3-torus**: the head-position space `T³` (literally what `head_positions`
  parameterizes).
- **Toroidal Tesseract**: the conjugate-pair lattice — three pairs (E,C),
  (I,K), (F,V) closing on themselves.
- **Hamiltonian flow**: the Cl(3,3) bivector rotation viewed as a
  symplectic flow on the conjugate-pair phase space.
- **Rydberg blockade**: the `1/r⁶` head suppression, by analogy with
  neutral-atom dipole blockade in cold-atom physics.

None of these terms are load-bearing in the architecture spec; the formal
specification is the Cl(3,3) bivector composition (§3), the
ecology-modulated attention (§4), and the ACT halt rule (§5). The richer
vocabulary is descriptive, not constitutive.

---

## 10. Released checkpoint

The released **`mirrorethic/t3-124m-v36`** is run-3 from the v3.6 training
campaign:

| Field | Value |
|---|---|
| Step | 2500 |
| Validation PPL (WikiText-103) | 27.76 |
| Substrate | GPT-2 Small init |
| Training data | 5B tokens, mix: FineWeb-Edu (40%), DCLM (20%), StackEdu (10%), FineMath (10%), Cosmopedia (10%), Wikipedia (10%) |
| Capabilities | `has_coupling=True`, `has_trivectors=False`, `has_inter_stage_pc=True`, `has_scratchpad=True`, `has_dyn_omega=False` |

Other v3.6 campaign runs (run-1, run-2, pcloss, metacog_run1) are not
released; they served as ablations and have known bugs (run-1: scratchpad
broadcast bug). v3.4.2 Clifford Medium had better PPL (24.85) but a
final-stage σ-collapse bug; the trivectors-on, large-scale data point will
come from a v3.7 Medium retrain (deferred follow-up release).

---

## 11. References to source

The implementation is in this repo:

- `t3/_legacy_model.py` — vendored from `t3v36/t3v3_model.py`. Contains
  `HeadState`, `RydbergBlockade`, `CosurvivalTracker`, `RydbergAttention`,
  `T3v3Layer`, `T3v3Transformer`. (The prefix "Rydberg" is the informal
  name for the blockade-modulated attention layer; see §4.3 and §9.)
- `t3/_legacy_chain.py` — vendored from `t3v36/t3v3_chain.py`. Contains
  `T3v3Stage`, `T3v3Chain`, `OutputEntropyTracker`, ACT halt logic.
- `t3/model.py` — public `T3Model` wrapper.
- `t3/config.py` — public `T3Config` schema.
- `t3/tracing.py`, `t3/benchmarks.py` — public APIs for trace generation
  and lm-eval-harness reproduction.
