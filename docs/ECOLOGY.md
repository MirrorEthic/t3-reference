# T³ Ecology — the Six Primitives

This is the long-form companion to `ARCHITECTURE.md §2`. It explains *why*
each primitive exists, how it's computed in detail, and what role it plays
during a forward pass.

For the formal mathematical structure (`Cl(3,3)`, conjugate pairs,
Hamiltonian rotation), see `ARCHITECTURE.md §3`. This document is the
**operational** reference.

---

## Why six?

Three observable axes (E, I, F) that the model can read directly off of
attention and activations, plus three conjugate axes (C, K, V) that the
model maintains alongside as a kind of "expected value" or commitment
signal. The pair structure is what enables the geometric-algebra rotation
to do useful work — the rotation moves uncertainty (E) into commitment (C),
intensity (I) into temporal lock (K), and friction (F) into valence (V).

If you stripped to just three primitives (no conjugates), you'd lose the
rotation. If you went to eight or twelve, you'd add more axes than the
ecology can usefully maintain at the parameter scale we work at. Six is
the smallest signature with non-trivial split-signature structure
(Cl(3,3)) that supports a full SO(N) bivector rotation.

---

## E — Entropy

**What it is.** The blended uncertainty of attention and output:

```
E_blended = blend_alpha · attn_entropy + (1 − blend_alpha) · output_entropy
          = 0.5 · attn_entropy + 0.5 · output_entropy
```

`attn_entropy` is the per-head softmax entropy of the attention weights at
this layer. `output_entropy` is the per-stage entropy of the *output
distribution* over the vocabulary — present only on stages whose output is
projected through `output_proj` (typically all stages, since weights are tied).

**Why blended.** Pure attention entropy is a local signal — it tells you
how diffuse the head's attention pattern is, but not whether the head is
contributing to a confident output. Pure output entropy is a global signal
— useful at the chain level but not per-head. Blending gives a per-head
quantity that responds both to local attention behavior and to the stage's
overall confidence.

**EMA.** Updated as `E ← entropy_ema_decay · E + (1 − entropy_ema_decay) · E_obs`
with `entropy_ema_decay = 0.95`.

---

## I — Intensity

**What it is.** Per-head activation magnitude — the L2 norm of the head's
attention output, normalized.

**Role.** Heads that are "doing something" have high I. Heads that are
silent have low I. I drives the friction signal (changes in I are part of
the friction computation).

---

## F — Friction

**What it is.** EMA of the absolute change in E (and a small contribution
from I — `friction_intensity_weight = 0.3`). Roughly:

```
F = EMA(|ΔE| + 0.3 · |ΔI|)
```

**Why it matters.** F captures whether the head's behavior is *changing*
or *stable*. High F means the head is in transition (responding to
something new in the input); low F means the head is settled. The σ MLP
reads F and uses it to modulate temperature — settled heads can run at
sharp temperature; transitioning heads should run softer.

---

## V — Valence (Fristonian)

**What it is.** A dual-EMA MACD signal on a free-energy proxy. Two EMAs of
attention entropy at different timescales (`valence_fast_decay = 0.95` and
`valence_slow_decay = 0.99`); V is the difference, scaled and
normalized:

```
fast = EMA_0.95(E_obs)
slow = EMA_0.99(E_obs)
V    = sigmoid(valence_scale · (slow − fast))   # high V = E is decreasing = improving
```

`valence_relative = True` normalizes V across heads so per-head V values
are comparable. There's a 3-call warmup before V activates (lets EMAs
settle).

**Role.** V is the *direction* signal: high V = the head is improving its
prediction (entropy dropping); low V = degrading. Cosurvival bonds are
modulated by V (heads that fire together when V is positive bond more
strongly), and V is the conjugate of F in the Cl(3,3) coupling.

The "Fristonian" name is from Karl Friston's free-energy framework: V is a
local proxy for the change in variational free energy.

---

## C — Coherence

**What it is.** The conjugate to E in `Cl(3,3)`. Negative-signature axis.

**Role.** C is a *commitment* signal — high C means the head is settled
into a specific role; low C means the head is uncommitted. The Hamiltonian
rotation (§3) couples C to E: when E is high (uncertain), C is pulled
down (commitment relaxes); when E is low (confident), C is pulled up
(commitment locks in).

C is maintained as a buffer that the rotation modifies; it is *not*
directly observed from attention behavior the way E and I are. It is a
genuine internal state of the ecology.

---

## K — Chronos

**What it is.** The conjugate to I. Negative-signature axis.

**Role.** K is a *temporal lock* — heads that have been intensely active
build up K, which biases them to remain active across forward passes.
Where I is the moment-to-moment activation, K is the integrated history of
activation. The I↔K rotation transfers between current and integrated
intensity.

K turned out to be the primitive whose inter-stage predictor learns the
most non-trivial cross-stage map (S1→S2 correlation r=0.59 after the v3.6
no_grad fix). This suggests K carries genuinely different information
across stages — the integrated history of one stage is not the integrated
history of the next.

---

## How σ is read off

```
σ_h = sigmoid( Linear(sigma_hidden, 1)( Tanh( Linear(n_primitives, sigma_hidden)( primitives_h ) ) ) )
```

`sigma_hidden = 16` for the released checkpoint. The σ MLP is trained
end-to-end with the rest of the model — its job is to map the 6-D
ecology vector into a single bounded uncertainty value that's actually
useful for downstream attention modulation. Empirically the MLP learns
that low σ should fire when (high E, low C, high V) — uncertain, uncommitted,
improving — i.e., the head is in a productive update state. High σ should
fire when (low E, high C, low V) — confident, committed, plateaued.

---

## Validity ranges and clamps

All six primitives are clamped to `[0.01, 0.99]`. This prevents the
exp/sigmoid feedback loops in the rotation and the σ MLP from saturating.

## Update order in a single forward

1. Run attention → observe `attn_entropy`, `intensity`, `output_entropy`
2. Compute `E_blended`, update E EMA
3. Update I from observation
4. Compute `ΔE`, `ΔI`; update F EMA
5. Update V from dual-EMA MACD on E_obs
6. Apply Hamiltonian rotation to all 6 primitives (pairwise + Ω + optional T)
7. Clamp to `[0.01, 0.99]`
8. Compute σ from the rotated primitives
9. (Optional) compute self-model surprise; nudge σ MLP toward σ_target
10. (Optional) compute inter-stage prediction; emit auxiliary loss

Steps 1–4 are observations. Step 5 is the slow-timescale derivative.
Step 6 is where the geometric algebra does its work. Step 8 is the
operational output. Steps 9–10 are training-time refinement signals.

---

## When primitives go missing

A T³ checkpoint that was trained with `n_primitives = 4` (older v2.5
lineage) only has E, I, F, V — no C, no K, and no Cl(3,3) rotation.
Released v3.6 has all six. A future Cl(4,4) variant would add two more
primitives (an extension under research as of mid-2026); the schema
permits up to 8.

The trace JSON's `meta.capabilities.n_primitives` and `meta.primitive_names`
declare which are present in the file at hand.
