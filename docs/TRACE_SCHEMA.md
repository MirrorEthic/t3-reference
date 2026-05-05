# T³ Trace Schema (v1)

A T³ ecology trace is a JSONL file capturing the live ecology state of a T³
chain across a forward pass (or autoregressive generation). One record per line.
The file is consumable without instantiating the T³ model — every quantity
needed to render or analyze the chain's behavior is materialized in the trace.

This schema is the *citeable artifact*; the viewer is one consumer of it.

---

## File structure

```
LINE 1:        meta             (one)
LINE 2..1+S:   stage_geom       (one per stage; static within a forward)
LINE 2+S..K:   chain_state      (one per full chain forward / generated token)
REMAINING:     frame            (one per stage call; varies through ACT pondering)
```

`S` = number of stages (typically 3, sometimes 4). `K` = number of chain forwards
(1 for non-generative, `n_tokens` for autoregressive).

---

## Record types

### `meta` (one per file, line 1)

```json
{
  "type": "meta",
  "n_stages": 3,
  "n_heads": 12,
  "d_head": 64,
  "n_layers_per_stage": [4, 3, 5],
  "primitive_names": ["E", "I", "F", "V", "C", "K"],
  "primitive_signature": [+1, +1, +1, -1, -1, -1],
  "prompt": "The capital of France is",
  "prompt_id": "factual",
  "lineage": "v3.7-run2",
  "ckpt": "best.pt",
  "n_tokens": 32,
  "n_frames": 232,
  "n_chain_states": 32,
  "created": "2026-04-30T22-44-11",
  "capabilities": { ... }
}
```

| Field | Type | Notes |
|---|---|---|
| `n_stages` | int | Stages in the chain. v3.x is typically 3. v3.4.2-clifford is 4. |
| `n_heads` | int | Heads per stage. GPT-2 family: 12. v3.4.2-clifford: 16. Gemma3-270M: 4. |
| `d_head` | int | Per-head attention dim. May be 0 if `cfg.d_head == 0` (auto = d_model // n_heads). |
| `n_layers_per_stage` | list[int] | Per-stage transformer layer count. Shape `[n_stages]`. |
| `primitive_names` | list[str] | Conventionally `[E, I, F, V, C, K]` for Cl(3,3); 8 entries for Cl(4,4). |
| `primitive_signature` | list[int] | Cl(p,q) signature. Spacelike = +1, timelike = -1. Drives the Q invariant. |
| `prompt` | str | Input prompt as a literal string. |
| `prompt_id` | str | Short tag from the prompt library (`factual`, `banana`, `selfref`, ...). |
| `lineage` | str | Architectural lineage (e.g. `v3.7-run2`, `gemma3-smoke10k`, `qwen2_5_1_5b-flex`). |
| `ckpt` | str | Checkpoint filename only (e.g. `step2500.pt`, `best.pt`). |
| `n_tokens` | int | Generated tokens beyond the prompt. 0 = single forward, no generation. |
| `n_frames` | int | Total frame records in this file. |
| `n_chain_states` | int | Total chain_state records. Equals `max(n_tokens, 1)` for generation. |
| `created` | str | ISO-like timestamp (`YYYY-MM-DDTHH-MM-SS`). |
| `capabilities` | object | See below. |

#### `capabilities`

Probe of the *checkpoint's* state dict + cfg, NOT of the chain code. Tells consumers
what dynamics this lineage actually has weights for. Honest record of what was vs
wasn't trained.

```json
"capabilities": {
  "has_coupling":   true,    // checkpoint has _coupling_params  (cross-pair Ω)
  "has_trivectors": true,    // checkpoint has _trivector_params (grade-3 state-dependent rotation)
  "has_dyn_omega":  true,    // checkpoint has _omega_shadow_*   (v3.7+ live Ω drift)
  "has_inter_stage_pc": true,  // inter_stage_predictor exists (predictive coding bridge)
  "has_scratchpad": true,    // scratchpad heads exist
  "n_primitives":   6,       // 6 = Cl(3,3); 8 = Cl(4,4) when v3.7+ Phase 3 lands
  "null_cone_strength":    0.02,  // restoring force toward Q=0
  "hamiltonian_coupling":  0.02,  // ω in E↔C, I↔K, F↔V kicks
  "sigma_hidden":          16,    // σ-MLP hidden width (Phase 1A sweep: 16/32/64)
  "scratchpad_inject_entropy": [0.0, 0.0, 0.0]
}
```

When a flag is `false`, the corresponding state in `frame` records will be
`null` or empty (e.g., `omega_flat: []` if `has_coupling: false`).

---

### `stage_geom` (one per stage, lines 2..1+S)

Static geometry per stage. Captured once before the recorded forward begins.

```json
{
  "type": "stage_geom",
  "stage_idx": 0,
  "head_positions": [[h0_x, h0_y, h0_z], ...],   // [n_heads, 3] in [0,1]^3 (3-torus)
  "distances": [[...], ...],                      // [n_heads, n_heads] geodesic on T³
  "blockade_kernel": [[...], ...],                // [n_heads, n_heads] 1/(1+(d/r)^exp), self-zeroed
  "blockade_radius": 0.27,                        // r₀, auto-calibrated to NN distance
  "blockade_exponent": 6.0,                       // Rydberg 1/r⁶
  "coupling_max": 0.2,                            // tanh scale for _coupling_params
  "has_trivectors": true,
  "cosurvival_matrix": [[...], ...],              // [n_heads, n_heads] fitness coupling
  "cosurvival_modulation": [[...], ...],          // [n_heads, n_heads] blockade modifier ∈ [0.3, 1.7]
  "cosurvival_head_loss_ema": [...],              // [n_heads] per-head running loss
  "cosurvival_protection_scores": [...]           // [n_heads] sum of positive bonds per head
}
```

#### Notes

- **`head_positions`** are points on the 3-torus T³ (`[0,1]^3` with opposite faces
  identified). Distances should be computed via wraparound: `min(|a-b|, 1-|a-b|)`
  per axis. Treating them as Euclidean is incorrect — see `geodesic_distance_t3`
  in `t3v3_model.py`.
- **`blockade_kernel`** is the geometric (positions-only) suppression graph.
- **`cosurvival_modulation`** is the multiplicative factor applied to the
  geometric kernel during forward — heads that cooperate (positive cosurvival)
  see their mutual blockade *reduced*; heads that interfere see it amplified.
- **`cosurvival_matrix`** values can be unbounded (raw EMA over training, can
  reach ±100s). Use `tanh(v / scale)` for visualization, where `scale` is e.g.
  the median of `|v|`.
- **Lineages without coupling** (e.g. `v3.4.1`, `v3.5-scratchpad`) still emit
  `stage_geom` records — the matrices are valid, just `_coupling_params` itself
  is absent at the chain level (see `frame.omega_flat`).

---

### `chain_state` (one per chain forward, lines 2+S..K)

Chain-level signals from the full forward. Captures ACT mechanics + retrospective
predictors. One record per generated token (or one total for non-generative).

```json
{
  "type": "chain_state",
  "token_idx": 0,
  "act_halt_probs": [0.0, 0.0, 1.0, ...],         // halt p per ponder step (skip-first-halt = first is 0)
  "act_strain_values": [0.13, 0.07, 0.02, ...],   // |σ_out - σ_in|.mean() per step
  "act_ponder_steps": 3,                          // how many full chain iterations occurred
  "act_per_stage_steps": [1, 2, 1],               // per-stage ACT calls (per-stage ACT mode only)
  "act_per_stage_strains": [[...], ...],          // per-stage strain history (per-stage ACT mode only)
  "act_ponder_cost": 0.18,                        // -Σ log(1-λ_t) + λ_p * steps
  "difficulty_pred": [0.95],                      // [B] retrospective difficulty in [0,1]
  "scratchpad_pred": [0.86, 0.65, 0.64, ...],     // [S] per-token "this token is hard" for the *prompt* tokens
  "omega_displacement_ema": 0.009,                // v3.7+ only: ‖shadow Ω - anchor Ω‖ EMA
  "omega_variance_ema": 1.15e-9                   // v3.7+ only: drift stability signal
}
```

#### Notes

- **`act_halt_probs[t]`** is the cumulative-product-style halt probability at
  ponder step `t`. With `act_skip_first_halt: true` (default), index 0 is always
  near zero and the chain commits to at least one full pass. The first index
  with `p ≈ 1.0` is the halt step.
- **`act_strain_values[t]`** is the σ convergence signal — small strain means
  ecology is stable, large strain means it's still moving. Drives halt.
- **`difficulty_pred`** is *retrospective*: trained against actual CE loss after
  the forward. A high value means "in hindsight this batch was hard." It then
  modulates the next forward's halt threshold.
- **`scratchpad_pred`** is per-position over the *current input sequence*.
  In a generation loop the sequence grows by one token per step; this field
  reflects the predictor's per-token output for the entire prompt-so-far.
- **`omega_displacement_ema`** only present when `has_dyn_omega: true`
  (v3.7+ dynamic Ω shadow buffers). Quantifies live deformation magnitude.

---

### `frame` (one per stage call, remaining lines)

The most-frequent record. Emitted once per `T3Stage.forward` call. Within a
single chain forward, `n_stages × n_act_calls` frames typically.

```json
{
  "type": "frame",
  "frame_idx": 17,                                // global counter within this trace
  "stage_idx": 1,                                 // 0-indexed
  "act_call": 2,                                  // how-many-th call to this stage (during ACT)
  "primitives": [[...], ...],                     // [n_heads, n_primitives] EMA values in [0, 1]
  "sigma": [...],                                 // [n_heads] per-head σ ∈ [0, 1]
  "omega_flat": [...],                            // [C(N,2)] bivector params in (i<j) lex order; [] if no coupling
  "trivectors": [...],                            // [C(N,3)] grade-3 params in (i<j<k) lex order; [] if absent
  "Q": [...],                                     // [n_heads] Cl(p,q) invariant: Σ sig[k] * prim[h,k]^2
  "kb_input_norms": [...],                        // [in_features] column norms of key_bias_proj weight (kb-SV1 proxy)
  "suppression": [...],                           // [n_heads] per-head dynamic blockade activity (avg across layers)
  "per_layer_suppression": [[...], ...],          // [n_layers, n_heads] before averaging
  "self_surprise": [...],                         // [n_heads] ‖predicted_prim - actual_prim‖ from WorldTrace self-model
  "output_entropy_ema": 0.089,                    // scalar, EMA of softmax entropy at the stage's output_proj
  "per_layer_attn_entropy": [[...], ...],         // [n_layers, n_heads] per-head attention entropy per layer
  "stage_top_tokens": [["the", 0.166], ...]       // [k] top-k decoded token + prob; ONLY non-null on stages with output_proj (typically last)
}
```

#### Notes

- **`primitives[h]`** has length equal to `meta.capabilities.n_primitives`.
  For Cl(3,3) the canonical order is `[E, I, F, V, C, K]`. Values are EMAs
  clamped to `[0, 1]`.
- **`Q`** is the Cl(p,q) invariant per head, computed as
  `Σ_k primitive_signature[k] * primitives[h, k]^2`.
  - Q > 0: spacelike (diverse-leaning)
  - Q < 0: timelike (constrained-leaning)
  - Q ≈ 0: near the null cone
- **`omega_flat`** length is `C(N, 2) = N*(N-1)/2`. For `N=6`: 15 entries.
  For `N=8` (Cl(4,4)): 28 entries. Lex order: `(0,1), (0,2), ..., (0,N-1), (1,2), ..., (N-2, N-1)`.
- **`trivectors`** length is `C(N, 3)`. For `N=6`: 20 entries. For `N=8`: 56.
  Lex order: `(i,j,k)` with `i<j<k`.
- **Reconstructing the rotation**: build antisymmetric Ω as
  `Ω[i,j] = tanh(omega_flat[idx]) * coupling_max`, plus state-dependent trivector
  contribution (see `_apply_coupling_rotation` in `t3v3_model.py`). Then
  `R = matrix_exp(Ω)` is the SO(N) rotation applied to centered primitives.
- **`stage_top_tokens`** is non-null only on stages that have an `output_proj`
  module — typically the final stage. Earlier stages don't project to vocab.
  Reading earlier-stage commitment requires a "logit lens" (project intermediate
  hiddens through the final stage's output_proj manually).
- **`per_layer_attn_entropy`** is the finer-grained version of `primitives[..., 0]` (E):
  per-layer attention entropy before being averaged into the stage's E primitive.

---

## Worked example: minimal Python loader

```python
import json
from pathlib import Path

def load_trace(path):
    """Load a trace into a typed dict. Frames returned as list (not stream)."""
    out = {"meta": None, "geoms": {}, "chain_states": [], "frames": []}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            t = r["type"]
            if t == "meta": out["meta"] = r
            elif t == "stage_geom": out["geoms"][r["stage_idx"]] = r
            elif t == "chain_state": out["chain_states"].append(r)
            elif t == "frame": out["frames"].append(r)
    return out

# Example: extract per-token ACT depth + σ entropy across stages
trace = load_trace("traces/v3.7-run2__best__factual.jsonl")
for cs in trace["chain_states"]:
    print(f"tok {cs['token_idx']}: ACT {cs['act_ponder_steps']} steps")

# Example: aggregate Q distribution across all heads × all frames in stage 1
import numpy as np
qs = []
for fr in trace["frames"]:
    if fr["stage_idx"] == 1:
        qs.extend(fr["Q"])
qs = np.array(qs)
print(f"S1 Q distribution: spacelike {np.mean(qs > 0.05):.2%}, "
      f"timelike {np.mean(qs < -0.05):.2%}, "
      f"null {np.mean(np.abs(qs) <= 0.05):.2%}")
```

---

## Schema versioning

This is **schema v1**. Future versions:

- **v1.1** would add fields without removing existing ones (forward-compatible
  consumers ignore unknown fields).
- **v2** would be a breaking change. If/when issued, `meta.schema_version: 2`
  will be present; v1 records will lack this key (treat as v1).

A trace produced by `export_trace.py` from this commit is v1.

---

## Provenance

- Capture code: `diagnostic_trace/export_trace.py`
- Sweep driver: `diagnostic_trace/sweep_traces.sh`
- Capture code: `t3/tracing.py` (`generate_trace`, `TraceRecorder`)
- Sweep driver: `scripts/sweep_traces.py`
- Prompt library: `t3/data/prompts.json`

The model code that produces the captured state is in `t3/ecology.py`,
`t3/attention.py`, and `t3/chain.py`. Major quantities cited in this schema:

| Quantity | Source |
|---|---|
| `_coupling_params` (Ω bivector) | `t3.ecology.HeadState` |
| `_trivector_params` | `t3.ecology.HeadState` |
| `_apply_coupling_rotation` | `t3.ecology.HeadState._apply_coupling_rotation` |
| `_self_surprise` | `t3.ecology.HeadState._update_self_model` |
| Blockade (1/r^N) | `t3.ecology.Blockade` |
| Cosurvival bond graph | `t3.ecology.Cosurvival` |
| `inter_stage_predictor` | `t3.ecology.HeadState` (registered, inert at inference) |
| ACT halt logic | `t3.chain.T3Chain._act_perstage_forward` |
| Difficulty predictor | `t3.chain.T3Chain.difficulty_head` |
| Scratchpad-need predictor | `t3.chain.T3Chain.scratchpad_need_head` |
