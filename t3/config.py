"""T3Config — architecture hyperparameters.

Faithful projection of the `config` dict embedded in published checkpoints.

Sections (top-to-bottom):
    1. Substrate shape (vocab, dims, layers)
    2. Stage partition (T³ chain)
    3. Ecology primitives (E, I, F, V, C, K)
    4. Cl(3,3) coupling (Hamiltonian, trivectors, null cone)
    5. Blockade (1/r^N head suppression)
    6. Cosurvival (head bond graph)
    7. Attention modulation (key bias, temperature range)
    8. ACT (output-entropy halt + confidence + difficulty predictor)
    9. Inter-stage signal passing (predictive coding, σ-flow, hidden-flow)
    10. Self-model (WorldTrace)
    11. Output / generation

Training-only fields (loss weights, optimizer hyperparameters, schedule knobs)
are intentionally absent from this dataclass — they are silently dropped by
`from_checkpoint_dict`. The training-time forward pass remains numerically
identical without them; only the auxiliary loss term computations differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class T3Config:
    # --- 1. substrate shape ---
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 0              # 0 → use n_heads (no GQA)
    n_layers: int = 5                # MAX layers across stages (not sum); used for buffer sizing
    d_ff: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.0
    norm_type: str = "layernorm"
    norm_eps: float = 1e-5
    attn_bias: bool = True
    attn_out_bias: bool = True
    ffn_bias: bool = True
    ffn_type: str = "gelu"
    use_rope: bool = False
    rope_base: float = 10000.0
    use_residual: bool = True
    skip_intermediate_norms: bool = False
    shared_positions: bool = False
    ignore_index: int = -100

    # --- 2. stage partition ---
    n_stages: int = 3
    layers_per_stage: Tuple[int, ...] = field(default_factory=lambda: (4, 3, 5))

    # --- 3. ecology primitives ---
    n_primitives: int = 6
    grounded_primitives: bool = True
    per_head_sigma: bool = True
    sigma_blend: float = 0.3                # σ-flow blend across stages
    sigma_stop_gradient: bool = False
    blend_alpha: float = 0.5                # 0.5 * attn_H + 0.5 * output_H
    prim_clamp_lo: float = 0.01
    prim_clamp_hi: float = 0.99
    entropy_ema_decay: float = 0.95
    friction_intensity_weight: float = 0.3
    valence_scale: float = 1.5
    valence_fast_decay: float = 0.95
    valence_slow_decay: float = 0.99
    valence_warmup_calls: int = 3
    valence_relative: bool = True

    # --- 4. Cl(3,3) coupling ---
    hamiltonian_coupling: float = 0.02      # ω
    hamiltonian_cross_coupling: bool = True
    hamiltonian_max_coupling: float = 0.2
    hamiltonian_trivectors: bool = False    # run-3 checkpoint: OFF (despite v3.6 convention)
    null_cone_strength: float = 0.02
    sigma_modulated_coupling: bool = False
    learned_ecology_params: bool = False
    grav_k: float = 0.2

    # --- 5. blockade ---
    blockade_enabled: bool = True
    blockade_strength: float = 0.3
    blockade_exponent: float = 6.0          # 1/r^N
    blockade_radius_init: float = 1.0
    blockade_radius_auto: bool = True
    blockade_learnable: bool = True
    blockade_warmup_steps: int = 200

    # --- 6. cosurvival ---
    cosurvival_enabled: bool = True
    cosurvival_decay: float = 0.999
    cosurvival_lr_coupling: float = 0.3
    cosurvival_update_interval: int = 50
    cosurvival_valence_modulation: bool = True
    complementarity_margin: float = 0.3

    # --- 7. attention modulation ---
    eco_key_bias: bool = True
    eco_key_bias_features: int = 6
    eco_key_bias_scale: float = 1.0
    temp_range_lo: float = 0.2
    temp_range_hi: float = 1.8
    bypass_ecology: bool = False

    # --- 8. ACT (adaptive computation time) ---
    act_enabled: bool = True
    act_per_stage: bool = True
    act_per_stage_max: int = 4              # per-stage halt; the global max is act_max_steps
    act_max_steps: int = 8
    act_entropy_halt: bool = True
    act_entropy_halt_threshold: float = 0.005
    act_entropy_halt_temperature: float = 0.005
    act_confidence_floor: float = 0.0
    act_difficulty_predictor: bool = True
    act_difficulty_scale: float = 0.8
    act_difficulty_ema_decay: float = 0.99
    act_skip_first_halt: bool = True
    act_hard_halt_eval: bool = True
    act_n_probe_positions: int = 2
    act_halt_epsilon: float = 0.01
    act_live_ecology: bool = True
    act_live_ecology_alpha: float = 0.3
    act_preponder_baseline: bool = True

    # --- 9. inter-stage signal passing ---
    inter_stage_pc: bool = True
    inter_stage_pc_weight: float = 0.05     # informational; loss term not run at inference
    pass_sigma: bool = True
    pass_hidden: bool = True

    # --- 10. self-model (WorldTrace) ---
    self_model_alpha: float = 0.3
    self_model_sensitivity: float = 15.0
    self_model_sigma_ceil: float = 0.85
    self_model_sigma_floor: float = 0.15

    # --- 11. output / generation ---
    logit_softcap: float | None = 30.0
    scratchpad_need_predictor: bool = True
    scratchpad_inject_entropy: Tuple[float, ...] = field(
        default_factory=lambda: (0.0, 0.0, 0.03)  # run-3 cleanest schedule (S2-only)
    )

    # --- provenance ---
    version: str = "v3.6-run3"
    source: str = "gpt2_t3_ultimate"

    # ---------------- helpers ----------------

    @property
    def total_layers(self) -> int:
        """Sum across stages (the actual transformer block count)."""
        return sum(self.layers_per_stage)

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        return self.d_model // self.n_heads

    @classmethod
    def from_checkpoint_dict(cls, ckpt_config: dict) -> "T3Config":
        """Project a training-script config dict (≈125 entries) onto T3Config.

        Unknown keys are silently dropped — these are training-only loss weights,
        optimizer state, and schedule knobs that have no architectural meaning.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kept = {k: v for k, v in ckpt_config.items() if k in known}
        # Tuples sometimes survive as lists in JSON / pickle
        for tup_field in ("layers_per_stage", "scratchpad_inject_entropy"):
            if tup_field in kept and isinstance(kept[tup_field], list):
                kept[tup_field] = tuple(kept[tup_field])
        return cls(**kept)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
