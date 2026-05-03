"""Vendored from t3v36/t3v3_chain.py — inference-only.

Stripped: file-level __main__ smoke test, T3v3Loss + scale_gradients_by_valence
imports (training-only). Triton fused-kernel path is hard-disabled (HAS_TRITON=False).

Do not edit directly — this is the validation copy. The clean module split
(t3.act, t3.chain, etc.) is the public API.
"""

#!/usr/bin/env python3
"""
t3v3_chain.py - T3 v3 Chain: Modular Geodesic Architecture with Rydberg Coupling
==================================================================================

Evolves from t3v2_chain.py. Key v3.0 changes:
- OutputEntropyTracker absorbed as native nn.Module (was monkey-patched in training scripts)
- Ecology state save/restore (get_ecology_state / restore_ecology_state) built in
- warmup_frac propagated through set_ecology_strength to all primitives
- Per-stage output entropy tracking drives blended E primitive natively

Preserved from v2:
- ACT forward (_act_forward, _act_perstage_forward)
- Variable layers_per_stage
- Shared positions
- Gradient checkpointing (layer-level and ACT-level)
- All diagnostic/compatibility methods

Author: Garret Sutherland, MirrorEthic LLC
Date: 2026-03-13
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from t3._legacy_model import (
    T3v3Config, T3v3Transformer, T3v3Layer,
    HeadState, RydbergBlockade, CosurvivalTracker,
    geodesic_distance_t3, RotaryEmbedding,
    T3RMSNorm, SwiGLUFFN, make_norm,
)

# Triton fused kernels (optional, vendored separately if needed)
HAS_TRITON = False


# ======================
# Chain Configuration
# ======================

@dataclass
class T3v3ChainConfig(T3v3Config):
    """T3 v3 Chain configuration. Extends T3v3Config."""

    # Chain structure
    n_stages: int = 2

    # Override: fewer layers per stage (each stage is shallow)
    # Total layers = n_stages * n_layers
    # Default n_layers=3 per stage x 2 stages = 6 total (same as monolith default)

    # Inter-stage communication
    pass_hidden: bool = True     # Pass hidden states between stages
    pass_sigma: bool = True      # Pass per-head sigma between stages
    use_residual: bool = True    # Skip connections between stages
    sigma_blend: float = 0.3     # Blend ratio: this x inherited + (1-this) x computed
    skip_intermediate_norms: bool = False  # Skip norm on non-last stages (only norm at final stage)

    # Shared geometry: all stages use one set of head positions
    # This gives blockade the full chain depth as surface area
    # while keeping E/I/F/V weights per-stage for local sigma computation
    shared_positions: bool = False

    # ACT (Adaptive Computation Time) -- pondering loops
    act_enabled: bool = False
    act_max_steps: int = 8          # Max ponder loops through stages 1+
    act_halt_epsilon: float = 0.01  # Halt when remaining probability < epsilon
    act_lambda_p: float = 0.3      # Geometric prior parameter (higher = halt sooner)
    act_ponder_weight: float = 0.01 # Weight for ponder regularization loss

    # Strain-based halting (T3-native physics -- replaces learned halt_head)
    act_strain_halt: bool = True        # Use strain tensor for halting (vs learned MLP)
    act_strain_threshold: float = 0.015  # delta-sigma threshold for 50% halt probability
    act_strain_temperature: float = 0.005  # Sharpness of sigmoid halt decision
    act_skip_first_halt: bool = True  # Skip halt decision on loop 1 (always run at least 1 full pass)

    # Adaptive strain threshold via EMA: threshold = ema * margin (replaces static threshold)
    # The system tracks a running average of strain and halts when current strain
    # drops below (ema * margin). This self-calibrates as the model improves.
    act_strain_ema_enabled: bool = False  # Use adaptive EMA threshold instead of static
    act_strain_ema_decay: float = 0.995   # EMA decay rate (0.995 ~ 200-step window)
    act_strain_ema_margin: float = 0.6    # Halt when strain < ema * margin (lower = halt sooner)

    # Output-entropy-based halting: halt when pondering stops reducing prediction uncertainty.
    # More direct than sigma strain -- measures actual prediction confidence improvement.
    # Requires output_proj weights (weight-tied embeddings). Subsamples 4 positions for efficiency.
    act_entropy_halt: bool = False       # Use output entropy improvement for halt decision
    act_entropy_halt_threshold: float = 0.005  # Min entropy improvement to justify continuing
    act_entropy_halt_temperature: float = 0.005  # Sigmoid temperature for soft halt
    act_n_probe_positions: int = 2       # v3.3 opt: number of positions to subsample for entropy probe (was 4)

    # v3.2: Pre-ponder entropy baseline — compute entropy of stage INPUT before ponder loop.
    # Gives prev_entropy at t=0 so first ponder step can make a real halt decision.
    # Without this, min_ponder is structurally 3 (skip_first_halt + need baseline).
    act_preponder_baseline: bool = True

    # v3.2: Per-stage adaptive thresholds — EMA-calibrated from training deltas.
    # Stages with larger typical entropy deltas get higher thresholds → halt earlier.
    # effective_threshold = max(ema_stage × margin, floor)
    act_adaptive_threshold: bool = False      # Requires training to calibrate
    act_adaptive_ema_decay: float = 0.99      # ~100-step window
    act_adaptive_margin: float = 0.5          # Halt when delta < ema * margin
    act_adaptive_floor: float = 0.001         # Minimum threshold (prevents halting at 0)

    # v3.2: Hard halt at eval — when lambda_t > 0.5, use that step's representation
    # directly and break. No probability-weighted average. Keeps soft PonderNet during training.
    act_hard_halt_eval: bool = True

    # v3.3: Confidence floor — second halt condition. Even if entropy halt says stop,
    # keep pondering if max softmax probability is below floor. Grounds halt in
    # "does the model actually have a confident prediction?" not just "did logits get sharper?"
    act_confidence_floor: float = 0.0  # 0 = disabled; ~0.05 good for 50K vocab

    # v3.3: Retrospective difficulty predictor — small MLP predicts batch difficulty
    # from pre-ponder hidden state. Trained with actual CE loss as supervision.
    # Difficulty modulates halt threshold: hard inputs get lower threshold → ponder more.
    act_difficulty_predictor: bool = False
    act_difficulty_scale: float = 0.8    # Hard inputs reduce threshold by up to 80%
    act_difficulty_ema_decay: float = 0.99  # Loss EMA for difficulty normalization

    # v3.5: Scratchpad-need predictor — per-token MLP on the final stage's hidden state
    # that predicts "this token is hard to commit to". Trained retrospectively against
    # the actual per-position CE loss. Phase-1 sidecar for marrying ACT with token-space
    # scratchpad: the head output (and/or raw S3 entropy) gates whether to emit a <think>
    # token instead of sampling normally.
    scratchpad_need_predictor: bool = False

    # v3.6: metacog → ecology feedback injection. Per-stage weight list. After the
    # scratchpad_need_head runs on the final hidden state, the mean of its per-token
    # prediction (pooled over batch × sequence) is injected into EACH stage's entropy
    # EMA as a stage-specific delta:  ΔE_s = α_s · (mean_pred − 0.5).
    #
    # Per-stage because our steering analysis showed stages have different roles: S0
    # mostly embed/restructure, S1 the responsive encode stage, S2 the commit stage
    # where decision-token metacognition matters most. Typical production setting:
    # [0.0, 0.01, 0.03] — silent at S0, mild at S1, strongest at S2.
    #
    # A single scalar α is also accepted (broadcast to all stages). Always detached
    # from gradient — the scratchpad head is trained by its own aux loss only; this
    # injection is a pure runtime feedback signal, not a training loss path.
    scratchpad_inject_entropy: tuple = (0.0, 0.0, 0.0)

    # ============================================================================
    # v3.7 ports from v5.3 — validated mechanisms (sigma fix, V1 residual, Dyn Ω).
    # All default OFF so v3.6 behavior is preserved when disabled.
    # ============================================================================
    # V1 residual: ecology-gated value blend (zero new params; -3.2 PPL on probe).
    # Layer 0 of each stage caches V; layers 1+ blend V_n with V1 by σ_h per head.
    v1_residual_enabled: bool = False
    v1_residual_gating: str = "sigma"        # "sigma" (validated) | "inverse_sigma" | "fixed"
    v1_residual_fixed_lambda: float = 0.5    # only used when gating="fixed" (rejected variant)

    # Dynamic Ω: medium-timescale Cl(3,3) coupling matrix evolution between forward passes.
    # 35 shadow params per stage; -2.7% on probe scale on top of V1.
    dynamic_omega_enabled: bool = False
    dynamic_omega_beta: float = 0.01         # EMA rate (~100 forwards to deform substantially)
    dynamic_omega_gamma: float = 0.5         # γ·ΔΩ_self + (1−γ)·ΔΩ_task — 0.5 validated
    dynamic_omega_max_delta: float = 0.01    # stability clamp per step

    # Per-stage ACT: each stage independently ponder-loops until its ecology converges.
    # Adaptive DEPTH (which stages work harder) instead of adaptive REPETITION (whole chain).
    act_per_stage: bool = False         # Per-stage strain halting (vs whole-chain loops)
    act_per_stage_max: int = 3          # Max ponder steps per individual stage

    # Gradient checkpointing for ACT ponder loops: trades compute for memory.
    # Recomputes stage forward during backward instead of storing all intermediates.
    # Essential for max_ponder >= 8 at 210M+ scale.
    act_gradient_checkpointing: bool = False
    # Layer-level gradient checkpointing: recompute each transformer layer's forward
    # during backward pass instead of storing activations. Essential for 3B+ models
    # on consumer GPUs (24GB). Trades ~30% speed for ~60% memory savings.
    layer_gradient_checkpointing: bool = False

    # Variable layers per stage: if set, overrides n_layers for each stage.
    # Length must equal n_stages. Allows deeper stages where compute is needed most.
    # If None, all stages use uniform n_layers (current behavior).
    layers_per_stage: Optional[List[int]] = None

    # v3.0: Output entropy tracker (per-stage, built into chain)
    entropy_ema_decay: float = 0.95     # EMA decay for output entropy tracking

    # v3.4: Logit softcap — prevents logit explosion via tanh squashing.
    # logits = cap * tanh(logits / cap). 0 = disabled.
    logit_softcap: float = 30.0

    def __post_init__(self):
        super().__post_init__() if hasattr(super(), '__post_init__') else None
        if self.layers_per_stage is not None:
            assert len(self.layers_per_stage) == self.n_stages, \
                f"layers_per_stage has {len(self.layers_per_stage)} entries but n_stages={self.n_stages}"
            assert all(n > 0 for n in self.layers_per_stage), \
                "All stages must have at least 1 layer"


# ======================
# Logit Softcap
# ======================

def compute_sigma_health(state: dict, device) -> tuple:
    """v3.7 sigma fix (ported from v5.3) — per-stage gated log barrier on live σ.

    The v3.6 sigma loss path in T3v3Chain.compute_aux_losses reads from
    `_last_head_sigmas`, a detached buffer — its gradient is zero, so the loss
    has no training signal regardless of weight. This helper instead reads
    `head_sigmas_tensor` from chain_states (live, with grad), iterates per-stage
    (NOT aggregated — aggregation hides per-stage saturation behind dilution),
    and uses a gated log-barrier (zero in [0.15, 0.85], blows up outside).

    Validated weights (v5 phase 3.3a/b): w_div=0.01, w_antisat=1.0.

    Usage in training:
        logits, state = chain(input_ids, return_state=True, return_chain_state=True)
        sigma_div, sigma_antisat = compute_sigma_health(state, device)
        loss = lm_loss + 0.01 * sigma_div + 1.0 * sigma_antisat

    Returns:
        (sigma_div, sigma_antisat) — both tensors with grad
    """
    sigma_div = torch.tensor(0.0, device=device)
    sigma_antisat = torch.tensor(0.0, device=device)
    n = 0
    for stage_state in state.get("chain_states", []):
        s = stage_state.get("head_sigmas_tensor", None)
        if s is None or s.numel() < 2:
            continue
        n += 1
        # Diversity: maximize std within stage (negative because we add to loss)
        sigma_div = sigma_div - s.std()
        # Gated log barrier — zero in [0.15, 0.85], log-barrier outside
        upper_mask = (s > 0.85).float()
        lower_mask = (s < 0.15).float()
        upper_barrier = -torch.log((1.05 - s.clamp(max=0.995)).clamp(min=1e-3)) * upper_mask
        lower_barrier = -torch.log((s.clamp(min=0.005) + 0.05).clamp(min=1e-3)) * lower_mask
        sigma_antisat = sigma_antisat + (upper_barrier + lower_barrier).mean()
    if n > 0:
        sigma_div = sigma_div / n
        sigma_antisat = sigma_antisat / n
    return sigma_div, sigma_antisat


def apply_logit_softcap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """Apply tanh softcap to prevent logit explosion. logits = cap * tanh(logits / cap)."""
    if cap > 0:
        logits = cap * torch.tanh(logits / cap)
    return logits


# ======================
# Output Entropy Tracker
# ======================

class OutputEntropyTracker(nn.Module):
    """Per-stage output entropy tracking for blended E primitive.

    Computes output entropy by projecting hidden states through
    the tied output embedding, subsampled at 4 positions for efficiency.
    Tracks per-stage entropy EMA and valence velocity.

    v2.5: This was a monkey-patched class in training scripts.
    v3.0: Native module owned by T3v3Stage.
    """

    def __init__(self, vocab_size: int, decay: float = 0.95, n_probe_positions: int = 2,
                 use_triton: bool = False):
        super().__init__()
        self.max_entropy = math.log(vocab_size)
        self.decay = decay
        self.n_probe_positions = n_probe_positions
        self.use_triton = use_triton and HAS_TRITON
        self.register_buffer("output_entropy_ema", torch.tensor(0.5))
        self.register_buffer("valence_velocity_ema", torch.tensor(0.0))
        self.register_buffer("_prev_entropy", torch.tensor(0.0))

    def compute(self, hidden: torch.Tensor, output_weight: torch.Tensor):
        """Compute output entropy from hidden + output projection weight.

        Args:
            hidden: (B, S, D) hidden state from stage
            output_weight: (vocab_size, D) output projection weight

        v3.3 opt: Reduced from 4 to 2 probe positions, bf16 matmul,
        removed @torch.compiler.disable.
        v3.3 Triton: Fused kernel eliminates [B, n_pos, V] materialization.
        """
        # FP32 guard: softmax+log and EMA accumulation overflow in FP16
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            B, S, D = hidden.shape
            n_pos = min(self.n_probe_positions, S)
            idx = torch.linspace(0, S - 1, n_pos).long().to(hidden.device)
            h_sub = hidden[:, idx, :].float()

            if self.use_triton:
                # Fused kernel: no [B, n_pos, V] intermediate in HBM
                ent_norm = fused_entropy_probe(h_sub, output_weight, self.max_entropy).detach()
            else:
                # PyTorch reference path
                logits = F.linear(h_sub, output_weight.float())
                probs = F.softmax(logits, dim=-1)
                log_probs = torch.log(probs + 1e-10)
                ent = -(probs * log_probs).sum(-1)
                ent_norm = (ent / self.max_entropy).mean().detach()
                del logits, probs, log_probs, ent

            del h_sub

            # EMA update
            alpha_ema = 1.0 - self.decay
            self.output_entropy_ema.lerp_(ent_norm, alpha_ema)

            # Valence velocity: -dH/dt
            prev = self._prev_entropy
            if prev > 0:
                delta = prev - ent_norm
                self.valence_velocity_ema.lerp_(delta, 1.0 - 0.99)
            self._prev_entropy.fill_(ent_norm)


# ======================
# Single T3 v3 Stage
# ======================

class T3v3Stage(nn.Module):
    """
    One stage of the T3 v3 Chain.

    Takes one geodesic step on the manifold with per-head
    Rydberg blockade and co-survival coupling.

    Receives hidden state and per-head sigma from previous stage.

    v3.0: Owns an OutputEntropyTracker for blended E primitive.
    """

    def __init__(self, cfg: T3v3ChainConfig, stage_idx: int,
                 is_first: bool = False, is_last: bool = False,
                 stage_n_layers: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.stage_idx = stage_idx
        self.is_first = is_first
        self.is_last = is_last

        # Determine layer count for this stage
        n_layers_this_stage = stage_n_layers if stage_n_layers is not None else cfg.n_layers
        self.n_layers = n_layers_this_stage

        # Embeddings only for first stage
        self.use_rope = getattr(cfg, 'use_rope', False)
        if is_first:
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
            if not self.use_rope:
                self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            self.embed_norm = make_norm(cfg.d_model, cfg)

        # RoPE embedding (shared across all stages, created once per stage but lightweight)
        if self.use_rope:
            rope_base = getattr(cfg, 'rope_base', 10000.0)
            rope_d_head = getattr(cfg, 'd_head', 0) or (cfg.d_model // cfg.n_heads)
            self.rope = RotaryEmbedding(rope_d_head, cfg.max_seq_len, rope_base)

        # Per-head T3 state (each stage has its own head positions + primitives)
        # v3.7+ Phase 1A staggered: if cfg.sigma_hidden_per_stage is set, pick this stage's value
        sigma_hidden_per_stage = getattr(cfg, 'sigma_hidden_per_stage', None)
        sigma_hidden_override = (sigma_hidden_per_stage[self.stage_idx]
                                  if sigma_hidden_per_stage is not None
                                  else None)
        self.head_state = HeadState(cfg.n_heads, cfg.d_model, cfg,
                                     sigma_hidden_override=sigma_hidden_override)

        # Co-survival tracker (per stage)
        self.cosurvival = CosurvivalTracker(cfg.n_heads, cfg)

        # Transformer layers (variable per stage when layers_per_stage is set)
        self.layers = nn.ModuleList([
            T3v3Layer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg, cfg.dropout)
            for _ in range(n_layers_this_stage)
        ])

        # Auto-calibrate blockade radius from head geometry.
        # Default radius=1.0 with NN distance ~0.2 gives (1/0.2)^6 = 15,625x suppression.
        # Setting radius=NN_distance makes blockade=50% at nearest-neighbor distance.
        if getattr(cfg, 'blockade_radius_auto', True) and cfg.blockade_enabled:
            nn_dist = self.head_state._init_nn_distance
            for layer in self.layers:
                layer.attention.blockade.blockade_radius = nn_dist

        # Layer norms
        self.norm = make_norm(cfg.d_model, cfg)

        # Output projection only for last stage
        if is_last:
            self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if is_first:  # Weight tying if single stage
                self.output_proj.weight = self.embed.weight

        self.dropout_layer = nn.Dropout(cfg.dropout)

        # Track step for blockade warmup (shared across stages via chain)
        self.register_buffer("_stage_step", torch.tensor(0, dtype=torch.long))

        # v3.0: Output entropy tracker (per-stage)
        self.entropy_tracker = OutputEntropyTracker(
            cfg.vocab_size, cfg.entropy_ema_decay,
            n_probe_positions=getattr(cfg, 'act_n_probe_positions', 2),
            use_triton=cfg.use_triton_kernels,
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)

    def _get_output_weight(self):
        """Get output projection weight for entropy computation."""
        if hasattr(self, 'output_proj'):
            return self.output_proj.weight
        return getattr(self, '_shared_output_weight', None)

    @torch.compiler.disable
    @torch.no_grad()
    def _update_ecology_state(self, all_head_entropy, distances, seq_len, bypass):
        """Update ecology buffers (co-survival, grounded primitives, step counters).
        Separated from forward() so torch.compile can trace the hot path.
        @torch.no_grad: all ops here are buffer mutations (EMA updates, co-survival,
        Hamiltonian kicks). Must not create autograd graph entries that conflict with
        sigma_b2/w2 being reused across ACT ponder steps (PyTorch 2.11+ strictness)."""
        self._stage_step += 1
        for layer in self.layers:
            layer.attention.blockade._global_step += 1

        if not bypass:
            avg_entropy = torch.stack(all_head_entropy).mean(dim=0)
            valence = self.head_state._valence_ema if self.cfg.grounded_primitives else None
            self.cosurvival.update(avg_entropy, distances, valence=valence)

            if self.cfg.grounded_primitives:
                warmup_frac = getattr(self, '_warmup_frac', 1.0)
                output_ent = self.entropy_tracker.output_entropy_ema.detach()

                # v3.1: Gather ecological signals for C and K grounding
                protection = self.cosurvival.get_protection_scores() if self.cfg.cosurvival_enabled else None
                # Aggregate per-head blockade suppression across layers
                suppression = None
                if self.cfg.blockade_enabled:
                    layer_supps = []
                    for layer in self.layers:
                        s = getattr(layer.attention.blockade, '_last_suppression', None)
                        if s is not None:
                            layer_supps.append(s)
                    if layer_supps:
                        suppression = torch.stack(layer_supps).mean(dim=0)  # avg across layers

                self.head_state.update_grounded_primitives(
                    avg_entropy, seq_len,
                    output_entropy=output_ent,
                    warmup_frac=warmup_frac,
                    protection_scores=protection,
                    blockade_suppression=suppression,
                )

            # v3.0: compute output entropy for blended E
            output_weight = self._get_output_weight()
            if output_weight is not None:
                # hidden is not available here -- caller must pass it
                # This is handled by the forward() method calling entropy_tracker.compute()
                pass

    def forward(
        self,
        x: Optional[torch.Tensor] = None,        # input_ids if first stage
        hidden: Optional[torch.Tensor] = None,    # hidden from prev stage
        sigma_prior: Optional[torch.Tensor] = None,  # per-head sigma from prev stage [n_heads]
        update_state: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Forward pass for one stage.

        Returns:
            hidden: [batch, seq, d_model]
            logits: [batch, seq, vocab] or None if not last stage
            stage_state: Dict with per-head sigma, blockade info, etc.
        """
        bypass = getattr(self.cfg, 'bypass_ecology', False)

        # Embedding (first stage, first call) or hidden passthrough (ponder / later stages)
        if self.is_first and x is not None:
            batch, seq_len = x.shape
            device = x.device
            if self.use_rope:
                h = self.embed(x)
            else:
                positions = torch.arange(seq_len, device=device)
                h = self.embed(x) + self.pos_embed(positions)
            # v3.7 audit fix (2026-04-29): apply embed_scale (Gemma-3 / Gemini-style sqrt(d_model)
            # scaling at input). Was a dead config field — set in cfg by the migration script but
            # never read by forward(). For GPT-2 transfers cfg.embed_scale=1.0 (no-op); for Gemma 3
            # 270M it's sqrt(640)≈25.3. Without this the residual stream is at the wrong scale —
            # RMSNorm masks most of the visible damage but the trained tied-output projection's
            # final logit magnitude is silently off, and any non-RMSNorm interaction (e.g. dropout,
            # gradient flow during continuation) sees the wrong scale.
            embed_scale = getattr(self.cfg, 'embed_scale', 1.0)
            if embed_scale != 1.0:
                h = h * embed_scale
            if not bypass and not getattr(self.cfg, 'skip_intermediate_norms', False):
                eco = getattr(self, '_ecology_strength', 1.0)
                if eco > 0.001:
                    h_normed = self.embed_norm(h)
                    h = h + eco * (h_normed - h)  # blend: at eco=0 skip norm, at eco=1 full norm
                else:
                    pass  # skip embed_norm entirely at eco=0
            h = self.dropout_layer(h)
        elif hidden is not None:
            h = hidden
            batch, seq_len, _ = h.shape
            device = h.device
        else:
            raise ValueError("Stage needs either x (input_ids) or hidden")

        # Invalidate distance/blockade caches from previous forward pass.
        # Critical for multi-step training — cached tensors retain autograd graph
        # from head_positions (learnable), causing "backward through graph twice" errors.
        self.head_state.invalidate_distance_cache()
        if self.cfg.cosurvival_enabled:
            self.cosurvival.invalidate_blockade_cache()

        # FP32 guard: all ecology computations (distances, blockade, sigma, K-bias)
        # must run in FP32. FP16 overflow in pow(6), division, and EMA values
        # produces NaN that contaminates buffers permanently.
        with torch.amp.autocast("cuda", enabled=False):
            # Compute pairwise head distances
            distances = self.head_state.get_pairwise_distances()

            # Co-survival blockade modulation (cooperating heads block less)
            blockade_mod = self.cosurvival.get_blockade_modulation() if self.cfg.cosurvival_enabled else None

            # Reshape to per-head view for sigma computation
            d_head = self.cfg.d_model // self.cfg.n_heads
            h_per_head = h.view(batch, seq_len, self.cfg.n_heads, d_head)

            # Compute per-head sigma (v3.0: pass warmup_frac)
            warmup_frac = getattr(self, '_warmup_frac', 1.0)
            own_sigmas = self.head_state.compute_head_sigmas(h_per_head, warmup_frac=warmup_frac)  # [n_heads]

            # sigma-flow: blend with inherited sigma from previous stage
            # NO detach -- let gradients flow between stages through sigma
            if self.cfg.pass_sigma and sigma_prior is not None:
                blend = self.cfg.sigma_blend
                head_sigmas = blend * sigma_prior + (1 - blend) * own_sigmas
            else:
                head_sigmas = own_sigmas

            # v3.4 Phase 3: Sigma complement — push bonded heads toward complementary uncertainty
            # Active at inference time. Uses co-survival bonds to determine who to diverge from.
            if getattr(self.cfg, 'cooperative_prediction', False) and not bypass and self.cfg.cosurvival_enabled:
                cs_matrix = self.cosurvival.cosurvival
                sigma_offset = self.head_state.compute_sigma_complement(head_sigmas, cs_matrix)
                eco = getattr(self, '_ecology_strength', 1.0)
                head_sigmas = (head_sigmas + eco * sigma_offset).clamp(0.01, 0.99)

        # Causal mask (skip for long sequences -- FlashAttention uses is_causal=True)
        flash_threshold = getattr(self.cfg, 'flash_seq_threshold', 2048)
        attn_mask = self._causal_mask(seq_len, device) if seq_len <= flash_threshold else None

        # Compute RoPE cos/sin if enabled
        rope_cos_sin = None
        if self.use_rope:
            rope_cos_sin = self.rope(seq_len)

        # v3.4 Phase 2: Compute eco K-bias offset (once per stage forward, shared by all layers)
        eco_k_offset = None
        if getattr(self.cfg, 'eco_key_bias', False) and not bypass and hasattr(self.head_state, 'key_bias_proj'):
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                hs = self.head_state
                n_eco = getattr(self.cfg, 'eco_key_bias_features', 4)
                if n_eco >= 6:
                    # v3.4.2: All 6 primitives feed K-bias — V is the coupling hub
                    eco_input = torch.stack([
                        head_sigmas.detach().float(),    # σ: [H]
                        hs._entropy_ema.float(),         # E: [H]
                        hs._friction_ema.float(),        # F: [H]
                        hs._valence_ema.float(),         # V: [H]
                        hs._last_excitation.float(),     # I proxy: [H]
                        hs._chronos_ema.float(),         # K: [H]
                    ], dim=-1)  # [H, 6]
                else:
                    # v3.4 original: 4 features
                    protection = hs.cosurvival_tracker.get_protection_scores() \
                        if hasattr(hs, 'cosurvival_tracker') else torch.zeros_like(head_sigmas)
                    eco_input = torch.stack([
                        head_sigmas.detach().float(),
                        hs._entropy_ema.float(),
                        hs._last_excitation.float(),
                        protection.detach().float(),
                    ], dim=-1)  # [H, 4]
            # K-offset WITH gradient through key_bias_proj weights — FP32
            with torch.amp.autocast("cuda", enabled=False):
                eco_k_offset = self.head_state.key_bias_proj(eco_input.float())  # [H, d_head]
                eco_k_offset = eco_k_offset * getattr(self.cfg, 'eco_key_bias_scale', 1.0)

        # Pass through layers
        all_head_entropy = []
        use_grad_ckpt = (getattr(self.cfg, 'layer_gradient_checkpointing', False)
                         and self.training and torch.is_grad_enabled())
        # v3.7: V1 residual — cache V from layer 0, pass to layers 1+ as raw-token grounding
        v1_residual_enabled = getattr(self.cfg, 'v1_residual_enabled', False)
        v1_cache = None
        for layer_idx, layer in enumerate(self.layers):
            is_first_layer = (layer_idx == 0)
            needs_return_v = v1_residual_enabled and is_first_layer
            layer_v1_residual = v1_cache if (v1_residual_enabled and not is_first_layer) else None

            if use_grad_ckpt:
                # Gradient checkpointing: recompute layer forward during backward
                # to save activation memory. Critical for 3B+ frozen-backbone training.
                def _layer_fn(_h, _sigmas, _dist, _mask, _bmod, _rope, _k_off, _v1, _layer=layer, _ret_v=needs_return_v):
                    return _layer(_h, _sigmas, _dist, attn_mask=_mask,
                                  blockade_mod=_bmod, rope_cos_sin=_rope, eco_k_offset=_k_off,
                                  v1_residual=_v1, return_v=_ret_v)
                result = torch.utils.checkpoint.checkpoint(
                    _layer_fn, h, head_sigmas, distances, attn_mask,
                    blockade_mod, rope_cos_sin, eco_k_offset, layer_v1_residual,
                    use_reentrant=False)
            else:
                result = layer(h, head_sigmas, distances, attn_mask=attn_mask,
                               blockade_mod=blockade_mod,
                               rope_cos_sin=rope_cos_sin,
                               eco_k_offset=eco_k_offset,
                               v1_residual=layer_v1_residual,
                               return_v=needs_return_v)
            if needs_return_v:
                h, head_entropy, v1_cache = result
            else:
                h, head_entropy = result
            all_head_entropy.append(head_entropy)

        # v3.4: Update cached excitation from last layer (for next forward's K-bias)
        if not bypass and self.layers and hasattr(self.head_state, '_last_excitation'):
            with torch.no_grad():
                self.head_state._last_excitation.copy_(
                    self.layers[-1].attention._head_activations.detach()
                )

        # Norm application:
        # - bypass mode: only last stage (matches Qwen's single model.norm)
        # - skip_intermediate_norms: only last stage (avoids redundant normalization)
        # - default: all stages get normed (original T3 behavior)
        # - ecology ramp: intermediate norms blended for transfer learning
        skip_norm = not self.is_last and (bypass or getattr(self.cfg, 'skip_intermediate_norms', False))
        if not skip_norm:
            if self.is_last:
                h = self.norm(h)  # last stage norm always applied (matches pretrained model.norm)
            else:
                # Intermediate norms: blend with ecology strength
                eco = getattr(self, '_ecology_strength', 1.0)
                if eco > 0.001:
                    h_normed = self.norm(h)
                    h = h + eco * (h_normed - h)
                else:
                    pass  # skip intermediate norm at eco=0

        # Update ecology state
        if update_state and self.training:
            # Full ecology update during training: co-survival + primitives + entropy tracker
            self._update_ecology_state(all_head_entropy, distances, seq_len, bypass)
            if not bypass and not getattr(self, '_skip_entropy_tracker', False):
                output_weight = self._get_output_weight()
                if output_weight is not None:
                    self.entropy_tracker.compute(h.detach(), output_weight)
        elif update_state and not self.training and getattr(self.cfg, 'eval_live_primitives', False):
            # v3.5: Live primitives during eval/generation — EMA + Hamiltonian + null cone
            # Skip co-survival (long-term bond tracking, not relevant for generation)
            # Skip output entropy tracker (training diagnostic)
            if not bypass and self.cfg.grounded_primitives and len(all_head_entropy) > 0:
                with torch.no_grad():
                    avg_entropy = torch.stack(all_head_entropy).mean(dim=0)
                    warmup_frac = getattr(self, '_warmup_frac', 1.0)
                    output_ent = self.entropy_tracker.output_entropy_ema.detach()
                    self.head_state.update_grounded_primitives(
                        avg_entropy, seq_len,
                        output_entropy=output_ent,
                        warmup_frac=warmup_frac,
                    )

        # Output logits (last stage only)
        # _skip_output_proj: pipeline parallel uses chunked output projection
        # to avoid materializing [S, V] logit tensor (saves ~300 MiB VRAM)
        logits = None
        if self.is_last and not getattr(self, '_skip_output_proj', False):
            if bypass:
                logits = self.output_proj(h)
            else:
                eco = getattr(self, '_ecology_strength', 1.0)
                # v3.1: stop-gradient — sigma modulates but CE can't hack it
                sigma_mean = head_sigmas.detach().mean()
                raw_temp = (1.5 - sigma_mean * 0.8).to(h.dtype)
                raw_temp = torch.clamp(raw_temp, 0.5, 2.0)
                temperature = 1.0 + eco * (raw_temp - 1.0)  # ramp: 1.0 at eco=0, raw at eco=1
                logits = self.output_proj(h) / temperature
            logits = apply_logit_softcap(logits, self.cfg.logit_softcap)

        # Build stage state
        stage_state = {
            "head_sigmas": head_sigmas.detach().cpu().tolist(),
            "head_sigmas_tensor": head_sigmas,  # Keep tensor for sigma-flow
            "global_sigma": float(head_sigmas.mean().item()),
            "head_positions": self.head_state.head_positions.detach().cpu().tolist(),
            "stage_idx": self.stage_idx,
        }

        if self.cfg.blockade_enabled:
            off_diag = distances[~torch.eye(self.cfg.n_heads, dtype=bool, device=device)]
            stage_state["blockade"] = {
                "min_head_distance": float(off_diag.min().item()),
                "max_head_distance": float(off_diag.max().item()),
                "mean_head_distance": float(off_diag.mean().item()),
            }

        if self.cfg.cosurvival_enabled:
            cs = self.cosurvival.cosurvival
            stage_state["cosurvival"] = {
                "mean": float(cs.mean().item()),
                "max": float(cs.max().item()),
                "min": float(cs.min().item()),
                "n_positive_bonds": int((cs > 0.01).sum().item()),
                "n_negative_bonds": int((cs < -0.01).sum().item()),
                "protection_scores": self.cosurvival.get_protection_scores().detach().cpu().tolist(),
            }

        if all_head_entropy:
            avg_ent = torch.stack(all_head_entropy).mean(dim=0)
            stage_state["head_activations"] = avg_ent.detach().cpu().tolist()
            # v3.4: Return tensors for live ecology during pondering (Phase 1 CAC)
            stage_state["head_entropy_tensors"] = all_head_entropy  # List of [n_heads] tensors

        # Per-head attention entropy (from last layer of this stage)
        if self.layers:
            stage_state["head_entropy"] = self.layers[-1].attention._head_entropy.detach().cpu().tolist()

        return h, logits, stage_state

    def reset_state(self):
        """Reset dynamic state."""
        self.head_state.head_tiers.fill_(self.cfg.initial_tier)
        self.head_state.head_dps.fill_(self.cfg.initial_dps)
        self.head_state._last_head_sigmas.fill_(0.5)
        self.cosurvival.cosurvival.zero_()
        self._stage_step.zero_()
        # v3.0: reset entropy tracker
        self.entropy_tracker.output_entropy_ema.fill_(0.5)
        self.entropy_tracker.valence_velocity_ema.fill_(0.0)
        self.entropy_tracker._prev_entropy.fill_(0.0)


# ======================
# Full T3 v3 Chain
# ======================

class T3v3Chain(nn.Module):
    """
    T3 v3 Chain: Modular Geodesic Architecture with Rydberg Coupling.

    Chains multiple small T3 v3 stages, each with per-head
    blockade and co-survival. Hidden states and per-head sigma
    flow between stages. Residual connections provide gradient highway.

    v3.0 additions:
    - OutputEntropyTracker per stage (native, not monkey-patched)
    - get_ecology_state / restore_ecology_state for checkpoint persistence
    - set_ecology_strength propagates warmup_frac to all modules

    Compatible with T3v3Transformer interface for benchmark integration.
    """

    def __init__(self, cfg: T3v3ChainConfig):
        super().__init__()
        self.cfg = cfg

        # Build stages (with optional variable layers per stage)
        self.stages = nn.ModuleList()
        for i in range(cfg.n_stages):
            is_first = (i == 0)
            is_last = (i == cfg.n_stages - 1)
            stage_n_layers = cfg.layers_per_stage[i] if cfg.layers_per_stage else None
            stage = T3v3Stage(cfg, stage_idx=i, is_first=is_first, is_last=is_last,
                              stage_n_layers=stage_n_layers)
            self.stages.append(stage)

        # Share head positions across all stages (single geometry for whole chain)
        # Each stage keeps its own E/I/F/V weights for local sigma computation
        # but blockade distances come from one shared set of positions.
        # This gives blockade the full chain depth as gradient surface area.
        if cfg.shared_positions and cfg.n_stages > 1:
            shared_pos = self.stages[0].head_state.head_positions
            for stage in self.stages[1:]:
                # Replace the later stage's head_positions with stage 0's parameter
                if isinstance(shared_pos, nn.Parameter):
                    # Delete the independent parameter and share stage 0's
                    del stage.head_state.head_positions
                    stage.head_state.head_positions = shared_pos

        # Residual projection (identity since d_model is constant)
        if cfg.use_residual and cfg.n_stages > 1:
            self.residual_proj = nn.Identity()

        # ACT: halt decision
        # Always register buffers if ACT is configured (even if starting disabled)
        # so that act_enabled can be toggled at runtime (e.g., P3 phase).
        if cfg.act_enabled and not cfg.act_strain_halt:
            # Learned halt head (PonderNet-style fallback)
            self.halt_head = nn.Sequential(
                nn.Linear(cfg.d_model, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
        # Ponder cost buffers -- needed whenever ACT might be used
        self.register_buffer("_last_ponder_cost", torch.tensor(0.0))
        self._last_ponder_steps = 1

        # Per-stage strain EMA for adaptive halting (all stages including stage 0)
        if cfg.act_strain_ema_enabled:
            self.register_buffer(
                "_strain_ema", torch.full((cfg.n_stages,), cfg.act_strain_threshold)
            )

        # v3.2: Per-stage entropy delta EMA for adaptive thresholds
        # Initialized to global entropy_halt_threshold for cold-start compatibility.
        # Calibrates during training: stages with larger typical deltas get higher thresholds.
        if cfg.act_adaptive_threshold or cfg.act_entropy_halt:
            self.register_buffer(
                "_entropy_delta_ema",
                torch.full((cfg.n_stages,), cfg.act_entropy_halt_threshold)
            )

        # v3.3: Difficulty predictor — predicts batch difficulty from pre-ponder hidden state
        # Trained retrospectively with actual CE loss. Modulates halt threshold.
        if cfg.act_difficulty_predictor:
            self.difficulty_head = nn.Sequential(
                nn.Linear(cfg.d_model, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            self.register_buffer("_loss_ema", torch.tensor(3.0))  # Initialize to typical CE loss
        self._last_difficulty_pred = None  # Stored during forward for retrospective loss

        # v3.5: Scratchpad-need head — per-token MLP on final stage hidden state.
        # Output [B, S] in [0, 1] = predicted probability that this position will be a
        # high-CE ("hard") commit. Trained retrospectively against per-position CE.
        if getattr(cfg, 'scratchpad_need_predictor', False):
            self.scratchpad_need_head = nn.Sequential(
                nn.Linear(cfg.d_model, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        self._last_scratchpad_pred = None  # [B, S] stored during forward
        self._last_final_hidden = None     # [B, S, D] post-S{K-1} hidden for downstream heads

        # v3.7 (port from v5.3): Dynamic Ω — per-stage EMA shadow of coupling params.
        # The actual nn.Parameters (_coupling_params, _trivector_params) stay as anchors;
        # the shadow holds the running deformed Ω that gets applied during forward.
        if getattr(cfg, 'dynamic_omega_enabled', False) and getattr(cfg, 'hamiltonian_cross_coupling', False):
            for si, stage in enumerate(self.stages):
                hs = stage.head_state
                if hasattr(hs, '_coupling_params'):
                    self.register_buffer(
                        f"_omega_shadow_{si}",
                        hs._coupling_params.data.clone()
                    )
                if hasattr(hs, '_trivector_params'):
                    self.register_buffer(
                        f"_omega_tri_shadow_{si}",
                        hs._trivector_params.data.clone()
                    )
            self.register_buffer("_omega_displacement_ema", torch.tensor(0.0))
            self.register_buffer("_omega_variance_ema", torch.tensor(0.0))

        # Initialize weights
        self.apply(self._init_weights)

        # v3.4: Re-init specialized projections after generic init
        # _init_weights applies normal_(std=0.02) to ALL nn.Linear, which destroys
        # zero-init (key_bias_proj) and identity-init (bond_predictor, inter_stage_predictor).
        for stage in self.stages:
            hs = stage.head_state
            if hasattr(hs, 'key_bias_proj'):
                nn.init.zeros_(hs.key_bias_proj.weight)
                nn.init.zeros_(hs.key_bias_proj.bias)
            if hasattr(hs, 'bond_predictor'):
                nn.init.eye_(hs.bond_predictor.weight)
                nn.init.zeros_(hs.bond_predictor.bias)
            if hasattr(hs, 'inter_stage_predictor'):
                nn.init.eye_(hs.inter_stage_predictor.weight)
                nn.init.zeros_(hs.inter_stage_predictor.bias)

        # Weight tying: last stage output = first stage embed
        if cfg.n_stages > 1:
            self.stages[-1].output_proj.weight = self.stages[0].embed.weight

        # v3.0: Share output weight with all stages for entropy computation
        output_weight = self.stages[-1].output_proj.weight
        for stage in self.stages[:-1]:
            stage._shared_output_weight = output_weight

        self.n_params = sum(p.numel() for p in self.parameters())
        self.params_per_stage = self.n_params // cfg.n_stages

        # Expose last stage's head_state/cosurvival for benchmark compatibility
        # (benchmark checks model.head_state, model.cosurvival)
        self._step_count = self.stages[-1]._stage_step

        # Global sigma buffer for compatibility
        self.register_buffer("_global_sigma", torch.tensor(0.5))

        # Spectral monitoring
        self._spectral_history = []

    @staticmethod
    def _stage_ponder_step(stage, h_in, sigma_loop):
        """Stage forward for gradient-checkpointed ponder steps."""
        return stage(x=None, hidden=h_in, sigma_prior=sigma_loop, update_state=False)

    @property
    def head_state(self):
        """Expose last stage's head state for benchmark compatibility."""
        return self.stages[-1].head_state

    @property
    def cosurvival(self):
        """Expose last stage's co-survival for benchmark compatibility."""
        return self.stages[-1].cosurvival

    @property
    def layers(self):
        """Expose all layers from all stages for spectral analysis hooks."""
        all_layers = []
        for stage in self.stages:
            all_layers.extend(stage.layers)
        return all_layers

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, T3RMSNorm):
            torch.nn.init.ones_(module.weight)

    # ========================================
    # Auxiliary ecology losses
    # ========================================

    def compute_auxiliary_losses(self, step: int = 0,
                                 cpa_warmup_steps: int = 500,
                                 cpa_bond_gate_threshold: float = 0.05) -> dict:
        """Compute all ecology auxiliary losses in one place.

        Returns dict of loss tensors (not yet weighted or divided by GRAD_ACCUM)
        and diagnostic scalars. Training script applies weights.

        Keys:
            sigma_diversity: negative sum of per-stage sigma std (tensor, has grad)
            sigma_antisat: sum of σ(1-σ) across stages (tensor, has grad)
            null_cone: mean Q² across stages (tensor, has grad through coupling)
            null_cone_Q_per_stage: list of [H] Q tensors (diagnostic)
            inter_stage_pc: inter-stage prediction MSE (tensor, has grad)
            cpa_pred: bond prediction loss (tensor, has grad)
            cpa_comp: complementarity loss (tensor, has grad)
            cpa_gate: float gating factor (step ramp × bond strength)
        """
        device = next(self.parameters()).device
        result = {}

        # --- Sigma diversity: encourage spread, penalize collapse ---
        # NOTE (v3.7): _last_head_sigmas is a DETACHED buffer (no gradient).
        # The legacy path below kept for backward compat but produces zero-grad
        # losses. Use compute_sigma_health() with chain_states for real gradient
        # signal — that's what v3.7 wires into the training loop.
        sigma_div = torch.tensor(0.0, device=device)
        sigma_antisat = torch.tensor(0.0, device=device)
        for stage in self.stages:
            s = getattr(stage.head_state, '_last_head_sigmas', None)
            if s is not None:
                sigma_div = sigma_div - s.std()
                sigma_antisat = sigma_antisat + (s * (1 - s)).sum()
        result['sigma_diversity'] = sigma_div
        result['sigma_antisat'] = sigma_antisat

        # --- Null cone Q(v) loss: teaches coupling rotation toward Q=0 ---
        null_cone_loss = torch.tensor(0.0, device=device)
        Q_per_stage = []
        for stage in self.stages:
            hs = stage.head_state
            if hasattr(hs, '_entropy_ema') and hasattr(hs, 'compute_null_cone_Q'):
                Q, Q_sq = hs.compute_null_cone_Q()
                null_cone_loss = null_cone_loss + Q_sq
                Q_per_stage.append(Q.detach())
        result['null_cone'] = null_cone_loss
        result['null_cone_Q_per_stage'] = Q_per_stage

        # --- Inter-stage predictive coding ---
        inter_loss = torch.tensor(0.0, device=device)
        for i in range(len(self.stages) - 1):
            hs = self.stages[i].head_state
            hs_next = self.stages[i + 1].head_state
            if hasattr(hs, 'inter_stage_predictor'):
                next_prims = hs_next.get_current_primitives()
                inter_loss = inter_loss + hs.compute_inter_stage_loss(next_prims)
        result['inter_stage_pc'] = inter_loss

        # --- CPA: cooperative prediction + complementarity ---
        cpa_pred = torch.tensor(0.0, device=device)
        cpa_comp = torch.tensor(0.0, device=device)
        cpa_step_ramp = min(1.0, step / max(cpa_warmup_steps, 1))
        mean_bond_strength = 0.0
        for stage in self.stages:
            if hasattr(stage.head_state, 'bond_predictor') and self.cfg.cosurvival_enabled:
                cs_matrix = stage.cosurvival.cosurvival
                mean_bond_strength = max(mean_bond_strength, cs_matrix.abs().mean().item())
                cpa_pred = cpa_pred + stage.head_state.compute_bond_prediction_loss(cs_matrix)
                cpa_comp = cpa_comp + stage.head_state.compute_complementarity_loss(cs_matrix)
        cpa_bond_gate = min(1.0, mean_bond_strength / max(cpa_bond_gate_threshold, 1e-8))
        result['cpa_pred'] = cpa_pred
        result['cpa_comp'] = cpa_comp
        result['cpa_gate'] = cpa_step_ramp * cpa_bond_gate

        return result

    # ========================================
    # Ecology strength / warmup management
    # ========================================

    def set_ecology_strength(self, strength: float):
        """Set ecology strength AND warmup_frac for all modules.

        Controls how strongly ecology modulations (attention temperature, FFN gating,
        locality bias, output temperature, inter-stage residual) are applied.
        0.0 = bypass (pretrained behavior), 1.0 = full ecology.

        v3.0: Also sets _warmup_frac on each stage so that grounded primitive
        updates and sigma computation can smoothly ramp during warmup.

        Critical for transfer learning: pretrained models need gradual ecology ramp
        to avoid immediate PPL explosion from FFN scaling (0.65x at sigma=0.5)
        compounding over deep (32+ layer) networks.
        """
        self._ecology_strength = strength
        for stage in self.stages:
            stage._ecology_strength = strength
            stage._warmup_frac = strength  # v3.0: warmup_frac = ecology strength
            for layer in stage.layers:
                layer._ecology_strength = strength
                layer.attention._ecology_strength = strength

        # v3.4.2: Temporal cache — sigma MLP trains but output=0.5 during warmup
        if getattr(self.cfg, 'sigma_temporal_cache', False):
            cache_steps = getattr(self.cfg, 'sigma_temporal_cache_steps', 0)
            if cache_steps == 0:
                # Use warmup duration: cache active while strength < 1.0
                cache_active = (strength < 1.0)
            else:
                # Step-based: set by set_temporal_cache_step()
                cache_active = getattr(self, '_sigma_cache_active_override', False)
            for stage in self.stages:
                stage.head_state._sigma_temporal_cache_active = cache_active

    def set_temporal_cache_step(self, step: int):
        """For step-based temporal cache: call each training step."""
        cache_steps = getattr(self.cfg, 'sigma_temporal_cache_steps', 0)
        if cache_steps > 0 and getattr(self.cfg, 'sigma_temporal_cache', False):
            self._sigma_cache_active_override = (step < cache_steps)
            for stage in self.stages:
                stage.head_state._sigma_temporal_cache_active = self._sigma_cache_active_override

    # ========================================
    # Ecology state save/restore (v3.0)
    # ========================================

    def get_ecology_state(self) -> dict:
        """Capture full ecology state for checkpoint persistence.

        Returns a dict containing per-stage OutputEntropyTracker state
        and HeadState EMA buffers. These are non-parameter buffers that
        need explicit save/restore across training restarts.

        v3.0: Replaces the per-training-script save_ecology_state / load_ecology_state
        monkey-patch pattern from v2.5.
        """
        eco = {'trackers': [], 'head_states': []}
        for stage in self.stages:
            # Tracker state
            eco['trackers'].append({
                'output_entropy_ema': stage.entropy_tracker.output_entropy_ema.cpu().clone(),
                'valence_velocity_ema': stage.entropy_tracker.valence_velocity_ema.cpu().clone(),
                '_prev_entropy': stage.entropy_tracker._prev_entropy.cpu().clone(),
            })
            # Head state EMAs
            hs = stage.head_state
            hs_state = {}
            for key in ['_entropy_ema', '_friction_ema', '_valence_ema', '_entropy_prev',
                        '_coherence_ema', '_chronos_ema', '_intensity_ema', '_last_intensity',
                        '_attn_fast_ema', '_attn_slow_ema', '_valence_init_count']:
                if hasattr(hs, key):
                    hs_state[key] = getattr(hs, key).cpu().clone()
            eco['head_states'].append(hs_state)
        # v3.2: Per-stage entropy delta EMA for adaptive thresholds
        if hasattr(self, '_entropy_delta_ema'):
            eco['_entropy_delta_ema'] = self._entropy_delta_ema.cpu().clone()
        # v3.3: Loss EMA for difficulty normalization
        if hasattr(self, '_loss_ema'):
            eco['_loss_ema'] = self._loss_ema.cpu().clone()
        return eco

    def reset_ecology_emas(self) -> dict:
        """v3.5: Re-initialize all ecology EMA buffers to defaults.

        Used by --reset_ecology_emas to force the model to rebuild its
        self-understanding under new dynamics (e.g. v3.5 null cone) instead
        of loading a frozen prior from a pre-v3.5 checkpoint.

        Resets per-stage and per-head:
          primitives: _entropy_ema=0.5, _entropy_prev=0.5, _friction_ema=0.1,
                     _valence_ema=0.5, _coherence_ema=0.5, _chronos_ema=0.5,
                     _intensity_ema=0.5, _last_intensity=0.5,
                     _attn_fast_ema=0.5, _attn_slow_ema=0.5
          tracker:    output_entropy_ema=0.5, valence_velocity_ema=0.0, _prev_entropy=0.0
        Resets chain-level:
          _entropy_delta_ema = act_entropy_halt_threshold (per-stage)  ← load-bearing for output-entropy halt
          _loss_ema = 3.0 (typical CE)
          _global_sigma = 0.5
          _strain_ema = act_strain_threshold (per-stage; legacy — strain halt path is dead in v3.3+ output-entropy configs)

        Returns a dict listing what was reset (for logging).
        """
        reset_log = {'stages': 0, 'fields': []}
        per_head_defaults = {
            '_entropy_ema': 0.5, '_entropy_prev': 0.5, '_friction_ema': 0.1,
            '_valence_ema': 0.5, '_coherence_ema': 0.5, '_chronos_ema': 0.5,
            '_intensity_ema': 0.5, '_last_intensity': 0.5,
            '_attn_fast_ema': 0.5, '_attn_slow_ema': 0.5,
        }
        for stage in self.stages:
            reset_log['stages'] += 1
            hs = stage.head_state
            for name, val in per_head_defaults.items():
                if hasattr(hs, name):
                    getattr(hs, name).fill_(val)
            if hasattr(hs, '_valence_init_count'):
                hs._valence_init_count.zero_()
            tr = stage.entropy_tracker
            tr.output_entropy_ema.fill_(0.5)
            tr.valence_velocity_ema.zero_()
            tr._prev_entropy.zero_()
        if hasattr(self, '_entropy_delta_ema'):
            self._entropy_delta_ema.fill_(self.cfg.act_entropy_halt_threshold)
        if hasattr(self, '_strain_ema'):
            self._strain_ema.fill_(self.cfg.act_strain_threshold)
        if hasattr(self, '_loss_ema'):
            self._loss_ema.fill_(3.0)
        if hasattr(self, '_global_sigma'):
            self._global_sigma.fill_(0.5)
        reset_log['fields'] = list(per_head_defaults.keys()) + [
            'output_entropy_ema', 'valence_velocity_ema', '_prev_entropy',
            '_entropy_delta_ema', '_strain_ema', '_loss_ema', '_global_sigma',
        ]
        return reset_log

    def restore_ecology_state(self, eco: dict):
        """Restore ecology state from checkpoint.

        Args:
            eco: dict from get_ecology_state(), typically stored as
                 checkpoint['ecology_state'].
        """
        for i, stage in enumerate(self.stages):
            if i < len(eco.get('trackers', [])):
                t_state = eco['trackers'][i]
                dev = stage.entropy_tracker.output_entropy_ema.device
                stage.entropy_tracker.output_entropy_ema.copy_(t_state['output_entropy_ema'].to(dev))
                stage.entropy_tracker.valence_velocity_ema.copy_(t_state['valence_velocity_ema'].to(dev))
                stage.entropy_tracker._prev_entropy.copy_(t_state['_prev_entropy'].to(dev))

            if i < len(eco.get('head_states', [])):
                hs_state = eco['head_states'][i]
                hs = stage.head_state
                dev = hs._entropy_ema.device if hasattr(hs, '_entropy_ema') else next(hs.parameters()).device
                for key in ['_entropy_ema', '_friction_ema', '_valence_ema', '_entropy_prev',
                            '_coherence_ema', '_chronos_ema', '_intensity_ema', '_last_intensity',
                            '_attn_fast_ema', '_attn_slow_ema', '_valence_init_count']:
                    if key in hs_state and hasattr(hs, key):
                        getattr(hs, key).copy_(hs_state[key].to(dev))
        # v3.2: Restore per-stage entropy delta EMA
        if '_entropy_delta_ema' in eco and hasattr(self, '_entropy_delta_ema'):
            dev = self._entropy_delta_ema.device
            self._entropy_delta_ema.copy_(eco['_entropy_delta_ema'].to(dev))
        # v3.3: Restore loss EMA
        if '_loss_ema' in eco and hasattr(self, '_loss_ema'):
            self._loss_ema.copy_(eco['_loss_ema'].to(self._loss_ema.device))

    # ========================================
    # Forward pass (non-ACT)
    # ========================================

    def forward(
        self,
        input_ids: torch.Tensor,
        return_state: bool = False,
        return_chain_state: bool = False,
        return_intermediate_logits: bool = False,
        update_state: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass through the chain.

        input_ids: [batch, seq_len]

        Args:
            return_intermediate_logits: If True, project each stage's hidden state
                through the tied output head to compute per-stage logits. Returns
                dict with 'intermediate_logits' key containing list of [batch, seq, vocab]
                tensors. Used for per-stage intermediate supervision during training.

        Returns:
            logits: [batch, seq_len, vocab]
            state: aggregated state dict (if return_state or return_intermediate_logits)
        """
        # v3.7 (port from v5.3): Apply deformed Ω shadow before forward (fixed during ponder).
        # We restore in the finally block; update happens at end of training-mode forward.
        omega_originals = self._apply_omega_shadow() if self.training else None

        try:
            if self.cfg.act_enabled:
                if self.cfg.act_per_stage:
                    result = self._act_perstage_forward(input_ids, return_state, return_chain_state, update_state)
                else:
                    result = self._act_forward(input_ids, return_state, return_chain_state, update_state)
                self._restore_omega(omega_originals)
                if self.training and update_state:
                    self._update_dynamic_omega()
                return result
        except Exception:
            self._restore_omega(omega_originals)
            raise

        bypass = getattr(self.cfg, 'bypass_ecology', False)
        chain_states = []
        hidden = None
        sigma_prior = None
        residual = None
        intermediate_logits = [] if return_intermediate_logits else None
        self._inter_stage_pc_errors = []  # v3.1e: Reset PC error cache

        # v3.3 opt: Invalidate distance/blockade caches
        for stage in self.stages:
            stage.head_state.invalidate_distance_cache()
            if self.cfg.cosurvival_enabled:
                stage.cosurvival.invalidate_blockade_cache()

        # Get shared output head for intermediate supervision
        output_proj = self.stages[-1].output_proj
        final_norm = self.stages[-1].norm

        for i, stage in enumerate(self.stages):
            if i == 0:
                # First stage: embed input_ids
                hidden, logits, stage_state = stage(
                    x=input_ids,
                    hidden=None,
                    sigma_prior=None,
                    update_state=update_state,
                )
                residual = hidden  # Save for skip connection
            else:
                # Later stages: receive hidden + sigma-flow
                hidden_in = hidden

                # Residual from first stage (skip in bypass mode -- Qwen has no inter-stage residual)
                if not bypass and self.cfg.use_residual and residual is not None:
                    eco = getattr(self, '_ecology_strength', 1.0)
                    hidden_in = hidden_in + eco * self.residual_proj(residual)

                hidden, logits, stage_state = stage(
                    x=None,
                    hidden=hidden_in,
                    sigma_prior=sigma_prior,
                    update_state=update_state,
                )

            # Per-stage intermediate supervision: project hidden -> logits via tied output head
            if return_intermediate_logits:
                # Apply final norm then output projection (same head used for all stages)
                h_normed = final_norm(hidden)
                stage_logits = apply_logit_softcap(output_proj(h_normed), self.cfg.logit_softcap)
                intermediate_logits.append(stage_logits)

            # sigma-flow: pass this stage's per-head sigma to next stage
            sigma_prior = stage_state.get("head_sigmas_tensor", None)

            # v3.1e: Inter-stage predictive coding (non-ACT path)
            # Exp B: no_grad removed so pc_error backprops into the predictor.
            # actual_prims comes from EMA register_buffers (no grad path); the predictor
            # input is .detach()'d in predict_next_stage_primitives, so gradients flow
            # only into inter_stage_predictor weights, not back into chain dynamics.
            if self.cfg.inter_stage_pc and i < len(self.stages) - 1 and self.training:
                if hasattr(stage.head_state, 'inter_stage_predictor'):
                    next_hs = self.stages[i + 1].head_state
                    actual_prims = next_hs.get_current_primitives()  # [H, 6]
                    predicted = stage.head_state.predict_next_stage_primitives()  # [H, 6]
                    pc_error = (predicted - actual_prims).pow(2).mean()
                    self._inter_stage_pc_errors.append(pc_error)

            chain_states.append(stage_state)

        # Update global sigma
        if chain_states:
            self._update_global_sigma(chain_states[-1].get("global_sigma", 0.5))

        # v3.7: restore Ω shadow originals + update for non-ACT path
        self._restore_omega(omega_originals)
        if self.training and update_state:
            self._update_dynamic_omega()

        if return_state or return_chain_state or return_intermediate_logits:
            state = self._aggregate_state(chain_states)
            if return_chain_state:
                state["chain_states"] = chain_states
            if return_intermediate_logits:
                state["intermediate_logits"] = intermediate_logits
            return logits, state

        return logits

    # ========================================
    # ACT forward (whole-chain pondering)
    # ========================================

    def _act_forward(
        self,
        input_ids: torch.Tensor,
        return_state: bool = False,
        return_chain_state: bool = False,
        update_state: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass with Adaptive Computation Time (pondering).

        Stage 0 runs once (embedding). Stages 1+ run in a ponder loop:
        each iteration re-processes through all computation stages,
        accumulating outputs weighted by halting probability.

        Halt decision:
          - strain mode (default): halt when delta-sigma converges (ecology reaches
            equilibrium). No learned parameters -- pure T3 physics.
          - learned mode: PonderNet-style halt_head MLP.
        """
        device = input_ids.device
        use_strain = self.cfg.act_strain_halt

        # v3.3 opt: Invalidate distance/blockade caches
        for stage in self.stages:
            stage.head_state.invalidate_distance_cache()
            if self.cfg.cosurvival_enabled:
                stage.cosurvival.invalidate_blockade_cache()

        # === Stage 0: embed once ===
        hidden, _, stage0_state = self.stages[0](
            x=input_ids, hidden=None, sigma_prior=None, update_state=update_state,
        )
        residual = hidden  # Embedding residual for skip connections
        sigma_prior = stage0_state.get("head_sigmas_tensor", None)

        # === ACT ponder loop through stages 1+ ===
        p_running = torch.ones(1, device=device)  # Remaining probability mass
        h_accum = torch.zeros_like(hidden)
        # logits_accum removed -- recompute from h_accum after loop (saves ~1-2GB VRAM)
        halt_probs = []
        strain_values = []
        last_chain_states = []

        h_loop = hidden  # Input to first ponder iteration
        prev_sigma = sigma_prior  # Strain baseline: stage 0's ecology

        for t in range(self.cfg.act_max_steps):
            # Run through stages 1 to K
            h = h_loop
            sigma = sigma_prior
            chain_states_t = []

            for i, stage in enumerate(self.stages[1:], 1):
                h_in = h
                if self.cfg.use_residual and residual is not None:
                    h_in = h_in + self.residual_proj(residual)

                h, logits_t, state_t = stage(
                    x=None, hidden=h_in, sigma_prior=sigma,
                    update_state=(update_state and t == 0),
                )
                sigma = state_t.get("head_sigmas_tensor", None)
                chain_states_t.append(state_t)

            # --- Halt decision ---
            if use_strain:
                # Strain = mean |delta-sigma| across heads
                # Measures how much the ecology shifted this ponder step.
                # When strain -> 0, the ecology has found its fixed point.
                current_sigma = sigma  # From last computation stage
                strain_t = (current_sigma - prev_sigma).abs().mean()

                # Loop 1 (t=0): ecology establishment -- always massive strain
                # because stage 0 sigma -> stages 1-K sigma is a phase transition.
                # Skip halt decision; start adaptive halting from loop 2.
                if t == 0 and self.cfg.act_skip_first_halt:
                    lambda_t = torch.zeros(1, device=device).squeeze()
                else:
                    # lambda = sigmoid((threshold - strain) / temperature)
                    #   strain >> threshold -> lambda ~ 0 -> keep pondering
                    #   strain << threshold -> lambda ~ 1 -> halt
                    lambda_t = torch.sigmoid(
                        (self.cfg.act_strain_threshold - strain_t)
                        / self.cfg.act_strain_temperature
                    )
                strain_values.append(strain_t)
                prev_sigma = current_sigma  # Update baseline for next step
            else:
                # Learned halt head (PonderNet fallback)
                halt_input = h.mean(dim=(0, 1))  # [d_model]
                lambda_t = torch.sigmoid(self.halt_head(halt_input)).squeeze()

            # Probability of halting at step t: p(t) = lambda_t x prod_{s<t}(1-lambda_s)
            if t < self.cfg.act_max_steps - 1:
                p_t = p_running * lambda_t
            else:
                p_t = p_running  # Assign ALL remaining mass to final step

            # Accumulate weighted outputs (hidden only -- logits recomputed after loop)
            h_accum = h_accum + p_t * h
            del logits_t  # Free vocab-sized tensor immediately

            halt_probs.append(p_t)
            p_running = p_running * (1 - lambda_t)
            last_chain_states = chain_states_t

            # Prepare next iteration input (feed output back)
            h_loop = h

            # Early exit when remaining probability is negligible
            if (p_running < self.cfg.act_halt_epsilon):
                break

        # Compute and store ponder cost
        ponder_cost = self._compute_ponder_cost(halt_probs, strain_values)
        self._store_ponder_state(ponder_cost, len(halt_probs))
        self._ponder_cost_live = ponder_cost  # Differentiable for training loop

        # Update global sigma
        if last_chain_states:
            self._update_global_sigma(last_chain_states[-1].get("global_sigma", 0.5))

        # Recompute logits from h_accum (linear proj, mathematically equivalent)
        last_stage = self.stages[-1]
        if last_chain_states:
            last_sigmas = last_chain_states[-1].get("head_sigmas_tensor", None)
            if last_sigmas is not None:
                eco = getattr(self, '_ecology_strength', 1.0)
                sigma_mean = last_sigmas.mean()
                raw_temp = torch.clamp(1.5 - sigma_mean * 0.8, 0.5, 2.0)
                temp = 1.0 + eco * (raw_temp - 1.0)
                logits = last_stage.output_proj(h_accum) / temp
            else:
                logits = last_stage.output_proj(h_accum)
        else:
            logits = last_stage.output_proj(h_accum)
        logits = apply_logit_softcap(logits, self.cfg.logit_softcap)

        if return_state or return_chain_state:
            all_states = [stage0_state] + last_chain_states
            state = self._aggregate_state(all_states)
            state["act_ponder_steps"] = len(halt_probs)
            state["act_ponder_cost"] = ponder_cost.item()
            state["act_halt_probs"] = [p.item() for p in halt_probs]
            if strain_values:
                state["act_strain_values"] = [s.item() for s in strain_values]
            if return_chain_state:
                state["chain_states"] = all_states
            return logits, state

        return logits

    # ========================================
    # ACT forward (per-stage pondering)
    # ========================================

    def _act_perstage_forward(
        self,
        input_ids: torch.Tensor,
        return_state: bool = False,
        return_chain_state: bool = False,
        update_state: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Per-stage ACT: each computation stage independently ponder-loops
        until its local ecology converges.

        Design:
          Stage 0: embed once (no pondering -- it's just lookup)
          Stage i (i=1..K):
            sigma_in = sigma from previous stage's converged output
            for t in range(max_ponder_per_stage):
              h, sigma_out = stage_i(h_in, sigma_in)
              strain_i = |sigma_out - sigma_in|.mean()
              halt decision based on strain_i
              accumulate h weighted by halt probability
              sigma_in = sigma_out  (feed converged sigma back)
            h = accumulated output -> becomes input to stage i+1

        This gives adaptive DEPTH: easy-to-process stages halt after 1 step,
        complex stages ponder longer. The ecology finds per-stage equilibrium.
        """
        device = input_ids.device
        max_per_stage = self.cfg.act_per_stage_max
        temperature = self.cfg.act_strain_temperature
        eps = self.cfg.act_halt_epsilon
        use_ema = self.cfg.act_strain_ema_enabled

        # v3.1e: Reset inter-stage PC error cache
        self._inter_stage_pc_errors = []

        # v3.3 opt: Invalidate distance/blockade caches at start of forward pass
        # so they're recomputed once then shared across all ponder steps
        for stage in self.stages:
            stage.head_state.invalidate_distance_cache()
            if self.cfg.cosurvival_enabled:
                stage.cosurvival.invalidate_blockade_cache()

        # === Pre-step: embed via stage 0 ===
        # Stage 0 always embeds first (converts input_ids -> hidden).
        # Then it enters the ponder loop like all other stages.
        hidden, _, stage0_state = self.stages[0](
            x=input_ids, hidden=None, sigma_prior=None, update_state=update_state,
        )
        residual = hidden  # Raw embedding for residual connections (frozen through pondering)
        sigma_prior = stage0_state.get("head_sigmas_tensor", None)

        # Track per-stage ponder info
        all_chain_states = []
        per_stage_ponder_steps = []
        per_stage_strains = []
        per_stage_entropy_deltas = []  # Track entropy improvement per ponder step per stage
        total_strain = torch.tensor(0.0, device=device)
        total_halt_probs = []

        # Output-entropy halt setup: grab output_proj weight once
        use_entropy_halt = self.cfg.act_entropy_halt
        if use_entropy_halt:
            output_weight = self.stages[-1].output_proj.weight  # [V, D]
            # Cast to bf16 for probing — under no_grad, precision is irrelevant
            # and bf16 halves the F.linear cost on [B, n_probe, V]
            if output_weight.dtype != torch.bfloat16:
                output_weight = output_weight.to(torch.bfloat16)
            entropy_threshold = self.cfg.act_entropy_halt_threshold
            entropy_temperature = self.cfg.act_entropy_halt_temperature
            max_entropy = math.log(output_weight.shape[0])  # log(V) for Triton kernel

        # v3.3: Difficulty prediction from pre-ponder hidden state
        use_difficulty = self.cfg.act_difficulty_predictor and hasattr(self, 'difficulty_head')
        if use_difficulty:
            h_pooled = hidden.mean(dim=1)  # [B, D]
            self._last_difficulty_pred = self.difficulty_head(h_pooled).squeeze(-1)  # [B]
            difficulty_scalar = self._last_difficulty_pred.mean().detach()  # keep as tensor, avoid CUDA sync
        else:
            difficulty_scalar = 0.0

        # v3.3: Confidence floor setup
        use_confidence_floor = self.cfg.act_confidence_floor > 0 and use_entropy_halt

        # === Per-stage ponder loops (ALL stages 0..K) ===
        for stage_idx, stage in enumerate(self.stages):
            # v3.3 opt: Skip redundant entropy tracker inside stage.forward()
            # during ACT — we feed converged entropy after the ponder loop instead
            if use_entropy_halt and self.training:
                stage._skip_entropy_tracker = True
            # PonderNet accumulators for this stage
            p_running = torch.ones(1, device=device)
            h_accum = torch.zeros_like(hidden)
            # logits_accum removed -- recompute from h_accum after loop (saves ~1-2GB VRAM)
            stage_halt_probs = []
            stage_strains = []

            # Output-entropy halt: track previous entropy for delta computation
            prev_entropy = None
            stage_entropy_deltas = []  # Track per-step entropy improvement

            # Input to this stage's ponder loop
            h_loop = hidden
            sigma_loop = sigma_prior  # sigma flowing in from previous stage

            # v3.4: Reset ponder-step entropy tracking for live ecology (Phase 1 CAC)
            stage._prev_ponder_entropy = None

            # Cache probe_idx for this stage (reused in preponder + ponder loop)
            if use_entropy_halt:
                B, S, D = hidden.shape
                n_probe = min(self.cfg.act_n_probe_positions, S)
                probe_idx = torch.linspace(0, S - 1, n_probe).long().to(device)

            # v3.2: Pre-ponder entropy baseline — compute entropy of stage INPUT
            # before the ponder loop. This gives prev_entropy at t=0 so the first
            # ponder step can make a real halt decision (min_ponder drops from 3 to 1).
            use_preponder = self.cfg.act_preponder_baseline and use_entropy_halt
            if use_preponder:
                with torch.no_grad():
                    h_probe_pre = hidden.index_select(1, probe_idx).to(output_weight.dtype)
                    probe_logits_pre = F.linear(h_probe_pre, output_weight)
                    probe_probs_pre = F.softmax(probe_logits_pre.float(), dim=-1)
                    log_probs_pre = torch.log(probe_probs_pre + 1e-10)
                    prev_entropy = -(probe_probs_pre * log_probs_pre).sum(dim=-1).mean()  # keep as tensor

            # v3.2: Per-stage adaptive threshold lookup
            use_adaptive = self.cfg.act_adaptive_threshold and use_entropy_halt
            if use_adaptive and hasattr(self, '_entropy_delta_ema'):
                stage_ent_threshold = max(
                    self._entropy_delta_ema[stage_idx].item() * self.cfg.act_adaptive_margin,
                    self.cfg.act_adaptive_floor,
                )
            else:
                stage_ent_threshold = entropy_threshold if use_entropy_halt else None

            # v3.3: Difficulty modulates halt threshold — hard inputs → lower threshold → ponder more
            if use_difficulty and stage_ent_threshold is not None and difficulty_scalar > 0:
                stage_ent_threshold = stage_ent_threshold * (1.0 - difficulty_scalar * self.cfg.act_difficulty_scale)
                stage_ent_threshold = max(stage_ent_threshold, self.cfg.act_adaptive_floor)

            # v3.2: Hard halt at eval — use representation directly when lambda_t > 0.5
            use_hard_halt = self.cfg.act_hard_halt_eval and not self.training

            for t in range(max_per_stage):
                h_in = h_loop
                # Residual skip connection from raw embedding (stage 0 doesn't add to itself)
                if stage_idx > 0 and self.cfg.use_residual and residual is not None:
                    h_in = h_in + self.residual_proj(residual)

                # Run this stage (stage 0 already embedded -- ponder uses hidden path)
                update_state_t = (update_state and t == 0 and stage_idx > 0)
                use_ckpt = (self.cfg.act_gradient_checkpointing and t > 0
                            and self.training)
                if use_ckpt:
                    # Gradient checkpointing: don't store intermediates, recompute on backward.
                    # Saves ~0.3-0.8 GB per ponder step at 210M scale.
                    h_out, logits_t, state_t = torch.utils.checkpoint.checkpoint(
                        self._stage_ponder_step, stage, h_in, sigma_loop,
                        use_reentrant=False,
                    )
                else:
                    h_out, logits_t, state_t = stage(
                        x=None, hidden=h_in, sigma_prior=sigma_loop,
                        update_state=update_state_t,
                    )
                del logits_t  # Free vocab-sized tensor immediately
                sigma_out = state_t.get("head_sigmas_tensor", None)

                # v3.4: Live ecology update during pondering (Phase 1 CAC)
                # Update E (entropy) and F (friction) between ponder steps using signals
                # already computed by the stage forward pass. Makes 3/6 primitives live
                # (E, I, F) instead of 1/6 (I only). Zero extra compute — just reconnecting wires.
                if t > 0 and self.cfg.act_live_ecology:
                    head_ent_tensors = state_t.get("head_entropy_tensors", None)
                    if head_ent_tensors and len(head_ent_tensors) > 0:
                        with torch.no_grad():
                            hs = stage.head_state
                            ponder_alpha = self.cfg.act_live_ecology_alpha

                            # Average head entropy across layers of this stage: [n_heads]
                            avg_entropy = torch.stack(head_ent_tensors).mean(dim=0)
                            # Normalize to [0, 1] — same as update_grounded_primitives
                            max_ent = math.log(max(hidden.shape[1], 2))
                            avg_entropy_norm = (avg_entropy / (max_ent + 1e-8)).clamp(0, 1).to(hs._entropy_ema.dtype)

                            # E update: lerp entropy EMA (ponder alpha >> training alpha 0.05)
                            hs._entropy_ema.lerp_(avg_entropy_norm, ponder_alpha)

                            # F update: |delta_entropy| as friction proxy
                            if stage._prev_ponder_entropy is not None:
                                delta = (avg_entropy_norm - stage._prev_ponder_entropy).abs()
                                hs._friction_ema.lerp_(delta, ponder_alpha)

                            stage._prev_ponder_entropy = avg_entropy_norm.clone()

                # Per-stage strain: how much did THIS stage's ecology shift?
                if sigma_out is not None and sigma_loop is not None:
                    strain_t = (sigma_out - sigma_loop).abs().mean()
                else:
                    strain_t = torch.tensor(0.0, device=device)

                # Halt decision
                # v3.2: With preponder baseline, skip_first_halt is bypassed because
                # prev_entropy is already set from the input hidden state.
                skip_halt = (t == 0 and self.cfg.act_skip_first_halt and not use_preponder)
                if skip_halt:
                    lambda_t = torch.zeros(1, device=device).squeeze()
                elif use_entropy_halt:
                    # Output-entropy halt: probe prediction confidence directly.
                    # v3.3 Triton: fused kernel avoids [B, 2, V] materialization.
                    # Falls back to PyTorch if confidence_floor is active (needs probe_probs).
                    with torch.no_grad():
                        h_probe = h_out.index_select(1, probe_idx).to(output_weight.dtype)  # [B, 2, D]
                        if self.cfg.use_triton_kernels and HAS_TRITON and not use_confidence_floor:
                            entropy = fused_entropy_probe(
                                h_probe, output_weight, max_entropy,
                                normalize=False,  # raw entropy for delta comparison
                            )
                        else:
                            # PyTorch reference path
                            probe_logits = F.linear(h_probe, output_weight)  # [B, 2, V]
                            probe_probs = F.softmax(probe_logits.float(), dim=-1)
                            log_probs = torch.log(probe_probs + 1e-10)
                            entropy = -(probe_probs * log_probs).sum(dim=-1).mean()  # scalar

                    if prev_entropy is not None:
                        # Improvement = entropy decrease (positive = more confident)
                        delta = prev_entropy - entropy
                        stage_entropy_deltas.append(delta.detach())
                        # v3.2: Use per-stage adaptive threshold if enabled
                        lambda_t = torch.sigmoid(
                            (stage_ent_threshold - delta) / entropy_temperature
                        )
                    else:
                        # First comparison step: always continue (need baseline)
                        lambda_t = torch.zeros(1, device=device).squeeze()
                    prev_entropy = entropy.detach()  # keep as tensor, avoid CUDA sync
                else:
                    # Adaptive EMA threshold: halt when strain drops below ema * margin
                    if use_ema:
                        ema_val = self._strain_ema[stage_idx]
                        effective_threshold = ema_val * self.cfg.act_strain_ema_margin
                    else:
                        effective_threshold = self.cfg.act_strain_threshold
                    lambda_t = torch.sigmoid(
                        (effective_threshold - strain_t) / temperature
                    )

                # v3.3: Confidence floor — override halt if model isn't confident enough
                # Uses probe_probs already computed for entropy halt (zero extra cost)
                if use_confidence_floor and (lambda_t > 0.5) and use_entropy_halt:
                    max_prob = probe_probs.max(dim=-1).values.mean()
                    if max_prob < self.cfg.act_confidence_floor:
                        lambda_t = torch.zeros(1, device=device).squeeze()  # Force continue

                # v3.2: Hard halt at eval — use this step's representation directly
                if use_hard_halt and (lambda_t > 0.5) and t < max_per_stage - 1:
                    h_accum = h_out  # Direct assignment, no weighted average
                    stage_halt_probs.append(torch.ones(1, device=device).squeeze())
                    stage_strains.append(strain_t)
                    total_strain = total_strain + strain_t
                    sigma_loop = sigma_out
                    break

                # Probability of halting at step t (soft PonderNet accumulation)
                if t < max_per_stage - 1:
                    p_t = p_running * lambda_t
                else:
                    p_t = p_running  # Assign ALL remaining mass to final step

                # Accumulate weighted output (hidden only -- logits recomputed after loop)
                h_accum = h_accum + p_t * h_out

                stage_halt_probs.append(p_t)
                stage_strains.append(strain_t)
                total_strain = total_strain + strain_t

                p_running = p_running * (1 - lambda_t)

                # Feed output back for next ponder iteration
                h_loop = h_out
                sigma_loop = sigma_out

                # Early exit when remaining probability is negligible
                if (p_running < eps):
                    break

            # v3.5: Allow EMA updates during eval when eval_live_primitives is on
            _eco_update = self.training or getattr(self.cfg, 'eval_live_primitives', False)

            # Update per-stage strain EMA
            if use_ema and _eco_update and stage_strains:
                with torch.no_grad():
                    mean_strain = torch.stack(stage_strains).mean()
                    decay = self.cfg.act_strain_ema_decay
                    self._strain_ema[stage_idx] = (
                        decay * self._strain_ema[stage_idx] + (1 - decay) * mean_strain
                    )

            # v3.2: Update per-stage entropy delta EMA
            if use_entropy_halt and _eco_update and stage_entropy_deltas and hasattr(self, '_entropy_delta_ema'):
                with torch.no_grad():
                    mean_delta = torch.stack([d.abs() if isinstance(d, torch.Tensor) else torch.tensor(abs(d), device=device) for d in stage_entropy_deltas]).mean()
                    decay = self.cfg.act_adaptive_ema_decay
                    self._entropy_delta_ema[stage_idx] = (
                        decay * self._entropy_delta_ema[stage_idx] + (1 - decay) * mean_delta
                    )

            # v3.3 opt: Feed last ACT entropy into OutputEntropyTracker
            if use_entropy_halt and _eco_update and prev_entropy is not None:
                with torch.no_grad():
                    max_ent = stage.entropy_tracker.max_entropy
                    ent_norm_t = prev_entropy / max_ent if isinstance(prev_entropy, torch.Tensor) else torch.tensor(prev_entropy / max_ent, device=device)
                    alpha_ema = 1.0 - stage.entropy_tracker.decay
                    stage.entropy_tracker.output_entropy_ema.lerp_(ent_norm_t, alpha_ema)
                    prev_ent_val = stage.entropy_tracker._prev_entropy
                    if prev_ent_val > 0:
                        delta = prev_ent_val - ent_norm_t
                        stage.entropy_tracker.valence_velocity_ema.lerp_(delta, 0.01)
                    stage.entropy_tracker._prev_entropy.fill_(ent_norm_t)

            # Clear skip flag
            if use_entropy_halt and _eco_update:
                stage._skip_entropy_tracker = False

            # v3.1e: Inter-stage predictive coding — modulate sigma-flow by prediction accuracy
            # When this stage's predictor is wrong about the next stage, reduce sigma blend
            # (less trust in inherited ecological state). Prediction errors cached for grad scaling.
            # Exp B: no_grad removed (see non-ACT path comment for safety argument).
            if self.cfg.inter_stage_pc and stage_idx < len(self.stages) - 1 and self.training:
                if hasattr(stage.head_state, 'inter_stage_predictor'):
                    next_hs = self.stages[stage_idx + 1].head_state
                    actual_prims = next_hs.get_current_primitives()  # [H, 6]
                    predicted = stage.head_state.predict_next_stage_primitives()  # [H, 6]
                    pc_error = (predicted - actual_prims).pow(2).mean()
                    self._inter_stage_pc_errors.append(pc_error)

            # This stage's converged output -> next stage's input
            hidden = h_accum
            sigma_prior = sigma_out  # Pass LAST sigma (not accumulated) to next stage
            # Recompute logits from h_accum (linear proj, mathematically equivalent)
            if stage.is_last:
                eco = getattr(self, '_ecology_strength', 1.0)
                sigma_mean = sigma_out.mean()
                raw_temp = torch.clamp(1.5 - sigma_mean * 0.8, 0.5, 2.0)
                temp = 1.0 + eco * (raw_temp - 1.0)
                logits = apply_logit_softcap(stage.output_proj(h_accum) / temp, self.cfg.logit_softcap)
            else:
                logits = None

            # Record per-stage stats
            n_steps = len(stage_halt_probs)
            per_stage_ponder_steps.append(n_steps)
            per_stage_strains.append([s.item() for s in stage_strains])
            per_stage_entropy_deltas.append(stage_entropy_deltas)  # e.g. [[0.012, 0.008, 0.003], [...], [...]]
            total_halt_probs.extend(stage_halt_probs)
            all_chain_states.append(state_t)  # Last iteration's state

        # Store ponder cost (sum of all per-stage strains)
        ponder_cost = total_strain
        self._store_ponder_state(ponder_cost, sum(per_stage_ponder_steps))
        self._last_per_stage_steps = per_stage_ponder_steps  # Per-stage breakdown
        self._last_per_stage_entropy_deltas = per_stage_entropy_deltas  # Per-stage entropy improvement trace
        self._ponder_cost_live = ponder_cost

        # v3.5: Expose final-stage hidden state for downstream heads (scratchpad-need, etc.).
        # `hidden` here is the converged h_accum from the last stage's ponder loop.
        self._last_final_hidden = hidden
        if getattr(self.cfg, 'scratchpad_need_predictor', False) and hasattr(self, 'scratchpad_need_head'):
            # [B, S, D] -> [B, S, 1] -> [B, S] in [0, 1]
            self._last_scratchpad_pred = self.scratchpad_need_head(hidden).squeeze(-1)
        else:
            self._last_scratchpad_pred = None

        # v3.6: Metacog → ecology feedback injection (per-stage).
        # If the scratchpad head ran, push the head's mean prediction (pooled over
        # batch and sequence) into each stage's E EMA as a STAGE-SPECIFIC delta:
        #    ΔE_s = α_s · (mean_pred − 0.5)
        # Per-stage weights allow us to route the metacog signal preferentially to
        # stages where it's most relevant (typically the commit stage S2). Detached —
        # the scratchpad head is trained via its own aux loss; this injection is a
        # pure runtime feedback signal. The injected delta carries into the NEXT
        # forward's sigma computation.
        inject_weights = getattr(self.cfg, 'scratchpad_inject_entropy', 0.0)
        if isinstance(inject_weights, (int, float)):
            inject_weights = tuple([float(inject_weights)] * len(self.stages))
        else:
            inject_weights = tuple(float(w) for w in inject_weights)
        # Pad or truncate to n_stages
        if len(inject_weights) < len(self.stages):
            inject_weights = inject_weights + (0.0,) * (len(self.stages) - len(inject_weights))
        elif len(inject_weights) > len(self.stages):
            inject_weights = inject_weights[:len(self.stages)]

        any_active = any(abs(w) > 0 for w in inject_weights)
        if any_active and self._last_scratchpad_pred is not None:
            with torch.no_grad():
                mean_pred = self._last_scratchpad_pred.detach().mean().item()
                cl = getattr(self.cfg, 'prim_clamp_lo', 0.01)
                ch = getattr(self.cfg, 'prim_clamp_hi', 0.99)
                per_stage_deltas = []
                for s_idx, stage in enumerate(self.stages):
                    alpha_s = inject_weights[s_idx]
                    delta_s = alpha_s * (mean_pred - 0.5)
                    per_stage_deltas.append(delta_s)
                    hs = stage.head_state
                    if hasattr(hs, '_entropy_ema') and abs(delta_s) > 0:
                        hs._entropy_ema.add_(delta_s).clamp_(cl, ch)
            self._last_metacog_entropy_delta = tuple(per_stage_deltas)
        else:
            self._last_metacog_entropy_delta = tuple(0.0 for _ in self.stages)

        # Update global sigma
        if all_chain_states:
            self._update_global_sigma(all_chain_states[-1].get("global_sigma", 0.5))

        if return_state or return_chain_state:
            state = self._aggregate_state(all_chain_states)
            state["act_ponder_steps"] = self._last_ponder_steps
            state["act_ponder_cost"] = ponder_cost.item()
            state["act_halt_probs"] = [p.item() for p in total_halt_probs]
            state["act_per_stage_steps"] = per_stage_ponder_steps
            state["act_per_stage_strains"] = per_stage_strains
            if use_ema:
                state["act_strain_ema"] = self._strain_ema.tolist()
            if hasattr(self, '_entropy_delta_ema'):
                state["act_entropy_delta_ema"] = self._entropy_delta_ema.tolist()
            if self._last_difficulty_pred is not None:
                state["act_difficulty_pred"] = self._last_difficulty_pred.mean().item()
            if hasattr(self, '_loss_ema'):
                state["act_loss_ema"] = self._loss_ema.item()
            if return_chain_state:
                state["chain_states"] = all_chain_states
            return logits, state

        return logits

    # ========================================
    # Ponder cost computation
    # ========================================

    def _compute_ponder_cost(
        self,
        halt_probs: List[torch.Tensor],
        strain_values: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute ponder regularization cost.

        Strain mode: total ecological work = sum strain_t
          Penalizes ecologies that take many steps to converge.
          The model learns to build ecologies that settle fast.

        Learned mode: KL(halt_dist || Geometric(lambda_p))
          PonderNet-style prior encouraging early halting.
        """
        if not halt_probs:
            return torch.tensor(0.0, device=halt_probs[0].device if halt_probs else "cpu")

        device = halt_probs[0].device

        if self.cfg.act_strain_halt and strain_values:
            # Total ecological work: sum of per-step strain
            # Model learns to produce ecologies that converge quickly
            cost = torch.tensor(0.0, device=device)
            for strain_t in strain_values:
                cost = cost + strain_t
            return cost
        else:
            # KL divergence from geometric prior (PonderNet fallback)
            lambda_p = self.cfg.act_lambda_p
            kl = torch.tensor(0.0, device=device)
            for t, p_t in enumerate(halt_probs):
                p_geom = lambda_p * ((1 - lambda_p) ** t)
                kl = kl + p_t * (torch.log(p_t + 1e-10) - math.log(max(p_geom, 1e-10)))
            return kl

    # ========================================
    # State aggregation and diagnostics
    # ========================================

    def _aggregate_state(self, chain_states: List[Dict]) -> Dict:
        """
        Aggregate state from all stages into single dict
        compatible with T3v3Transformer state format.
        """
        if not chain_states:
            return {}

        last = chain_states[-1]

        state = {
            "global_sigma": last.get("global_sigma", 0.5),
            "head_sigmas": last.get("head_sigmas", []),
            "head_positions": last.get("head_positions", []),
            "step": int(self.stages[-1]._stage_step.item()),
            "n_stages": len(chain_states),
        }

        # Aggregate blockade info from last stage
        if "blockade" in last:
            state["blockade"] = last["blockade"]

        # Aggregate co-survival from last stage
        if "cosurvival" in last:
            state["cosurvival"] = last["cosurvival"]

        # Head activations from last stage
        if "head_activations" in last:
            state["head_activations"] = last["head_activations"]

        # Head entropy from last stage
        if "head_entropy" in last:
            state["head_entropy"] = last["head_entropy"]

        # Live sigma tensors from ALL stages (for differentiable loss)
        all_sigma_tensors = [s.get("head_sigmas_tensor") for s in chain_states
                             if s.get("head_sigmas_tensor") is not None]
        if all_sigma_tensors:
            # Stack and mean -> single [n_heads] tensor with gradients from all stages
            state["head_sigmas_tensor"] = torch.stack(all_sigma_tensors).mean(dim=0)

        # Per-stage sigma values (useful for analysis)
        state["per_stage_sigmas"] = [
            s.get("head_sigmas", []) for s in chain_states
        ]

        # Per-stage entropy
        state["per_stage_entropy"] = [
            s.get("head_entropy", []) for s in chain_states
        ]

        return state

    def compute_spectral_rank(self, h: Optional[torch.Tensor] = None) -> Dict:
        """Compute spectral rank (same as T3v3Transformer)."""
        if h is None:
            return {"spectral_rank": -1}

        with torch.no_grad():
            h_flat = h.reshape(-1, h.shape[-1])
            h_centered = h_flat - h_flat.mean(dim=0, keepdim=True)

            try:
                U, S, Vh = torch.linalg.svd(h_centered, full_matrices=False)
            except Exception:
                return {"spectral_rank": -1}

            S_norm = S / S.sum()
            S_norm = S_norm[S_norm > 1e-10]
            entropy = -(S_norm * S_norm.log()).sum()
            effective_rank = entropy.exp().item()

            cumulative_energy = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
            rank_90 = (cumulative_energy < 0.9).sum().item() + 1
            rank_95 = (cumulative_energy < 0.95).sum().item() + 1
            rank_99 = (cumulative_energy < 0.99).sum().item() + 1

            return {
                "spectral_rank": effective_rank,
                "rank_90": rank_90,
                "rank_95": rank_95,
                "rank_99": rank_99,
                "total_dims": h.shape[-1],
            }

    def get_chain_state(self) -> List[Dict]:
        """Diagnostic: per-stage state."""
        return [
            {
                "stage": i,
                "head_sigmas": stage.head_state._last_head_sigmas.tolist(),
                "cosurvival_bonds": int((stage.cosurvival.cosurvival > 0.01).sum().item()),
                "output_entropy_ema": stage.entropy_tracker.output_entropy_ema.item(),
                "valence_velocity_ema": stage.entropy_tracker.valence_velocity_ema.item(),
            }
            for i, stage in enumerate(self.stages)
        ]

    def reset_state(self):
        """Reset all stages."""
        for stage in self.stages:
            stage.reset_state()
        self._global_sigma.fill_(0.5)

    @torch.compiler.disable
    def _update_global_sigma(self, sigma_val):
        with torch.no_grad():
            self._global_sigma.fill_(sigma_val)

    @torch.compiler.disable
    def _store_ponder_state(self, ponder_cost, ponder_steps):
        with torch.no_grad():
            self._last_ponder_cost.fill_(ponder_cost.item())
        self._last_ponder_steps = ponder_steps

    # ====================================================================
    # v3.7 Dynamic Ω (port from v5.3) — coupling matrix EMA between forwards
    # ====================================================================

    @torch.compiler.disable
    def _apply_omega_shadow(self):
        """Swap dynamic shadow values into _coupling_params before forward.

        Returns a list of (head_state, attr_name, original_tensor) for restoring.
        Idempotent no-op when Dynamic Ω is disabled.
        """
        if not getattr(self.cfg, 'dynamic_omega_enabled', False):
            return None
        originals = []
        for si, stage in enumerate(self.stages):
            hs = stage.head_state
            shadow_name = f"_omega_shadow_{si}"
            tri_shadow_name = f"_omega_tri_shadow_{si}"
            if hasattr(self, shadow_name) and hasattr(hs, '_coupling_params'):
                orig = hs._coupling_params.data.clone()
                hs._coupling_params.data.copy_(getattr(self, shadow_name))
                originals.append((hs, '_coupling_params', orig))
            if hasattr(self, tri_shadow_name) and hasattr(hs, '_trivector_params'):
                orig = hs._trivector_params.data.clone()
                hs._trivector_params.data.copy_(getattr(self, tri_shadow_name))
                originals.append((hs, '_trivector_params', orig))
        return originals

    @torch.compiler.disable
    def _restore_omega(self, originals):
        """Restore original (anchor) coupling params after forward."""
        if originals is None:
            return
        for hs, attr_name, orig_val in originals:
            getattr(hs, attr_name).data.copy_(orig_val)

    @torch.compiler.disable
    def _update_dynamic_omega(self):
        """EMA-update Ω shadows based on self-surprise + task strain.

        ΔΩ = γ · ΔΩ_self + (1−γ) · ΔΩ_task
          ΔΩ_self: WorldTrace surprise-weighted antisymmetric covariance of primitives
          ΔΩ_task: pull toward learned anchor scaled by per-stage ponder strain
        Clamped to ±max_delta per step. Validated γ=0.5, β=0.01 (v5 phase 3.1a).
        """
        if not getattr(self.cfg, 'dynamic_omega_enabled', False):
            return
        beta = self.cfg.dynamic_omega_beta
        gamma = self.cfg.dynamic_omega_gamma
        max_delta = self.cfg.dynamic_omega_max_delta

        for si, stage in enumerate(self.stages):
            hs = stage.head_state
            shadow_name = f"_omega_shadow_{si}"
            tri_shadow_name = f"_omega_tri_shadow_{si}"
            if not hasattr(self, shadow_name):
                continue
            shadow = getattr(self, shadow_name)

            # ΔΩ_self: surprise-weighted antisymmetrized covariance
            surprise = getattr(hs, '_self_surprise', None)
            if surprise is None:
                continue
            prims = hs.get_current_primitives()
            prims_centered = prims - prims.mean(dim=0, keepdim=True)
            surprise_w = surprise / (surprise.sum() + 1e-8)
            cov = (prims_centered * surprise_w.unsqueeze(-1)).t() @ prims_centered
            delta_self = (cov - cov.t()) * 0.5
            delta_self_flat = torch.zeros_like(shadow)
            idx = 0
            for i in range(6):
                for j in range(i + 1, 6):
                    delta_self_flat[idx] = delta_self[i, j]
                    idx += 1

            # ΔΩ_task: pull toward anchor scaled by per-stage strain
            if hasattr(self, '_last_per_stage_steps'):
                pss = self._last_per_stage_steps
                strain_scale = (pss[si] / max(sum(pss), 1)) if si < len(pss) else 0.0
            else:
                strain_scale = 0.0
            anchor = hs._coupling_params.data
            delta_task_flat = (anchor - shadow) * strain_scale

            delta_omega = gamma * delta_self_flat + (1 - gamma) * delta_task_flat
            delta_omega = delta_omega.clamp(-max_delta, max_delta)
            shadow.lerp_(shadow + delta_omega, beta)

            # Trivectors: same logic, simpler (only task-strain pull for now)
            if hasattr(self, tri_shadow_name) and hasattr(hs, '_trivector_params'):
                tri_shadow = getattr(self, tri_shadow_name)
                tri_anchor = hs._trivector_params.data
                delta_tri = (tri_anchor - tri_shadow) * strain_scale
                delta_tri = delta_tri.clamp(-max_delta, max_delta)
                tri_shadow.lerp_(tri_shadow + delta_tri, beta)

        # Track Ω trajectory for diagnostics
        if hasattr(self, '_omega_displacement_ema'):
            total_disp = 0.0
            for si in range(len(self.stages)):
                shadow_name = f"_omega_shadow_{si}"
                if hasattr(self, shadow_name):
                    shadow = getattr(self, shadow_name)
                    anchor = self.stages[si].head_state._coupling_params.data
                    total_disp += (shadow - anchor).norm().item()
            avg_disp = total_disp / max(len(self.stages), 1)
            disp_var = (avg_disp - self._omega_displacement_ema.item()) ** 2
            self._omega_displacement_ema.lerp_(
                torch.tensor(avg_disp, device=self._omega_displacement_ema.device), 0.05)
            self._omega_variance_ema.lerp_(
                torch.tensor(disp_var, device=self._omega_variance_ema.device), 0.05)


# ======================
# Quick Test
# ======================

