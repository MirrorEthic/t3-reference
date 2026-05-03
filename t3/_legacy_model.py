"""Vendored from t3v36/t3v3_model.py — inference-only.

Stripped: T3v3Loss (training loss head), scale_gradients_by_valence,
scale_gradients_by_stage_surprise (training-only gradient surgery), and
the file-level __main__ smoke-test block.

Do not edit directly — this is the validation copy. The clean module split
(t3.ecology, t3.attention, t3.act, etc.) is the public API.
"""

#!/usr/bin/env python3
"""
t3v3_model.py - T3 v3.0: Rydberg-Coupled Transformer with 6-Primitive Ecology
===============================================================================

Direct evolution of t3v2_model.py. Key changes from v2:

1. 6-PRIMITIVE SYSTEM (E, I, F, V, C, K):
   - E = Blended entropy (output entropy + attention entropy, alpha=0.5)
   - I = Activation magnitude (live, differentiable)
   - F = Friction (|delta entropy| EMA)
   - V = Fristonian valence (dual-EMA MACD on pure attention entropy)
   - C = Coherence placeholder (1 - E, opposing entropy)
   - K = Chronos placeholder (1 - F, temporal stability)
   - Sigma projection: nn.Linear(6, 16) -> Tanh -> nn.Linear(16, 1) -> Sigmoid

2. BLENDED E (output entropy + attention entropy):
   - E = alpha * attn_entropy + (1-alpha) * output_entropy
   - Grounds entire ecological stack in prediction confidence
   - Configurable via blend_alpha (default 0.5)

3. FRISTONIAN VALENCE (V = -dH/dt):
   - Dual-EMA MACD on pure attention entropy
   - Fast EMA (0.95 decay) vs slow EMA (0.99 decay)
   - V = sigmoid((slow - fast) * scale) -- centered, per-head
   - Warmup period (3 calls) lets EMAs settle before V activates

4. PER-HEAD GRADIENT SCALING FROM VALENCE:
   - scale_gradients_by_valence(): call after backward(), before step()
   - High valence (improving) -> more gradient (faster learning)
   - Low valence (worsening) -> less gradient (more cautious)

5. SIGMA WARMUP:
   - compute_head_sigmas(warmup_frac) interpolates sigma toward 0.5
   - Prevents ecology shock at training start

All other components (RoPE, RydbergBlockade, CosurvivalTracker, RydbergAttention,
T3v3Layer, T3v3Transformer) are structurally identical to v2.

Author: Garret Sutherland, MirrorEthic LLC
Date: 2026-03-13
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# FlexAttention (PyTorch 2.5+) — lazy import with availability check
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX_ATTENTION = True

    # Compiled wrapper — torch.compile is REQUIRED for FlexAttention performance.
    # Without it, flex_attention falls back to eager (slower than einsum).
    # The score_mod closure is defined inside so torch.compile captures it fully.
    @torch.compile(dynamic=False)
    def _compiled_flex_attn(q, k, v, temperatures, locality_strengths, block_mask, scale, enable_gqa):
        def score_mod(score, b, h, q_idx, kv_idx):
            score = score / temperatures[h]
            dist = (q_idx - kv_idx).to(score.dtype).abs()
            score = score - dist * locality_strengths[h]
            return score
        return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask,
                             scale=scale, enable_gqa=enable_gqa)
except ImportError:
    HAS_FLEX_ATTENTION = False


# ======================
# Configuration
# ======================

@dataclass
class T3v3Config:
    """T3 v3.0 Configuration."""

    # Transformer basics
    vocab_size: int = 50_257
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_head: int = 0              # 0 = auto (d_model // n_heads); >0 = explicit (for wide-head models like Gemma 3 with d_head=256 > d_model/n_heads)
    d_ff: int = 2048
    max_seq_len: int = 512
    embed_scale: float = 1.0     # multiplier applied after embed lookup; Gemma 3 uses sqrt(d_model) (embeds and lm_head share weight, so scale lives at forward not weight)
    use_qk_norm: bool = False    # per-head RMSNorm on Q and K before attention; Gemma 3 / Llama-3.1 style
    use_post_norms: bool = False  # RMSNorm after attention AND FFN sublayers (before residual add); Gemma 3 style
    dropout: float = 0.1

    # Rydberg blockade (from QuEra + Genesis validation)
    blockade_enabled: bool = True
    blockade_exponent: float = 6.0       # 1/r^6 (Rydberg C6 interaction)
    blockade_strength: float = 0.3       # How much active heads suppress neighbors (validated: 0.3)
    blockade_radius_init: float = 1.0    # Blockade interaction radius: distance at which blockade = 50%. Default 1.0 = backward compatible (unit scale). Set to NN distance for calibrated ecology.
    blockade_radius_auto: bool = True    # Auto-calibrate blockade_radius from NN head distance at construction. Prevents catastrophic 1/r^6 suppression when default 1.0 >> actual head spacing.
    blockade_learnable: bool = True      # Head positions are learnable
    blockade_warmup_steps: int = 200     # Ramp blockade 0->full over this many steps (genesis: Phase I has no blockade)

    # Co-survival coupling (from Genesis validation)
    cosurvival_enabled: bool = True
    cosurvival_decay: float = 0.999      # Very slow forgetting (Genesis: 0.999)
    cosurvival_update_interval: int = 50  # Update every N forward passes
    cosurvival_lr_coupling: float = 0.3  # How much co-survival affects gradient flow
    grav_k: float = 0.2                  # Distance coefficient (Genesis: 0.2)
    cosurvival_valence_modulation: bool = True   # v2.5: weight co-survival bonds by valence agreement

    # Per-head primitives
    per_head_sigma: bool = True          # Each head computes own sigma
    grounded_primitives: bool = True     # v2.3+: EMA-based E/F/V + magnitude I (not learned)

    # v3.0: 6-primitive system (E, I, F, V, C, K)
    n_primitives: int = 6               # Number of primitives for sigma projection input

    # v3.0: Blended E (output entropy + attention entropy)
    blend_alpha: float = 0.5            # Alpha for blended E: alpha*attn + (1-alpha)*output

    # v3.0: Fristonian Valence (dual-EMA MACD on pure attention entropy)
    valence_fast_decay: float = 0.95    # Fast EMA decay for MACD
    valence_slow_decay: float = 0.99    # Slow EMA decay for MACD
    valence_scale: float = 1.5          # Sigmoid scale for normalized valence (±1σ → 0.82/0.18)
    valence_relative: bool = True        # Relative valence: normalize across heads
    valence_warmup_calls: int = 3       # Number of calls before V activates (let EMAs settle)

    # v3.0: Per-head gradient scaling from valence
    valence_grad_scale: float = 0.1     # Gradient scale strength (0 = off)

    # v3.0: Hamiltonian coupling between conjugate pairs (E↔C, I↔K, F↔V)
    hamiltonian_coupling: float = 0.02  # Oscillatory kick strength (omega)
    friction_intensity_weight: float = 0.3  # Weight of intensity change in friction (0=pure dE, 1=pure dI)

    # v3.5: Null cone regularization — Cl(3,3) conformal geometry
    # Q = E²+I²+F² - C²-K²-V² (split-signature metric). The null cone (Q=0)
    # is the critical surface between spacelike (diverse) and timelike (trapped).
    # Restoring force pushes primitives toward Q=0 during Hamiltonian phase.
    null_cone_strength: float = 0.0  # Restoring force strength (0 = off). Try 0.01-0.05.

    # v3.5: Live ecology during inference
    # When True, primitive EMAs + Hamiltonian kicks update during eval/generation.
    # Without this, the ecology is frozen at checkpoint state and can't respond
    # to generated tokens — sigma reads stale buffers.
    eval_live_primitives: bool = False

    # v3.4.1: Ecology signal chain — validated via experiment_ecology_ablation.py (Variant F)
    # Key insight: CE must flow through sigma to teach useful temperatures. WorldTrace sigma
    # target removed — replaced with diversity regularizer in training loop. sigma_spread_weight
    # should be 0 (no WorldTrace target); use sigma_diversity_weight instead.
    # PREVIOUS (v3.1-v3.4): sigma_stop_gradient=True, temp_range=[0.5,1.5]
    sigma_stop_gradient: bool = False    # Let CE flow through sigma→attention (Variant F winner)
    temp_range_lo: float = 0.2           # Wider range = meaningful modulation [0.2, 1.8]
    temp_range_hi: float = 1.8           # (was [0.5, 1.5] — too narrow, ±10% invisible to model)
    prim_clamp_lo: float = 0.01          # Primitive EMA lower clamp
    prim_clamp_hi: float = 0.99          # Primitive EMA upper clamp
    sigma_diversity_weight: float = 0.01  # -std(sigmas) diversity penalty (prevents collapse)
    sigma_antisat_weight: float = 1e-4   # σ*(1-σ) regularizer — prevents head crystallization at 0 or 1
    learned_ecology_params: bool = False  # Make omega, temp range, clamp bounds learnable nn.Parameter
    sigma_hidden: int = 16                # v3.7+ Phase 1A: width of per-head σ MLP hidden layer.
                                           # 16 = v3.7 baseline (preserves bit-perfect default).
                                           # 32 / 64 = ecology DoF expansion (smoother σ response).
    sigma_hidden_per_stage: Optional[List[int]] = None  # v3.7+ Phase 1A staggered: per-stage override.
                                           # If set (list of length n_stages), overrides scalar sigma_hidden
                                           # at each stage. e.g. [64, 32, 16] = encoder-heavy capacity.
                                           # Useful for testing whether ecology wants different σ resolution
                                           # at different stages — smoketest evidence (||Ω|| saturated at
                                           # S0/S1 not S2) suggests this is worth exploring.

    # v3.1d: WorldTrace self-model (predictive coding for sigma)
    # Each head predicts its own primitives. Surprise (prediction error) drives sigma target.
    # High surprise → low sigma (constrict, increase precision). Low surprise → high sigma (explore).
    self_model_alpha: float = 0.3      # How fast self-model tracks actual primitives (v6: 0.3)
    self_model_sigma_floor: float = 0.15  # Minimum sigma target (never fully constrict)
    self_model_sigma_ceil: float = 0.85  # Maximum sigma target (never fully explore)
    self_model_sensitivity: float = 15.0  # How quickly surprise drives sigma down (exp scale)

    # v3.1e: Inter-stage predictive coding
    # Each stage predicts the NEXT stage's per-head primitives.
    # Creates hierarchical prediction chain: S0→S1→S2→...→S(N-1).
    # Prediction error = auxiliary loss driving each stage to model downstream dynamics.
    # Rao & Ballard / Friston hierarchy: higher stages predict lower, errors propagate up.
    inter_stage_pc: bool = True          # Enable inter-stage prediction
    inter_stage_pc_weight: float = 0.05  # Weight for inter-stage prediction loss

    # v3.1e: Stage-surprise gradient scaling
    # Per-stage mean surprise modulates all parameter gradients in that stage.
    # High surprise → more gradient (stage needs to adapt). Low surprise → less gradient.
    # Dynamic version of manual stage freeze — driven by self-model, not diagnostics.
    # NOTE: Different from per-stage LR (which was null result) because this is dynamic —
    # changes based on model's self-assessment each step, not a static multiplier.
    stage_surprise_grad_scale: float = 0.5  # How much surprise modulates stage gradients (0=off)

    # Positional encoding
    use_rope: bool = False               # Use Rotary Position Embeddings instead of learned absolute
    rope_base: float = 10000.0           # RoPE base frequency

    # GQA (Grouped Query Attention) -- for weight transfer from models like Qwen, LLaMA
    n_kv_heads: int = 0                  # 0 = MHA (all heads have own K/V), >0 = GQA
    attn_bias: bool = True               # Bias on Q/K/V projections
    attn_out_bias: bool = True           # Bias on output projection

    # FFN type
    ffn_type: str = "gelu"               # "gelu" (standard) or "swiglu" (gated, used by Qwen/LLaMA)
    ffn_bias: bool = True                # Bias on FFN projections

    # Normalization
    norm_type: str = "layernorm"         # "layernorm" or "rmsnorm"
    norm_eps: float = 1e-5               # Norm epsilon

    # Bypass mode -- disables all T3-specific mechanisms for logit verification
    # When True, the chain becomes structurally equivalent to a standard transformer
    # (no embed_norm, no intermediate stage norms, no residuals, no sigma effects)
    bypass_ecology: bool = False

    # Triton kernel fusion — use fused kernels for entropy probe and attention.
    # Both paths (Triton and PyTorch) coexist; this flag selects at runtime.
    # Requires triton package. Falls back to PyTorch if triton unavailable.
    use_triton_kernels: bool = False

    # FlexAttention — use torch.nn.attention.flex_attention for fused attention.
    # Provides O(S) memory (flash-attention), native GQA (no repeat_kv), and
    # fused score_mod for temperature + locality. Requires torch.compile for perf.
    # Attention entropy is approximated via subsampled probe positions.
    use_flex_attention: bool = False

    # v3.4: Cooperative Adaptive Computation (CAC) — Phase 1: Live ecology during pondering
    act_live_ecology: bool = True           # Update E, F between ponder steps (zero compute cost)
    act_live_ecology_alpha: float = 0.3     # Ponder-time EMA decay (fast — ponder steps are correlated)

    # v3.4: CAC Phase 2 — Eco-conditioned K-bias
    eco_key_bias: bool = True               # Eco-conditioned K-bias in attention (changes WHAT heads attend to)
    eco_key_bias_scale: float = 1.0         # Scale factor for K offset (for tuning/warmup)

    # v3.4: CAC Phase 3 — Cooperative Predictive Attention (CPA)
    cooperative_prediction: bool = False              # Intra-head predictive coding via co-survival bonds
    cooperative_prediction_weight: float = 0.01       # L_pred loss weight (bond prediction)
    cooperative_prediction_bond_threshold: float = 0.01  # Min |cosurv| for active bonds
    complementarity_weight: float = 0.005             # L_comp loss weight (anti-redundancy)
    complementarity_margin: float = 0.3               # Cosine sim threshold for penalty
    sigma_complement_strength: float = 0.02           # Forward-pass sigma offset magnitude

    # v3.4.2: Cross-pair Hamiltonian coupling (Clifford experiment)
    hamiltonian_cross_coupling: bool = False           # Learn full 6x6 antisymmetric coupling
    hamiltonian_max_coupling: float = 0.2              # Max coupling strength (tanh * max)
    hamiltonian_trivectors: bool = True                # Grade-3: state-dependent coupling (20 params/stage).
                                                       # v3.7+: ON by default (orbital morning audit 2026-04-28 found
                                                       # that without trivectors, R is identical per stage for every
                                                       # input — bivector rotation is "decorative" for per-sample
                                                       # purposes. With trivectors, Ω is state-dependent through
                                                       # x_mean coupling, giving real per-input geometry variation.
    sigma_modulated_coupling: bool = False             # Scale coupling rotation by per-head sigma
    eco_key_bias_features: int = 4                     # K-bias input dim: 4 (original) or 6 (all primitives)

    # v3.4.2: Temporal cache warmup — sigma observes before committing
    sigma_temporal_cache: bool = False                 # During warmup, sigma MLP trains but output=0.5
    sigma_temporal_cache_steps: int = 0                # 0 = use main warmup duration

    # Training
    ignore_index: int = -100

    # Spectral monitoring
    track_spectral: bool = True
    spectral_interval: int = 100         # Compute spectral rank every N steps


# ======================
# Rotary Position Embeddings (RoPE)
# ======================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (Su et al., 2021).

    Precomputes sin/cos tables for efficient application to Q/K tensors.
    Supports arbitrary sequence lengths up to max_seq_len.
    """

    def __init__(self, d_head: int, max_seq_len: int = 32768, base: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        # Inverse frequency bands: theta_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Precompute for common lengths
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, d_head/2)
        cos_cached = freqs.cos()
        sin_cached = freqs.sin()
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, seq_len: int) -> tuple:
        """Returns (cos, sin) each of shape (seq_len, d_head/2)."""
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                          cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    """Apply rotary embeddings to Q and K tensors.

    Args:
        q: (batch, seq, n_heads, d_head)
        k: (batch, seq, n_heads, d_head)
        cos: (seq, d_head/2)
        sin: (seq, d_head/2)

    Returns:
        (q_rotated, k_rotated) with same shapes
    """
    d_half = cos.shape[-1]
    # Split into even/odd dimensions
    q1, q2 = q[..., :d_half], q[..., d_half:]
    k1, k2 = k[..., :d_half], k[..., d_half:]
    # Reshape cos/sin for broadcasting: (1, seq, 1, d_half)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    # Apply rotation
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot


# ======================
# RMSNorm (for Qwen/LLaMA weight transfer)
# ======================

class T3RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Weight-only (no bias), used by Qwen2, LLaMA, Mistral, etc."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


def make_norm(d_model: int, cfg: 'T3v3Config') -> nn.Module:
    """Create normalization layer based on config."""
    norm_type = getattr(cfg, 'norm_type', 'layernorm')
    eps = getattr(cfg, 'norm_eps', 1e-5)
    if norm_type == 'rmsnorm':
        return T3RMSNorm(d_model, eps=eps)
    return nn.LayerNorm(d_model, eps=eps)


# ======================
# SwiGLU FFN (for Qwen/LLaMA weight transfer)
# ======================

class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (Shazeer, 2020).
    gate_proj + up_proj with SiLU gating, then down_proj.
    Used by Qwen2, LLaMA, Mistral, etc."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GeGLUFFN(nn.Module):
    """GeGLU Feed-Forward Network — same structure as SwiGLU but GeLU activation.
    Used by Gemma 3 (gelu_pytorch_tanh)."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.gelu(self.gate_proj(x), approximate='tanh') * self.up_proj(x)))


# ======================
# GQA Utilities
# ======================

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match Q head count for Grouped Query Attention.
    x: [batch, seq, n_kv_heads, d_head] -> [batch, seq, n_heads, d_head]"""
    if n_rep == 1:
        return x
    batch, seq, n_kv_heads, d_head = x.shape
    x = x[:, :, :, None, :].expand(batch, seq, n_kv_heads, n_rep, d_head)
    return x.reshape(batch, seq, n_kv_heads * n_rep, d_head)


# ======================
# Geodesic Utilities
# ======================

def geodesic_distance_1d(a: torch.Tensor, b: torch.Tensor, period: float = 1.0) -> torch.Tensor:
    """Geodesic distance on S1. Vectorized."""
    diff = (a - b).abs()
    return torch.min(diff, period - diff)


def geodesic_distance_t3(
    pos_a: torch.Tensor,  # (..., 3)
    pos_b: torch.Tensor,  # (..., 3)
    periods: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> torch.Tensor:
    """Geodesic distance on T3. Vectorized, no offset."""
    d_sq = torch.zeros(pos_a.shape[:-1], device=pos_a.device)
    for i in range(3):
        d = geodesic_distance_1d(pos_a[..., i], pos_b[..., i], periods[i])
        d_sq = d_sq + d * d
    return torch.sqrt(d_sq + 1e-8)


# ======================
# Per-Head T3 State
# ======================

class HeadState(nn.Module):
    """
    Each attention head has its own position in T3 space
    and its own primitive estimators.

    The position determines which heads compete (blockade)
    and which heads cooperate (co-survival).

    v3.0: 6-primitive system (E, I, F, V, C, K) with Fristonian valence
    and blended E (output entropy + attention entropy).
    """

    def __init__(self, n_heads: int, d_model: int, cfg: T3v3Config,
                 sigma_hidden_override: Optional[int] = None):
        super().__init__()
        self.n_heads = n_heads
        # NOTE: HeadState's d_head partitions the residual stream (always d_model//n_heads),
        # distinct from RydbergAttention's d_head which can be wider (e.g. Gemma 3 with d_head=256
        # on d_model=640, n_heads=4). The entropy_heads / key_bias_proj projections here read
        # from h_per_head = h.view(B, S, n_heads, d_model//n_heads) in t3v3_chain.
        self.d_head = d_model // n_heads
        self.cfg = cfg
        # v3.7+ Phase 1A: sigma_hidden_override lets T3v3Stage pass per-stage value
        # from cfg.sigma_hidden_per_stage[stage_idx] if set; falls back to cfg.sigma_hidden.
        self._sigma_hidden_override = sigma_hidden_override

        # Learnable head positions on T3
        # Initialize spread out to avoid initial blockade collapse
        if cfg.blockade_learnable:
            # Spread heads evenly in T3 space initially
            init_positions = torch.zeros(n_heads, 3)
            for i in range(n_heads):
                # Fibonacci spiral on T3 for even coverage
                golden = (1 + math.sqrt(5)) / 2
                init_positions[i, 0] = (i / n_heads) % 1.0
                init_positions[i, 1] = (i * golden / n_heads) % 1.0
                init_positions[i, 2] = (i * golden * golden / n_heads) % 1.0
            self.head_positions = nn.Parameter(init_positions)
        else:
            init_positions = torch.zeros(n_heads, 3)
            for i in range(n_heads):
                golden = (1 + math.sqrt(5)) / 2
                init_positions[i, 0] = (i / n_heads) % 1.0
                init_positions[i, 1] = (i * golden / n_heads) % 1.0
                init_positions[i, 2] = (i * golden * golden / n_heads) % 1.0
            self.register_buffer("head_positions", init_positions)

        # Compute initial nearest-neighbor distance for diagnostics
        with torch.no_grad():
            _pos = init_positions % 1.0
            _d = geodesic_distance_t3(_pos.unsqueeze(1), _pos.unsqueeze(0))
            _d.fill_diagonal_(float('inf'))
            self._init_nn_distance = _d.min(dim=1).values.mean().item()

        # Per-head primitive estimators
        # Each head senses pressure differently (specialization pressure)
        n_prims = cfg.n_primitives  # v3.0: 6 primitives (E, I, F, V, C, K)

        if cfg.per_head_sigma:
            if cfg.grounded_primitives:
                # v2.3+: EMA buffers for grounded E/F/V from actual signals
                # E = attention entropy EMA, I = activation magnitude (live),
                # F = |delta entropy| EMA, V = entropy improvement direction EMA
                self.register_buffer("_entropy_ema", torch.full((n_heads,), 0.5))
                self.register_buffer("_entropy_prev", torch.full((n_heads,), 0.5))
                self.register_buffer("_friction_ema", torch.full((n_heads,), 0.1))
                self.register_buffer("_valence_ema", torch.full((n_heads,), 0.5))

                # v3.0: Coherence and Chronos (grounded signals, not placeholders)
                # C = cross-head entropy agreement (epistemic conjugate to E)
                # K = temporal self-predictability (dynamic conjugate to I)
                self.register_buffer("_coherence_ema", torch.full((n_heads,), 0.5))
                self.register_buffer("_chronos_ema", torch.full((n_heads,), 0.5))

                # v3.0: Intensity EMA for Hamiltonian coupling and friction enrichment
                self.register_buffer("_intensity_ema", torch.full((n_heads,), 0.5))
                self.register_buffer("_last_intensity", torch.full((n_heads,), 0.5))

                # v3.0: Dual-EMA for Fristonian valence (MACD on pure attention entropy)
                self.register_buffer("_attn_fast_ema", torch.full((n_heads,), 0.5))
                self.register_buffer("_attn_slow_ema", torch.full((n_heads,), 0.5))
                self.register_buffer("_valence_init_count", torch.tensor(0, dtype=torch.long))

                # v3.1d: WorldTrace self-model — predicts own primitives
                # Each head maintains an EMA prediction of what its E,I,F,V,C,K should be.
                # Surprise = L2 distance between predicted and actual.
                # Maps to v6 controller's WorldTrace.predicted + WorldTrace.surprise()
                self.register_buffer("_pred_E", torch.full((n_heads,), 0.5))
                self.register_buffer("_pred_I", torch.full((n_heads,), 0.5))
                self.register_buffer("_pred_F", torch.full((n_heads,), 0.1))
                self.register_buffer("_pred_V", torch.full((n_heads,), 0.5))
                self.register_buffer("_pred_C", torch.full((n_heads,), 0.5))
                self.register_buffer("_pred_K", torch.full((n_heads,), 0.5))
                self.register_buffer("_self_surprise", torch.zeros(n_heads))  # cached per-head surprise

                # v3.1e: Inter-stage predictive coding predictor
                # Predicts next stage's primitives from this stage's primitives.
                # Shared across heads (one Linear for all heads).
                # Creates hierarchical prediction chain: each stage models downstream.
                if cfg.inter_stage_pc:
                    self.inter_stage_predictor = nn.Linear(n_prims, n_prims)
                    # Initialize close to identity — predict "same as me" as default
                    with torch.no_grad():
                        nn.init.eye_(self.inter_stage_predictor.weight)
                        nn.init.zeros_(self.inter_stage_predictor.bias)

                # v3.4 Phase 2: Cached excitation for eco K-bias (one-step delay)
                self.register_buffer("_last_excitation", torch.zeros(n_heads))

                # v3.4 Phase 2: Eco-conditioned K-bias projection
                # Maps [sigma, E, excitation, protection] → d_head offset per head.
                # Zero-init: no effect at start, model learns when to shift attention.
                # v3.4.2: Expanded from 4 to 6 features — V is the coupling hub but
                # K-bias couldn't see it. Now all 6 primitives feed attention routing.
                # PREVIOUS: Linear(4, d_head) with [σ, E, excitation, protection]
                # v3.7 audit (Gemma): the K-bias adds to attention's K, so it must match
                # attention's d_head, not the residual-stream partition (cfg.d_head if set).
                if getattr(cfg, 'eco_key_bias', False):
                    attn_d_head = getattr(cfg, 'd_head', 0) or (d_model // n_heads)
                    n_eco_features = getattr(cfg, 'eco_key_bias_features', 4)
                    self.key_bias_proj = nn.Linear(n_eco_features, attn_d_head)
                    nn.init.zeros_(self.key_bias_proj.weight)
                    nn.init.zeros_(self.key_bias_proj.bias)

                # v3.4 Phase 3: Bond predictor for cooperative predictive attention
                # Predicts bonded partners' primitive vectors via co-survival bonds.
                # Same pattern as inter_stage_predictor: Linear(6→6), identity init.
                if getattr(cfg, 'cooperative_prediction', False):
                    self.bond_predictor = nn.Linear(n_prims, n_prims)
                    with torch.no_grad():
                        nn.init.eye_(self.bond_predictor.weight)
                        nn.init.zeros_(self.bond_predictor.bias)

                # v3.4.1: Learned ecology parameters (experiment variant E)
                # When enabled, omega, temp range, and clamp bounds become nn.Parameter
                # initialized from current config values. The model learns its own operating regime.
                if getattr(cfg, 'learned_ecology_params', False):
                    # Hamiltonian omega — sigmoid-bounded [0, 0.2]
                    init_omega = cfg.hamiltonian_coupling
                    self._learned_omega = nn.Parameter(
                        torch.tensor(math.log(init_omega / (0.2 - init_omega + 1e-8))))
                    # Temperature range — lo sigmoid [0, 1], hi softplus from 1.0
                    self._learned_temp_lo = nn.Parameter(torch.tensor(cfg.temp_range_lo))
                    self._learned_temp_hi = nn.Parameter(torch.tensor(cfg.temp_range_hi))
                    # Clamp bounds — sigmoid-bounded [0, 0.5] for lo, [0.5, 1.0] for hi
                    self._learned_clamp_lo = nn.Parameter(torch.tensor(cfg.prim_clamp_lo))
                    self._learned_clamp_hi = nn.Parameter(torch.tensor(cfg.prim_clamp_hi))

                # v3.4.2: Cross-pair Hamiltonian coupling (Clifford experiment)
                # Tests whether ecology needs the full 15-dim grade-2 space of Cl(3,3)
                # or just the 3 within-pair bivectors. 15 params for the upper triangle
                # of a 6x6 antisymmetric coupling matrix. Intra-pair terms init to omega,
                # cross-pair terms init to 0. If cross-pair terms grow during training,
                # the ecology genuinely needs cross-axis dynamics.
                # Primitive ordering: [E=0, I=1, F=2, V=3, C=4, K=5]
                # Intra-pair: (0,4)=E-C, (1,5)=I-K, (2,3)=F-V → indices 3, 8, 9
                # Cross-pair: all other 12 upper-triangle entries → init 0
                if getattr(cfg, 'hamiltonian_cross_coupling', False):
                    init_omega = cfg.hamiltonian_coupling
                    max_coupling = getattr(cfg, 'hamiltonian_max_coupling', 0.2)
                    # 15 raw params → tanh * max_coupling gives bounded [-max, +max]
                    coupling_init = torch.zeros(15)
                    # Intra-pair indices in upper triangle (i<j):
                    # idx 3 = (0,4) E-C, idx 8 = (1,5) I-K, idx 9 = (2,3) F-V
                    # Init to atanh(omega/max_coupling) so tanh maps back to omega
                    intra_val = math.atanh(min(init_omega / max_coupling, 0.99))
                    coupling_init[3] = intra_val   # E-C
                    coupling_init[8] = intra_val   # I-K
                    coupling_init[9] = intra_val   # F-V
                    self._coupling_params = nn.Parameter(coupling_init)
                    self._coupling_max = max_coupling
                    # Map from flat index to (i,j) for logging
                    self._coupling_labels = [
                        'E-I', 'E-F', 'E-V', 'E-C', 'E-K',
                        'I-F', 'I-V', 'I-C', 'I-K',
                        'F-V', 'F-C', 'F-K',
                        'V-C', 'V-K',
                        'C-K',
                    ]

                    # v3.4.2: Grade-3 trivectors — state-dependent coupling
                    # Each trivector α_ijk makes the i-j coupling depend on x_k,
                    # the j-k coupling depend on x_i, and the k-i coupling depend on x_j.
                    # C(6,3) = 20 trivectors. Zero-init: model discovers which matter.
                    if getattr(cfg, 'hamiltonian_trivectors', False):
                        self._trivector_params = nn.Parameter(torch.zeros(20))
                        # Build the (i,j,k) triple index for each of 20 trivectors
                        triples = []
                        pnames = ['E', 'I', 'F', 'V', 'C', 'K']
                        labels = []
                        for i in range(6):
                            for j in range(i + 1, 6):
                                for k in range(j + 1, 6):
                                    triples.append((i, j, k))
                                    labels.append(f'{pnames[i]}{pnames[j]}{pnames[k]}')
                        self._trivector_triples = triples  # list of (i,j,k) tuples
                        self._trivector_labels = labels
            else:
                # v2.2: Learned primitive projections (KEY: separate per head)
                self.entropy_heads = nn.ModuleList([nn.Linear(self.d_head, 1) for _ in range(n_heads)])
                self.intensity_heads = nn.ModuleList([nn.Linear(self.d_head, 1) for _ in range(n_heads)])
                self.friction_heads = nn.ModuleList([nn.Linear(self.d_head, 1) for _ in range(n_heads)])
                self.valence_heads = nn.ModuleList([nn.Linear(self.d_head, 1) for _ in range(n_heads)])

            # Per-head sigma projection — batched for GPU efficiency
            # v3.0: 6 input primitives (E, I, F, V, C, K)
            # v3.3: Vectorized — replaces per-head ModuleList with batched params.
            # Architecture per head: Linear(n_prims→sigma_hidden) → Tanh → Linear(sigma_hidden→1) → Sigmoid
            # Init: Kaiming uniform matching v2.5's nn.Linear default
            # (N(0,0.1) was too small — sigma stuck at 0.5, no differentiation)
            # v3.7+ Phase 1A: sigma_hidden is now configurable (cfg.sigma_hidden, default 16
            # to preserve v3.7 baseline; 32 or 64 increases σ MLP capacity for ecology DoF tests).
            # If T3v3Stage passes a per-stage override (from cfg.sigma_hidden_per_stage),
            # use that; else fall back to cfg.sigma_hidden.
            sigma_hidden = (self._sigma_hidden_override
                            if self._sigma_hidden_override is not None
                            else getattr(cfg, 'sigma_hidden', 16))
            # Kaiming: bound = 1/sqrt(fan_in)
            w1_bound = 1.0 / (n_prims ** 0.5)   # 1/sqrt(6) ≈ 0.408
            w2_bound = 1.0 / (sigma_hidden ** 0.5)  # 1/sqrt(16) = 0.25
            self.sigma_w1 = nn.Parameter(torch.empty(n_heads, sigma_hidden, n_prims).uniform_(-w1_bound, w1_bound))
            self.sigma_b1 = nn.Parameter(torch.zeros(n_heads, sigma_hidden))
            self.sigma_w2 = nn.Parameter(torch.empty(n_heads, 1, sigma_hidden).uniform_(-w2_bound, w2_bound))
            self.sigma_b2 = nn.Parameter(torch.zeros(n_heads, 1))
        else:
            # Fall back to global sigma (v1 behavior)
            self.global_entropy_head = nn.Linear(d_model, 1)
            self.global_intensity_head = nn.Linear(d_model, 1)
            self.global_friction_head = nn.Linear(d_model, 1)
            self.global_valence_head = nn.Linear(d_model, 1)
            self.global_sigma_projection = nn.Sequential(
                nn.Linear(n_prims, 16),
                nn.Tanh(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        self.register_buffer("_last_head_sigmas", torch.full((n_heads,), 0.5))
        self.register_buffer("_last_head_fitness", torch.zeros(n_heads))

    def _apply_coupling_rotation(self, prims: torch.Tensor) -> torch.Tensor:
        """Apply cross-pair coupling rotation to primitives.

        Differentiable through _coupling_params and _trivector_params.
        Used by compute_head_sigmas (sigma path) and compute_null_cone_Q (null cone loss).

        Args:
            prims: [H, 6] primitive values (detached EMA buffers)
        Returns:
            [H, 6] rotated primitives (gradient flows through coupling params)
        """
        if getattr(self, '_coupling_params', None) is None:
            return prims

        vals = torch.tanh(self._coupling_params) * self._coupling_max
        Omega = torch.zeros(6, 6, device=vals.device, dtype=vals.dtype)
        idx = 0
        for i in range(6):
            for j in range(i + 1, 6):
                Omega[i, j] = vals[idx]
                Omega[j, i] = -vals[idx]
                idx += 1

        # Grade-3: trivectors make coupling state-dependent.
        # α_ijk: the i-j coupling depends on x_k, j-k on x_i, k-i on x_j.
        if getattr(self, '_trivector_params', None) is not None:
            x_centered = prims - 0.5  # [H, 6] detached primitive state
            x_mean = x_centered.mean(dim=0)  # [6]
            tri_vals = torch.tanh(self._trivector_params) * self._coupling_max
            for t_idx, (i, j, k) in enumerate(self._trivector_triples):
                alpha = tri_vals[t_idx]
                Omega[i, j] = Omega[i, j] + alpha * x_mean[k]
                Omega[j, i] = Omega[j, i] - alpha * x_mean[k]
                Omega[j, k] = Omega[j, k] + alpha * x_mean[i]
                Omega[k, j] = Omega[k, j] - alpha * x_mean[i]
                Omega[k, i] = Omega[k, i] + alpha * x_mean[j]
                Omega[i, k] = Omega[i, k] - alpha * x_mean[j]

        R = torch.linalg.matrix_exp(Omega)  # [6, 6]
        prims_centered = prims - 0.5
        prims_rotated = (prims_centered @ R.t()) + 0.5
        if getattr(self.cfg, 'sigma_modulated_coupling', False):
            blend = self._last_head_sigmas.detach().unsqueeze(-1)  # [H, 1]
            prims = prims * (1 - blend) + prims_rotated * blend
        else:
            prims = prims_rotated
        return prims

    def compute_null_cone_Q(self) -> tuple:
        """Compute Cl(3,3) quadratic form Q(v) = E²+I²+F² - V²-C²-K².

        Routes through coupling rotation when available, giving gradient
        to _coupling_params. The coupling learns rotations that keep
        the system near the null cone. Without coupling, returns a
        no-grad diagnostic (Hamiltonian restoring force handles dynamics).

        Returns:
            Q: [n_heads] per-head Q values
            Q_loss: scalar mean(Q²) loss
        """
        prims = self.get_current_primitives()  # [H, 6] from EMA buffers
        prims = self._apply_coupling_rotation(prims)
        # Q(v) = ||observables||² - ||conjugates||²
        # Primitive order: [E, I, F, V, C, K]
        obs_sq = prims[:, :3].pow(2).sum(dim=-1)   # E² + I² + F²
        conj_sq = prims[:, 3:].pow(2).sum(dim=-1)  # V² + C² + K²
        Q = obs_sq - conj_sq
        Q_loss = (Q ** 2).mean()
        return Q, Q_loss

    # v3.3: @torch._dynamo.disable removed — batched sigma is compile-friendly
    def compute_head_sigmas(self, h_per_head: torch.Tensor, warmup_frac: float = 1.0) -> torch.Tensor:
        """
        Compute per-head sigma values from per-head hidden states.

        h_per_head: [batch, seq, n_heads, d_head]
        warmup_frac: 0.0 = all sigmas at 0.5 (neutral), 1.0 = full ecology
        Returns: [n_heads] sigma values (differentiable)
        """
        if self.cfg.per_head_sigma:
            # Pool over batch and sequence for each head
            h_pooled = h_per_head.mean(dim=(0, 1))  # [n_heads, d_head]

            if self.cfg.grounded_primitives:
                # v3.0: Grounded 6-primitive system
                # I is differentiable (from current hidden state)
                # E, F, V, C, K are from EMA buffers (stable, one-step delay)
                norms = h_pooled.float().norm(dim=-1)  # [n_heads]
                I_all = norms / (norms.max() + 1e-8)  # [0, 1], differentiable

                # Store current intensity for update_grounded_primitives
                with torch.no_grad():
                    self._last_intensity.copy_(I_all.detach())

                # v3.3: Batched sigma — no per-head loop
                # Stack all 6 primitives into [n_heads, 6]
                # .detach() breaks version tracking so in-place EMA mutations
                # (Hamiltonian kicks, WorldTrace, ACT live ecology) don't invalidate
                # the computation graph. Gradient flows through sigma_w/b and
                # _coupling_params only, not through the buffer values themselves.
                prims = torch.stack([
                    self._entropy_ema.detach(),    # E [n_heads]
                    I_all.detach(),                 # I [n_heads]
                    self._friction_ema.detach(),   # F [n_heads]
                    self._valence_ema.detach(),    # V [n_heads]
                    self._coherence_ema.detach(),  # C [n_heads]
                    self._chronos_ema.detach(),    # K [n_heads]
                ], dim=-1)  # [n_heads, 6]

                # v3.4.2: Cross-pair coupling — inject into sigma's differentiable path
                # The Hamiltonian rotation in update_grounded_primitives mutates EMA
                # buffers in-place (no grad). To give _coupling_params gradient signal,
                # we apply the coupling rotation HERE in the computation graph, so
                # loss → sigma → rotated_prims → coupling_params is differentiable.
                # Also used by compute_null_cone_Q() for null cone loss gradient.
                prims = self._apply_coupling_rotation(prims)

                # Batched MLP: [H, 6] → [H, 16] → [H, 1] → [H]
                # FP32 guard: sigma MLP has tanh+sigmoid that saturate in FP16
                # .clone() on parameters breaks version tracking so ACT's multiple
                # calls to compute_head_sigmas per forward don't conflict on PyTorch 2.11+.
                # Gradient still flows: clone() preserves the autograd graph to the original param.
                with torch.amp.autocast("cuda", enabled=False):
                    prims_f = prims.float()
                    w1 = self.sigma_w1.float().clone()
                    b1 = self.sigma_b1.float().clone()
                    w2 = self.sigma_w2.float().clone()
                    b2 = self.sigma_b2.float().clone()
                    h1 = torch.bmm(w1, prims_f.unsqueeze(-1)).squeeze(-1) + b1
                    h1 = torch.tanh(h1)
                    h2 = torch.bmm(w2, h1.unsqueeze(-1)).squeeze(-1) + b2
                    sigmas = torch.sigmoid(h2.squeeze(-1))  # [H]

                # v3.4.2: Temporal cache warmup — sigma MLP trains but output stays neutral.
                # The MLP learns which directions in primitive space matter BEFORE committing.
                # Prevents premature crystallization from random first-step gradients.
                if getattr(self, '_sigma_temporal_cache_active', False):
                    # Straight-through estimator: output is 0.5, but gradient flows
                    # through sigmas as if they were used. The MLP learns which directions
                    # matter BEFORE sigma commits to actual temperature modulation.
                    sigmas = sigmas - sigmas.detach() + 0.5
            else:
                # v2.2: Learned primitive projections (4 primitives + 2 placeholders)
                sigmas_list = []
                for i in range(self.n_heads):
                    h_i = h_pooled[i]  # [d_head]
                    E = torch.sigmoid(self.entropy_heads[i](h_i.unsqueeze(0)))
                    I = torch.sigmoid(self.intensity_heads[i](h_i.unsqueeze(0)))
                    F_val = torch.sigmoid(self.friction_heads[i](h_i.unsqueeze(0)))
                    V = torch.sigmoid(self.valence_heads[i](h_i.unsqueeze(0)))
                    C = 1.0 - E
                    K = 1.0 - F_val
                    prims = torch.cat([E, I, F_val, V, C, K], dim=-1)  # [1, 6]
                    # v2.2 still uses per-head sigma_projections (not migrated)
                    sigma_i = torch.sigmoid(
                        (self.sigma_w2[i] @ torch.tanh(
                            self.sigma_w1[i] @ prims.squeeze(0).unsqueeze(-1) + self.sigma_b1[i].unsqueeze(-1)
                        ) + self.sigma_b2[i]).squeeze()
                    )
                    sigmas_list.append(sigma_i)
                sigmas = torch.stack(sigmas_list)  # [n_heads]

            # Symmetry breaking: small noise prevents sigma saddle point
            if self.training:
                sigmas = sigmas + torch.randn_like(sigmas) * 0.02
                sigmas = sigmas.clamp(0.01, 0.99)

            # v3.0: Sigma warmup -- interpolate toward 0.5 during ecology ramp
            if warmup_frac < 1.0:
                sigmas = 0.5 + warmup_frac * (sigmas - 0.5)

            with torch.no_grad():
                self._last_head_sigmas.copy_(sigmas.detach())

            # v3.4.2: Detach live reference — prevents PyTorch 2.11+ in-place version
            # conflicts when ACT calls compute_head_sigmas multiple times per forward.
            # Gradient to sigma_w/b flows through the LAST ponder step's return value.
            self._live_head_sigmas = sigmas.detach()

            return sigmas
        else:
            # Global sigma computation (v1 fallback)
            h_flat = h_per_head.reshape(h_per_head.shape[0], h_per_head.shape[1], -1)
            h_pooled = h_flat.mean(dim=(0, 1))  # [d_model]
            E = torch.sigmoid(self.global_entropy_head(h_pooled.unsqueeze(0)))
            I = torch.sigmoid(self.global_intensity_head(h_pooled.unsqueeze(0)))
            F_val = torch.sigmoid(self.global_friction_head(h_pooled.unsqueeze(0)))
            V = torch.sigmoid(self.global_valence_head(h_pooled.unsqueeze(0)))
            C = 1.0 - E
            K = 1.0 - F_val
            p = torch.cat([E, I, F_val, V, C, K], dim=-1)
            sigma = self.global_sigma_projection(p).squeeze()
            # Broadcast to all heads
            result = sigma.expand(self.n_heads)
            if warmup_frac < 1.0:
                result = 0.5 + warmup_frac * (result - 0.5)
            return result

    @torch.compiler.disable
    def update_grounded_primitives(self, head_entropy: torch.Tensor, max_seq_len: int = 512,
                                    output_entropy: Optional[float] = None,
                                    warmup_frac: float = 1.0,
                                    protection_scores: Optional[torch.Tensor] = None,
                                    blockade_suppression: Optional[torch.Tensor] = None):
        """
        Update grounded primitive EMA buffers from actual attention entropy.

        Called after attention runs each step. One-step delay is intentional:
        primitives reflect recent *trend*, sigma MLP learns the mapping.

        v3.0: 6-primitive grounded system with Hamiltonian conjugate pair coupling.
        v3.1: C grounded in blockade suppression, K grounded in co-survival protection.

        Three conjugate pairs (from formal grounding — symplectic phase space):
          E ↔ C (Epistemic): Entropy ↔ Coherence — "truth/structure" resource
          I ↔ K (Dynamic):   Intensity ↔ Chronos — "attention/processing" resource
          F ↔ V (Telic):     Friction ↔ Valence — "purpose/goal" resource

        Each pair has:
          1. Independent forcing (EMA update from observations)
          2. Hamiltonian coupling (oscillatory kick between conjugates)

        head_entropy: [n_heads] average attention entropy across layers
        output_entropy: scalar output entropy from output_proj (None during warmup)
        warmup_frac: ecology warmup fraction (0.0 = off, 1.0 = full)
        protection_scores: [n_heads] co-survival protection (high = well-supported head)
        blockade_suppression: [n_heads] blockade suppression (high = heavily silenced head)
        """
        if not self.cfg.grounded_primitives:
            return

        # FP32 guard: EMA lerp_, Hamiltonian coupling, exp/std overflow in FP16
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            # Normalize entropy to [0, 1], cast to buffer dtype for bf16 compat
            max_ent = math.log(max(max_seq_len, 2))
            norm_entropy = (head_entropy / max_ent).clamp(0, 1).to(self._entropy_ema.dtype)

            # Blended E: alpha*attn + (1-alpha)*output_entropy
            # v3.3 fix: constant blend from step 0 — output entropy is the grounding
            # signal linking ecology to prediction quality. Ramping it in late (old
            # formula: 1.0 - warmup_frac * 0.5) caused ecology to spin without
            # prediction feedback during the critical bootstrap period.
            if output_entropy is not None and warmup_frac > 0.01:
                alpha = self.cfg.blend_alpha  # constant 0.5
                oe_tensor = torch.full_like(norm_entropy, output_entropy)
                blended = alpha * norm_entropy + (1 - alpha) * oe_tensor
            else:
                blended = norm_entropy

            alpha = 0.05  # EMA blend rate

            # ==============================
            # PHASE 1: Observation-driven forcing (EMA updates)
            # ==============================

            # E: blended entropy EMA
            self._entropy_ema.lerp_(blended, alpha)

            # I: intensity EMA from activation magnitude (stored by compute_head_sigmas)
            self._intensity_ema.lerp_(self._last_intensity, alpha)

            # F: friction = entropy change rate + intensity change rate
            # The formal grounding defines F as "resistance, effort, constraint violation"
            # |dE| captures information resistance, |dI| captures energy volatility
            delta_e = (blended - self._entropy_prev).abs()
            # v3.4.1: FIX — delta_i should be step-to-step intensity change, not deviation from EMA.
            # PREVIOUS (broken at convergence — deviation shrinks even if I oscillates):
            #   delta_i = (self._last_intensity - self._intensity_ema).abs()
            # NEW: compare current intensity to PREVIOUS step's intensity (true rate of change)
            _prev_I = getattr(self, '_last_intensity_prev', self._last_intensity)
            delta_i = (self._last_intensity - _prev_I).abs()
            self._last_intensity_prev = self._last_intensity.clone()
            w_i = self.cfg.friction_intensity_weight  # 0.3 default
            friction_raw = (1.0 - w_i) * delta_e + w_i * delta_i
            self._friction_ema.lerp_(friction_raw, alpha)

            # V: Fristonian dual-EMA valence (MACD on PURE attention entropy)
            fast_alpha = 1.0 - self.cfg.valence_fast_decay  # 0.05
            slow_alpha = 1.0 - self.cfg.valence_slow_decay  # 0.01
            self._attn_fast_ema.lerp_(norm_entropy, fast_alpha)
            self._attn_slow_ema.lerp_(norm_entropy, slow_alpha)

            self._valence_init_count += 1
            if self._valence_init_count <= self.cfg.valence_warmup_calls:
                pass  # V stays at 0.5, let EMAs settle
            elif self._valence_init_count == self.cfg.valence_warmup_calls + 1:
                # First step after warmup: sync slow=fast for clean start
                self._attn_slow_ema.copy_(self._attn_fast_ema)
            else:
                # v3.4: Relative valence with adaptive gain
                # Raw MACD signal collapses as training converges (both EMAs → same value).
                # Fix: normalize across heads so V measures "is this head's entropy
                # changing faster or slower than the population" — always informative.
                v_diff = self._attn_slow_ema - self._attn_fast_ema
                v_centered = v_diff - v_diff.mean()

                if self.cfg.valence_relative:
                    # Adaptive gain: normalize by cross-head std
                    # Inputs become ~N(0,1), sigmoid(x*3) gives [0.05, 0.95] at ±1σ
                    v_std = v_centered.std() + 1e-8
                    v_normalized = v_centered / v_std
                    v_scaled = torch.sigmoid(v_normalized * self.cfg.valence_scale)
                else:
                    # Legacy absolute mode (scale=30 for raw diffs)
                    v_scaled = torch.sigmoid(v_centered * self.cfg.valence_scale)

                self._valence_ema.lerp_(v_scaled, 0.1)

            # C: Coherence = cross-head agreement × ecological standing
            # v3.0: Gaussian of deviation from mean entropy (cross-head agreement)
            # v3.1: Blended with blockade suppression — heavily suppressed heads
            #        have low coherence (ecology is silencing them → low structural match)
            head_mean = norm_entropy.mean()
            head_std = norm_entropy.std() + 1e-8
            c_agreement = torch.exp(-0.5 * ((norm_entropy - head_mean) / head_std) ** 2)
            if blockade_suppression is not None:
                # Suppressed heads get coherence pulled down.
                # c_ecological = agreement * (1 - suppression): fully suppressed → C=0
                c_ecological = c_agreement * (1.0 - blockade_suppression.clamp(0, 1))
                # 70% agreement + 30% ecological standing
                c_raw = 0.7 * c_agreement + 0.3 * c_ecological
            else:
                c_raw = c_agreement
            self._coherence_ema.lerp_(c_raw, alpha)

            # K: Chronos = temporal self-predictability × cooperative support
            # v3.0: Prediction error (how well EMA predicts current entropy)
            # v3.1: Blended with co-survival protection — well-supported heads
            #        have high chronos (cooperative bonds = stable processing capacity)
            pred_error = (norm_entropy - self._entropy_ema).abs()
            max_error = pred_error.max() + 1e-8
            k_temporal = (1.0 - pred_error / max_error).clamp(0.01, 0.99)
            if protection_scores is not None:
                # Well-protected heads get chronos pulled up.
                # 70% temporal + 30% cooperative support
                k_raw = 0.7 * k_temporal + 0.3 * protection_scores.clamp(0, 1)
            else:
                k_raw = k_temporal
            self._chronos_ema.lerp_(k_raw, alpha)

            # ==============================
            # PHASE 2: Hamiltonian conjugate pair coupling
            # ==============================
            # Damped Hamiltonian flow: each pair oscillates via coupled kicks.
            # From formal grounding (Full_Formal_Grounding.md §4.4):
            #   dq/dt = ∂H/∂p - γ(q - q₀) + forcing
            #   dp/dt = -∂H/∂q - γ(p - p₀) + forcing
            # Phase 1 above is the forcing + damping (EMA toward observation).
            # Phase 2 adds the Hamiltonian term (oscillatory coupling).
            # v3.4.1: Configurable omega and clamp bounds. Variant E makes these learnable.
            # PREVIOUS: omega = self.cfg.hamiltonian_coupling (constant 0.02)
            #           clamp_(0.01, 0.99) hard-coded
            if getattr(self, '_learned_omega', None) is not None:
                omega = torch.sigmoid(self._learned_omega) * 0.2  # learned, bounded [0, 0.2]
            else:
                omega = self.cfg.hamiltonian_coupling
            cl = getattr(self.cfg, 'prim_clamp_lo', 0.01)
            ch = getattr(self.cfg, 'prim_clamp_hi', 0.99)

            # v3.4.2: When cross-pair coupling is enabled, the full matrix_exp rotation
            # lives in compute_sigma's differentiable path (provides gradient to _coupling_params).
            # The Hamiltonian phase always uses the standard 3-pair Euler kicks on EMA buffers.
            # no_grad prevents in-place .add_() on buffers from conflicting with autograd's
            # version tracking (buffers were read by compute_sigma's torch.stack earlier).
            # no_grad: EMA buffer mutations must not conflict with autograd version tracking.
            # compute_sigma reads these buffers via torch.stack; in-place .add_() here would
            # invalidate the graph. Gradient to _coupling_params flows through compute_sigma only.
            with torch.no_grad():
                if (omega if isinstance(omega, float) else omega.item()) > 1e-6:
                    # 3 independent planar rotations (u(1)^3 Cartan subalgebra)
                    e_kick = omega * (self._coherence_ema - 0.5)
                    c_kick = -omega * (self._entropy_ema - 0.5)
                    self._entropy_ema.add_(e_kick).clamp_(cl, ch)
                    self._coherence_ema.add_(c_kick).clamp_(cl, ch)

                    i_kick = omega * (self._chronos_ema - 0.5)
                    k_kick = -omega * (self._intensity_ema - 0.5)
                    self._intensity_ema.add_(i_kick).clamp_(cl, ch)
                    self._chronos_ema.add_(k_kick).clamp_(cl, ch)

                    f_kick = omega * (self._valence_ema - 0.5)
                    v_kick = -omega * (self._friction_ema - 0.5)
                    self._friction_ema.add_(f_kick).clamp_(cl, ch)
                    self._valence_ema.add_(v_kick).clamp_(cl, ch)

                # v3.5: Null cone restoring force
                # Q = E²+I²+F² - C²-K²-V² measures distance from null cone.
                # Gradient descent on Q²: observables get pushed up when Q<0
                # (timelike/trapped), conjugates get pushed down. Reverses when Q>0.
                nc = getattr(self.cfg, 'null_cone_strength', 0.0)
                if nc > 0:
                    E = self._entropy_ema
                    I_p = self._intensity_ema
                    F_p = self._friction_ema
                    C = self._coherence_ema
                    K = self._chronos_ema
                    V = self._valence_ema
                    Q = (E**2 + I_p**2 + F_p**2) - (C**2 + K**2 + V**2)
                    # ∂(Q²)/∂x = 2Q·∂Q/∂x. For observables ∂Q/∂E = 2E, for conjugates ∂Q/∂C = -2C
                    # Update: x -= nc * 2Q * ∂Q/∂x  (gradient descent on Q²)
                    self._entropy_ema.add_(-nc * 2 * Q * E).clamp_(cl, ch)
                    self._intensity_ema.add_(-nc * 2 * Q * I_p).clamp_(cl, ch)
                    self._friction_ema.add_(-nc * 2 * Q * F_p).clamp_(cl, ch)
                    self._coherence_ema.add_(nc * 2 * Q * C).clamp_(cl, ch)
                    self._chronos_ema.add_(nc * 2 * Q * K).clamp_(cl, ch)
                    self._valence_ema.add_(nc * 2 * Q * V).clamp_(cl, ch)

            # Save current for next delta
            self._entropy_prev.copy_(blended)

            # ==============================
            # PHASE 3: WorldTrace self-model update (predictive coding)
            # ==============================
            # Key: surprise is computed against RAW observations (pre-EMA),
            # NOT against the smoothed EMA values. This matches v6 controller's
            # WorldTrace.surprise(obs) which compares predictions to raw input.
            # Raw values captured in local scope: blended (E), friction_raw (F),
            # c_raw (C), k_raw (K), plus _last_intensity (I) and v_scaled (V).
            self._update_self_model(
                raw_E=blended,
                raw_I=self._last_intensity,
                raw_F=friction_raw,
                raw_V=self._valence_ema,  # V is already smoothed (MACD output)
                raw_C=c_raw,
                raw_K=k_raw,
            )

    @torch.compiler.disable
    def _update_self_model(self, raw_E, raw_I, raw_F, raw_V, raw_C, raw_K):
        """Update the WorldTrace self-model: compute surprise vs RAW, then track.

        Maps to v6 controller's WorldTrace:
          - predicted{} → _pred_E, _pred_I, _pred_F, _pred_V, _pred_C, _pred_K
          - surprise(obs) → L2 distance between predicted and RAW observation
          - update(obs) → EMA toward RAW observation (alpha=0.3)

        v3.1e: Confidence-weighted energy (intensity × prediction error)
        ---------------------------------------------------------------
        Intensity is a proxy for attention weight concentration.
        Energy is prediction error (how wrong the self-model was).
        Their interaction gives 4 regimes:

          High I, Low E → Calibrated specialist. Leave alone.
          High I, High E → Confident & WRONG. Most dangerous. Steep σ drop.
          Low I, Low E  → Background/redundant. Ecological reallocation candidate.
          Low I, High E → Lost. Needs neighborhood guidance (bonds).

        The raw surprise (L2 across all primitives) treats all errors equally.
        Confidence-weighted surprise: energy * (0.5 + intensity) upweights
        errors from committed heads and downweights errors from diffuse heads.
        This gives ∂Energy/∂attention_weights from thermodynamic observables alone
        — the shadow model reads the weights from their signatures, like inferring
        molecular behavior from pressure and temperature.
        """
        # FP32 guard: sqrt, exp in sigma_target overflow in FP16
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            # Stack predicted vs raw observations
            predicted = torch.stack([
                self._pred_E, self._pred_I, self._pred_F,
                self._pred_V, self._pred_C, self._pred_K,
            ], dim=-1)  # [n_heads, 6]

            raw_obs = torch.stack([
                raw_E, raw_I, raw_F, raw_V, raw_C, raw_K,
            ], dim=-1)  # [n_heads, 6]

            # Base energy: L2 distance between predicted and raw
            energy = (predicted - raw_obs).pow(2).sum(dim=-1).sqrt()  # [n_heads]

            # Confidence weighting: intensity modulates how seriously we take the error.
            # High intensity = head is attending sharply (committed) → amplify error signal
            # Low intensity = head is diffuse (uncommitted) → dampen error signal
            # Scale: (0.5 + I) gives range [0.5, 1.5] — never fully ignore, but up to 3× ratio
            intensity = raw_I.clamp(0.0, 1.0)
            confidence_weight = 0.5 + intensity  # [n_heads], range [0.5, 1.5]
            surprise = energy * confidence_weight  # confidence-weighted energy

            self._self_surprise.copy_(surprise)

            # Update predictions toward raw observations (EMA, slower than raw)
            sm_alpha = self.cfg.self_model_alpha
            self._pred_E.lerp_(raw_E, sm_alpha)
            self._pred_I.lerp_(raw_I, sm_alpha)
            self._pred_F.lerp_(raw_F, sm_alpha)
            self._pred_V.lerp_(raw_V, sm_alpha)
            self._pred_C.lerp_(raw_C, sm_alpha)
            self._pred_K.lerp_(raw_K, sm_alpha)

    def compute_sigma_target(self) -> torch.Tensor:
        """Compute per-head sigma target from confidence-weighted surprise.

        v3.1e: Surprise now incorporates intensity (confidence weighting).
        The 4 regimes emerge naturally from the exponential mapping:

          High I, Low E → Low surprise → σ ≈ ceil (0.85) — calibrated specialist
          High I, High E → HIGH surprise → σ ≈ floor (0.15) — confident & WRONG, steep drop
          Low I, Low E → Moderate-low surprise → σ ≈ 0.5-0.6 — background head
          Low I, High E → Moderate surprise → σ ≈ 0.3-0.4 — lost, needs bonds

        High I amplifies both correct and incorrect signals — but when a head is
        confident AND wrong, the confidence weight pushes surprise into crisis range.
        This is the "natural humility from thermodynamics" — overconfident errors
        get hammered harder than uncertain errors.

        Uses exponential decay: sigma_target = floor + (ceil-floor) * exp(-sensitivity * surprise)

        Returns: [n_heads] sigma targets (detached, used as loss target)
        """
        # FP32 guard: exp() saturates in FP16 for surprise > ~5
        with torch.amp.autocast("cuda", enabled=False):
            surprise = self._self_surprise.float()  # [n_heads], confidence-weighted

            floor = self.cfg.self_model_sigma_floor
            ceil = self.cfg.self_model_sigma_ceil
            k = self.cfg.self_model_sensitivity

            # Exponential mapping: surprise=0 → ceil, surprise→∞ → floor
            sigma_target = floor + (ceil - floor) * torch.exp(-k * surprise)

        return sigma_target.detach()

    def get_current_primitives(self) -> torch.Tensor:
        """Return current per-head primitive values as a tensor.

        Returns: [n_heads, 6] tensor of [E, I, F, V, C, K] per head.
        All values are from EMA buffers (detached, no gradient).
        Used for inter-stage predictive coding targets.
        """
        return torch.stack([
            self._entropy_ema,
            self._last_intensity,
            self._friction_ema,
            self._valence_ema,
            self._coherence_ema,
            self._chronos_ema,
        ], dim=-1)  # [n_heads, 6]

    def predict_next_stage_primitives(self) -> torch.Tensor:
        """Predict the next stage's per-head primitives from current primitives.

        Uses inter_stage_predictor (nn.Linear) to map this stage's [E,I,F,V,C,K]
        to predicted next stage's [E,I,F,V,C,K]. Gradients flow through the predictor.

        Returns: [n_heads, 6] predicted primitives (with grad through predictor weights).
        """
        current = self.get_current_primitives().detach()  # [n_heads, 6]
        return self.inter_stage_predictor(current)  # [n_heads, 6]

    def compute_inter_stage_loss(self, next_stage_primitives: torch.Tensor) -> torch.Tensor:
        """Compute inter-stage prediction loss.

        MSE between this stage's prediction and the actual next stage's primitives.
        Gradients flow back through the predictor weights only (targets are detached).

        Args:
            next_stage_primitives: [n_heads, 6] actual primitives from the next stage
        Returns:
            scalar MSE loss
        """
        predicted = self.predict_next_stage_primitives()  # [n_heads, 6], has grad
        return F.mse_loss(predicted, next_stage_primitives.detach())

    # ============================================================
    # v3.4 Phase 3: Cooperative Predictive Attention (CPA) methods
    # ============================================================

    def predict_bonded_partners(self, cosurvival_matrix: torch.Tensor):
        """Predict bonded partners' primitive vectors.

        Uses shared bond_predictor to map this head's primitives → predicted partner primitives.
        Bond-weighted aggregation gives each head a summary of "what my partners are expected to do."

        Args:
            cosurvival_matrix: [H, H] co-survival bond strengths

        Returns:
            predicted_prims: [H, 6] — each head's predicted primitive vector
            partner_predicted_prims: [H, 6] — co-survival-weighted partner expectation
        """
        my_prims = self.get_current_primitives()  # [H, 6] (detached EMA buffers)
        predicted_prims = self.bond_predictor(my_prims.detach())  # [H, 6] — grad through predictor

        # Co-survival-weighted partner expectations
        cs_pos = cosurvival_matrix.clamp(min=0)  # Only positive bonds
        cs_norm = cs_pos / (cs_pos.sum(dim=1, keepdim=True) + 1e-8)  # [H, H] row-normalized

        # partner_predicted_prims[i] = weighted sum of predicted_prims[j] for all partners j
        partner_predicted_prims = torch.mm(cs_norm, predicted_prims.detach())  # [H, 6]

        return predicted_prims, partner_predicted_prims

    def compute_bond_prediction_loss(self, cosurvival_matrix: torch.Tensor) -> torch.Tensor:
        """MSE between predicted and actual partner primitives, weighted by bond strength.

        Gradient flows through bond_predictor weights only (targets are detached).
        Stronger bonds → stronger predictive pressure.

        Args:
            cosurvival_matrix: [H, H] co-survival bond strengths
        Returns:
            scalar prediction loss
        """
        my_prims = self.get_current_primitives()           # [H, 6]
        predicted = self.bond_predictor(my_prims.detach())  # [H, 6] — grad through predictor only

        actual = my_prims.detach()  # [H, 6] — stop gradient on targets

        # Each head's prediction error vs each other head's actual primitives
        errors = (predicted.unsqueeze(1) - actual.unsqueeze(0)).pow(2).sum(dim=-1)  # [H, H]

        # Weight by co-survival bond strength
        cs_pos = cosurvival_matrix.detach().clamp(min=0)
        mask = cs_pos > self.cfg.cooperative_prediction_bond_threshold

        if mask.sum() == 0:
            return torch.tensor(0.0, device=my_prims.device)

        weighted_errors = (errors * cs_pos)[mask]
        return weighted_errors.mean()

    def compute_complementarity_loss(self, cosurvival_matrix: torch.Tensor) -> torch.Tensor:
        """Penalize bonded heads with too-similar primitive vectors.

        Cosine similarity hinge loss: penalize if sim > margin for bonded pairs.
        Pushes bonded heads toward non-redundant ecological states.

        Args:
            cosurvival_matrix: [H, H] co-survival bond strengths
        Returns:
            scalar complementarity loss
        """
        my_prims = self.get_current_primitives()  # [H, 6]
        margin = self.cfg.complementarity_margin

        # Pairwise cosine similarity
        prims_norm = F.normalize(my_prims, dim=-1)  # [H, 6]
        sim = torch.mm(prims_norm, prims_norm.t())   # [H, H]

        # Only penalize bonded pairs (positive co-survival)
        cs_pos = cosurvival_matrix.detach().clamp(min=0)
        mask = cs_pos > self.cfg.cooperative_prediction_bond_threshold

        if mask.sum() == 0:
            return torch.tensor(0.0, device=my_prims.device)

        # Hinge: penalize similarity above margin
        excess_sim = (sim - margin).clamp(min=0)
        weighted = (excess_sim * cs_pos)[mask]
        return weighted.mean()

    def compute_sigma_complement(self, own_sigma: torch.Tensor,
                                  cosurvival_matrix: torch.Tensor) -> torch.Tensor:
        """Compute sigma offset that pushes bonded heads toward complementary uncertainty.

        If my sigma ≈ my bonded partners' mean sigma, nudge mine away.
        Active at inference time — this IS the adaptive mechanism.

        Args:
            own_sigma: [H] per-head sigma values
            cosurvival_matrix: [H, H] co-survival bond strengths
        Returns:
            [H] sigma offset to ADD to own_sigma
        """
        cs_pos = cosurvival_matrix.detach().clamp(min=0)
        cs_norm = cs_pos / (cs_pos.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted mean of partner sigmas
        partner_mean_sigma = torch.mv(cs_norm, own_sigma.detach())  # [H]

        # Complement: push apart from partner mean
        diff = own_sigma.detach() - partner_mean_sigma
        offset = diff.sign() * self.cfg.sigma_complement_strength

        # Only apply for heads with meaningful bonds
        bond_strength = cs_pos.sum(dim=1)
        bond_mask = (bond_strength > 0.1).float()
        return offset * bond_mask

    def compute_bond_diagnostics(self, cosurvival_matrix: torch.Tensor,
                                  head_sigmas: torch.Tensor) -> Dict:
        """Compute per-bonded-pair diagnostics for CPA health monitoring.

        Tracks whether bonded pairs are becoming:
        - coordinated and complementary (good: different primitives, different sigma)
        - locked into exaggerated divergence theater (bad: max divergence, no actual cooperation)
        - overbinding (bad: everything collapses to the same state)

        Returns dict with aggregate statistics over active bonds.
        """
        with torch.no_grad():
            my_prims = self.get_current_primitives()  # [H, 6]
            cs_pos = cosurvival_matrix.clamp(min=0)
            threshold = getattr(self.cfg, 'cooperative_prediction_bond_threshold', 0.01)
            mask = cs_pos > threshold  # [H, H] active bonds

            n_bonds = mask.sum().item()
            if n_bonds == 0:
                return {"n_active_bonds": 0}

            # Per-pair cosine similarity of primitives
            prims_norm = F.normalize(my_prims, dim=-1)
            prim_cosine = torch.mm(prims_norm, prims_norm.t())  # [H, H]

            # Per-pair sigma difference
            sigma_diff = (head_sigmas.unsqueeze(0) - head_sigmas.unsqueeze(1)).abs()  # [H, H]

            # Per-pair entropy difference
            e = self._entropy_ema
            entropy_diff = (e.unsqueeze(0) - e.unsqueeze(1)).abs()  # [H, H]

            # Co-survival strengths of active bonds
            bond_strengths = cs_pos[mask]

            # Blockade suppression (from last layer, if available)
            blockade_supp = getattr(self, '_last_excitation', torch.zeros_like(head_sigmas))
            supp_diff = (blockade_supp.unsqueeze(0) - blockade_supp.unsqueeze(1)).abs()

            return {
                "n_active_bonds": int(n_bonds // 2),  # symmetric, count unique pairs
                "bond_strength_mean": float(bond_strengths.mean().item()),
                "bond_strength_max": float(bond_strengths.max().item()),
                "prim_cosine_mean": float(prim_cosine[mask].mean().item()),
                "prim_cosine_min": float(prim_cosine[mask].min().item()),
                "sigma_diff_mean": float(sigma_diff[mask].mean().item()),
                "sigma_diff_max": float(sigma_diff[mask].max().item()),
                "entropy_diff_mean": float(entropy_diff[mask].mean().item()),
                "excitation_diff_mean": float(supp_diff[mask].mean().item()),
                # Health indicators:
                # coordination = low cosine (complementary) + high sigma diff (diverse)
                # divergence_theater = near-zero cosine + near-max sigma diff (suspicious)
                # overbinding = high cosine (redundant) + low sigma diff (stuck together)
            }

    def get_pairwise_distances(self) -> torch.Tensor:
        """
        Compute pairwise geodesic distances between heads on T3.
        Returns: [n_heads, n_heads] distance matrix

        v3.3 opt: Cached per forward pass. Call invalidate_distance_cache()
        at start of chain forward to refresh.
        """
        if hasattr(self, '_dist_cache') and self._dist_cache is not None:
            return self._dist_cache
        # Wrap positions to [0, 1]
        pos = self.head_positions % 1.0
        # Compute all pairwise distances
        pos_i = pos.unsqueeze(1)  # [n_heads, 1, 3]
        pos_j = pos.unsqueeze(0)  # [1, n_heads, 3]
        self._dist_cache = geodesic_distance_t3(pos_i, pos_j)  # [n_heads, n_heads]
        return self._dist_cache

    def invalidate_distance_cache(self):
        """Clear cached distance matrix. Call at start of each forward pass."""
        self._dist_cache = None


# ======================
# Per-Head Gradient Scaling by Valence
# ======================

class RydbergBlockade(nn.Module):
    """
    Computes blockade suppression between attention heads.

    Physics: V_ij = C6 / r^6
    - 1/r^6 is nearly a step function (distance 1: full, distance 2: 0.016)
    - Active heads suppress nearby heads
    - Forces local differentiation in T3 space

    Validated by QuEra quantum experiment:
    - 4.5 um: full blockade (competition)
    - 41.5 um: zero interaction (independence)
    - r = 0.646, p < 10^-6
    """

    def __init__(self, cfg: T3v3Config):
        super().__init__()
        self.cfg = cfg
        self.exponent = cfg.blockade_exponent
        self.strength = cfg.blockade_strength
        self.blockade_radius = cfg.blockade_radius_init  # Distance at which blockade = 50%
        self.register_buffer("_global_step", torch.tensor(0, dtype=torch.long))
        # v3.1: Store per-head suppression for feeding into Coherence primitive
        self.register_buffer("_last_suppression", torch.zeros(cfg.n_heads))

    def forward(
        self,
        head_excitation: torch.Tensor,       # [n_heads] excitation (inverse entropy)
        distances: torch.Tensor,              # [n_heads, n_heads]
        blockade_modulation: Optional[torch.Tensor] = None,  # [n_heads, n_heads] from co-survival
    ) -> torch.Tensor:
        """
        Compute per-head suppression from Rydberg blockade.

        head_excitation: inverse attention entropy (sharp = excited = blockades neighbors)
        blockade_modulation: co-survival modulation (< 1 = cooperating, > 1 = interfering)

        Returns: [n_heads] suppression factor in [0, 1]
                 0 = no suppression, 1 = fully suppressed
        """
        if not self.cfg.blockade_enabled:
            return torch.zeros(distances.shape[0], device=distances.device)

        # Warmup: ramp blockade from 0 -> full over warmup_steps
        # Genesis insight: don't blockade during Phase I (primordial soup)
        warmup = self.cfg.blockade_warmup_steps
        if warmup > 0:
            # torch.compile-friendly: no .item() call
            ramp = (self._global_step.float() / warmup).clamp(max=1.0)
        else:
            ramp = 1.0

        # FP32 guard: pow(6) and 1/r^6 underflow in FP16
        with torch.amp.autocast("cuda", enabled=False):
            dist_f = distances.float()
            # Blockade interaction weight: 1 / (1 + (d/r0)^exponent)
            blockade_weight = 1.0 / (1.0 + (dist_f / self.blockade_radius).pow(self.exponent))

            # Zero self-interaction
            blockade_weight = blockade_weight * (1.0 - torch.eye(
                distances.shape[0], device=distances.device
            ))

            # Co-survival modulation: cooperating heads block less, interfering block more
            if blockade_modulation is not None:
                blockade_weight = blockade_weight * blockade_modulation.float()

            # Excitation already [n_heads], normalize to [0, 1]
            excitation = head_excitation.float()
            if excitation.max() > excitation.min() + 1e-8:
                excitation = (excitation - excitation.min()) / (excitation.max() - excitation.min() + 1e-8)

            # Suppression from neighbors: sum(neighbor_excitation * blockade_weight)
            suppression = torch.matmul(blockade_weight, excitation)  # [n_heads]

            # Scale by strength AND warmup ramp, then clamp
            suppression = (suppression * self.strength * ramp).clamp(0.0, 0.95)

        # v3.1: Store for Coherence primitive coupling
        with torch.no_grad():
            self._last_suppression.copy_(suppression.detach())

        return suppression.to(head_excitation.dtype)


# ======================
# Co-Survival Tracker
# ======================

class CosurvivalTracker(nn.Module):
    """
    Tracks fitness correlation between attention heads.

    Physics: gravitational coupling with 1/r decay
    - Heads with correlated loss reduction form bonds
    - Heads that interfere get decoupled
    - Bonds are weighted by geodesic distance (nearby bonds stronger)

    Validated by Genesis lattice:
    - 196K+ protected connections from co-survival
    - 70x more stable under noise than fresh connections
    - 2.76x more bonds than alternatives
    """

    def __init__(self, n_heads: int, cfg: T3v3Config):
        super().__init__()
        self.n_heads = n_heads
        self.cfg = cfg

        # Co-survival matrix: [n_heads, n_heads]
        # Positive = correlated success (bond)
        # Negative = interference (decouple)
        self.register_buffer("cosurvival", torch.zeros(n_heads, n_heads))

        # Running mean of per-head loss for correlation computation
        self.register_buffer("head_loss_ema", torch.zeros(n_heads))
        self.register_buffer("head_loss_var", torch.ones(n_heads))

        # Step counter
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    @torch.compiler.disable
    def update(
        self,
        per_head_loss: torch.Tensor,   # [n_heads]
        distances: torch.Tensor,        # [n_heads, n_heads]
        valence: torch.Tensor = None,   # [n_heads] optional valence EMA for modulation
    ):
        """
        Update co-survival scores based on fitness correlation.

        Called periodically (every cosurvival_update_interval steps).
        If valence is provided and cosurvival_valence_modulation=True,
        bonds are weighted by valence agreement between head pairs.
        """
        if not self.cfg.cosurvival_enabled:
            return

        self._step += 1

        if self._step % self.cfg.cosurvival_update_interval != 0:
            # Update running stats but don't update bonds
            alpha = 0.01
            self.head_loss_ema = (1 - alpha) * self.head_loss_ema + alpha * per_head_loss.detach()
            self.head_loss_var = (1 - alpha) * self.head_loss_var + alpha * (per_head_loss.detach() - self.head_loss_ema).pow(2)
            return

        # FP32 guard: outer product + division sensitive to FP16 precision
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            # Fitness = negative loss (lower loss = higher fitness)
            fitness = -per_head_loss.detach().float()

            # Center fitness
            fitness_centered = fitness - fitness.mean()

            # Fitness agreement: outer product of centered fitness
            agreement = torch.outer(fitness_centered, fitness_centered)

            # Normalize by variance
            std = fitness_centered.abs().max() + 1e-8
            agreement = agreement / (std * std)

            # v2.5: Valence modulation
            if self.cfg.cosurvival_valence_modulation and valence is not None:
                v = valence.detach().float()
                valence_agreement = 1.0 - (v.unsqueeze(0) - v.unsqueeze(1)).abs()
                agreement = agreement * valence_agreement

            # Gravitational weighting: nearby heads' co-survival matters more
            grav_weight = 1.0 / (1.0 + self.cfg.grav_k * distances.float())

            # Update co-survival with gravitational weighting
            increment = agreement * grav_weight * 0.1

            # Accumulate
            self.cosurvival = self.cosurvival * self.cfg.cosurvival_decay + increment

            # Update running stats
            alpha = 0.01
            self.head_loss_ema = (1 - alpha) * self.head_loss_ema + alpha * per_head_loss.detach().float()

    def get_coupling_matrix(self) -> torch.Tensor:
        """
        Get coupling matrix for gradient modulation.

        Returns: [n_heads, n_heads] matrix where:
        - Positive entries = correlated heads (couple gradients)
        - Negative entries = interfering heads (decouple gradients)
        """
        if not self.cfg.cosurvival_enabled:
            return torch.eye(self.n_heads, device=self.cosurvival.device)

        # Normalize co-survival to [-1, 1]
        cs_max = self.cosurvival.abs().max() + 1e-8
        cs_norm = self.cosurvival / cs_max

        # Protection factor: positive co-survival protects, negative accelerates decay
        # This modulates how much heads' gradients are correlated
        coupling = torch.eye(self.n_heads, device=self.cosurvival.device)
        coupling = coupling + cs_norm * self.cfg.cosurvival_lr_coupling

        return coupling

    def get_protection_scores(self) -> torch.Tensor:
        """
        Per-head protection score from co-survival bonds.

        High protection = many positive bonds = stable, protected head
        Low protection = few bonds or negative = vulnerable, plastic
        """
        if not self.cfg.cosurvival_enabled:
            return torch.ones(self.n_heads, device=self.cosurvival.device)

        # Sum positive co-survival per head
        positive_bonds = self.cosurvival.clamp(min=0).sum(dim=1)
        # Normalize
        max_bonds = positive_bonds.max() + 1e-8
        return positive_bonds / max_bonds

    def get_blockade_modulation(self) -> torch.Tensor:
        """
        Modulate blockade strength based on co-survival relationships.

        Returns [n_heads, n_heads] where:
        - Values < 1.0: reduced blockade (cooperating pairs)
        - Values > 1.0: increased blockade (interfering pairs)

        Physics: cooperative heads shouldn't suppress each other.
        Redundant/interfering heads should feel stronger pressure.
        The universe doesn't like redundancy.

        v3.3 opt: Cached per forward pass. Call invalidate_blockade_cache()
        at start of chain forward to refresh.
        """
        if hasattr(self, '_blockade_mod_cache') and self._blockade_mod_cache is not None:
            return self._blockade_mod_cache

        if not self.cfg.cosurvival_enabled:
            return torch.ones(self.n_heads, self.n_heads, device=self.cosurvival.device)

        cs_max = self.cosurvival.abs().max() + 1e-8
        cs_norm = self.cosurvival / cs_max

        # Positive co-survival -> reduce blockade (cooperation, multiply < 1)
        # Negative co-survival -> increase blockade (interference, multiply > 1)
        modulation = 1.0 - cs_norm * 0.5  # Range: [0.5, 1.5]
        self._blockade_mod_cache = modulation.clamp(0.3, 1.7)
        return self._blockade_mod_cache

    def invalidate_blockade_cache(self):
        """Clear cached blockade modulation. Call at start of each forward pass."""
        self._blockade_mod_cache = None


# ======================
# Rydberg Attention
# ======================

class RydbergAttention(nn.Module):
    """
    Multi-head attention with Rydberg blockade and co-survival.

    Each head:
    1. Has a position in T3 space
    2. Computes its own sigma from per-head primitives
    3. Is suppressed by nearby active heads (blockade -> sparsity)
    4. Forms bonds with co-surviving heads (topology preservation)
    """

    def __init__(self, d_model: int, n_heads: int, cfg: T3v3Config, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = getattr(cfg, 'd_head', 0) or (d_model // n_heads)
        assert (getattr(cfg, 'd_head', 0) > 0) or (d_model % n_heads == 0), \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads}) when cfg.d_head is not set"
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.cfg = cfg
        self.use_rope = getattr(cfg, 'use_rope', False)

        # GQA support: separate KV head count
        self.n_kv_heads = getattr(cfg, 'n_kv_heads', 0)
        if self.n_kv_heads <= 0:
            self.n_kv_heads = n_heads  # MHA fallback
        assert n_heads % self.n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        self.n_kv_groups = n_heads // self.n_kv_heads

        # Projections (Q always full, K/V may be smaller for GQA)
        attn_bias = getattr(cfg, 'attn_bias', True)
        attn_out_bias = getattr(cfg, 'attn_out_bias', True)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=attn_bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=attn_bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=attn_bias)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=attn_out_bias)

        # QK-norm (Gemma 3 / Llama-3.1-style): per-head RMSNorm of Q and K before attention
        self.use_qk_norm = getattr(cfg, 'use_qk_norm', False)
        if self.use_qk_norm:
            norm_eps = getattr(cfg, 'norm_eps', 1e-6)
            self.q_norm = T3RMSNorm(self.d_head, eps=norm_eps)
            self.k_norm = T3RMSNorm(self.d_head, eps=norm_eps)

        self.dropout = nn.Dropout(dropout)

        # Blockade computation
        self.blockade = RydbergBlockade(cfg)

        # Store per-head activation magnitudes for blockade
        self.register_buffer("_head_activations", torch.zeros(n_heads))
        # Store per-head attention entropy for specialization measurement
        self.register_buffer("_head_entropy", torch.zeros(n_heads))

    def forward(
        self,
        x: torch.Tensor,           # [batch, seq, d_model]
        head_sigmas: torch.Tensor,  # [n_heads]
        distances: torch.Tensor,    # [n_heads, n_heads]
        mask: Optional[torch.Tensor] = None,
        blockade_mod: Optional[torch.Tensor] = None,  # [n_heads, n_heads] co-survival modulation
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # RoPE (cos, sin)
        eco_k_offset: Optional[torch.Tensor] = None,  # v3.4: [n_heads, d_head] eco K-bias
        v1_residual: Optional[torch.Tensor] = None,   # v3.7 (v5.1 port): V from layer 0 of this stage
        return_v: bool = False,                        # v3.7: whether to return v_pre_blend
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (output, per_head_entropy) or (output, per_head_entropy, v_pre_blend) if return_v

        v3.7 (V1 residual port from v5.1): when v1_residual is provided (layers 1+ in
        a stage), V is blended:
            V_blend = V_n + λ·(V1 − V_n)
        where λ is ecology-gated per head (λ_h = σ_h):
            High σ (uncertain) → lean on V1 (raw tokens)
            Low σ (confident specialist) → trust V_n (own transformation)
        Zero new params; V1 is just a cache of layer-0's pre-blend V.
        """
        batch, seq_len, _ = x.shape
        bypass = getattr(self.cfg, 'bypass_ecology', False)
        use_flex = (getattr(self.cfg, 'use_flex_attention', False)
                    and HAS_FLEX_ATTENTION and not bypass)

        # Project to Q, K, V and reshape to heads
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.d_head)

        # QK-norm (Gemma 3): per-head RMSNorm over the d_head dimension before RoPE
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # v3.7: Save unblended V for return_v (layer 0 caches as V1)
        v_pre_blend = v if return_v else None

        # v3.7: V1 residual blend — ecology-gated per-head, applied BEFORE K-bias and RoPE
        # V_blend = V_n + λ·(V1 − V_n) = (1−λ)·V_n + λ·V1
        if v1_residual is not None and not bypass:
            v1_gating = getattr(self.cfg, 'v1_residual_gating', 'sigma')
            if v1_gating == 'sigma':
                # Per-head λ = σ_h. For GQA, mean σ over Q heads sharing each KV group.
                if self.n_kv_groups > 1:
                    sigma_kv = head_sigmas.detach().view(self.n_kv_heads, self.n_kv_groups).mean(dim=1)
                else:
                    sigma_kv = head_sigmas.detach()
                lam = sigma_kv.view(1, 1, self.n_kv_heads, 1).to(v.dtype)
            elif v1_gating == 'inverse_sigma':
                if self.n_kv_groups > 1:
                    sigma_kv = head_sigmas.detach().view(self.n_kv_heads, self.n_kv_groups).mean(dim=1)
                else:
                    sigma_kv = head_sigmas.detach()
                lam = (1.0 - sigma_kv).view(1, 1, self.n_kv_heads, 1).to(v.dtype)
            else:  # "fixed" — validated to hurt when ecology is alive (3.3a result)
                fixed_lam = getattr(self.cfg, 'v1_residual_fixed_lambda', 0.5)
                lam = torch.tensor(fixed_lam, device=v.device, dtype=v.dtype)
            v = v + lam * (v1_residual.to(v.dtype) - v)

        # v3.4 Phase 2: Apply eco-conditioned K-bias (changes WHAT heads attend to)
        # Offset is pre-computed by HeadState and passed through from stage forward.
        # Applied to K BEFORE RoPE and score computation. For GQA, offset is
        # broadcast from Q heads to KV heads (mean over KV group).
        if eco_k_offset is not None and not bypass:
            eco = getattr(self, '_ecology_strength', 1.0)
            if self.n_kv_groups > 1:
                # GQA: average offset across heads sharing each KV group
                offset_grouped = eco_k_offset.view(self.n_kv_heads, self.n_kv_groups, -1).mean(dim=1)
                k = k + (eco * offset_grouped).unsqueeze(0).unsqueeze(1).to(k.dtype)
            else:
                k = k + (eco * eco_k_offset).unsqueeze(0).unsqueeze(1).to(k.dtype)

        # Apply RoPE if provided (before KV repeat -- saves compute)
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if use_flex:
            flex_result = self._forward_flex(x, q, k, v, head_sigmas, distances,
                                             mask, blockade_mod, batch, seq_len)
            if return_v:
                # Re-pack to add v_pre_blend
                out, head_entropy = flex_result
                return out, head_entropy, v_pre_blend
            return flex_result

        # ========================
        # Standard PyTorch path
        # ========================

        # GQA: repeat K/V to match Q head count
        if self.n_kv_groups > 1:
            k = repeat_kv(k, self.n_kv_groups)
            v = repeat_kv(v, self.n_kv_groups)

        # Compute raw attention scores: [batch, n_heads, seq, seq]
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * self.scale

        if not bypass:
            # _ecology_strength: ramps ecology effects from 0 (bypass) to 1 (full).
            # Set externally by training loop during warmup. Defaults to 1.0 for existing code.
            eco = getattr(self, '_ecology_strength', 1.0)

            # === PER-HEAD sigma MODULATION ===
            # v3.4.1: Configurable stop-gradient and temperature range.
            # PREVIOUS (v3.1):
            #   sigma_sg = head_sigmas.detach()
            #   raw_temps = 0.5 + sigma_sg * 1.0  # [0.5, 1.5]
            #   temperatures = 1.0 + eco * (raw_temps - 1.0)
            # NEW: sigma_stop_gradient=False lets CE flow through sigma→attention,
            # and temp_range_lo/hi widen the modulation range for meaningful effect.
            cfg = getattr(self, '_head_state_cfg', None)
            if cfg is not None and not getattr(cfg, 'sigma_stop_gradient', True):
                sigma_mod = head_sigmas  # WITH gradient — CE can teach sigma
            else:
                sigma_mod = head_sigmas.detach()  # Original behavior
            t_lo = getattr(cfg, 'temp_range_lo', 0.5) if cfg is not None else 0.5
            t_hi = getattr(cfg, 'temp_range_hi', 1.5) if cfg is not None else 1.5
            raw_temps = t_lo + sigma_mod * (t_hi - t_lo)  # [t_lo, t_hi]
            temperatures = 1.0 + eco * (raw_temps - 1.0)  # ramp: 1.0 at eco=0, raw at eco=1
            scores = scores / temperatures.view(1, -1, 1, 1)

            # === PER-HEAD LOCALITY BIAS ===
            positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            distance_matrix = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
            locality_strengths = (0.5 - sigma_mod.detach()).clamp(min=0) * 0.2 * eco  # ramped
            locality_bias = -distance_matrix.unsqueeze(0) * locality_strengths.view(-1, 1, 1)
            scores = scores + locality_bias.unsqueeze(0).to(scores.dtype)

        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1).to(v.dtype)  # ensure dtype match for compiled einsum
        attn = self.dropout(attn)

        # === ATTENTION ENTROPY (excitation signal for blockade) ===
        # Compute BEFORE blockade so it drives suppression decisions
        # Sharp attention (low entropy) = excited head = suppresses neighbors
        # FP32 guard: FP16 softmax outputs flush small probs to 0,
        # then 0 * log(0) = 0 * -inf = NaN (IEEE 754). Must compute in FP32.
        with torch.amp.autocast("cuda", enabled=False):
            attn_f = attn.float()
            attn_for_entropy = attn_f.clamp(min=1e-8)
            entropy_per_pos = -(attn_for_entropy * attn_for_entropy.log()).sum(dim=-1)  # [batch, n_heads, seq]
            head_entropy = entropy_per_pos.mean(dim=(0, 2))  # [n_heads]

        if not bypass:
            # Excitation = inverse entropy (sharp attention = excited = blockades neighbors)
            max_entropy = math.log(max(seq_len, 2))
            excitation = ((max_entropy - head_entropy) / (max_entropy + 1e-8)).to(x.dtype)  # [n_heads], ~[0, 1]

            # === RYDBERG BLOCKADE ===
            # Excited heads suppress nearby heads (modulated by co-survival)
            suppression = self.blockade(excitation, distances, blockade_mod)  # [n_heads]

            # Apply suppression to attention output (post-softmax)
            # The head still "sees" everything but its contribution is reduced
            out = torch.einsum("bhqk,bkhd->bqhd", attn, v)  # [batch, seq, n_heads, d_head]
            # Ramp suppression with ecology strength: at eco=0, no suppression (head_scale=1)
            head_scale = (1.0 - eco * suppression).view(1, 1, -1, 1).to(out.dtype)
            out = out * head_scale
        else:
            # Bypass: vanilla attention, no blockade
            out = torch.einsum("bhqk,bkhd->bqhd", attn, v)  # [batch, seq, n_heads, d_head]

        # Reshape and project (n_heads * d_head may exceed d_model for wide-head models like Gemma 3)
        out = out.reshape(batch, seq_len, self.n_heads * self.d_head)
        out = self.out_proj(out)

        # Store for diagnostics (outside compiled region)
        self._store_diagnostics(head_entropy, excitation if not bypass else None)

        if return_v:
            return out, head_entropy, v_pre_blend
        return out, head_entropy

    def _forward_flex(
        self,
        x: torch.Tensor,
        q: torch.Tensor,    # [B, S, H_q, D]
        k: torch.Tensor,    # [B, S, H_kv, D]
        v: torch.Tensor,    # [B, S, H_kv, D]
        head_sigmas: torch.Tensor,
        distances: torch.Tensor,
        mask: Optional[torch.Tensor],
        blockade_mod: Optional[torch.Tensor],
        batch: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FlexAttention path: fused flash attention with score_mod for
        temperature + locality. GQA handled natively (no repeat_kv).
        Attention entropy computed via subsampled probe positions.
        """
        eco = getattr(self, '_ecology_strength', 1.0)

        # === SIGMA-DERIVED MODULATIONS (stop-gradient) ===
        sigma_sg = head_sigmas.detach()
        raw_temps = 0.5 + sigma_sg * 1.0  # [0.5, 1.5]
        temperatures = 1.0 + eco * (raw_temps - 1.0)
        locality_strengths = (0.5 - sigma_sg).clamp(min=0) * 0.2 * eco

        # === FLEX ATTENTION ===
        # FlexAttention expects [B, H, S, D] layout, all same dtype.
        # RoPE may upcast Q/K to fp32 (sin/cos tables), so cast back to match V.
        compute_dtype = v.dtype
        q_t = q.to(compute_dtype).transpose(1, 2).contiguous()  # [B, H_q, S, D]
        k_t = k.to(compute_dtype).transpose(1, 2).contiguous()  # [B, H_kv, S, D]
        v_t = v.transpose(1, 2).contiguous()  # [B, H_kv, S, D]

        # Cache block mask — seq_len is constant during training
        if not hasattr(self, '_cached_block_mask') or self._cached_seq_len != seq_len:
            def causal_mask_fn(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx
            self._cached_block_mask = create_block_mask(
                causal_mask_fn, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len,
                device=x.device,
            )
            self._cached_seq_len = seq_len

        # flex_attention with GQA — compiled wrapper handles score_mod + fusion
        out_t = _compiled_flex_attn(
            q_t, k_t, v_t,
            temperatures, locality_strengths,
            self._cached_block_mask,
            self.scale, self.n_kv_groups > 1,
        )
        # Back to [B, S, H, D]
        out = out_t.transpose(1, 2)

        # === SUBSAMPLED ATTENTION ENTROPY ===
        # FlexAttention doesn't expose attention weights, so we compute
        # approximate head entropy from N=8 probe positions. This feeds
        # the EMA (alpha=0.05) so noise is heavily smoothed.
        head_entropy = self._subsampled_entropy(
            q_t, k, seq_len, temperatures, locality_strengths, x.device, x.dtype
        )

        # === BLOCKADE (post-hoc on flex output) ===
        max_entropy = math.log(max(seq_len, 2))
        excitation = ((max_entropy - head_entropy) / (max_entropy + 1e-8)).to(x.dtype)
        suppression = self.blockade(excitation, distances, blockade_mod)
        head_scale = (1.0 - eco * suppression).view(1, 1, -1, 1).to(out.dtype)
        out = out * head_scale

        # Reshape and project (n_heads * d_head may exceed d_model for wide-head models like Gemma 3)
        out = out.reshape(batch, seq_len, self.n_heads * self.d_head)
        out = self.out_proj(out)

        self._store_diagnostics(head_entropy, excitation)
        return out, head_entropy

    @torch.no_grad()
    def _subsampled_entropy(
        self,
        q_t: torch.Tensor,    # [B, H_q, S, D]
        k_raw: torch.Tensor,  # [B, S, H_kv, D] (pre-repeat)
        seq_len: int,
        temperatures: torch.Tensor,  # [H_q]
        locality_strengths: torch.Tensor,  # [H_q]
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Compute approximate per-head attention entropy by probing N=8
        query positions from the latter 3/4 of the sequence. Includes
        temperature and locality bias to match standard path.
        Cost: O(N * H * S) vs O(H * S^2).
        """
        # Sample from latter 3/4 of sequence (avoid early positions with trivially low entropy)
        n_probe = min(8, seq_len)
        start_pos = max(1, seq_len // 4)
        probe_indices = torch.linspace(start_pos, seq_len - 1, n_probe, device=device).long()

        # Probe queries: [B, H_q, N, D]
        q_probe = q_t[:, :, probe_indices, :].to(dtype)

        # Expand K for all Q heads (GQA) — only for N probe positions, negligible memory
        if self.n_kv_groups > 1:
            k_expanded = repeat_kv(k_raw.to(dtype), self.n_kv_groups)  # [B, S, H_q, D]
        else:
            k_expanded = k_raw.to(dtype)
        k_expanded_t = k_expanded.transpose(1, 2)  # [B, H_q, S, D]

        # Probe scores: [B, H_q, N, S]
        probe_scores = torch.einsum("bhnd,bhsd->bhns", q_probe, k_expanded_t) * self.scale

        # Apply temperature (matches standard path)
        probe_scores = probe_scores / temperatures.view(1, -1, 1, 1)

        # Apply locality bias (matches standard path)
        if locality_strengths.abs().max() > 1e-8:
            seq_range_f = torch.arange(seq_len, device=device, dtype=torch.float32)
            probe_pos = probe_indices.float()  # [N]
            dist = (probe_pos.unsqueeze(1) - seq_range_f.unsqueeze(0)).abs()  # [N, S]
            loc_bias = -dist.unsqueeze(0) * locality_strengths.view(-1, 1, 1)  # [H, N, S]
            probe_scores = probe_scores + loc_bias.unsqueeze(0).to(probe_scores.dtype)

        # Causal mask for probe positions
        seq_range = torch.arange(seq_len, device=device)
        causal = probe_indices.unsqueeze(1) >= seq_range.unsqueeze(0)  # [N, S]
        probe_scores.masked_fill_(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax + entropy — FP32 guard: 0*log(0) = NaN in FP16
        with torch.amp.autocast("cuda", enabled=False):
            probe_attn = F.softmax(probe_scores.float(), dim=-1)
            probe_attn = probe_attn.clamp(min=1e-8)
            probe_entropy = -(probe_attn * probe_attn.log()).sum(dim=-1)  # [B, H_q, N]
            head_entropy = probe_entropy.mean(dim=(0, 2))  # [H_q]

        return head_entropy

    @torch.compiler.disable
    def _store_diagnostics(self, head_entropy, excitation=None):
        with torch.no_grad():
            if excitation is not None:
                self._head_activations.copy_(excitation.detach())
            self._head_entropy.copy_(head_entropy.detach())


# ======================
# T3 v3 Layer
# ======================

class T3v3Layer(nn.Module):
    """Transformer layer with Rydberg blockade and co-survival coupling."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, cfg: T3v3Config, dropout: float):
        super().__init__()

        self.attention = RydbergAttention(d_model, n_heads, cfg, dropout)
        self.attention._head_state_cfg = cfg  # v3.4.1: config ref for sigma/temp control

        # FFN: standard GELU or SwiGLU (for Qwen/LLaMA transfer) or GeGLU (Gemma 3)
        ffn_type = getattr(cfg, 'ffn_type', 'gelu')
        ffn_bias = getattr(cfg, 'ffn_bias', True)
        if ffn_type == 'swiglu':
            self.ff = SwiGLUFFN(d_model, d_ff, bias=ffn_bias, dropout=dropout)
        elif ffn_type == 'geglu':
            self.ff = GeGLUFFN(d_model, d_ff, bias=ffn_bias, dropout=dropout)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=ffn_bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model, bias=ffn_bias),
                nn.Dropout(dropout),
            )

        self.norm1 = make_norm(d_model, cfg)
        self.norm2 = make_norm(d_model, cfg)

        # Post-sublayer norms (Gemma 3 style: applied before residual add)
        self.use_post_norms = getattr(cfg, 'use_post_norms', False)
        if self.use_post_norms:
            self.post_attn_norm = make_norm(d_model, cfg)
            self.post_ff_norm = make_norm(d_model, cfg)

        self.cfg = cfg

    def forward(
        self,
        x: torch.Tensor,
        head_sigmas: torch.Tensor,
        distances: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        blockade_mod: Optional[torch.Tensor] = None,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        eco_k_offset: Optional[torch.Tensor] = None,  # v3.4: [n_heads, d_head] eco K-bias
        v1_residual: Optional[torch.Tensor] = None,   # v3.7 (v5.1 port): cached V from layer 0
        return_v: bool = False,                        # v3.7: return v_pre_blend for layer 0 caching
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (output, per_head_entropy) or (output, per_head_entropy, v_pre_blend) if return_v
        """
        # Pre-norm attention with blockade + co-survival modulation
        attn_result = self.attention(
            self.norm1(x), head_sigmas, distances, mask=attn_mask,
            blockade_mod=blockade_mod, rope_cos_sin=rope_cos_sin,
            eco_k_offset=eco_k_offset,
            v1_residual=v1_residual, return_v=return_v,
        )
        if return_v:
            attn_out, head_entropy, v_pre_blend = attn_result
        else:
            attn_out, head_entropy = attn_result
        if self.use_post_norms:
            attn_out = self.post_attn_norm(attn_out)
        x = x + attn_out

        # FFN with mean sigma gating (average across heads)
        ff_out = self.ff(self.norm2(x))
        if self.use_post_norms:
            ff_out = self.post_ff_norm(ff_out)
        if getattr(self.cfg, 'bypass_ecology', False):
            x = x + ff_out
        else:
            eco = getattr(self, '_ecology_strength', 1.0)
            # v3.1: stop-gradient — sigma modulates but CE can't hack the MLP
            sigma_mean = head_sigmas.detach().mean()
            raw_ff_scale = (0.3 + 0.7 * sigma_mean).to(x.dtype)
            ff_scale = 1.0 + eco * (raw_ff_scale - 1.0)  # ramp: 1.0 at eco=0, raw at eco=1
            x = x + ff_out * ff_scale

        if return_v:
            return x, head_entropy, v_pre_blend
        return x, head_entropy


# ======================
# Full T3 v3 Model
# ======================

class T3v3Transformer(nn.Module):
    """
    T3 v3.0: Rydberg-Coupled Transformer with 6-Primitive Ecology.

    Key additions over v2:
    - 6-primitive system (E, I, F, V, C, K)
    - Blended E (output entropy + attention entropy)
    - Fristonian valence (dual-EMA MACD)
    - Per-head gradient scaling by valence
    - Sigma warmup interpolation
    """

    def __init__(self, cfg: T3v3Config):
        super().__init__()
        self.cfg = cfg

        # Embeddings
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # Per-head T3 state
        self.head_state = HeadState(cfg.n_heads, cfg.d_model, cfg)

        # Co-survival tracker
        self.cosurvival = CosurvivalTracker(cfg.n_heads, cfg)

        # Transformer layers
        self.layers = nn.ModuleList([
            T3v3Layer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        # Output
        self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.embed_norm = make_norm(cfg.d_model, cfg)
        self.final_norm = make_norm(cfg.d_model, cfg)
        self.dropout_layer = nn.Dropout(cfg.dropout)

        # Weight tying
        self.output_proj.weight = self.embed.weight

        # sigma-aware output temperature
        self.register_buffer("_global_sigma", torch.tensor(0.5))

        # Spectral monitoring
        self.register_buffer("_step_count", torch.tensor(0, dtype=torch.long))
        self._spectral_history = []

        # Init
        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, T3RMSNorm):
            torch.nn.init.ones_(module.weight)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_state: bool = False,
        update_state: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        input_ids: [batch, seq_len]
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Embedding
        positions = torch.arange(seq_len, device=device)
        h = self.embed(input_ids) + self.pos_embed(positions)
        h = self.embed_norm(h)
        h = self.dropout_layer(h)

        # Invalidate caches from previous forward pass (prevents graph retention)
        self.head_state.invalidate_distance_cache()
        if self.cfg.cosurvival_enabled:
            self.cosurvival.invalidate_blockade_cache()

        # Compute pairwise head distances (for blockade + co-survival)
        distances = self.head_state.get_pairwise_distances()  # [n_heads, n_heads]

        # Co-survival blockade modulation (cooperating heads block less)
        blockade_mod = self.cosurvival.get_blockade_modulation() if self.cfg.cosurvival_enabled else None

        # Reshape hidden to per-head view for sigma computation
        h_per_head = h.view(batch, seq_len, self.cfg.n_heads, self.cfg.d_model // self.cfg.n_heads)

        # Compute per-head sigma values (DIFFERENTIABLE)
        head_sigmas = self.head_state.compute_head_sigmas(h_per_head)  # [n_heads]

        # Store global sigma for output temperature
        global_sigma = head_sigmas.mean()
        with torch.no_grad():
            self._global_sigma.copy_(global_sigma.detach())

        # Causal mask
        attn_mask = self._causal_mask(seq_len, device)

        # Pass through layers, collecting per-head entropy
        all_head_entropy = []
        for layer in self.layers:
            h, head_entropy = layer(h, head_sigmas, distances, attn_mask=attn_mask,
                                     blockade_mod=blockade_mod)
            all_head_entropy.append(head_entropy)

        h = self.final_norm(h)

        # Output with sigma-aware temperature (v3.1: stop-gradient)
        logits = self.output_proj(h)
        temperature = 1.5 - global_sigma.detach() * 0.8  # [0.7, 1.5]
        temperature = torch.clamp(temperature, 0.5, 2.0)
        logits = logits / temperature

        # Update co-survival and blockade step counter (if training)
        if update_state and self.training:
            self._step_count += 1

            # Increment blockade warmup counter across all layers
            for layer in self.layers:
                layer.attention.blockade._global_step += 1

            # Co-survival update: entropy as fitness proxy
            # (lower entropy = more focused = fitter; update() negates to get fitness)
            avg_entropy = torch.stack(all_head_entropy).mean(dim=0)  # [n_heads]
            # Pass valence EMA for valence-modulated co-survival (v2.5)
            valence = self.head_state._valence_ema if self.cfg.grounded_primitives else None
            self.cosurvival.update(avg_entropy, distances, valence=valence)

            # v3.0+: Update grounded primitive EMAs from actual attention entropy
            if self.cfg.grounded_primitives:
                # v3.1: Gather ecological signals for C and K grounding
                protection = self.cosurvival.get_protection_scores() if self.cfg.cosurvival_enabled else None
                suppression = None
                if self.cfg.blockade_enabled:
                    layer_supps = []
                    for layer in self.layers:
                        s = getattr(layer.attention.blockade, '_last_suppression', None)
                        if s is not None:
                            layer_supps.append(s)
                    if layer_supps:
                        suppression = torch.stack(layer_supps).mean(dim=0)
                self.head_state.update_grounded_primitives(
                    avg_entropy, seq_len,
                    protection_scores=protection,
                    blockade_suppression=suppression,
                )

        if return_state:
            state = self._build_state_dict(head_sigmas, distances, all_head_entropy)
            # Keep live tensor for differentiable loss (sigma diversity pressure)
            state["head_sigmas_tensor"] = head_sigmas
            return logits, state

        return logits

    def compute_per_head_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-head loss for co-survival updates.

        This is an approximation: we measure how much each head
        contributes to the total loss by checking the gradient
        magnitude w.r.t. each head's output projection weights.
        """
        # Simple proxy: per-head activation correlation with loss
        # Real implementation would use per-head ablation or gradient attribution
        vocab = logits.size(-1)
        loss = F.cross_entropy(
            logits.view(-1, vocab),
            targets.view(-1),
            ignore_index=self.cfg.ignore_index,
            reduction='none',
        ).view(logits.shape[0], logits.shape[1])

        # Per-head contribution estimated from stored activations
        # For now, use head activation magnitude as fitness proxy
        # (active heads contributing more = higher fitness)
        per_head_fitness = self.head_state._last_head_sigmas  # sigma as fitness proxy
        return per_head_fitness

    def compute_spectral_rank(self, h: Optional[torch.Tensor] = None) -> Dict:
        """
        Compute effective spectral rank of hidden representations.

        This is the key metric: standard transformers collapse to ~16.
        We expect blockade + co-survival to increase this.
        """
        if h is None:
            return {"spectral_rank": -1}

        with torch.no_grad():
            # h: [batch, seq, d_model]
            # Flatten batch and seq
            h_flat = h.reshape(-1, h.shape[-1])  # [N, d_model]

            # Center
            h_centered = h_flat - h_flat.mean(dim=0, keepdim=True)

            # SVD
            try:
                U, S, Vh = torch.linalg.svd(h_centered, full_matrices=False)
            except Exception:
                return {"spectral_rank": -1}

            # Effective rank: exp(entropy of normalized singular values)
            S_norm = S / S.sum()
            S_norm = S_norm[S_norm > 1e-10]
            entropy = -(S_norm * S_norm.log()).sum()
            effective_rank = entropy.exp().item()

            # Also compute where spectrum drops off (90% energy)
            cumulative_energy = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
            rank_90 = (cumulative_energy < 0.9).sum().item() + 1
            rank_95 = (cumulative_energy < 0.95).sum().item() + 1
            rank_99 = (cumulative_energy < 0.99).sum().item() + 1

            # Top 20 singular values for spectrum analysis
            top_20 = S[:20].tolist() if len(S) >= 20 else S.tolist()

            return {
                "spectral_rank": effective_rank,
                "rank_90": rank_90,
                "rank_95": rank_95,
                "rank_99": rank_99,
                "top_singular_values": top_20,
                "total_dims": h.shape[-1],
            }

    def _build_state_dict(
        self,
        head_sigmas: torch.Tensor,
        distances: torch.Tensor,
        all_head_activations: List[torch.Tensor],
    ) -> Dict:
        """Build state dictionary for inspection."""
        state = {
            "global_sigma": float(self._global_sigma.item()),
            "head_sigmas": head_sigmas.detach().cpu().tolist(),
            "head_positions": self.head_state.head_positions.detach().cpu().tolist(),
            "step": int(self._step_count.item()),
        }

        # Blockade info
        if self.cfg.blockade_enabled:
            min_dist = distances[~torch.eye(self.cfg.n_heads, dtype=bool, device=distances.device)].min().item()
            max_dist = distances[~torch.eye(self.cfg.n_heads, dtype=bool, device=distances.device)].max().item()
            state["blockade"] = {
                "min_head_distance": min_dist,
                "max_head_distance": max_dist,
                "mean_head_distance": distances[~torch.eye(self.cfg.n_heads, dtype=bool, device=distances.device)].mean().item(),
            }

        # Co-survival info
        if self.cfg.cosurvival_enabled:
            cs = self.cosurvival.cosurvival
            state["cosurvival"] = {
                "mean": float(cs.mean().item()),
                "max": float(cs.max().item()),
                "min": float(cs.min().item()),
                "n_positive_bonds": int((cs > 0.01).sum().item()),
                "n_negative_bonds": int((cs < -0.01).sum().item()),
                "protection_scores": self.cosurvival.get_protection_scores().detach().cpu().tolist(),
            }

        # Head activation patterns
        if all_head_activations:
            avg_acts = torch.stack(all_head_activations).mean(dim=0)
            state["head_activations"] = avg_acts.detach().cpu().tolist()

        # Per-head attention entropy (from last layer)
        if self.layers:
            state["head_entropy"] = self.layers[-1].attention._head_entropy.detach().cpu().tolist()

        return state

    def reset_state(self):
        """Reset all dynamic state."""
        self.head_state._last_head_sigmas.fill_(0.5)
        self.cosurvival.cosurvival.zero_()
        self._step_count.zero_()
        self._global_sigma.fill_(0.5)


# ======================
# Loss Function
# ======================

def migrate_sigma_projections(state_dict: dict) -> dict:
    """Migrate old per-head sigma_projections (ModuleList) to batched params.

    Old format (per-head ModuleList):
      head_state.sigma_projections.0.0.weight  [16, 6]  (Linear layer 0, head 0)
      head_state.sigma_projections.0.0.bias    [16]
      head_state.sigma_projections.0.2.weight  [1, 16]  (Linear layer 2, head 0)
      head_state.sigma_projections.0.2.bias    [1]
      ... repeated for each head

    New format (batched):
      head_state.sigma_w1  [n_heads, 16, 6]
      head_state.sigma_b1  [n_heads, 16]
      head_state.sigma_w2  [n_heads, 1, 16]
      head_state.sigma_b2  [n_heads, 1]

    Returns modified state_dict (in-place).
    """
    # Find all sigma_projections keys grouped by stage prefix
    sigma_keys = [k for k in state_dict if 'sigma_projections' in k]
    if not sigma_keys:
        return state_dict  # Already migrated or no sigma projections

    # Group by stage prefix (e.g., "stages.0.head_state")
    from collections import defaultdict
    stage_groups = defaultdict(dict)
    for k in sigma_keys:
        # Key format: stages.X.head_state.sigma_projections.H.L.{weight,bias}
        # where X=stage, H=head, L=layer (0=first linear, 2=second linear)
        parts = k.split('.')
        # Find the stage prefix up to head_state
        sp_idx = parts.index('sigma_projections')
        prefix = '.'.join(parts[:sp_idx])  # e.g., "stages.0.head_state"
        head_idx = int(parts[sp_idx + 1])
        layer_idx = int(parts[sp_idx + 2])
        param_type = parts[sp_idx + 3]  # weight or bias
        stage_groups[prefix][(head_idx, layer_idx, param_type)] = k

    migrated_count = 0
    for prefix, keys_map in stage_groups.items():
        # Determine n_heads from the keys
        head_indices = set(h for (h, l, p) in keys_map)
        n_heads = max(head_indices) + 1

        # Build batched tensors
        w1_list, b1_list, w2_list, b2_list = [], [], [], []
        for h in range(n_heads):
            w1_key = keys_map.get((h, 0, 'weight'))
            b1_key = keys_map.get((h, 0, 'bias'))
            w2_key = keys_map.get((h, 2, 'weight'))
            b2_key = keys_map.get((h, 2, 'bias'))
            if w1_key and b1_key and w2_key and b2_key:
                w1_list.append(state_dict[w1_key])   # [16, 6]
                b1_list.append(state_dict[b1_key])   # [16]
                w2_list.append(state_dict[w2_key])   # [1, 16]
                b2_list.append(state_dict[b2_key])   # [1]

        if len(w1_list) == n_heads:
            state_dict[f'{prefix}.sigma_w1'] = torch.stack(w1_list)  # [H, 16, 6]
            state_dict[f'{prefix}.sigma_b1'] = torch.stack(b1_list)  # [H, 16]
            state_dict[f'{prefix}.sigma_w2'] = torch.stack(w2_list)  # [H, 1, 16]
            state_dict[f'{prefix}.sigma_b2'] = torch.stack(b2_list)  # [H, 1]
            migrated_count += n_heads

    # Remove old keys
    for k in sigma_keys:
        del state_dict[k]

    if migrated_count > 0:
        print(f"  Migrated {migrated_count} sigma projections to batched format")

    return state_dict


# ======================
# Quick Test
# ======================

