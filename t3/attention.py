"""Ecology-modulated multi-head attention.

Standard MHA, augmented per-head by:
    - σ-modulated attention temperature  (per-head softmax sharpness)
    - σ-modulated locality bias          (low σ → more local attention)
    - learned key-bias projection from ecology primitives (eco K-bias)
    - blockade suppression of head outputs based on neighbor excitation
    - optional V1-residual blending (layer-0 V re-injected, σ-gated)

Supports:
    - GQA (n_kv_heads < n_heads), via key/value head repetition
    - QK-norm (RMSNorm of Q and K before attention)
    - RoPE (caller-supplied (cos, sin), applied before scoring)

Does **not** include the FlexAttention fast path that lived in the training
codebase. The standard PyTorch path here matches FlexAttention numerically
within the entropy-EMA tolerance, and avoids depending on torch internals
that have been unstable across versions.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from t3.ecology import Blockade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Weight-only RMSNorm. Used by attention's optional QK-norm and by the
    chain's stage/layer norms when `cfg.norm_type='rmsnorm'`."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, S, n_kv, D] → [B, S, n_kv * n_rep, D]. Identity when n_rep == 1."""
    if n_rep == 1:
        return x
    batch, seq, n_kv, d = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch, seq, n_kv, n_rep, d)
        .reshape(batch, seq, n_kv * n_rep, d)
    )


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K. Shapes: q,k=[B,S,H,D]; cos,sin=[S, D/2]."""
    d_half = cos.shape[-1]
    q1, q2 = q[..., :d_half], q[..., d_half:]
    k1, k2 = k[..., :d_half], k[..., d_half:]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# EcologyAttention
# ---------------------------------------------------------------------------


class EcologyAttention(nn.Module):
    """Multi-head attention with ecology-modulated sharpness, locality, and
    head-output suppression. The HeadState supplies per-head sigma; the
    Blockade applies neighbor-excitation suppression after softmax. The
    Cosurvival graph (when enabled) modulates blockade strength per pair.
    """

    def __init__(self, d_model: int, n_heads: int, cfg, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = getattr(cfg, "d_head", 0) or (d_model // n_heads)
        assert (getattr(cfg, "d_head", 0) > 0) or (d_model % n_heads == 0), (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads}) "
            f"unless cfg.d_head is set"
        )
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.cfg = cfg
        self.use_rope = getattr(cfg, "use_rope", False)

        # GQA.
        self.n_kv_heads = getattr(cfg, "n_kv_heads", 0) or n_heads
        assert n_heads % self.n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        self.n_kv_groups = n_heads // self.n_kv_heads

        attn_bias = getattr(cfg, "attn_bias", True)
        attn_out_bias = getattr(cfg, "attn_out_bias", True)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=attn_bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=attn_bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=attn_bias)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=attn_out_bias)

        self.use_qk_norm = getattr(cfg, "use_qk_norm", False)
        if self.use_qk_norm:
            norm_eps = getattr(cfg, "norm_eps", 1e-6)
            self.q_norm = RMSNorm(self.d_head, eps=norm_eps)
            self.k_norm = RMSNorm(self.d_head, eps=norm_eps)

        self.dropout = nn.Dropout(dropout)

        self.blockade = Blockade(cfg)

        # Diagnostic snapshots (one-step-delayed feed for ecology updates).
        self.register_buffer("_head_activations", torch.zeros(n_heads))
        self.register_buffer("_head_entropy", torch.zeros(n_heads))

    def forward(
        self,
        x: torch.Tensor,                                        # [B, S, D]
        head_sigmas: torch.Tensor,                              # [n_heads]
        distances: torch.Tensor,                                # [n_heads, n_heads]
        mask: Optional[torch.Tensor] = None,
        blockade_mod: Optional[torch.Tensor] = None,            # [n_heads, n_heads]
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        eco_k_offset: Optional[torch.Tensor] = None,            # [n_heads, d_head]
        v1_residual: Optional[torch.Tensor] = None,             # [B, S, n_kv, d_head]
        return_v: bool = False,
    ):
        """Returns (output, head_entropy) — or (output, head_entropy, v_pre_blend)
        if `return_v` is True (used by stages that re-inject layer-0 V).
        """
        batch, seq_len, _ = x.shape
        bypass = getattr(self.cfg, "bypass_ecology", False)

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.d_head)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        v_pre_blend = v if return_v else None

        # V1 residual blend (layers 1+ in a stage). λ is per-head sigma by default.
        if v1_residual is not None and not bypass:
            v1_gating = getattr(self.cfg, "v1_residual_gating", "sigma")
            if v1_gating == "sigma":
                if self.n_kv_groups > 1:
                    sigma_kv = (
                        head_sigmas.detach()
                        .view(self.n_kv_heads, self.n_kv_groups)
                        .mean(dim=1)
                    )
                else:
                    sigma_kv = head_sigmas.detach()
                lam = sigma_kv.view(1, 1, self.n_kv_heads, 1).to(v.dtype)
            elif v1_gating == "inverse_sigma":
                if self.n_kv_groups > 1:
                    sigma_kv = (
                        head_sigmas.detach()
                        .view(self.n_kv_heads, self.n_kv_groups)
                        .mean(dim=1)
                    )
                else:
                    sigma_kv = head_sigmas.detach()
                lam = (1.0 - sigma_kv).view(1, 1, self.n_kv_heads, 1).to(v.dtype)
            else:  # "fixed"
                fixed_lam = getattr(self.cfg, "v1_residual_fixed_lambda", 0.5)
                lam = torch.tensor(fixed_lam, device=v.device, dtype=v.dtype)
            v = v + lam * (v1_residual.to(v.dtype) - v)

        # Eco-conditioned K-bias (changes WHAT heads attend to). Pre-RoPE.
        if eco_k_offset is not None and not bypass:
            eco = getattr(self, "_ecology_strength", 1.0)
            if self.n_kv_groups > 1:
                offset_grouped = (
                    eco_k_offset
                    .view(self.n_kv_heads, self.n_kv_groups, -1)
                    .mean(dim=1)
                )
                k = k + (eco * offset_grouped).unsqueeze(0).unsqueeze(1).to(k.dtype)
            else:
                k = k + (eco * eco_k_offset).unsqueeze(0).unsqueeze(1).to(k.dtype)

        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: bring K, V up to Q's head count.
        if self.n_kv_groups > 1:
            k = _repeat_kv(k, self.n_kv_groups)
            v = _repeat_kv(v, self.n_kv_groups)

        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * self.scale

        if not bypass:
            eco = getattr(self, "_ecology_strength", 1.0)

            # Per-head temperature: linearly maps σ ∈ (0,1) to [t_lo, t_hi].
            # `sigma_stop_gradient=False` lets cross-entropy reach σ.
            if not getattr(self.cfg, "sigma_stop_gradient", True):
                sigma_mod = head_sigmas
            else:
                sigma_mod = head_sigmas.detach()
            t_lo = getattr(self.cfg, "temp_range_lo", 0.5)
            t_hi = getattr(self.cfg, "temp_range_hi", 1.5)
            raw_temps = t_lo + sigma_mod * (t_hi - t_lo)
            temperatures = 1.0 + eco * (raw_temps - 1.0)
            scores = scores / temperatures.view(1, -1, 1, 1)

            # Per-head locality bias: low σ → more local attention.
            positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            distance_matrix = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
            locality_strengths = (0.5 - sigma_mod.detach()).clamp(min=0) * 0.2 * eco
            locality_bias = -distance_matrix.unsqueeze(0) * locality_strengths.view(-1, 1, 1)
            scores = scores + locality_bias.unsqueeze(0).to(scores.dtype)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1).to(v.dtype)
        attn = self.dropout(attn)

        # Per-head attention entropy → drives blockade excitation.
        with torch.amp.autocast("cuda", enabled=False):
            attn_for_entropy = attn.float().clamp(min=1e-8)
            entropy_per_pos = -(attn_for_entropy * attn_for_entropy.log()).sum(dim=-1)
            head_entropy = entropy_per_pos.mean(dim=(0, 2))

        if not bypass:
            max_entropy = math.log(max(seq_len, 2))
            excitation = ((max_entropy - head_entropy) / (max_entropy + 1e-8)).to(x.dtype)
            suppression = self.blockade(excitation, distances, blockade_mod)

            out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
            head_scale = (1.0 - eco * suppression).view(1, 1, -1, 1).to(out.dtype)
            out = out * head_scale
        else:
            out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
            excitation = None

        out = out.reshape(batch, seq_len, self.n_heads * self.d_head)
        out = self.out_proj(out)

        self._store_diagnostics(head_entropy, excitation)

        if return_v:
            return out, head_entropy, v_pre_blend
        return out, head_entropy

    @torch.compiler.disable
    def _store_diagnostics(
        self, head_entropy: torch.Tensor, excitation: Optional[torch.Tensor]
    ) -> None:
        with torch.no_grad():
            if excitation is not None:
                self._head_activations.copy_(excitation.detach())
            self._head_entropy.copy_(head_entropy.detach())
