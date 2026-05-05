"""T³ chain — layers, stages, and the per-stage ACT chain.

This module assembles the public modules from `t3.ecology` and `t3.attention`
into the full T³ architecture: a sequence of stages, each with its own
ecology state, that pass hidden + σ-flow forward and ponder adaptively.

The released checkpoint runs with `act_enabled=True, act_per_stage=True`,
so `T3Chain.forward` routes to `_act_perstage_forward`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from t3.attention import EcologyAttention, RMSNorm, _apply_rotary_pos_emb
from t3.ecology import Cosurvival, HeadState


# ---------------------------------------------------------------------------
# Norm + FFN helpers
# ---------------------------------------------------------------------------


def _make_norm(d_model: int, cfg) -> nn.Module:
    norm_type = getattr(cfg, "norm_type", "layernorm")
    eps = getattr(cfg, "norm_eps", 1e-5)
    if norm_type == "rmsnorm":
        return RMSNorm(d_model, eps=eps)
    return nn.LayerNorm(d_model, eps=eps)


class _SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class _GeGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
        )


def _make_ffn(d_model: int, d_ff: int, cfg, dropout: float) -> nn.Module:
    ffn_type = getattr(cfg, "ffn_type", "gelu")
    ffn_bias = getattr(cfg, "ffn_bias", True)
    if ffn_type == "swiglu":
        return _SwiGLUFFN(d_model, d_ff, bias=ffn_bias, dropout=dropout)
    if ffn_type == "geglu":
        return _GeGLUFFN(d_model, d_ff, bias=ffn_bias, dropout=dropout)
    return nn.Sequential(
        nn.Linear(d_model, d_ff, bias=ffn_bias),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model, bias=ffn_bias),
        nn.Dropout(dropout),
    )


# ---------------------------------------------------------------------------
# OutputEntropyTracker — feeds blended-E primitive
# ---------------------------------------------------------------------------


class OutputEntropyTracker(nn.Module):
    """Per-stage tracker of output-distribution entropy.

    Probes the stage's hidden state at a few sequence positions, projects
    through the (tied) output embedding, and updates a running EMA of the
    normalized output entropy. The chain reads this EMA when computing the
    blended-E primitive in HeadState's update path.
    """

    def __init__(
        self,
        vocab_size: int,
        decay: float = 0.95,
        n_probe_positions: int = 2,
    ):
        super().__init__()
        self.max_entropy = math.log(vocab_size)
        self.decay = decay
        self.n_probe_positions = n_probe_positions
        self.register_buffer("output_entropy_ema", torch.tensor(0.5))
        self.register_buffer("valence_velocity_ema", torch.tensor(0.0))
        self.register_buffer("_prev_entropy", torch.tensor(0.0))

    def compute(self, hidden: torch.Tensor, output_weight: torch.Tensor) -> None:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            B, S, D = hidden.shape
            n_pos = min(self.n_probe_positions, S)
            idx = torch.linspace(0, S - 1, n_pos).long().to(hidden.device)
            h_sub = hidden[:, idx, :].float()

            logits = F.linear(h_sub, output_weight.float())
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-10)
            ent = -(probs * log_probs).sum(-1)
            ent_norm = (ent / self.max_entropy).mean().detach()
            del logits, probs, log_probs, ent, h_sub

            alpha_ema = 1.0 - self.decay
            self.output_entropy_ema.lerp_(ent_norm, alpha_ema)

            prev = self._prev_entropy
            if prev > 0:
                delta = prev - ent_norm
                self.valence_velocity_ema.lerp_(delta, 1.0 - 0.99)
            self._prev_entropy.fill_(ent_norm)


# ---------------------------------------------------------------------------
# T3Layer — pre-norm transformer block with ecology-modulated attention + FFN
# ---------------------------------------------------------------------------


class T3Layer(nn.Module):
    """One transformer layer inside a stage: pre-norm attention + FFN with
    σ-gated FFN scale. No mutable state of its own; the ecology state lives
    on the parent stage's `HeadState` and `Cosurvival`."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, cfg, dropout: float):
        super().__init__()
        self.cfg = cfg

        self.attention = EcologyAttention(d_model, n_heads, cfg, dropout)

        self.ff = _make_ffn(d_model, d_ff, cfg, dropout)
        self.norm1 = _make_norm(d_model, cfg)
        self.norm2 = _make_norm(d_model, cfg)

        self.use_post_norms = getattr(cfg, "use_post_norms", False)
        if self.use_post_norms:
            self.post_attn_norm = _make_norm(d_model, cfg)
            self.post_ff_norm = _make_norm(d_model, cfg)

    def forward(
        self,
        x: torch.Tensor,
        head_sigmas: torch.Tensor,
        distances: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        blockade_mod: Optional[torch.Tensor] = None,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        eco_k_offset: Optional[torch.Tensor] = None,
        v1_residual: Optional[torch.Tensor] = None,
        return_v: bool = False,
    ):
        attn_result = self.attention(
            self.norm1(x),
            head_sigmas,
            distances,
            mask=attn_mask,
            blockade_mod=blockade_mod,
            rope_cos_sin=rope_cos_sin,
            eco_k_offset=eco_k_offset,
            v1_residual=v1_residual,
            return_v=return_v,
        )
        if return_v:
            attn_out, head_entropy, v_pre_blend = attn_result
        else:
            attn_out, head_entropy = attn_result
        if self.use_post_norms:
            attn_out = self.post_attn_norm(attn_out)
        x = x + attn_out

        ff_out = self.ff(self.norm2(x))
        if self.use_post_norms:
            ff_out = self.post_ff_norm(ff_out)
        if getattr(self.cfg, "bypass_ecology", False):
            x = x + ff_out
        else:
            eco = getattr(self, "_ecology_strength", 1.0)
            sigma_mean = head_sigmas.detach().mean()
            raw_ff_scale = (0.3 + 0.7 * sigma_mean).to(x.dtype)
            ff_scale = 1.0 + eco * (raw_ff_scale - 1.0)
            x = x + ff_out * ff_scale

        if return_v:
            return x, head_entropy, v_pre_blend
        return x, head_entropy


# ---------------------------------------------------------------------------
# Logit softcap
# ---------------------------------------------------------------------------


def _apply_logit_softcap(logits: torch.Tensor, cap: Optional[float]) -> torch.Tensor:
    if cap is not None and cap > 0:
        logits = cap * torch.tanh(logits / cap)
    return logits


# ---------------------------------------------------------------------------
# RoPE — light table-builder (only built when cfg.use_rope=True)
# ---------------------------------------------------------------------------


class _RotaryEmbedding(nn.Module):
    """Caches (cos, sin) tables for RoPE. Built lazily; not in checkpoints."""

    def __init__(self, d_head: int, max_seq_len: int = 32768, base: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


# ---------------------------------------------------------------------------
# T3Stage — one stage of the chain
# ---------------------------------------------------------------------------


class T3Stage(nn.Module):
    """One stage of the T³ chain: own ecology + cosurvival + N transformer
    layers. The first stage owns the input embedding; the last stage owns
    the output projection (typically tied to the embedding weight)."""

    def __init__(
        self,
        cfg,
        stage_idx: int,
        is_first: bool = False,
        is_last: bool = False,
        stage_n_layers: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.stage_idx = stage_idx
        self.is_first = is_first
        self.is_last = is_last

        n_layers_this_stage = stage_n_layers if stage_n_layers is not None else cfg.n_layers
        self.n_layers = n_layers_this_stage

        self.use_rope = getattr(cfg, "use_rope", False)
        if is_first:
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
            if not self.use_rope:
                self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            self.embed_norm = _make_norm(cfg.d_model, cfg)

        if self.use_rope:
            rope_base = getattr(cfg, "rope_base", 10000.0)
            rope_d_head = getattr(cfg, "d_head", 0) or (cfg.d_model // cfg.n_heads)
            self.rope = _RotaryEmbedding(rope_d_head, cfg.max_seq_len, rope_base)

        sigma_hidden_per_stage = getattr(cfg, "sigma_hidden_per_stage", None)
        sigma_hidden_override = (
            sigma_hidden_per_stage[self.stage_idx]
            if sigma_hidden_per_stage is not None
            else None
        )
        self.head_state = HeadState(
            cfg.n_heads, cfg.d_model, cfg, sigma_hidden_override=sigma_hidden_override
        )
        self.cosurvival = Cosurvival(cfg.n_heads, cfg)

        self.layers = nn.ModuleList(
            [
                T3Layer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg, cfg.dropout)
                for _ in range(n_layers_this_stage)
            ]
        )

        # Auto-calibrate blockade radius from head geometry. Default radius=1.0
        # with NN distance ~0.2 yields (1/0.2)^6 = 15,625× suppression; setting
        # radius = NN distance puts blockade at 50% on nearest neighbors.
        if getattr(cfg, "blockade_radius_auto", True) and cfg.blockade_enabled:
            nn_dist = self.head_state._init_nn_distance
            for layer in self.layers:
                layer.attention.blockade.blockade_radius = nn_dist

        self.norm = _make_norm(cfg.d_model, cfg)

        if is_last:
            self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if is_first:
                self.output_proj.weight = self.embed.weight

        self.dropout_layer = nn.Dropout(cfg.dropout)
        self.register_buffer("_stage_step", torch.tensor(0, dtype=torch.long))

        self.entropy_tracker = OutputEntropyTracker(
            cfg.vocab_size,
            cfg.entropy_ema_decay,
            n_probe_positions=getattr(cfg, "act_n_probe_positions", 2),
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)

    def _get_output_weight(self) -> Optional[torch.Tensor]:
        if hasattr(self, "output_proj"):
            return self.output_proj.weight
        return getattr(self, "_shared_output_weight", None)

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        sigma_prior: Optional[torch.Tensor] = None,
        update_state: bool = True,
    ):
        """Run one stage forward.

        Returns `(hidden [B, S, D], logits [B, S, V] or None, state dict)`.
        """
        bypass = getattr(self.cfg, "bypass_ecology", False)

        if self.is_first and x is not None:
            batch, seq_len = x.shape
            device = x.device
            if self.use_rope:
                h = self.embed(x)
            else:
                positions = torch.arange(seq_len, device=device)
                h = self.embed(x) + self.pos_embed(positions)
            embed_scale = getattr(self.cfg, "embed_scale", 1.0)
            if embed_scale != 1.0:
                h = h * embed_scale
            if not bypass and not getattr(self.cfg, "skip_intermediate_norms", False):
                eco = getattr(self, "_ecology_strength", 1.0)
                if eco > 0.001:
                    h_normed = self.embed_norm(h)
                    h = h + eco * (h_normed - h)
            h = self.dropout_layer(h)
        elif hidden is not None:
            h = hidden
            batch, seq_len, _ = h.shape
            device = h.device
        else:
            raise ValueError("Stage needs either x (input_ids) or hidden")

        self.head_state.invalidate_distance_cache()
        if self.cfg.cosurvival_enabled:
            self.cosurvival.invalidate_blockade_cache()

        # FP32 guard around all ecology arithmetic.
        with torch.amp.autocast("cuda", enabled=False):
            distances = self.head_state.get_pairwise_distances()
            blockade_mod = (
                self.cosurvival.get_blockade_modulation()
                if self.cfg.cosurvival_enabled
                else None
            )

            d_head = self.cfg.d_model // self.cfg.n_heads
            h_per_head = h.view(batch, seq_len, self.cfg.n_heads, d_head)

            warmup_frac = getattr(self, "_warmup_frac", 1.0)
            own_sigmas = self.head_state.compute_head_sigmas(h_per_head, warmup_frac=warmup_frac)

            if self.cfg.pass_sigma and sigma_prior is not None:
                blend = self.cfg.sigma_blend
                head_sigmas = blend * sigma_prior + (1 - blend) * own_sigmas
            else:
                head_sigmas = own_sigmas

            # Sigma complement: bonded heads pushed to complementary uncertainty.
            if (
                getattr(self.cfg, "cooperative_prediction", False)
                and not bypass
                and self.cfg.cosurvival_enabled
            ):
                cs_matrix = self.cosurvival.cosurvival
                sigma_offset = self.head_state.compute_sigma_complement(head_sigmas, cs_matrix)
                eco = getattr(self, "_ecology_strength", 1.0)
                head_sigmas = (head_sigmas + eco * sigma_offset).clamp(0.01, 0.99)

        attn_mask = self._causal_mask(seq_len, device)
        rope_cos_sin = self.rope(seq_len) if self.use_rope else None

        # Eco-conditioned K-bias offset (shared by all layers in this stage).
        eco_k_offset = None
        if (
            getattr(self.cfg, "eco_key_bias", False)
            and not bypass
            and hasattr(self.head_state, "key_bias_proj")
        ):
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                hs = self.head_state
                n_eco = getattr(self.cfg, "eco_key_bias_features", 4)
                if n_eco >= 6:
                    eco_input = torch.stack(
                        [
                            head_sigmas.detach().float(),
                            hs._entropy_ema.float(),
                            hs._friction_ema.float(),
                            hs._valence_ema.float(),
                            hs._last_excitation.float(),
                            hs._chronos_ema.float(),
                        ],
                        dim=-1,
                    )
                else:
                    protection = (
                        hs.cosurvival_tracker.get_protection_scores()
                        if hasattr(hs, "cosurvival_tracker")
                        else torch.zeros_like(head_sigmas)
                    )
                    eco_input = torch.stack(
                        [
                            head_sigmas.detach().float(),
                            hs._entropy_ema.float(),
                            hs._last_excitation.float(),
                            protection.detach().float(),
                        ],
                        dim=-1,
                    )
            with torch.amp.autocast("cuda", enabled=False):
                eco_k_offset = self.head_state.key_bias_proj(eco_input.float())
                eco_k_offset = eco_k_offset * getattr(self.cfg, "eco_key_bias_scale", 1.0)

        # Layer loop. V1-residual (layer-0 V re-injected to layers 1+) is
        # config-gated; default off.
        all_head_entropy: List[torch.Tensor] = []
        v1_residual_enabled = getattr(self.cfg, "v1_residual_enabled", False)
        v1_cache: Optional[torch.Tensor] = None
        for layer_idx, layer in enumerate(self.layers):
            is_first_layer = layer_idx == 0
            needs_return_v = v1_residual_enabled and is_first_layer
            layer_v1_residual = (
                v1_cache if (v1_residual_enabled and not is_first_layer) else None
            )

            result = layer(
                h,
                head_sigmas,
                distances,
                attn_mask=attn_mask,
                blockade_mod=blockade_mod,
                rope_cos_sin=rope_cos_sin,
                eco_k_offset=eco_k_offset,
                v1_residual=layer_v1_residual,
                return_v=needs_return_v,
            )
            if needs_return_v:
                h, head_entropy, v1_cache = result
            else:
                h, head_entropy = result
            all_head_entropy.append(head_entropy)

        # Update cached excitation from the last layer (feeds next forward's K-bias).
        if not bypass and self.layers and hasattr(self.head_state, "_last_excitation"):
            with torch.no_grad():
                self.head_state._last_excitation.copy_(
                    self.layers[-1].attention._head_activations.detach()
                )

        skip_norm = not self.is_last and (
            bypass or getattr(self.cfg, "skip_intermediate_norms", False)
        )
        if not skip_norm:
            if self.is_last:
                h = self.norm(h)
            else:
                eco = getattr(self, "_ecology_strength", 1.0)
                if eco > 0.001:
                    h_normed = self.norm(h)
                    h = h + eco * (h_normed - h)

        # Live ecology update at inference. The training-time co-survival /
        # entropy-tracker update path (gated on `self.training`) is not present
        # in this public reference; only the inference-time live primitive
        # update remains.
        if (
            update_state
            and not self.training
            and getattr(self.cfg, "eval_live_primitives", False)
        ):
            if not bypass and self.cfg.grounded_primitives and len(all_head_entropy) > 0:
                with torch.no_grad():
                    avg_entropy = torch.stack(all_head_entropy).mean(dim=0)
                    output_ent = self.entropy_tracker.output_entropy_ema.detach()
                    self.head_state.update_grounded_primitives(
                        avg_entropy,
                        seq_len,
                        output_entropy=output_ent,
                        warmup_frac=warmup_frac,
                    )

        logits = None
        if self.is_last:
            if bypass:
                logits = self.output_proj(h)
            else:
                eco = getattr(self, "_ecology_strength", 1.0)
                sigma_mean = head_sigmas.detach().mean()
                raw_temp = (1.5 - sigma_mean * 0.8).to(h.dtype)
                raw_temp = torch.clamp(raw_temp, 0.5, 2.0)
                temperature = 1.0 + eco * (raw_temp - 1.0)
                logits = self.output_proj(h) / temperature
            logits = _apply_logit_softcap(logits, self.cfg.logit_softcap)

        stage_state: Dict = {
            "head_sigmas": head_sigmas.detach().cpu().tolist(),
            "head_sigmas_tensor": head_sigmas,
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
            stage_state["head_entropy_tensors"] = all_head_entropy
        if self.layers:
            stage_state["head_entropy"] = (
                self.layers[-1].attention._head_entropy.detach().cpu().tolist()
            )

        return h, logits, stage_state


# ---------------------------------------------------------------------------
# T3Chain — full architecture: stages + ACT + output heads
# ---------------------------------------------------------------------------


class T3Chain(nn.Module):
    """The full T³ model: a sequence of stages with shared ecology dynamics
    and per-stage adaptive computation time. The released checkpoint runs
    with `act_enabled=True, act_per_stage=True`, so `forward` routes to
    `_act_perstage_forward`."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.stages = nn.ModuleList()
        for i in range(cfg.n_stages):
            is_first = i == 0
            is_last = i == cfg.n_stages - 1
            stage_n_layers = cfg.layers_per_stage[i] if cfg.layers_per_stage else None
            self.stages.append(
                T3Stage(
                    cfg,
                    stage_idx=i,
                    is_first=is_first,
                    is_last=is_last,
                    stage_n_layers=stage_n_layers,
                )
            )

        # Optional shared head positions across stages (single chain geometry).
        if cfg.shared_positions and cfg.n_stages > 1:
            shared_pos = self.stages[0].head_state.head_positions
            for stage in self.stages[1:]:
                if isinstance(shared_pos, nn.Parameter):
                    del stage.head_state.head_positions
                    stage.head_state.head_positions = shared_pos

        if cfg.use_residual and cfg.n_stages > 1:
            self.residual_proj = nn.Identity()

        # Learned PonderNet halt head — registered (and loaded from ckpt) when
        # `act_strain_halt` is False. The released ckpt has this even when the
        # live halt criterion is entropy-based, so we register either way to
        # keep checkpoint loading strict.
        if cfg.act_enabled and not getattr(cfg, "act_strain_halt", False):
            self.halt_head = nn.Sequential(
                nn.Linear(cfg.d_model, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )

        self.register_buffer("_last_ponder_cost", torch.tensor(0.0))
        self._last_ponder_steps = 1

        if getattr(cfg, "act_strain_ema_enabled", False):
            self.register_buffer(
                "_strain_ema",
                torch.full((cfg.n_stages,), getattr(cfg, "act_strain_threshold", 0.05)),
            )

        if getattr(cfg, "act_adaptive_threshold", False) or cfg.act_entropy_halt:
            self.register_buffer(
                "_entropy_delta_ema",
                torch.full((cfg.n_stages,), cfg.act_entropy_halt_threshold),
            )

        if cfg.act_difficulty_predictor:
            self.difficulty_head = nn.Sequential(
                nn.Linear(cfg.d_model, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            self.register_buffer("_loss_ema", torch.tensor(3.0))
        self._last_difficulty_pred: Optional[torch.Tensor] = None

        if getattr(cfg, "scratchpad_need_predictor", False):
            self.scratchpad_need_head = nn.Sequential(
                nn.Linear(cfg.d_model, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        self._last_scratchpad_pred: Optional[torch.Tensor] = None
        self._last_final_hidden: Optional[torch.Tensor] = None

        self.apply(self._init_weights)

        # Re-init specialized projections that the generic init clobbers.
        for stage in self.stages:
            hs = stage.head_state
            if hasattr(hs, "key_bias_proj"):
                nn.init.zeros_(hs.key_bias_proj.weight)
                nn.init.zeros_(hs.key_bias_proj.bias)
            if hasattr(hs, "bond_predictor"):
                nn.init.eye_(hs.bond_predictor.weight)
                nn.init.zeros_(hs.bond_predictor.bias)
            if hasattr(hs, "inter_stage_predictor"):
                nn.init.eye_(hs.inter_stage_predictor.weight)
                nn.init.zeros_(hs.inter_stage_predictor.bias)

        # Tie last stage's output projection to first stage's embedding.
        if cfg.n_stages > 1:
            self.stages[-1].output_proj.weight = self.stages[0].embed.weight

        # Earlier stages share a reference to the output weight for entropy probing.
        output_weight = self.stages[-1].output_proj.weight
        for stage in self.stages[:-1]:
            stage._shared_output_weight = output_weight

        self._step_count = self.stages[-1]._stage_step
        self.register_buffer("_global_sigma", torch.tensor(0.5))

    # ----------------- compatibility properties ------------------

    @property
    def head_state(self) -> HeadState:
        return self.stages[-1].head_state

    @property
    def cosurvival(self) -> Cosurvival:
        return self.stages[-1].cosurvival

    @property
    def layers(self) -> List[T3Layer]:
        all_layers: List[T3Layer] = []
        for stage in self.stages:
            all_layers.extend(stage.layers)
        return all_layers

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # ----------------- forward dispatch ------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        return_state: bool = False,
        return_chain_state: bool = False,
        return_intermediate_logits: bool = False,
        update_state: bool = True,
    ):
        """Inference forward pass.

        Routes to `_act_perstage_forward` when `act_per_stage=True` (the
        released-checkpoint configuration).
        """
        if self.cfg.act_enabled and self.cfg.act_per_stage:
            return self._act_perstage_forward(
                input_ids,
                return_state=return_state,
                return_chain_state=return_chain_state,
                update_state=update_state,
            )
        # The non-per-stage ACT path and the no-ACT path are not part of this
        # public reference. The released checkpoint uses act_per_stage=True.
        raise NotImplementedError(
            "T3Chain only ships the act_per_stage forward path. "
            "Set cfg.act_enabled=True and cfg.act_per_stage=True."
        )

    # ----------------- per-stage ACT forward ------------------

    def _act_perstage_forward(
        self,
        input_ids: torch.Tensor,
        return_state: bool = False,
        return_chain_state: bool = False,
        update_state: bool = True,
    ):
        device = input_ids.device
        max_per_stage = self.cfg.act_per_stage_max
        eps = self.cfg.act_halt_epsilon
        use_ema = getattr(self.cfg, "act_strain_ema_enabled", False)

        for stage in self.stages:
            stage.head_state.invalidate_distance_cache()
            if self.cfg.cosurvival_enabled:
                stage.cosurvival.invalidate_blockade_cache()

        # Stage 0: embed first (no pondering on lookup).
        hidden, _, stage0_state = self.stages[0](
            x=input_ids, hidden=None, sigma_prior=None, update_state=update_state
        )
        residual = hidden
        sigma_prior = stage0_state.get("head_sigmas_tensor")

        all_chain_states: List[Dict] = []
        per_stage_ponder_steps: List[int] = []
        per_stage_strains: List[List[float]] = []
        per_stage_entropy_deltas: List[List[torch.Tensor]] = []
        total_strain = torch.tensor(0.0, device=device)
        total_halt_probs: List[torch.Tensor] = []

        use_entropy_halt = self.cfg.act_entropy_halt
        if use_entropy_halt:
            output_weight = self.stages[-1].output_proj.weight
            if output_weight.dtype != torch.bfloat16:
                output_weight = output_weight.to(torch.bfloat16)
            entropy_threshold = self.cfg.act_entropy_halt_threshold
            entropy_temperature = self.cfg.act_entropy_halt_temperature
            max_entropy = math.log(output_weight.shape[0])

        use_difficulty = self.cfg.act_difficulty_predictor and hasattr(self, "difficulty_head")
        if use_difficulty:
            h_pooled = hidden.mean(dim=1)
            self._last_difficulty_pred = self.difficulty_head(h_pooled).squeeze(-1)
            difficulty_scalar = self._last_difficulty_pred.mean().detach()
        else:
            difficulty_scalar = torch.tensor(0.0, device=device)

        use_confidence_floor = self.cfg.act_confidence_floor > 0 and use_entropy_halt
        use_hard_halt = self.cfg.act_hard_halt_eval and not self.training

        for stage_idx, stage in enumerate(self.stages):
            p_running = torch.ones(1, device=device)
            h_accum = torch.zeros_like(hidden)
            stage_halt_probs: List[torch.Tensor] = []
            stage_strains: List[torch.Tensor] = []

            prev_entropy: Optional[torch.Tensor] = None
            stage_entropy_deltas: List[torch.Tensor] = []

            h_loop = hidden
            sigma_loop = sigma_prior
            stage._prev_ponder_entropy = None

            if use_entropy_halt:
                B, S, D = hidden.shape
                n_probe = min(self.cfg.act_n_probe_positions, S)
                probe_idx = torch.linspace(0, S - 1, n_probe).long().to(device)

            # Pre-ponder entropy baseline (lets t=0 already make a halt decision).
            use_preponder = self.cfg.act_preponder_baseline and use_entropy_halt
            if use_preponder:
                with torch.no_grad():
                    h_probe_pre = hidden.index_select(1, probe_idx).to(output_weight.dtype)
                    probe_logits_pre = F.linear(h_probe_pre, output_weight)
                    probe_probs_pre = F.softmax(probe_logits_pre.float(), dim=-1)
                    log_probs_pre = torch.log(probe_probs_pre + 1e-10)
                    prev_entropy = -(probe_probs_pre * log_probs_pre).sum(dim=-1).mean()

            # Adaptive per-stage threshold.
            use_adaptive = (
                getattr(self.cfg, "act_adaptive_threshold", False) and use_entropy_halt
            )
            if use_adaptive and hasattr(self, "_entropy_delta_ema"):
                stage_ent_threshold = max(
                    self._entropy_delta_ema[stage_idx].item()
                    * getattr(self.cfg, "act_adaptive_margin", 1.0),
                    getattr(self.cfg, "act_adaptive_floor", 0.0),
                )
            else:
                stage_ent_threshold = entropy_threshold if use_entropy_halt else None

            if use_difficulty and stage_ent_threshold is not None and difficulty_scalar > 0:
                stage_ent_threshold = stage_ent_threshold * (
                    1.0 - difficulty_scalar.item() * self.cfg.act_difficulty_scale
                )
                stage_ent_threshold = max(
                    stage_ent_threshold, getattr(self.cfg, "act_adaptive_floor", 0.0)
                )

            for t in range(max_per_stage):
                h_in = h_loop
                if stage_idx > 0 and self.cfg.use_residual and residual is not None:
                    h_in = h_in + self.residual_proj(residual)

                # Stage's own ecology update only fires on the very first step
                # of stages 1+ (stage 0 already updated during the embed pass).
                update_state_t = update_state and t == 0 and stage_idx > 0

                h_out, logits_t, state_t = stage(
                    x=None, hidden=h_in, sigma_prior=sigma_loop, update_state=update_state_t
                )
                del logits_t
                sigma_out = state_t.get("head_sigmas_tensor")

                # Live ecology between ponder steps: update E (entropy) and F (friction)
                # using signals already produced by the stage forward.
                if t > 0 and self.cfg.act_live_ecology:
                    head_ent_tensors = state_t.get("head_entropy_tensors")
                    if head_ent_tensors and len(head_ent_tensors) > 0:
                        with torch.no_grad():
                            hs = stage.head_state
                            ponder_alpha = self.cfg.act_live_ecology_alpha
                            avg_entropy = torch.stack(head_ent_tensors).mean(dim=0)
                            max_ent = math.log(max(hidden.shape[1], 2))
                            avg_entropy_norm = (
                                (avg_entropy / (max_ent + 1e-8)).clamp(0, 1).to(hs._entropy_ema.dtype)
                            )
                            hs._entropy_ema.lerp_(avg_entropy_norm, ponder_alpha)
                            if stage._prev_ponder_entropy is not None:
                                delta = (avg_entropy_norm - stage._prev_ponder_entropy).abs()
                                hs._friction_ema.lerp_(delta, ponder_alpha)
                            stage._prev_ponder_entropy = avg_entropy_norm.clone()

                if sigma_out is not None and sigma_loop is not None:
                    strain_t = (sigma_out - sigma_loop).abs().mean()
                else:
                    strain_t = torch.tensor(0.0, device=device)

                skip_halt = (
                    t == 0 and self.cfg.act_skip_first_halt and not use_preponder
                )
                if skip_halt:
                    lambda_t = torch.zeros(1, device=device).squeeze()
                elif use_entropy_halt:
                    with torch.no_grad():
                        h_probe = h_out.index_select(1, probe_idx).to(output_weight.dtype)
                        probe_logits = F.linear(h_probe, output_weight)
                        probe_probs = F.softmax(probe_logits.float(), dim=-1)
                        log_probs = torch.log(probe_probs + 1e-10)
                        entropy = -(probe_probs * log_probs).sum(dim=-1).mean()

                    if prev_entropy is not None:
                        delta = prev_entropy - entropy
                        stage_entropy_deltas.append(delta.detach())
                        lambda_t = torch.sigmoid(
                            (stage_ent_threshold - delta) / entropy_temperature
                        )
                    else:
                        lambda_t = torch.zeros(1, device=device).squeeze()
                    prev_entropy = entropy.detach()
                else:
                    if use_ema:
                        ema_val = self._strain_ema[stage_idx]
                        effective_threshold = ema_val * getattr(
                            self.cfg, "act_strain_ema_margin", 1.0
                        )
                    else:
                        effective_threshold = getattr(
                            self.cfg, "act_strain_threshold", 0.05
                        )
                    lambda_t = torch.sigmoid(
                        (effective_threshold - strain_t)
                        / getattr(self.cfg, "act_strain_temperature", 0.01)
                    )

                if use_confidence_floor and (lambda_t > 0.5) and use_entropy_halt:
                    max_prob = probe_probs.max(dim=-1).values.mean()
                    if max_prob < self.cfg.act_confidence_floor:
                        lambda_t = torch.zeros(1, device=device).squeeze()

                if use_hard_halt and (lambda_t > 0.5) and t < max_per_stage - 1:
                    h_accum = h_out
                    stage_halt_probs.append(torch.ones(1, device=device).squeeze())
                    stage_strains.append(strain_t)
                    total_strain = total_strain + strain_t
                    sigma_loop = sigma_out
                    break

                if t < max_per_stage - 1:
                    p_t = p_running * lambda_t
                else:
                    p_t = p_running

                h_accum = h_accum + p_t * h_out

                stage_halt_probs.append(p_t)
                stage_strains.append(strain_t)
                total_strain = total_strain + strain_t
                p_running = p_running * (1 - lambda_t)
                h_loop = h_out
                sigma_loop = sigma_out

                if (p_running < eps):
                    break

            _eco_update = getattr(self.cfg, "eval_live_primitives", False)

            if use_ema and _eco_update and stage_strains:
                with torch.no_grad():
                    mean_strain = torch.stack(stage_strains).mean()
                    decay = getattr(self.cfg, "act_strain_ema_decay", 0.99)
                    self._strain_ema[stage_idx] = (
                        decay * self._strain_ema[stage_idx] + (1 - decay) * mean_strain
                    )

            if (
                use_entropy_halt
                and _eco_update
                and stage_entropy_deltas
                and hasattr(self, "_entropy_delta_ema")
            ):
                with torch.no_grad():
                    mean_delta = torch.stack(
                        [
                            d.abs() if isinstance(d, torch.Tensor) else torch.tensor(abs(d), device=device)
                            for d in stage_entropy_deltas
                        ]
                    ).mean()
                    decay = getattr(self.cfg, "act_adaptive_ema_decay", 0.99)
                    self._entropy_delta_ema[stage_idx] = (
                        decay * self._entropy_delta_ema[stage_idx] + (1 - decay) * mean_delta
                    )

            if use_entropy_halt and _eco_update and prev_entropy is not None:
                with torch.no_grad():
                    max_ent_t = stage.entropy_tracker.max_entropy
                    ent_norm_t = (
                        prev_entropy / max_ent_t
                        if isinstance(prev_entropy, torch.Tensor)
                        else torch.tensor(prev_entropy / max_ent_t, device=device)
                    )
                    alpha_ema = 1.0 - stage.entropy_tracker.decay
                    stage.entropy_tracker.output_entropy_ema.lerp_(ent_norm_t, alpha_ema)
                    prev_ent_val = stage.entropy_tracker._prev_entropy
                    if prev_ent_val > 0:
                        delta = prev_ent_val - ent_norm_t
                        stage.entropy_tracker.valence_velocity_ema.lerp_(delta, 0.01)
                    stage.entropy_tracker._prev_entropy.fill_(ent_norm_t)

            hidden = h_accum
            sigma_prior = sigma_out

            if stage.is_last:
                eco = getattr(self, "_ecology_strength", 1.0)
                sigma_mean = sigma_out.mean()
                raw_temp = torch.clamp(1.5 - sigma_mean * 0.8, 0.5, 2.0)
                temp = 1.0 + eco * (raw_temp - 1.0)
                logits = _apply_logit_softcap(
                    stage.output_proj(h_accum) / temp, self.cfg.logit_softcap
                )
            else:
                logits = None

            n_steps = len(stage_halt_probs)
            per_stage_ponder_steps.append(n_steps)
            per_stage_strains.append([s.item() for s in stage_strains])
            per_stage_entropy_deltas.append(stage_entropy_deltas)
            total_halt_probs.extend(stage_halt_probs)
            all_chain_states.append(state_t)

        ponder_cost = total_strain
        self._store_ponder_state(ponder_cost, sum(per_stage_ponder_steps))
        self._last_per_stage_steps = per_stage_ponder_steps
        self._last_per_stage_entropy_deltas = per_stage_entropy_deltas

        self._last_final_hidden = hidden
        if (
            getattr(self.cfg, "scratchpad_need_predictor", False)
            and hasattr(self, "scratchpad_need_head")
        ):
            self._last_scratchpad_pred = self.scratchpad_need_head(hidden).squeeze(-1)
        else:
            self._last_scratchpad_pred = None

        # Metacog → ecology feedback: per-stage scratchpad-need pulls E.
        inject_weights = getattr(self.cfg, "scratchpad_inject_entropy", 0.0)
        if isinstance(inject_weights, (int, float)):
            inject_weights = tuple([float(inject_weights)] * len(self.stages))
        else:
            inject_weights = tuple(float(w) for w in inject_weights)
        if len(inject_weights) < len(self.stages):
            inject_weights = inject_weights + (0.0,) * (len(self.stages) - len(inject_weights))
        elif len(inject_weights) > len(self.stages):
            inject_weights = inject_weights[: len(self.stages)]

        if any(abs(w) > 0 for w in inject_weights) and self._last_scratchpad_pred is not None:
            with torch.no_grad():
                mean_pred = self._last_scratchpad_pred.detach().mean().item()
                cl = getattr(self.cfg, "prim_clamp_lo", 0.01)
                ch = getattr(self.cfg, "prim_clamp_hi", 0.99)
                for s_idx, stage in enumerate(self.stages):
                    delta_s = inject_weights[s_idx] * (mean_pred - 0.5)
                    hs = stage.head_state
                    if hasattr(hs, "_entropy_ema") and abs(delta_s) > 0:
                        hs._entropy_ema.add_(delta_s).clamp_(cl, ch)

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
            if hasattr(self, "_entropy_delta_ema"):
                state["act_entropy_delta_ema"] = self._entropy_delta_ema.tolist()
            if self._last_difficulty_pred is not None:
                state["act_difficulty_pred"] = self._last_difficulty_pred.mean().item()
            if hasattr(self, "_loss_ema"):
                state["act_loss_ema"] = self._loss_ema.item()
            if return_chain_state:
                state["chain_states"] = all_chain_states
            return logits, state

        return logits

    # ----------------- aggregate state + helpers ------------------

    def _aggregate_state(self, chain_states: List[Dict]) -> Dict:
        if not chain_states:
            return {}
        last = chain_states[-1]
        state: Dict = {
            "global_sigma": last.get("global_sigma", 0.5),
            "head_sigmas": last.get("head_sigmas", []),
            "head_positions": last.get("head_positions", []),
            "step": int(self.stages[-1]._stage_step.item()),
            "n_stages": len(chain_states),
        }
        for k in ("blockade", "cosurvival", "head_activations", "head_entropy"):
            if k in last:
                state[k] = last[k]
        all_sigma_tensors = [
            s.get("head_sigmas_tensor")
            for s in chain_states
            if s.get("head_sigmas_tensor") is not None
        ]
        if all_sigma_tensors:
            state["head_sigmas_tensor"] = torch.stack(all_sigma_tensors).mean(dim=0)
        state["per_stage_sigmas"] = [s.get("head_sigmas", []) for s in chain_states]
        state["per_stage_entropy"] = [s.get("head_entropy", []) for s in chain_states]
        return state

    @torch.compiler.disable
    def _update_global_sigma(self, sigma_val) -> None:
        with torch.no_grad():
            self._global_sigma.fill_(sigma_val)

    @torch.compiler.disable
    def _store_ponder_state(self, ponder_cost: torch.Tensor, ponder_steps: int) -> None:
        with torch.no_grad():
            self._last_ponder_cost.fill_(ponder_cost.item())
        self._last_ponder_steps = ponder_steps

    # ----------------- utilities ------------------

    def set_ecology_strength(self, strength: float) -> None:
        """Set the ecology warmup ramp on every submodule. 1.0 = full ecology
        (default at inference); lower values blend toward bypass."""
        self._ecology_strength = strength
        for stage in self.stages:
            stage._ecology_strength = strength
            for layer in stage.layers:
                layer._ecology_strength = strength
                layer.attention._ecology_strength = strength
