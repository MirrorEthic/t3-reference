"""Per-head ecology: primitives, blockade, cosurvival.

This module is the inference-time implementation of the T³ ecology — the
state that each attention head carries between forward passes and that
modulates attention dynamics across the chain.

Six per-head primitives (Cl(3,3) signature):

    E (Entropy)   — blended attention + output entropy        signature  +1
    I (Intensity) — activation magnitude                      signature  +1
    F (Friction)  — |dE/dt| EMA                               signature  +1
    V (Valence)   — Fristonian dual-EMA on free-energy proxy  signature  -1
    C (Coherence) — conjugate to E                            signature  -1
    K (Chronos)   — conjugate to I                            signature  -1

Conjugate pairs (Hamiltonian rotation): E↔C, I↔K, F↔V.

The three classes here own:

    HeadState   — the six primitives, sigma MLP, position embedding, and the
                  learned Cl(3,3) coupling parameters.
    Blockade    — 1/r^N suppression of attention from neighboring heads.
    Cosurvival  — bond graph between heads, loaded from checkpoint and frozen
                  at inference (the inference path consumes bonds; training
                  updates them, and that update lives outside this module).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Geodesic distance on T³ (the 3-torus the heads live on)
# ---------------------------------------------------------------------------


def _geodesic_1d(a: torch.Tensor, b: torch.Tensor, period: float = 1.0) -> torch.Tensor:
    diff = (a - b).abs()
    return torch.min(diff, period - diff)


def _geodesic_t3(
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
    periods: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    d_sq = torch.zeros(pos_a.shape[:-1], device=pos_a.device)
    for i in range(3):
        d = _geodesic_1d(pos_a[..., i], pos_b[..., i], periods[i])
        d_sq = d_sq + d * d
    return torch.sqrt(d_sq + 1e-8)


# ---------------------------------------------------------------------------
# Blockade
# ---------------------------------------------------------------------------


class Blockade(nn.Module):
    """1/r^N suppression of attention from neighboring heads.

    Each head emits an "excitation" (sharp attention = high excitation), and
    nearby heads (small geodesic distance in the head-position embedding) feel
    a suppression proportional to neighbor excitation. Falloff exponent and
    strength are config-set; radius is config-set with optional learnable
    refinement on `HeadState.head_positions`.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.exponent = cfg.blockade_exponent
        self.strength = cfg.blockade_strength
        self.blockade_radius = cfg.blockade_radius_init
        self.register_buffer("_global_step", torch.tensor(0, dtype=torch.long))
        # Per-head suppression from the most recent forward, fed into the
        # Coherence primitive on the next stage.
        self.register_buffer("_last_suppression", torch.zeros(cfg.n_heads))

    def forward(
        self,
        head_excitation: torch.Tensor,         # [n_heads]
        distances: torch.Tensor,                # [n_heads, n_heads]
        blockade_modulation: Optional[torch.Tensor] = None,  # [n_heads, n_heads]
    ) -> torch.Tensor:
        """Per-head suppression in [0, 0.95].

        head_excitation: inverse attention entropy (sharper = more excited).
        blockade_modulation: cosurvival-driven scaling (< 1 = cooperating pair,
            > 1 = interfering pair).
        """
        if not self.cfg.blockade_enabled:
            return torch.zeros(distances.shape[0], device=distances.device)

        warmup = self.cfg.blockade_warmup_steps
        if warmup > 0:
            ramp = (self._global_step.float() / warmup).clamp(max=1.0)
        else:
            ramp = 1.0

        with torch.amp.autocast("cuda", enabled=False):
            dist_f = distances.float()
            blockade_weight = 1.0 / (1.0 + (dist_f / self.blockade_radius).pow(self.exponent))
            blockade_weight = blockade_weight * (
                1.0 - torch.eye(distances.shape[0], device=distances.device)
            )

            if blockade_modulation is not None:
                blockade_weight = blockade_weight * blockade_modulation.float()

            excitation = head_excitation.float()
            if excitation.max() > excitation.min() + 1e-8:
                excitation = (excitation - excitation.min()) / (
                    excitation.max() - excitation.min() + 1e-8
                )

            suppression = torch.matmul(blockade_weight, excitation)
            suppression = (suppression * self.strength * ramp).clamp(0.0, 0.95)

        with torch.no_grad():
            self._last_suppression.copy_(suppression.detach())

        return suppression.to(head_excitation.dtype)


# ---------------------------------------------------------------------------
# Cosurvival
# ---------------------------------------------------------------------------


class Cosurvival(nn.Module):
    """Head bond graph — frozen at inference.

    The `cosurvival` matrix encodes which heads have correlated success during
    training. At inference it's loaded from the checkpoint and consumed (via
    `get_coupling_matrix`, `get_protection_scores`, `get_blockade_modulation`)
    but never updated — bond *learning* is a training-time mechanism that does
    not live in this reference implementation.
    """

    def __init__(self, n_heads: int, cfg):
        super().__init__()
        self.n_heads = n_heads
        self.cfg = cfg

        # cosurvival[i, j] > 0 → heads i, j correlated; < 0 → interfering.
        self.register_buffer("cosurvival", torch.zeros(n_heads, n_heads))
        self.register_buffer("head_loss_ema", torch.zeros(n_heads))
        self.register_buffer("head_loss_var", torch.ones(n_heads))
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

        self._blockade_mod_cache: Optional[torch.Tensor] = None

    def get_coupling_matrix(self) -> torch.Tensor:
        """[n_heads, n_heads] coupling matrix used downstream as a per-pair
        scalar on attention contributions. Identity when cosurvival is off."""
        if not self.cfg.cosurvival_enabled:
            return torch.eye(self.n_heads, device=self.cosurvival.device)
        cs_max = self.cosurvival.abs().max() + 1e-8
        cs_norm = self.cosurvival / cs_max
        coupling = torch.eye(self.n_heads, device=self.cosurvival.device)
        coupling = coupling + cs_norm * self.cfg.cosurvival_lr_coupling
        return coupling

    def get_protection_scores(self) -> torch.Tensor:
        """[n_heads] — sum of positive bonds per head, normalized to [0, 1]."""
        if not self.cfg.cosurvival_enabled:
            return torch.ones(self.n_heads, device=self.cosurvival.device)
        positive_bonds = self.cosurvival.clamp(min=0).sum(dim=1)
        max_bonds = positive_bonds.max() + 1e-8
        return positive_bonds / max_bonds

    def get_blockade_modulation(self) -> torch.Tensor:
        """[n_heads, n_heads] in [0.3, 1.7] — cooperating pairs (< 1) reduce
        each other's blockade pressure, interfering pairs (> 1) amplify it.
        Cached per forward; call `invalidate_blockade_cache` to refresh.
        """
        if self._blockade_mod_cache is not None:
            return self._blockade_mod_cache
        if not self.cfg.cosurvival_enabled:
            return torch.ones(self.n_heads, self.n_heads, device=self.cosurvival.device)
        cs_max = self.cosurvival.abs().max() + 1e-8
        cs_norm = self.cosurvival / cs_max
        modulation = 1.0 - cs_norm * 0.5
        self._blockade_mod_cache = modulation.clamp(0.3, 1.7)
        return self._blockade_mod_cache

    def invalidate_blockade_cache(self) -> None:
        self._blockade_mod_cache = None


# ---------------------------------------------------------------------------
# HeadState — per-head ecology primitives
# ---------------------------------------------------------------------------


class HeadState(nn.Module):
    """Per-head, per-stage ecology state.

    Owns:
      - 6 primitive EMAs (E, I, F, V, C, K) and their conjugate predictions
      - Per-head sigma MLP (the projection from primitives to attention temp)
      - Head positions on T³ (geodesic distances feed Blockade and Cosurvival)
      - Optional Cl(3,3) cross-pair coupling parameters
      - Optional WorldTrace self-model predictions (`_pred_*`)

    The two main entry points are:
      - `compute_head_sigmas(h_per_head)` — differentiable; produces per-head
        attention temperature modulation from current hidden state and EMA buffers
      - `update_grounded_primitives(...)` — in-place EMA update of the buffers
        (called once per stage forward; runs at inference when the chain is
        configured with `eval_live_primitives=True`)

    Several training-only methods on the legacy class are intentionally absent:
    the inter-stage prediction-coding loss, bond prediction loss, sigma-target
    loss, complementarity loss, and bond diagnostics. The parameters those
    losses learned (`inter_stage_predictor`, `bond_predictor`) are still
    registered so that released checkpoints load cleanly; they're inert here.
    """

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        cfg,
        sigma_hidden_override: Optional[int] = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        # HeadState's d_head partitions the residual stream (d_model // n_heads),
        # which can differ from the attention's d_head when those are wider.
        self.d_head = d_model // n_heads
        self.cfg = cfg
        self._sigma_hidden_override = sigma_hidden_override

        # Head positions on T³, initialized via Fibonacci spiral for even coverage.
        init_positions = torch.zeros(n_heads, 3)
        golden = (1 + math.sqrt(5)) / 2
        for i in range(n_heads):
            init_positions[i, 0] = (i / n_heads) % 1.0
            init_positions[i, 1] = (i * golden / n_heads) % 1.0
            init_positions[i, 2] = (i * golden * golden / n_heads) % 1.0
        if cfg.blockade_learnable:
            self.head_positions = nn.Parameter(init_positions)
        else:
            self.register_buffer("head_positions", init_positions)

        with torch.no_grad():
            _pos = init_positions % 1.0
            _d = _geodesic_t3(_pos.unsqueeze(1), _pos.unsqueeze(0))
            _d.fill_diagonal_(float("inf"))
            self._init_nn_distance = _d.min(dim=1).values.mean().item()

        n_prims = cfg.n_primitives

        if cfg.per_head_sigma:
            if cfg.grounded_primitives:
                # 6-primitive EMAs.
                self.register_buffer("_entropy_ema", torch.full((n_heads,), 0.5))
                self.register_buffer("_entropy_prev", torch.full((n_heads,), 0.5))
                self.register_buffer("_friction_ema", torch.full((n_heads,), 0.1))
                self.register_buffer("_valence_ema", torch.full((n_heads,), 0.5))
                self.register_buffer("_coherence_ema", torch.full((n_heads,), 0.5))
                self.register_buffer("_chronos_ema", torch.full((n_heads,), 0.5))
                self.register_buffer("_intensity_ema", torch.full((n_heads,), 0.5))
                self.register_buffer("_last_intensity", torch.full((n_heads,), 0.5))

                # Dual EMAs for Fristonian valence (MACD on attention entropy).
                self.register_buffer("_attn_fast_ema", torch.full((n_heads,), 0.5))
                self.register_buffer("_attn_slow_ema", torch.full((n_heads,), 0.5))
                self.register_buffer("_valence_init_count", torch.tensor(0, dtype=torch.long))

                # WorldTrace self-model predictions.
                self.register_buffer("_pred_E", torch.full((n_heads,), 0.5))
                self.register_buffer("_pred_I", torch.full((n_heads,), 0.5))
                self.register_buffer("_pred_F", torch.full((n_heads,), 0.1))
                self.register_buffer("_pred_V", torch.full((n_heads,), 0.5))
                self.register_buffer("_pred_C", torch.full((n_heads,), 0.5))
                self.register_buffer("_pred_K", torch.full((n_heads,), 0.5))
                self.register_buffer("_self_surprise", torch.zeros(n_heads))

                # Inter-stage primitive predictor — present for ckpt-load fidelity,
                # not invoked at inference. Trained via the inter-stage PC loss.
                if cfg.inter_stage_pc:
                    self.inter_stage_predictor = nn.Linear(n_prims, n_prims)
                    with torch.no_grad():
                        nn.init.eye_(self.inter_stage_predictor.weight)
                        nn.init.zeros_(self.inter_stage_predictor.bias)

                # One-step-delayed excitation, used by attention's K-bias projection.
                self.register_buffer("_last_excitation", torch.zeros(n_heads))

                # Eco-conditioned K-bias projection.
                if getattr(cfg, "eco_key_bias", False):
                    attn_d_head = getattr(cfg, "d_head", 0) or (d_model // n_heads)
                    n_eco_features = getattr(cfg, "eco_key_bias_features", 4)
                    self.key_bias_proj = nn.Linear(n_eco_features, attn_d_head)
                    nn.init.zeros_(self.key_bias_proj.weight)
                    nn.init.zeros_(self.key_bias_proj.bias)

                # Bond predictor — ditto: registered for ckpt fidelity, inert here.
                if getattr(cfg, "cooperative_prediction", False):
                    self.bond_predictor = nn.Linear(n_prims, n_prims)
                    with torch.no_grad():
                        nn.init.eye_(self.bond_predictor.weight)
                        nn.init.zeros_(self.bond_predictor.bias)

                # Optionally learnable Hamiltonian / temperature / clamp params.
                if getattr(cfg, "learned_ecology_params", False):
                    init_omega = cfg.hamiltonian_coupling
                    self._learned_omega = nn.Parameter(
                        torch.tensor(math.log(init_omega / (0.2 - init_omega + 1e-8)))
                    )
                    self._learned_temp_lo = nn.Parameter(torch.tensor(cfg.temp_range_lo))
                    self._learned_temp_hi = nn.Parameter(torch.tensor(cfg.temp_range_hi))
                    self._learned_clamp_lo = nn.Parameter(torch.tensor(cfg.prim_clamp_lo))
                    self._learned_clamp_hi = nn.Parameter(torch.tensor(cfg.prim_clamp_hi))

                # Cl(3,3) cross-pair coupling: 15 grade-2 bivectors.
                # Primitive order [E=0, I=1, F=2, V=3, C=4, K=5].
                # Intra-conjugate-pair indices in the upper-triangular flat layout:
                # (0,4)=E↔C → idx 3, (1,5)=I↔K → idx 8, (2,3)=F↔V → idx 9.
                if getattr(cfg, "hamiltonian_cross_coupling", False):
                    init_omega = cfg.hamiltonian_coupling
                    max_coupling = getattr(cfg, "hamiltonian_max_coupling", 0.2)
                    coupling_init = torch.zeros(15)
                    intra_val = math.atanh(min(init_omega / max_coupling, 0.99))
                    coupling_init[3] = intra_val
                    coupling_init[8] = intra_val
                    coupling_init[9] = intra_val
                    self._coupling_params = nn.Parameter(coupling_init)
                    self._coupling_max = max_coupling

                    # Optional grade-3 trivectors (state-dependent coupling).
                    if getattr(cfg, "hamiltonian_trivectors", False):
                        self._trivector_params = nn.Parameter(torch.zeros(20))
                        triples = []
                        for i in range(6):
                            for j in range(i + 1, 6):
                                for k in range(j + 1, 6):
                                    triples.append((i, j, k))
                        self._trivector_triples = triples
            else:
                # Non-grounded fallback: per-head linear primitive heads.
                self.entropy_heads = nn.ModuleList(
                    [nn.Linear(self.d_head, 1) for _ in range(n_heads)]
                )
                self.intensity_heads = nn.ModuleList(
                    [nn.Linear(self.d_head, 1) for _ in range(n_heads)]
                )
                self.friction_heads = nn.ModuleList(
                    [nn.Linear(self.d_head, 1) for _ in range(n_heads)]
                )
                self.valence_heads = nn.ModuleList(
                    [nn.Linear(self.d_head, 1) for _ in range(n_heads)]
                )

            # Per-head sigma MLP (batched across heads).
            sigma_hidden = (
                self._sigma_hidden_override
                if self._sigma_hidden_override is not None
                else getattr(cfg, "sigma_hidden", 16)
            )
            w1_bound = 1.0 / (n_prims ** 0.5)
            w2_bound = 1.0 / (sigma_hidden ** 0.5)
            self.sigma_w1 = nn.Parameter(
                torch.empty(n_heads, sigma_hidden, n_prims).uniform_(-w1_bound, w1_bound)
            )
            self.sigma_b1 = nn.Parameter(torch.zeros(n_heads, sigma_hidden))
            self.sigma_w2 = nn.Parameter(
                torch.empty(n_heads, 1, sigma_hidden).uniform_(-w2_bound, w2_bound)
            )
            self.sigma_b2 = nn.Parameter(torch.zeros(n_heads, 1))
        else:
            # Global (non-per-head) sigma fallback.
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

        # Non-buffer caches initialized lazily.
        self._dist_cache: Optional[torch.Tensor] = None

    # -------------------- coupling rotation --------------------

    def _apply_coupling_rotation(self, prims: torch.Tensor) -> torch.Tensor:
        """Cl(3,3) cross-pair rotation of the primitive vector. Differentiable
        through `_coupling_params` (and `_trivector_params` if enabled). Used by
        the sigma MLP path so gradient reaches the coupling parameters."""
        if getattr(self, "_coupling_params", None) is None:
            return prims

        vals = torch.tanh(self._coupling_params) * self._coupling_max
        Omega = torch.zeros(6, 6, device=vals.device, dtype=vals.dtype)
        idx = 0
        for i in range(6):
            for j in range(i + 1, 6):
                Omega[i, j] = vals[idx]
                Omega[j, i] = -vals[idx]
                idx += 1

        if getattr(self, "_trivector_params", None) is not None:
            x_centered = prims - 0.5
            x_mean = x_centered.mean(dim=0)
            tri_vals = torch.tanh(self._trivector_params) * self._coupling_max
            for t_idx, (i, j, k) in enumerate(self._trivector_triples):
                alpha = tri_vals[t_idx]
                Omega[i, j] = Omega[i, j] + alpha * x_mean[k]
                Omega[j, i] = Omega[j, i] - alpha * x_mean[k]
                Omega[j, k] = Omega[j, k] + alpha * x_mean[i]
                Omega[k, j] = Omega[k, j] - alpha * x_mean[i]
                Omega[k, i] = Omega[k, i] + alpha * x_mean[j]
                Omega[i, k] = Omega[i, k] - alpha * x_mean[j]

        R = torch.linalg.matrix_exp(Omega)
        prims_centered = prims - 0.5
        prims_rotated = (prims_centered @ R.t()) + 0.5
        if getattr(self.cfg, "sigma_modulated_coupling", False):
            blend = self._last_head_sigmas.detach().unsqueeze(-1)
            prims = prims * (1 - blend) + prims_rotated * blend
        else:
            prims = prims_rotated
        return prims

    # -------------------- sigma path (differentiable) --------------------

    def compute_head_sigmas(
        self, h_per_head: torch.Tensor, warmup_frac: float = 1.0
    ) -> torch.Tensor:
        """Per-head sigma in (0, 1).

        h_per_head: [B, T, n_heads, d_head]. Returns [n_heads]. Differentiable
        through the sigma MLP and (when enabled) the coupling parameters.
        """
        if self.cfg.per_head_sigma:
            h_pooled = h_per_head.mean(dim=(0, 1))  # [n_heads, d_head]

            if self.cfg.grounded_primitives:
                norms = h_pooled.float().norm(dim=-1)
                I_all = norms / (norms.max() + 1e-8)
                with torch.no_grad():
                    self._last_intensity.copy_(I_all.detach())

                # Stack EMAs (detached) with current intensity (live).
                prims = torch.stack(
                    [
                        self._entropy_ema.detach(),
                        I_all.detach(),
                        self._friction_ema.detach(),
                        self._valence_ema.detach(),
                        self._coherence_ema.detach(),
                        self._chronos_ema.detach(),
                    ],
                    dim=-1,
                )

                # Cross-pair coupling routes gradient to `_coupling_params`.
                prims = self._apply_coupling_rotation(prims)

                with torch.amp.autocast("cuda", enabled=False):
                    prims_f = prims.float()
                    w1 = self.sigma_w1.float().clone()
                    b1 = self.sigma_b1.float().clone()
                    w2 = self.sigma_w2.float().clone()
                    b2 = self.sigma_b2.float().clone()
                    h1 = torch.bmm(w1, prims_f.unsqueeze(-1)).squeeze(-1) + b1
                    h1 = torch.tanh(h1)
                    h2 = torch.bmm(w2, h1.unsqueeze(-1)).squeeze(-1) + b2
                    sigmas = torch.sigmoid(h2.squeeze(-1))

                if getattr(self, "_sigma_temporal_cache_active", False):
                    sigmas = sigmas - sigmas.detach() + 0.5
            else:
                sigmas_list = []
                for i in range(self.n_heads):
                    h_i = h_pooled[i]
                    E = torch.sigmoid(self.entropy_heads[i](h_i.unsqueeze(0)))
                    I = torch.sigmoid(self.intensity_heads[i](h_i.unsqueeze(0)))
                    F_val = torch.sigmoid(self.friction_heads[i](h_i.unsqueeze(0)))
                    V = torch.sigmoid(self.valence_heads[i](h_i.unsqueeze(0)))
                    C = 1.0 - E
                    K = 1.0 - F_val
                    prims = torch.cat([E, I, F_val, V, C, K], dim=-1)
                    sigma_i = torch.sigmoid(
                        (
                            self.sigma_w2[i]
                            @ torch.tanh(
                                self.sigma_w1[i]
                                @ prims.squeeze(0).unsqueeze(-1)
                                + self.sigma_b1[i].unsqueeze(-1)
                            )
                            + self.sigma_b2[i]
                        ).squeeze()
                    )
                    sigmas_list.append(sigma_i)
                sigmas = torch.stack(sigmas_list)

            if self.training:
                sigmas = sigmas + torch.randn_like(sigmas) * 0.02
                sigmas = sigmas.clamp(0.01, 0.99)

            if warmup_frac < 1.0:
                sigmas = 0.5 + warmup_frac * (sigmas - 0.5)

            with torch.no_grad():
                self._last_head_sigmas.copy_(sigmas.detach())

            self._live_head_sigmas = sigmas.detach()
            return sigmas

        # Global-sigma fallback.
        h_flat = h_per_head.reshape(h_per_head.shape[0], h_per_head.shape[1], -1)
        h_pooled = h_flat.mean(dim=(0, 1))
        E = torch.sigmoid(self.global_entropy_head(h_pooled.unsqueeze(0)))
        I = torch.sigmoid(self.global_intensity_head(h_pooled.unsqueeze(0)))
        F_val = torch.sigmoid(self.global_friction_head(h_pooled.unsqueeze(0)))
        V = torch.sigmoid(self.global_valence_head(h_pooled.unsqueeze(0)))
        C = 1.0 - E
        K = 1.0 - F_val
        p = torch.cat([E, I, F_val, V, C, K], dim=-1)
        sigma = self.global_sigma_projection(p).squeeze()
        result = sigma.expand(self.n_heads)
        if warmup_frac < 1.0:
            result = 0.5 + warmup_frac * (result - 0.5)
        return result

    # -------------------- EMA / Hamiltonian / self-model update --------------------

    @torch.compiler.disable
    def update_grounded_primitives(
        self,
        head_entropy: torch.Tensor,
        max_seq_len: int = 512,
        output_entropy: Optional[float] = None,
        warmup_frac: float = 1.0,
        protection_scores: Optional[torch.Tensor] = None,
        blockade_suppression: Optional[torch.Tensor] = None,
    ):
        """In-place update of the ecology EMAs from the most recent attention.

        Three phases run in order:
          1. Observation-driven EMA updates of E, I, F, V, C, K from the
             stage's average attention entropy and the ecological signals.
          2. Hamiltonian conjugate-pair coupling — three planar rotations
             (E↔C, I↔K, F↔V) with optional null-cone restoring force.
          3. WorldTrace self-model update (`_pred_*` + `_self_surprise`).
        """
        if not self.cfg.grounded_primitives:
            return

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            max_ent = math.log(max(max_seq_len, 2))
            norm_entropy = (head_entropy / max_ent).clamp(0, 1).to(self._entropy_ema.dtype)

            if output_entropy is not None and warmup_frac > 0.01:
                alpha = self.cfg.blend_alpha
                oe_tensor = torch.full_like(norm_entropy, output_entropy)
                blended = alpha * norm_entropy + (1 - alpha) * oe_tensor
            else:
                blended = norm_entropy

            alpha = 0.05

            # ---- Phase 1: observation-driven forcing ----
            self._entropy_ema.lerp_(blended, alpha)
            self._intensity_ema.lerp_(self._last_intensity, alpha)

            delta_e = (blended - self._entropy_prev).abs()
            _prev_I = getattr(self, "_last_intensity_prev", self._last_intensity)
            delta_i = (self._last_intensity - _prev_I).abs()
            self._last_intensity_prev = self._last_intensity.clone()
            w_i = self.cfg.friction_intensity_weight
            friction_raw = (1.0 - w_i) * delta_e + w_i * delta_i
            self._friction_ema.lerp_(friction_raw, alpha)

            fast_alpha = 1.0 - self.cfg.valence_fast_decay
            slow_alpha = 1.0 - self.cfg.valence_slow_decay
            self._attn_fast_ema.lerp_(norm_entropy, fast_alpha)
            self._attn_slow_ema.lerp_(norm_entropy, slow_alpha)

            self._valence_init_count += 1
            if self._valence_init_count <= self.cfg.valence_warmup_calls:
                pass
            elif self._valence_init_count == self.cfg.valence_warmup_calls + 1:
                self._attn_slow_ema.copy_(self._attn_fast_ema)
            else:
                v_diff = self._attn_slow_ema - self._attn_fast_ema
                v_centered = v_diff - v_diff.mean()
                if self.cfg.valence_relative:
                    v_std = v_centered.std() + 1e-8
                    v_normalized = v_centered / v_std
                    v_scaled = torch.sigmoid(v_normalized * self.cfg.valence_scale)
                else:
                    v_scaled = torch.sigmoid(v_centered * self.cfg.valence_scale)
                self._valence_ema.lerp_(v_scaled, 0.1)

            head_mean = norm_entropy.mean()
            head_std = norm_entropy.std() + 1e-8
            c_agreement = torch.exp(-0.5 * ((norm_entropy - head_mean) / head_std) ** 2)
            if blockade_suppression is not None:
                c_ecological = c_agreement * (1.0 - blockade_suppression.clamp(0, 1))
                c_raw = 0.7 * c_agreement + 0.3 * c_ecological
            else:
                c_raw = c_agreement
            self._coherence_ema.lerp_(c_raw, alpha)

            pred_error = (norm_entropy - self._entropy_ema).abs()
            max_error = pred_error.max() + 1e-8
            k_temporal = (1.0 - pred_error / max_error).clamp(0.01, 0.99)
            if protection_scores is not None:
                k_raw = 0.7 * k_temporal + 0.3 * protection_scores.clamp(0, 1)
            else:
                k_raw = k_temporal
            self._chronos_ema.lerp_(k_raw, alpha)

            # ---- Phase 2: Hamiltonian conjugate-pair coupling ----
            if getattr(self, "_learned_omega", None) is not None:
                omega = torch.sigmoid(self._learned_omega) * 0.2
            else:
                omega = self.cfg.hamiltonian_coupling
            cl = getattr(self.cfg, "prim_clamp_lo", 0.01)
            ch = getattr(self.cfg, "prim_clamp_hi", 0.99)

            with torch.no_grad():
                if (omega if isinstance(omega, float) else omega.item()) > 1e-6:
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

                # Null-cone restoring force on Q = E²+I²+F² − V²−C²−K².
                nc = getattr(self.cfg, "null_cone_strength", 0.0)
                if nc > 0:
                    E = self._entropy_ema
                    I_p = self._intensity_ema
                    F_p = self._friction_ema
                    C = self._coherence_ema
                    K = self._chronos_ema
                    V = self._valence_ema
                    Q = (E ** 2 + I_p ** 2 + F_p ** 2) - (C ** 2 + K ** 2 + V ** 2)
                    self._entropy_ema.add_(-nc * 2 * Q * E).clamp_(cl, ch)
                    self._intensity_ema.add_(-nc * 2 * Q * I_p).clamp_(cl, ch)
                    self._friction_ema.add_(-nc * 2 * Q * F_p).clamp_(cl, ch)
                    self._coherence_ema.add_(nc * 2 * Q * C).clamp_(cl, ch)
                    self._chronos_ema.add_(nc * 2 * Q * K).clamp_(cl, ch)
                    self._valence_ema.add_(nc * 2 * Q * V).clamp_(cl, ch)

            self._entropy_prev.copy_(blended)

            # ---- Phase 3: WorldTrace self-model ----
            self._update_self_model(
                raw_E=blended,
                raw_I=self._last_intensity,
                raw_F=friction_raw,
                raw_V=self._valence_ema,
                raw_C=c_raw,
                raw_K=k_raw,
            )

    @torch.compiler.disable
    def _update_self_model(self, raw_E, raw_I, raw_F, raw_V, raw_C, raw_K):
        """Update WorldTrace predictions and write `_self_surprise`. Surprise is
        confidence-weighted (intensity-amplified) L2 between predictions and
        the raw observations seen this forward."""
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            predicted = torch.stack(
                [self._pred_E, self._pred_I, self._pred_F, self._pred_V, self._pred_C, self._pred_K],
                dim=-1,
            )
            raw_obs = torch.stack(
                [raw_E, raw_I, raw_F, raw_V, raw_C, raw_K], dim=-1
            )
            energy = (predicted - raw_obs).pow(2).sum(dim=-1).sqrt()
            intensity = raw_I.clamp(0.0, 1.0)
            confidence_weight = 0.5 + intensity
            surprise = energy * confidence_weight
            self._self_surprise.copy_(surprise)

            sm_alpha = self.cfg.self_model_alpha
            self._pred_E.lerp_(raw_E, sm_alpha)
            self._pred_I.lerp_(raw_I, sm_alpha)
            self._pred_F.lerp_(raw_F, sm_alpha)
            self._pred_V.lerp_(raw_V, sm_alpha)
            self._pred_C.lerp_(raw_C, sm_alpha)
            self._pred_K.lerp_(raw_K, sm_alpha)

    # -------------------- inference accessors --------------------

    def get_current_primitives(self) -> torch.Tensor:
        """[n_heads, 6] stack of [E, I, F, V, C, K] from EMA buffers."""
        return torch.stack(
            [
                self._entropy_ema,
                self._last_intensity,
                self._friction_ema,
                self._valence_ema,
                self._coherence_ema,
                self._chronos_ema,
            ],
            dim=-1,
        )

    def compute_sigma_complement(
        self, own_sigma: torch.Tensor, cosurvival_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Per-head sigma offset that nudges bonded heads toward complementary
        attention temperature. Active at inference: bonds in the checkpoint
        produce a deterministic shift on each forward."""
        cs_pos = cosurvival_matrix.detach().clamp(min=0)
        cs_norm = cs_pos / (cs_pos.sum(dim=1, keepdim=True) + 1e-8)
        partner_mean_sigma = torch.mv(cs_norm, own_sigma.detach())
        diff = own_sigma.detach() - partner_mean_sigma
        offset = diff.sign() * self.cfg.sigma_complement_strength
        bond_strength = cs_pos.sum(dim=1)
        bond_mask = (bond_strength > 0.1).float()
        return offset * bond_mask

    def get_pairwise_distances(self) -> torch.Tensor:
        """[n_heads, n_heads] geodesic distance on T³. Cached per forward;
        invalidate via `invalidate_distance_cache`."""
        if self._dist_cache is not None:
            return self._dist_cache
        pos = self.head_positions % 1.0
        pos_i = pos.unsqueeze(1)
        pos_j = pos.unsqueeze(0)
        self._dist_cache = _geodesic_t3(pos_i, pos_j)
        return self._dist_cache

    def invalidate_distance_cache(self) -> None:
        self._dist_cache = None
