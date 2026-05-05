"""T3Model — public inference wrapper for the T³ reference architecture."""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import torch
from torch import nn

from t3.chain import T3Chain
from t3.config import T3Config


def _project_config(ckpt_config: dict) -> T3Config:
    """Project a checkpoint's config dict onto `T3Config`. Unknown keys are
    silently dropped — they are training-time hyperparameters that have no
    architectural meaning."""
    known = {f.name for f in dataclass_fields(T3Config)}
    kept = {k: v for k, v in ckpt_config.items() if k in known}
    for tup_field in ("layers_per_stage", "scratchpad_inject_entropy"):
        if tup_field in kept and isinstance(kept[tup_field], list):
            kept[tup_field] = tuple(kept[tup_field])
    return T3Config(**kept)


def _augment_with_runtime_fields(cfg: T3Config, ckpt_config: dict) -> T3Config:
    """Attach extra runtime fields the chain consults via `getattr` but which
    are not architectural parameters (e.g. `eval_live_primitives`,
    `act_strain_halt`, `act_strain_ema_enabled`). Setting them on the config
    object preserves the legacy chain's `getattr(cfg, ..., default)` pattern
    without polluting the public `T3Config` dataclass."""
    runtime_keys = (
        "eval_live_primitives",
        "act_strain_halt",
        "act_strain_ema_enabled",
        "act_adaptive_threshold",
        "act_strain_threshold",
        "act_strain_temperature",
        "act_strain_ema_decay",
        "act_strain_ema_margin",
        "act_adaptive_margin",
        "act_adaptive_floor",
        "act_adaptive_ema_decay",
        "blockade_radius_auto",
        "use_qk_norm",
        "use_post_norms",
        "use_triton_kernels",
        "v1_residual_enabled",
        "v1_residual_gating",
        "v1_residual_fixed_lambda",
        "embed_scale",
        "sigma_hidden",
        "sigma_hidden_per_stage",
        "sigma_complement_strength",
        "cooperative_prediction",
        "cooperative_prediction_bond_threshold",
        "eco_key_bias_features",
        "eco_key_bias_scale",
        "hamiltonian_max_coupling",
        "learned_ecology_params",
        "use_flex_attention",
        "d_head",
    )
    for k in runtime_keys:
        if k in ckpt_config and not hasattr(cfg, k):
            object.__setattr__(cfg, k, ckpt_config[k])
    return cfg


def _strip_compile_prefix(state: dict) -> dict:
    """Remove the `_orig_mod.` prefix that torch.compile inserts."""
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


class T3Model(nn.Module):
    """Top-level T³ reference model.

    Built from a `T3Config` or loaded from a published checkpoint via
    `T3Model.from_checkpoint`. The forward delegates to `T3Chain`, which
    runs per-stage adaptive computation with the ecology dynamics.
    """

    def __init__(self, config: T3Config):
        super().__init__()
        self.config = config
        self.chain = T3Chain(config)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any):
        return self.chain(input_ids, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
        strict: bool = False,
    ) -> "T3Model":
        """Load a published T³ checkpoint.

        `strict=False` by default because some checkpoints carry training-only
        tracker buffers (e.g. dynamic-Ω shadows) that the reference model does
        not register. Missing/unexpected keys are exposed via `_load_report`.
        """
        ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
        cfg = _project_config(ckpt["config"])
        cfg = _augment_with_runtime_fields(cfg, ckpt["config"])
        model = cls(cfg)
        state = _strip_compile_prefix(ckpt["model_state"])
        report = model.chain.load_state_dict(state, strict=strict)
        model._load_report = {
            "missing": list(report.missing_keys),
            "unexpected": list(report.unexpected_keys),
            "checkpoint_step": ckpt.get("step"),
            "checkpoint_val_ppl": ckpt.get("val_ppl"),
            "run_id": ckpt.get("run_id"),
        }
        return model

    def get_load_report(self) -> dict | None:
        return getattr(self, "_load_report", None)
