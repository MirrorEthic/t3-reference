"""T3Model — public inference wrapper around the vendored chain.

The clean module split (t3.ecology, t3.attention, t3.act, t3.chain) is in
progress; until it lands, T3Model delegates to t3._legacy_chain.T3v3Chain so
that the forward pass is bit-identical to the training-script implementation
that produced the released checkpoints.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import torch
from torch import nn

from t3._legacy_chain import T3v3Chain, T3v3ChainConfig
from t3.config import T3Config


def _build_legacy_config(ckpt_config: dict) -> T3v3ChainConfig:
    """Project a checkpoint's config dict onto T3v3ChainConfig kwargs."""
    known = {f.name for f in dataclass_fields(T3v3ChainConfig)}
    kept = {k: v for k, v in ckpt_config.items() if k in known}
    # `tuple` is the declared type for scratchpad_inject_entropy etc.; pickle may
    # bring it back as a list.
    for tup_field in ("scratchpad_inject_entropy",):
        if tup_field in kept and isinstance(kept[tup_field], list):
            kept[tup_field] = tuple(kept[tup_field])
    return T3v3ChainConfig(**kept)


def _strip_compile_prefix(state: dict) -> dict:
    """Remove the `_orig_mod.` prefix that torch.compile inserts into state dicts."""
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


class T3Model(nn.Module):
    """Top-level T³ reference model.

    Constructed from a `T3Config` (clean public schema) or loaded directly from
    a published checkpoint via `T3Model.from_checkpoint`.

    Forward signature (delegates to T3v3Chain):
        logits, aux = model(input_ids, attention_mask=None, return_dict=True)
    """

    def __init__(self, config: T3Config | T3v3ChainConfig):
        super().__init__()
        if isinstance(config, T3Config):
            # Build a legacy config from the public schema.
            legacy_kwargs = {
                k: v for k, v in config.to_dict().items()
                if k in {f.name for f in dataclass_fields(T3v3ChainConfig)}
            }
            self._legacy_config = T3v3ChainConfig(**legacy_kwargs)
        else:
            self._legacy_config = config
        self.config = config
        self.chain = T3v3Chain(self._legacy_config)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any):
        return self.chain(input_ids, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
        strict: bool = False,
    ) -> "T3Model":
        """Load a published T³ checkpoint into the reference model.

        `strict=False` by default because the training-script `model_state` may
        carry tracker buffers (`_last_ponder_cost`, `_entropy_delta_ema`, etc.)
        that the reference model registers via different code paths. Missing /
        unexpected keys are reported on the returned model as `_load_report`.
        """
        ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
        legacy_cfg = _build_legacy_config(ckpt["config"])
        model = cls(legacy_cfg)
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
        """Return the strict-load diagnostic from the most recent from_checkpoint call."""
        return getattr(self, "_load_report", None)
