"""Ecology-modulated multi-head attention.

Standard MHA augmented with:
    - σ-modulated attention temperature (per head)
    - learned key-bias projection from ecology primitives (`key_bias_proj`)
    - blockade suppression of off-diagonal head correlations

PLACEHOLDER: extraction from t3v36/t3v3_model.py.
"""

from __future__ import annotations

from torch import nn


class EcologyAttention(nn.Module):  # pragma: no cover
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Extraction pending.")
