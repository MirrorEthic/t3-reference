"""Adaptive computation time (ACT) — output-entropy halt + confidence floor.

Halt condition (per stage, per token):
    halt = (output_entropy < threshold * (1 - difficulty * scale))
           AND (max_softmax_prob >= confidence_floor)

Difficulty predictor:
    h_pooled = hidden.mean(dim=1)
    difficulty = sigmoid( Linear(64, 1)( Tanh( Linear(d_model, 64)(h_pooled) ) ) )

PLACEHOLDER: extraction from t3v36/t3v3_chain.py.
"""

from __future__ import annotations

from torch import nn


class ACTController(nn.Module):  # pragma: no cover
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Extraction pending.")
