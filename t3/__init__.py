"""T³ — Clifford-algebra-augmented transformer (reference implementation).

Inference-only reference. Training infrastructure is not included.
See https://t3atlas.dev for the public trace library and benchmarks.
"""

from t3.config import T3Config
from t3.model import T3Model

__version__ = "0.1.0"
__all__ = ["T3Config", "T3Model"]
