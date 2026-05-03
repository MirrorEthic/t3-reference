"""Shared test fixtures.

The integration tests need a real T³ checkpoint. Resolution order:

    1. $T3_LOCAL_CKPT — explicit override
    2. The path the dev machine has (skipped silently if absent)
    3. Download from Hugging Face (mirrorethic/t3-124m-v36)

If none work — typically only when offline + no local copy — the test
skips with an explanatory message rather than failing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

DEV_PATH = Path(
    "/home/garret-sutherland/CVMP/T3_sims/t3v2/t3v3/t3v36/"
    "checkpoints_v36_run3/best.pt"
)
HF_REPO = "mirrorethic/t3-124m-v36"
HF_FILE = "pytorch_model.bin"


def _resolve_checkpoint() -> str | None:
    env = os.environ.get("T3_LOCAL_CKPT")
    if env and Path(env).exists():
        return env
    if DEV_PATH.exists():
        return str(DEV_PATH)
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(HF_REPO, HF_FILE)
    except Exception:
        return None


@pytest.fixture(scope="session")
def run3_checkpoint() -> str:
    path = _resolve_checkpoint()
    if path is None:
        pytest.skip(
            "Run-3 checkpoint unavailable: set T3_LOCAL_CKPT, place it at "
            f"{DEV_PATH}, or ensure huggingface_hub can download "
            f"{HF_REPO}/{HF_FILE}."
        )
    return path
