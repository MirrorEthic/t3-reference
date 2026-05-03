"""End-to-end: load run-3 best.pt, run a forward pass, sanity-check logits."""

from pathlib import Path

import pytest
import torch

from t3 import T3Model

RUN3 = Path(
    "/home/garret-sutherland/CVMP/T3_sims/t3v2/t3v3/t3v36/"
    "checkpoints_v36_run3/best.pt"
)


@pytest.fixture(scope="module")
def model():
    if not RUN3.exists():
        pytest.skip(f"Run-3 checkpoint not present at {RUN3}")
    m = T3Model.from_checkpoint(RUN3)
    m.eval()
    return m


def test_load_report_shape(model):
    rep = model.get_load_report()
    assert rep is not None
    assert rep["checkpoint_val_ppl"] == pytest.approx(27.7592, rel=1e-4)
    assert rep["checkpoint_step"] == 2500
    assert rep["run_id"] == "v36_run3"
    # Missing keys we tolerate (training-loop trackers); print for diagnostics.
    print(f"\n  missing  ({len(rep['missing'])}): {rep['missing'][:10]}")
    print(f"  unexpect ({len(rep['unexpected'])}): {rep['unexpected'][:10]}")


def test_forward_runs_and_returns_logits(model):
    torch.manual_seed(0)
    input_ids = torch.randint(0, 50257, (1, 16))
    with torch.no_grad():
        out = model(input_ids)
    # The legacy chain's forward returns (logits, aux) when return_dict=True; check both shapes.
    if isinstance(out, tuple):
        logits = out[0]
    elif isinstance(out, dict):
        logits = out.get("logits") or out.get("hidden")
    else:
        logits = out
    assert logits.shape[0] == 1
    assert logits.shape[1] == 16
    assert logits.shape[-1] == 50257
    assert torch.isfinite(logits).all(), "non-finite logits"
    print(f"\n  logits[0, -1, :5] = {logits[0, -1, :5].tolist()}")
