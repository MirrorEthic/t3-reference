"""Harness self-test + port-equivalence gate.

Loads the saved capture from `.claude/harness/reference.pt` (a frozen
fingerprint of the released-checkpoint forward pass, captured before any
porting work) and asserts the current model reproduces it at tight tolerance.
Any drift in the architecture port shows up here as a logits or per-stage
hidden-state mismatch.

If the reference file does not yet exist, this test bootstraps it from the
current model (after first proving the model is deterministic across two
captures).
"""

from __future__ import annotations

from pathlib import Path

import torch

from t3 import T3Model
from harness import (
    capture,
    compare,
    load_reference,
    make_fixture,
    save_reference,
)

REF_DIR = Path(__file__).resolve().parents[1] / ".claude" / "harness"
REF_PATH = REF_DIR / "reference.pt"
KEYS_PATH = REF_DIR / "state_dict_keys.txt"
TOL = 1e-5


def test_port_equivalent_to_phase0_reference(run3_checkpoint):
    torch.manual_seed(0)
    model = T3Model.from_checkpoint(run3_checkpoint, map_location="cpu")
    ids = make_fixture()

    cap_a = capture(model, ids)
    cap_b = capture(model, ids)

    # Determinism on the current model (snapshot/restore must hold).
    assert torch.equal(cap_a.logits, cap_b.logits), (
        "Current model is nondeterministic across repeated captures — harness "
        "snapshot/restore is broken."
    )
    assert len(cap_a.stage_hiddens) == len(cap_b.stage_hiddens)
    for i, (a, b) in enumerate(zip(cap_a.stage_hiddens, cap_b.stage_hiddens)):
        assert torch.equal(a, b), f"Per-stage hidden[{i}] is not deterministic"

    if not REF_PATH.exists():
        # Bootstrap: no reference yet, save the current capture as ground truth.
        save_reference(cap_a, REF_PATH)
        REF_DIR.mkdir(parents=True, exist_ok=True)
        KEYS_PATH.write_text("\n".join(cap_a.state_dict_keys) + "\n")
        return

    ref = load_reference(REF_PATH)
    diff = compare(ref, cap_a, tol=TOL)
    assert diff["pass"], (
        f"Port-equivalence failed against saved reference:\n"
        f"  logits_max_abs={diff['logits_max_abs']:.3e}\n"
        f"  per_stage_max_abs={diff['per_stage_max_abs']}\n"
        f"  n_stage_calls ref={diff['n_stage_calls_ref']} cand={diff['n_stage_calls_cand']}\n"
        f"  state_keys missing on candidate: {diff['state_keys_missing_in_candidate'][:5]}\n"
        f"  state_keys extra on candidate: {diff['state_keys_extra_in_candidate'][:5]}"
    )
