"""Invariant harness for the legacy → clean-modules port.

Captures a numerical fingerprint of a model's forward pass on a fixed
token-id fixture, so a candidate implementation can be compared against a
saved reference at tight tolerance.

Usage:

    from harness import capture, save_reference, load_reference, compare, make_fixture

    ids = make_fixture()
    ref_cap = capture(legacy_model, ids)
    save_reference(ref_cap, ".claude/harness/reference_<run3-sha>.pt")

    # ...later, after porting...
    cand_cap = capture(ported_model, ids)
    diff = compare(load_reference(".claude/harness/reference_<run3-sha>.pt"), cand_cap)
    assert diff["pass"], diff
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

DEFAULT_TOL = 1e-5


@dataclass
class Capture:
    logits: torch.Tensor                  # [B, T, V]
    stage_hiddens: list[torch.Tensor]     # one entry per stage forward call (ACT may call > n_stages)
    state_dict_keys: list[str]


def make_fixture(
    seed: int = 0,
    batch: int = 2,
    seq_len: int = 16,
    vocab_size: int = 50257,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (batch, seq_len), generator=g)


def capture(model, ids: torch.Tensor) -> Capture:
    """Run a deterministic forward and capture logits + per-stage hiddens.

    `model` must expose `.chain` with `.stages` (true for both legacy `T3Model`
    and the ported implementation).

    Determinism: the chain has ~49 buffers that mutate on every forward (the
    `_last_*` snapshot registers and per-stage `_entropy_ema`) regardless of
    `update_state`, because they store within-forward feedback. To make the
    capture reproducible, this function snapshots the chain's full state_dict
    before the forward and restores it after — so successive calls observe
    identical input state and produce identical outputs.
    """
    model.eval()
    chain = model.chain
    stage_hiddens: list[torch.Tensor] = []

    def _hook(_mod, _inp, out):
        # T3Stage forward returns (hidden, logits, stage_state).
        hidden = out[0] if isinstance(out, tuple) else out
        stage_hiddens.append(hidden.detach().cpu())

    snapshot = {k: v.detach().clone() for k, v in chain.state_dict().items()}
    handles = [stage.register_forward_hook(_hook) for stage in chain.stages]
    try:
        with torch.no_grad():
            out = chain(ids, return_state=True, update_state=False)
    finally:
        for h in handles:
            h.remove()
        chain.load_state_dict(snapshot, strict=True)

    logits = out[0] if isinstance(out, tuple) else out
    return Capture(
        logits=logits.detach().cpu(),
        stage_hiddens=stage_hiddens,
        state_dict_keys=list(snapshot.keys()),
    )


def save_reference(cap: Capture, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "logits": cap.logits,
            "stage_hiddens": cap.stage_hiddens,
            "state_dict_keys": cap.state_dict_keys,
        },
        path,
    )
    return path


def load_reference(path: str | Path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def compare(ref: dict, cap: Capture, tol: float = DEFAULT_TOL) -> dict:
    """Diff a capture against a saved reference. Returns a diagnostic dict.

    `pass` is True iff every numeric max-abs-diff < tol AND the per-stage call
    count matches AND no required state-dict keys are missing on the candidate.
    """
    diffs: dict = {}

    diffs["logits_max_abs"] = float((ref["logits"] - cap.logits).abs().max())

    ref_n, cand_n = len(ref["stage_hiddens"]), len(cap.stage_hiddens)
    diffs["n_stage_calls_ref"] = ref_n
    diffs["n_stage_calls_cand"] = cand_n
    if ref_n == cand_n:
        diffs["per_stage_max_abs"] = [
            float((r - c).abs().max())
            for r, c in zip(ref["stage_hiddens"], cap.stage_hiddens)
        ]
    else:
        diffs["per_stage_max_abs"] = None  # sentinel: structural mismatch

    ref_keys = set(ref["state_dict_keys"])
    cand_keys = set(cap.state_dict_keys)
    diffs["state_keys_missing_in_candidate"] = sorted(ref_keys - cand_keys)
    diffs["state_keys_extra_in_candidate"] = sorted(cand_keys - ref_keys)

    numeric = [diffs["logits_max_abs"]]
    if diffs["per_stage_max_abs"] is not None:
        numeric.extend(diffs["per_stage_max_abs"])
    diffs["max_abs"] = max(numeric)

    diffs["pass"] = (
        diffs["max_abs"] < tol
        and ref_n == cand_n
        and not diffs["state_keys_missing_in_candidate"]
    )
    return diffs


# ---------------------------------------------------------------------------
# Loose comparator — for production-acceptance gates (atlas verify, benchmarks)
# rather than port-vs-legacy semantic equivalence.
#
# Tight `compare()` runs both impls from snapshot-restored state and demands
# bit-near-equal logits. That's right for catching port bugs, wrong for asking
# "does this artifact reproduce well enough to ship." Real inference does live
# EMA updates; CPU vs GPU, BLAS choice, and torch version all introduce drift
# that's irrelevant to whether the model is the same model.
#
# `compare_loose()` checks the things downstream actually depends on:
#   - top-k overlap on argmax tokens (semantic match without bit-equality)
#   - relative logit error in a sane range
#   - structural match on stage-call count
#   - per-stage hidden ranges within scale (catches "wrong layer norm" bugs
#     without flagging numerical noise)
# ---------------------------------------------------------------------------


def compare_loose(
    ref: dict,
    cap: Capture,
    *,
    logits_rel_tol: float = 1e-2,
    topk: int = 5,
    topk_overlap_min: float = 0.95,
    hidden_rel_tol: float = 5e-2,
) -> dict:
    """Production-acceptance comparator: structural + statistical, not bit-exact.

    Pass criteria:
      - per-token top-`topk` overlap >= `topk_overlap_min` (default 95%)
      - max relative logit error <= `logits_rel_tol` (default 1%)
      - stage-call count matches (architecture is the same shape)
      - per-stage hidden state max-abs drift <= `hidden_rel_tol * ref_scale`
    """
    diffs: dict = {}

    ref_logits = ref["logits"]
    cand_logits = cap.logits

    # Top-k overlap per token: fraction of positions where the top-k sets agree.
    rk = ref_logits.topk(topk, dim=-1).indices  # [B, T, K]
    ck = cand_logits.topk(topk, dim=-1).indices
    rk_set = rk.sort(dim=-1).values
    ck_set = ck.sort(dim=-1).values
    # For each [B, T], count overlap by broadcasting comparison.
    overlap = (rk_set.unsqueeze(-1) == ck_set.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)
    diffs[f"top{topk}_overlap_mean"] = float(overlap.mean())
    diffs[f"top{topk}_overlap_min"] = float(overlap.min())

    # Relative logit error on the magnitudes that actually drive sampling
    # (we softmax / temperature these — small absolute errors at large logit
    # values barely move probabilities; large relative errors do).
    abs_diff = (ref_logits - cand_logits).abs()
    scale = ref_logits.abs().clamp(min=1.0)
    diffs["logits_rel_err_max"] = float((abs_diff / scale).max())
    diffs["logits_abs_err_max"] = float(abs_diff.max())

    # Stage-call structural match.
    ref_n, cand_n = len(ref["stage_hiddens"]), len(cap.stage_hiddens)
    diffs["n_stage_calls_ref"] = ref_n
    diffs["n_stage_calls_cand"] = cand_n

    # Per-stage hidden drift, scaled by stage activation magnitude so a drift of
    # 0.01 on hiddens with mean-abs 10 reads as 0.1% rel, not "fail."
    if ref_n == cand_n:
        per_stage = []
        for r, c in zip(ref["stage_hiddens"], cap.stage_hiddens):
            scale_s = r.abs().mean().clamp(min=1e-3)
            rel = (r - c).abs().max() / scale_s
            per_stage.append(float(rel))
        diffs["per_stage_rel_err"] = per_stage
        max_stage_rel = max(per_stage)
    else:
        diffs["per_stage_rel_err"] = None
        max_stage_rel = float("inf")

    diffs["pass"] = (
        diffs[f"top{topk}_overlap_min"] >= topk_overlap_min
        and diffs["logits_rel_err_max"] <= logits_rel_tol
        and ref_n == cand_n
        and max_stage_rel <= hidden_rel_tol
    )
    return diffs
