"""End-to-end tracing: generate a JSONL from run-3 and verify schema v1 conformance."""

from pathlib import Path

import pytest
import torch

from t3 import T3Model
from t3.tracing import generate_trace, load_trace, builtin_prompts

RUN3 = Path(
    "/home/garret-sutherland/CVMP/T3_sims/t3v2/t3v3/t3v36/"
    "checkpoints_v36_run3/best.pt"
)


@pytest.fixture(scope="module")
def model():
    if not RUN3.exists():
        pytest.skip(f"Run-3 checkpoint not present at {RUN3}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = T3Model.from_checkpoint(RUN3, map_location=device)
    m.to(device)
    m.eval()
    return m


def test_builtin_prompts_load():
    prompts = builtin_prompts()
    assert isinstance(prompts, list)
    assert len(prompts) > 0
    sample = prompts[0]
    assert "id" in sample or "prompt_id" in sample or "text" in sample or "prompt" in sample, \
        f"unexpected prompt schema: {sample.keys()}"


def test_generate_and_load_single_forward(model, tmp_path):
    out = generate_trace(
        model,
        prompt="The capital of France is",
        prompt_id="factual_smoke",
        n_tokens=0,
        lineage="t3-124m-v36",
        out_path=tmp_path / "smoke_single.jsonl",
    )
    trace = load_trace(out)

    meta = trace["meta"]
    assert meta["type"] == "meta"
    assert meta["n_stages"] == 3
    assert meta["n_heads"] == 12
    assert meta["n_layers_per_stage"] == [4, 3, 5]
    assert meta["primitive_names"] == ["E", "I", "F", "V", "C", "K"]
    assert meta["primitive_signature"] == [1, 1, 1, -1, -1, -1]
    assert meta["lineage"] == "t3-124m-v36"
    assert meta["prompt_id"] == "factual_smoke"
    assert meta["n_tokens"] == 0
    cap = meta["capabilities"]
    assert cap["n_primitives"] == 6
    assert cap["has_inter_stage_pc"] is True
    assert cap["has_trivectors"] is False  # run-3 was trained without trivectors
    assert cap["has_coupling"] is True

    assert len(trace["geoms"]) == 3
    geom0 = trace["geoms"][0]
    assert len(geom0["head_positions"]) == 12
    assert len(geom0["head_positions"][0]) == 3   # 3-torus
    assert len(geom0["distances"]) == 12
    assert len(geom0["blockade_kernel"]) == 12

    assert len(trace["frames"]) > 0
    fr = trace["frames"][0]
    assert fr["type"] == "frame"
    assert "stage_idx" in fr
    assert len(fr["primitives"]) == 12  # n_heads
    assert len(fr["primitives"][0]) == 6  # n_primitives
    assert len(fr["sigma"]) == 12
    assert len(fr["Q"]) == 12

    print(f"\n  frames: {len(trace['frames'])}  chain_states: {len(trace['chain_states'])}")
    print(f"  capabilities: {cap}")


def test_generate_with_short_generation(model, tmp_path):
    out = generate_trace(
        model,
        prompt="The cat sat on the",
        prompt_id="cat_smoke",
        n_tokens=4,
        temperature=0.8,
        out_path=tmp_path / "smoke_gen.jsonl",
    )
    trace = load_trace(out)
    assert trace["meta"]["n_tokens"] == 4
    assert len(trace["chain_states"]) == 4  # one per generated token
    # sanity: ACT pondered at least once for each token
    for cs in trace["chain_states"]:
        assert cs["token_idx"] >= 0
        assert "act_ponder_steps" in cs
