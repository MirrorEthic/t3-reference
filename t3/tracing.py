"""Trace generation matching docs/TRACE_SCHEMA.md (schema v1).

Public API:

    from t3 import T3Model
    from t3.tracing import generate_trace, load_trace

    model = T3Model.from_checkpoint("path/to/best.pt")
    out_path = generate_trace(
        model,
        prompt="The capital of France is",
        prompt_id="factual",
        n_tokens=32,
        out_path="traces/v3.6-run3__best__factual.jsonl",
    )

    trace = load_trace(out_path)
    print(trace["meta"]["n_frames"])

The heavy lifting (TraceRecorder hooks, geometry snapshot, schema-conformant
serialization) lives in t3._legacy_trace. This module is the inference-time
glue that wires our T3Model into that recorder.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from t3._legacy_trace import (
    TraceRecorder,
    snapshot_stage_geometry,
)
from t3.model import T3Model

VOCAB_TO_TOKENIZER = {
    50257: "gpt2",
    50304: "gpt2",
    256000: "google/gemma-3-1b-it",
    262144: "google/gemma-3-1b-it",
    151936: "Qwen/Qwen2.5-0.5B",
    152064: "Qwen/Qwen2.5-0.5B",
}


def _capabilities_from_model(model: T3Model) -> dict:
    """Probe the loaded chain's state dict for which dynamics are present."""
    sd_keys = list(model.chain.state_dict().keys())
    cfg = model._legacy_config
    return {
        "has_coupling":   any("_coupling_params" in k for k in sd_keys),
        "has_trivectors": any("_trivector_params" in k for k in sd_keys),
        "has_dyn_omega":  any("_omega_shadow"     in k for k in sd_keys),
        "has_inter_stage_pc": any("inter_stage_predictor" in k for k in sd_keys),
        "has_scratchpad": any("scratchpad" in k for k in sd_keys),
        "n_primitives":   getattr(cfg, "n_primitives", 6),
        "null_cone_strength": float(getattr(cfg, "null_cone_strength", 0.0)),
        "hamiltonian_coupling": float(getattr(cfg, "hamiltonian_coupling", 0.0)),
        "sigma_hidden":   int(getattr(cfg, "sigma_hidden", 16)),
        "scratchpad_inject_entropy": list(
            getattr(cfg, "scratchpad_inject_entropy", []) or []
        ),
    }


def _capture_chain_state(chain, state_dict: Any, token_idx: int) -> dict | None:
    if not isinstance(state_dict, dict):
        return None
    entry: dict = {"token_idx": token_idx}
    for k in ("act_halt_probs", "act_strain_values", "act_ponder_steps",
              "act_per_stage_steps", "act_per_stage_strains", "act_ponder_cost"):
        v = state_dict.get(k)
        if v is None:
            continue
        if torch.is_tensor(v):
            entry[k] = v.detach().cpu().tolist()
        elif isinstance(v, list):
            entry[k] = [x.detach().cpu().tolist() if torch.is_tensor(x) else x for x in v]
        else:
            entry[k] = v
    for attr, jkey in (("_last_difficulty_pred", "difficulty_pred"),
                       ("_last_scratchpad_pred", "scratchpad_pred")):
        v = getattr(chain, attr, None)
        if v is None:
            continue
        try:
            arr = v.detach().cpu()
            if attr == "_last_scratchpad_pred" and arr.dim() == 2:
                entry[jkey] = arr[-1].tolist()
            else:
                entry[jkey] = arr.tolist()
        except Exception:
            pass
    for k in ("_omega_displacement_ema", "_omega_variance_ema"):
        v = getattr(chain, k, None)
        if v is None:
            continue
        try:
            entry[k.lstrip("_")] = float(v.item() if torch.is_tensor(v) else v)
        except Exception:
            pass
    return entry


def _autodetect_tokenizer(vocab_size: int) -> str:
    return VOCAB_TO_TOKENIZER.get(vocab_size, "gpt2")


def generate_trace(
    model: T3Model,
    prompt: str,
    prompt_id: str = "custom",
    n_tokens: int = 0,
    temperature: float = 0.8,
    tokenizer_name: str | None = None,
    lineage: str | None = None,
    ckpt_filename: str = "best.pt",
    out_path: str | Path | None = None,
) -> Path:
    """Run the model on `prompt` (optionally generating `n_tokens`) and write
    a schema-v1 JSONL trace.

    Returns the path to the written trace.
    """
    from transformers import AutoTokenizer

    chain = model.chain
    cfg = model._legacy_config
    device = next(chain.parameters()).device
    capabilities = _capabilities_from_model(model)

    # Force live ecology so EMAs update through the forward — otherwise primitives
    # stay frozen at checkpoint state and the trace has no per-step variation.
    if hasattr(cfg, "eval_live_primitives"):
        cfg.eval_live_primitives = True
        for s in chain.stages:
            if hasattr(s.cfg, "eval_live_primitives"):
                s.cfg.eval_live_primitives = True

    tok_name = tokenizer_name or _autodetect_tokenizer(cfg.vocab_size)
    tok = AutoTokenizer.from_pretrained(tok_name)
    ids = torch.tensor([tok.encode(prompt)], device=device)

    rec = TraceRecorder(tokenizer=tok)
    handles = [stage.register_forward_hook(rec.hook(i))
               for i, stage in enumerate(chain.stages)]

    chain.eval()
    stage_geoms = []
    chain_states: list[dict] = []
    with torch.no_grad():
        # warm forward to materialize positions/distances buffers
        _ = chain(ids[:, :2])
        for i, stage in enumerate(chain.stages):
            stage_geoms.append(snapshot_stage_geometry(stage, i))
        rec.frames.clear()
        rec._frame_idx = 0
        rec._stage_call_counts = {}

        if n_tokens > 0:
            cur = ids
            for t in range(n_tokens):
                out = chain(cur, return_state=True)
                logits = out[0] if isinstance(out, tuple) else out
                state = out[-1] if isinstance(out, tuple) and isinstance(out[-1], dict) else None
                if state is not None:
                    cs = _capture_chain_state(chain, state, t)
                    if cs:
                        chain_states.append(cs)
                next_logits = logits[:, -1, :] if logits.dim() == 3 else logits
                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)
                else:
                    next_tok = next_logits.argmax(dim=-1, keepdim=True)
                cur = torch.cat([cur, next_tok], dim=1)
        else:
            out = chain(ids, return_state=True)
            state = out[-1] if isinstance(out, tuple) and isinstance(out[-1], dict) else None
            if state:
                cs = _capture_chain_state(chain, state, 0)
                if cs:
                    chain_states.append(cs)

    for h in handles:
        h.remove()

    if out_path is None:
        out_dir = Path.cwd() / "traces"
        out_dir.mkdir(exist_ok=True)
        lin = lineage or "t3-124m-v36"
        out_path = out_dir / f"{lin}__{Path(ckpt_filename).stem}__{prompt_id}.jsonl"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    n_heads = cfg.n_heads
    d_head = getattr(cfg, "d_head", 0) or (cfg.d_model // n_heads)
    n_prim = capabilities["n_primitives"]
    meta = {
        "type": "meta",
        "n_stages": len(chain.stages),
        "n_heads": n_heads,
        "d_head": d_head,
        "n_layers_per_stage": [len(s.layers) for s in chain.stages],
        "primitive_names": ["E", "I", "F", "V", "C", "K"][:n_prim],
        "primitive_signature": [+1, +1, +1, -1, -1, -1][:n_prim],
        "prompt": prompt,
        "prompt_id": prompt_id,
        "lineage": lineage or "t3-124m-v36",
        "ckpt": ckpt_filename,
        "n_tokens": n_tokens,
        "n_frames": len(rec.frames),
        "n_chain_states": len(chain_states),
        "created": datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
        "capabilities": capabilities,
    }

    with open(out_path, "w") as f:
        f.write(json.dumps(meta) + "\n")
        for sg in stage_geoms:
            f.write(json.dumps(sg) + "\n")
        for cs in chain_states:
            f.write(json.dumps({"type": "chain_state", **cs}) + "\n")
        for fr in rec.frames:
            f.write(json.dumps(fr) + "\n")

    return out_path


def load_trace(path: str | Path) -> dict:
    """Load a schema-v1 trace into a typed dict.

    Returns: {"meta": ..., "geoms": {stage_idx: ...}, "chain_states": [...], "frames": [...]}
    """
    out: dict = {"meta": None, "geoms": {}, "chain_states": [], "frames": []}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            t = r["type"]
            if t == "meta":
                out["meta"] = r
            elif t == "stage_geom":
                out["geoms"][r["stage_idx"]] = r
            elif t == "chain_state":
                out["chain_states"].append(r)
            elif t == "frame":
                out["frames"].append(r)
    return out


def builtin_prompts() -> list[dict]:
    """Return the bundled prompt library (matches t3atlas trace sweeps)."""
    import importlib.resources as ir
    with ir.files("t3.data").joinpath("prompts.json").open() as f:
        return json.load(f)
