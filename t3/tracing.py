"""Trace generation matching `docs/TRACE_SCHEMA.md` (schema v1).

A T³ trace is a JSONL file capturing the live ecology state of a chain across
a forward pass (or autoregressive generation). One record per line. The trace
is consumable without instantiating the model: every quantity needed to
analyze the chain's behavior is materialized.

Public API:

    from t3 import T3Model
    from t3.tracing import generate_trace, load_trace, builtin_prompts

    model = T3Model.from_checkpoint("path/to/best.pt")
    out_path = generate_trace(
        model,
        prompt="The capital of France is",
        prompt_id="factual",
        n_tokens=32,
        out_path="traces/v3.6-run3__best__factual.jsonl",
    )
    trace = load_trace(out_path)

The recorder hooks each `T3Stage.forward` and snapshots the per-head
ecology primitives, sigma, dynamic blockade activity, and self-surprise on
every call. Under per-stage ACT, each stage is called multiple times per
token, so frames-per-token = sum of per-stage ponder steps.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

from t3.chain import T3Stage
from t3.model import T3Model

VOCAB_TO_TOKENIZER = {
    50257: "gpt2",
    50304: "gpt2",
    256000: "google/gemma-3-1b-it",
    262144: "google/gemma-3-1b-it",
    151936: "Qwen/Qwen2.5-0.5B",
    152064: "Qwen/Qwen2.5-0.5B",
}


# ---------------------------------------------------------------------------
# Capability + capture helpers
# ---------------------------------------------------------------------------


def _capabilities_from_model(model: T3Model) -> dict:
    """Probe the loaded chain's state dict for which dynamics are present.

    The schema's `meta.capabilities` is the honest record of what was vs
    wasn't trained — distinct from what the *code* supports. A consumer
    seeing `has_trivectors: false` should not try to render trivector data.
    """
    sd_keys = list(model.chain.state_dict().keys())
    cfg = model.config
    return {
        "has_coupling": any("_coupling_params" in k for k in sd_keys),
        "has_trivectors": any("_trivector_params" in k for k in sd_keys),
        "has_dyn_omega": any("_omega_shadow" in k for k in sd_keys),
        "has_inter_stage_pc": any("inter_stage_predictor" in k for k in sd_keys),
        "has_scratchpad": any("scratchpad" in k for k in sd_keys),
        "n_primitives": getattr(cfg, "n_primitives", 6),
        "null_cone_strength": float(getattr(cfg, "null_cone_strength", 0.0)),
        "hamiltonian_coupling": float(getattr(cfg, "hamiltonian_coupling", 0.0)),
        "sigma_hidden": int(getattr(cfg, "sigma_hidden", 16)),
        "scratchpad_inject_entropy": list(
            getattr(cfg, "scratchpad_inject_entropy", []) or []
        ),
    }


def _autodetect_tokenizer(vocab_size: int) -> str:
    return VOCAB_TO_TOKENIZER.get(vocab_size, "gpt2")


def _compute_blockade_kernel(
    distances: torch.Tensor, radius: float, exponent: float
) -> torch.Tensor:
    """Reconstruct the geometric blockade interaction kernel
    (`1 / (1 + (d/r₀)^exponent)`) with self-edge zeroed. Excludes the live
    cosurvival modulation; that's emitted separately."""
    H = distances.shape[0]
    bw = 1.0 / (1.0 + (distances / radius).pow(exponent))
    bw = bw * (1.0 - torch.eye(H, device=distances.device, dtype=distances.dtype))
    return bw


def _snapshot_stage_geometry(stage: T3Stage, stage_idx: int) -> dict:
    """Per-stage one-shot geometry: head positions, distances, blockade kernel,
    cosurvival matrix + derived signals. Static within a single forward."""
    hs = stage.head_state
    positions = hs.head_positions.detach().cpu().tolist()
    distances = hs.get_pairwise_distances().detach()
    radius = float(stage.layers[0].attention.blockade.blockade_radius)
    exponent = float(stage.layers[0].attention.blockade.exponent)
    bk = _compute_blockade_kernel(distances, radius, exponent).cpu().tolist()
    coupling_max = float(getattr(hs, "_coupling_max", 0.2))
    has_trivectors = hasattr(hs, "_trivector_params")

    cs_matrix = stage.cosurvival.cosurvival.detach().cpu().tolist()
    mod = stage.cosurvival.get_blockade_modulation()
    cs_modulation = mod.detach().cpu().tolist() if torch.is_tensor(mod) else None
    cs_head_loss_ema = stage.cosurvival.head_loss_ema.detach().cpu().tolist()
    ps = stage.cosurvival.get_protection_scores()
    cs_protection_scores = ps.detach().cpu().tolist() if torch.is_tensor(ps) else None

    return {
        "type": "stage_geom",
        "stage_idx": stage_idx,
        "head_positions": positions,
        "distances": distances.cpu().tolist(),
        "blockade_kernel": bk,
        "blockade_radius": radius,
        "blockade_exponent": exponent,
        "coupling_max": coupling_max,
        "has_trivectors": has_trivectors,
        "cosurvival_matrix": cs_matrix,
        "cosurvival_modulation": cs_modulation,
        "cosurvival_head_loss_ema": cs_head_loss_ema,
        "cosurvival_protection_scores": cs_protection_scores,
    }


def _snapshot_headstate(hs) -> dict:
    """Per-frame primitive snapshot. Six EMAs stacked into [n_heads, 6] plus
    the Cl(3,3) invariant Q per head and the bivector/trivector parameters."""
    prims = torch.stack(
        [
            hs._entropy_ema,
            hs._last_intensity,
            hs._friction_ema,
            hs._valence_ema,
            hs._coherence_ema,
            hs._chronos_ema,
        ],
        dim=-1,
    ).detach().cpu()
    omega = (
        hs._coupling_params.detach().cpu().tolist()
        if hasattr(hs, "_coupling_params")
        else []
    )
    tri = (
        hs._trivector_params.detach().cpu().tolist()
        if hasattr(hs, "_trivector_params")
        else []
    )
    Q = (prims[:, :3].pow(2).sum(-1) - prims[:, 3:].pow(2).sum(-1)).tolist()
    return {
        "primitives": prims.tolist(),
        "omega_flat": omega,
        "trivectors": tri,
        "Q": Q,
    }


# ---------------------------------------------------------------------------
# TraceRecorder — per-stage forward hook
# ---------------------------------------------------------------------------


class TraceRecorder:
    """Stage-level forward hook that emits one frame per stage call."""

    def __init__(self, tokenizer=None, top_k: int = 5):
        self.frames: list[dict] = []
        self.chain_states: list[dict] = []
        self._frame_idx = 0
        self._stage_call_counts: dict[int, int] = {}
        self.tokenizer = tokenizer
        self.top_k = top_k

    def hook(self, stage_idx: int):
        def _post(module: T3Stage, inputs, output):
            stage_state: dict = {}
            stage_logits = None
            if isinstance(output, (tuple, list)):
                if len(output) == 3:
                    _, stage_logits, stage_state = output
                elif len(output) == 2:
                    _, stage_state = output
            if not isinstance(stage_state, dict):
                stage_state = {}

            top_tokens = None
            if stage_logits is not None and self.tokenizer is not None:
                try:
                    last = stage_logits[0, -1, :] if stage_logits.dim() == 3 else stage_logits[-1]
                    probs = torch.softmax(last.float(), dim=-1)
                    top_p, top_i = probs.topk(self.top_k)
                    toks = [self.tokenizer.decode([int(i)]) for i in top_i.tolist()]
                    top_tokens = list(
                        zip(toks, [round(p, 4) for p in top_p.tolist()])
                    )
                except Exception:
                    pass

            sigma = []
            if "head_sigmas_tensor" in stage_state:
                sigma = stage_state["head_sigmas_tensor"].detach().cpu().tolist()

            snap = _snapshot_headstate(module.head_state)

            kb = []
            try:
                w = module.head_state.key_bias_proj.weight.detach().cpu()
                kb = w.norm(dim=0).tolist()
            except Exception:
                pass

            suppression = None
            per_layer_suppression = None
            try:
                accs = []
                for layer in module.layers:
                    s = getattr(layer.attention.blockade, "_last_suppression", None)
                    if s is not None:
                        accs.append(s.detach().cpu())
                if accs:
                    suppression = torch.stack(accs).mean(dim=0).tolist()
                    per_layer_suppression = [a.tolist() for a in accs]
            except Exception:
                pass

            self_surprise = None
            try:
                ss = getattr(module.head_state, "_self_surprise", None)
                if ss is not None:
                    self_surprise = ss.detach().cpu().tolist()
            except Exception:
                pass

            output_entropy_ema = None
            try:
                tracker = getattr(module, "entropy_tracker", None)
                if tracker is not None:
                    val = getattr(tracker, "output_entropy_ema", None)
                    if val is not None:
                        output_entropy_ema = float(
                            val.item() if torch.is_tensor(val) else val
                        )
            except Exception:
                pass

            per_layer_attn_entropy = None
            try:
                layer_ents = []
                for layer in module.layers:
                    he = getattr(layer.attention, "_head_entropy", None)
                    if he is not None:
                        layer_ents.append(he.detach().cpu().tolist())
                if layer_ents:
                    per_layer_attn_entropy = layer_ents
            except Exception:
                pass

            call_k = self._stage_call_counts.get(stage_idx, 0)
            self._stage_call_counts[stage_idx] = call_k + 1
            self.frames.append(
                {
                    "type": "frame",
                    "frame_idx": self._frame_idx,
                    "stage_idx": stage_idx,
                    "act_call": call_k,
                    "sigma": sigma,
                    **snap,
                    "kb_input_norms": kb,
                    "suppression": suppression,
                    "per_layer_suppression": per_layer_suppression,
                    "self_surprise": self_surprise,
                    "output_entropy_ema": output_entropy_ema,
                    "per_layer_attn_entropy": per_layer_attn_entropy,
                    "stage_top_tokens": top_tokens,
                }
            )
            self._frame_idx += 1

        return _post


# ---------------------------------------------------------------------------
# Chain-state capture (per-token ACT signals)
# ---------------------------------------------------------------------------


def _capture_chain_state(chain, state_dict: Any, token_idx: int) -> Optional[dict]:
    if not isinstance(state_dict, dict):
        return None
    entry: dict = {"token_idx": token_idx}
    for k in (
        "act_halt_probs",
        "act_strain_values",
        "act_ponder_steps",
        "act_per_stage_steps",
        "act_per_stage_strains",
        "act_ponder_cost",
    ):
        v = state_dict.get(k)
        if v is None:
            continue
        if torch.is_tensor(v):
            entry[k] = v.detach().cpu().tolist()
        elif isinstance(v, list):
            entry[k] = [
                x.detach().cpu().tolist() if torch.is_tensor(x) else x for x in v
            ]
        else:
            entry[k] = v

    for attr, jkey in (
        ("_last_difficulty_pred", "difficulty_pred"),
        ("_last_scratchpad_pred", "scratchpad_pred"),
    ):
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


# ---------------------------------------------------------------------------
# Public: generate_trace, load_trace, builtin_prompts, update_manifest
# ---------------------------------------------------------------------------


def generate_trace(
    model: T3Model,
    prompt: str,
    prompt_id: str = "custom",
    n_tokens: int = 0,
    temperature: float = 0.8,
    tokenizer_name: Optional[str] = None,
    lineage: Optional[str] = None,
    ckpt_filename: str = "best.pt",
    out_path: Optional[Any] = None,
) -> Path:
    """Run a forward (and optionally `n_tokens` of generation) and write a
    schema-v1 JSONL trace.

    Forces `eval_live_primitives=True` on the model's config so the per-stage
    ecology EMAs update through the forward — without that, primitives stay
    frozen at checkpoint values and the trace has no per-step variation.
    Returns the path to the written trace.
    """
    from transformers import AutoTokenizer

    chain = model.chain
    cfg = model.config
    device = next(chain.parameters()).device
    capabilities = _capabilities_from_model(model)

    # Live-ecology toggle for non-trivial trace dynamics.
    object.__setattr__(cfg, "eval_live_primitives", True)
    for s in chain.stages:
        object.__setattr__(s.cfg, "eval_live_primitives", True)

    tok_name = tokenizer_name or _autodetect_tokenizer(cfg.vocab_size)
    tok = AutoTokenizer.from_pretrained(tok_name)
    ids = torch.tensor([tok.encode(prompt)], device=device)

    rec = TraceRecorder(tokenizer=tok)
    handles = [
        stage.register_forward_hook(rec.hook(i)) for i, stage in enumerate(chain.stages)
    ]

    chain.eval()
    stage_geoms: list[dict] = []
    chain_states: list[dict] = []
    try:
        with torch.no_grad():
            # Warm forward to materialize position/distance buffers.
            _ = chain(ids[:, :2])
            for i, stage in enumerate(chain.stages):
                stage_geoms.append(_snapshot_stage_geometry(stage, i))
            rec.frames.clear()
            rec._frame_idx = 0
            rec._stage_call_counts = {}

            if n_tokens > 0:
                cur = ids
                for t in range(n_tokens):
                    out = chain(cur, return_state=True)
                    logits = out[0] if isinstance(out, tuple) else out
                    state = (
                        out[-1]
                        if isinstance(out, tuple) and isinstance(out[-1], dict)
                        else None
                    )
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
                state = (
                    out[-1]
                    if isinstance(out, tuple) and isinstance(out[-1], dict)
                    else None
                )
                if state:
                    cs = _capture_chain_state(chain, state, 0)
                    if cs:
                        chain_states.append(cs)
    finally:
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


def load_trace(path: Any) -> dict:
    """Load a schema-v1 trace into a typed dict.

    Returns: `{"meta": ..., "geoms": {stage_idx: ...}, "chain_states": [...], "frames": [...]}`.
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


def update_manifest(out_path: Path, meta: dict, traces_dir: Optional[Path] = None) -> None:
    """Append/refresh this trace's entry in `<traces_dir>/index.json`. Used by
    sweep drivers to keep a manifest of generated traces."""
    if traces_dir is None:
        traces_dir = out_path.parent
    traces_dir.mkdir(exist_ok=True)
    manifest_path = traces_dir / "index.json"
    existing: list = []
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except Exception:
            existing = []
    rel = out_path.name
    entry = {
        "file": rel,
        "lineage": meta.get("lineage"),
        "ckpt": meta.get("ckpt"),
        "prompt_id": meta.get("prompt_id"),
        "prompt": meta.get("prompt"),
        "n_stages": meta.get("n_stages"),
        "n_heads": meta.get("n_heads"),
        "n_frames": meta.get("n_frames"),
        "n_tokens": meta.get("n_tokens", 0),
        "created": meta.get("created"),
        "capabilities": meta.get("capabilities", {}),
    }
    existing = [e for e in existing if e.get("file") != rel]
    existing.append(entry)
    existing.sort(key=lambda e: e.get("created") or "", reverse=True)
    manifest_path.write_text(json.dumps(existing, indent=2))
