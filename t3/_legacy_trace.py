"""Vendored from t3v36/diagnostic_trace/export_trace.py — inference-time tracer.

Emits JSONL conforming to t3.docs.TRACE_SCHEMA. The clean public API is
in `t3.tracing`; this file holds the proven implementation. Do not edit.
"""

"""
T3 v3.6 forward-pass ecology trace exporter.

Wraps each T3v3Stage with a forward hook that snapshots HeadState ecology
after every call. Because ACT calls each stage N times per forward, each
hook firing = one animation frame.

Output: JSONL, one record per line. First line is meta; rest are frames.

Schema:
  {"type":"meta", "n_stages", "n_heads", "d_head", "primitive_names":["E","I","F","V","C","K"],
   "blockade":{"radius":[per-stage], "exponent":int}}
  {"type":"stage_geom", "stage_idx", "head_positions":[[H][3]],
   "distances":[[H][H]], "blockade_kernel":[[H][H]]}   # one per stage, emitted before frames
  {"type":"frame", "frame_idx", "stage_idx", "act_call":k,
   "primitives":[[H][6]],  # rows = heads, cols = E,I,F,V,C,K
   "sigma":[H],
   "omega_flat":[15],      # so(6) bivector (upper triangle)
   "trivectors":[20],
   "Q":[H],                # Cl(3,3) invariant per head: E^2+I^2+F^2-V^2-C^2-K^2
   "suppression":[H],      # _last_suppression: dynamic per-head blockade activity
   "kb_input_norms":[in_features]}

Usage:
  python export_trace.py --ckpt path/to/step7500.pt --out trace.jsonl --prompt "Hello world"
"""
import argparse, json, os, sys, math
from pathlib import Path
import torch

THIS = Path(__file__).resolve()

from t3._legacy_model import HeadState, T3v3Config  # noqa
from t3._legacy_chain import T3v3Chain, T3v3Stage, T3v3ChainConfig  # noqa
from dataclasses import fields as _dc_fields


def compute_blockade_kernel(distances: torch.Tensor, radius: float, exponent: float) -> torch.Tensor:
    """Reconstruct the [H,H] blockade interaction kernel exactly as RydbergBlockade does.

    Matches t3v3_model.py:1610: 1 / (1 + (d/r0)^exp), with self-edge zeroed.
    Excludes co-survival modulation (which varies through training but is roughly
    static within a single forward pass — capture separately if needed).
    """
    H = distances.shape[0]
    bw = 1.0 / (1.0 + (distances / radius).pow(exponent))
    bw = bw * (1.0 - torch.eye(H, device=distances.device, dtype=distances.dtype))
    return bw


def snapshot_stage_geometry(stage, stage_idx: int):
    """One-time per-stage capture: head positions + pairwise distances + blockade kernel
    + initial cosurvival matrix snapshot (it changes slowly; the static snapshot is the
    baseline, with frame-by-frame override emitted when it does evolve)."""
    hs = stage.head_state
    positions = hs.head_positions.detach().cpu().tolist()
    distances = hs.get_pairwise_distances().detach()
    radius = float(stage.layers[0].attention.blockade.blockade_radius)
    exponent = float(stage.layers[0].attention.blockade.exponent)
    bk = compute_blockade_kernel(distances, radius, exponent).cpu().tolist()
    coupling_max = float(getattr(hs, "_coupling_max", 0.2))
    has_trivectors = hasattr(hs, "_trivector_params")

    # Cosurvival snapshot — [H,H] + per-head [H] signals.
    cs_matrix = None
    cs_modulation = None
    cs_head_loss_ema = None
    cs_protection_scores = None
    try:
        cs_matrix = stage.cosurvival.cosurvival.detach().cpu().tolist()
    except Exception:
        pass
    try:
        mod = stage.cosurvival.get_blockade_modulation()
        if torch.is_tensor(mod):
            cs_modulation = mod.detach().cpu().tolist()
    except Exception:
        pass
    try:
        hle = stage.cosurvival.head_loss_ema.detach().cpu().tolist()
        cs_head_loss_ema = hle
    except Exception:
        pass
    try:
        # protection_scores = clamp_min(cosurvival, 0).sum(dim=-1)  per-head bond strength
        ps = stage.cosurvival.get_protection_scores()
        if torch.is_tensor(ps):
            cs_protection_scores = ps.detach().cpu().tolist()
    except Exception:
        pass

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


def snapshot_headstate(hs: HeadState):
    """Pull the six primitive EMAs + Omega + trivectors off a HeadState."""
    prims = torch.stack([
        hs._entropy_ema,    # E
        hs._last_intensity, # I
        hs._friction_ema,   # F
        hs._valence_ema,    # V
        hs._coherence_ema,  # C
        hs._chronos_ema,    # K
    ], dim=-1).detach().cpu()  # [H, 6]
    omega = hs._coupling_params.detach().cpu().tolist() if hasattr(hs, "_coupling_params") else []
    tri = hs._trivector_params.detach().cpu().tolist() if hasattr(hs, "_trivector_params") else []
    # Cl(3,3) invariant Q = E^2+I^2+F^2 - V^2-C^2-K^2 per head
    Q = (prims[:, :3].pow(2).sum(-1) - prims[:, 3:].pow(2).sum(-1)).tolist()
    return {
        "primitives": prims.tolist(),
        "omega_flat": omega,
        "trivectors": tri,
        "Q": Q,
    }


class TraceRecorder:
    def __init__(self, tokenizer=None, top_k=5):
        self.frames = []
        self.chain_states = []
        self._frame_idx = 0
        self._stage_call_counts = {}
        self.tokenizer = tokenizer
        self.top_k = top_k

    def hook(self, stage_idx: int):
        def _post(module: T3v3Stage, inputs, output):
            stage_state = {}
            stage_logits = None
            if isinstance(output, (tuple, list)):
                if len(output) == 3:
                    _, stage_logits, stage_state = output
                elif len(output) == 2:
                    _, stage_state = output
            if not isinstance(stage_state, dict):
                stage_state = {}

            # Per-stage logit top-K from the LAST token position. Tells us what
            # this stage thinks the next token is. Reveals stage-by-stage commitment.
            top_tokens = None
            if stage_logits is not None and self.tokenizer is not None:
                try:
                    last = stage_logits[0, -1, :] if stage_logits.dim() == 3 else stage_logits[-1]
                    probs = torch.softmax(last.float(), dim=-1)
                    top_p, top_i = probs.topk(self.top_k)
                    toks = [self.tokenizer.decode([int(i)]) for i in top_i.tolist()]
                    top_tokens = list(zip(toks, [round(p, 4) for p in top_p.tolist()]))
                except Exception:
                    pass
            sigma = []
            if isinstance(stage_state, dict) and "head_sigmas_tensor" in stage_state:
                sigma = stage_state["head_sigmas_tensor"].detach().cpu().tolist()

            snap = snapshot_headstate(module.head_state)
            kb = []
            try:
                w = module.head_state.key_bias_proj.weight.detach().cpu()  # [d_head, in_features]
                kb = w.norm(dim=0).tolist()
            except Exception:
                pass

            # Per-head dynamic suppression: aggregate _last_suppression across this stage's
            # attention layers (chain doesn't expose a per-stage aggregate after the fact).
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

            # Self-surprise per head: the WorldTrace prediction-error signal.
            self_surprise = None
            try:
                ss = getattr(module.head_state, "_self_surprise", None)
                if ss is not None:
                    self_surprise = ss.detach().cpu().tolist()
            except Exception:
                pass

            # Per-stage output entropy EMA — the signal that grounds blended E.
            output_entropy_ema = None
            try:
                tracker = getattr(module, "entropy_tracker", None)
                if tracker is not None:
                    val = getattr(tracker, "output_entropy_ema", None)
                    if val is not None:
                        output_entropy_ema = float(val.item() if torch.is_tensor(val) else val)
            except Exception:
                pass

            # Per-layer attention entropy [n_layers x [H]] — finer than aggregated E.
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
            self.frames.append({
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
            })
            self._frame_idx += 1
        return _post


def load_chain(ckpt_path: str, device="cuda" if torch.cuda.is_available() else "cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_obj = ck.get("cfg") or ck.get("config")
    if cfg_obj is None:
        raise SystemExit("checkpoint has no cfg/config")
    if isinstance(cfg_obj, dict):
        valid = {f.name for f in _dc_fields(T3v3ChainConfig)}
        kept = {k: v for k, v in cfg_obj.items() if k in valid}
        dropped = sorted(set(cfg_obj.keys()) - valid)
        if dropped:
            print(f"cfg: dropped {len(dropped)} unknown keys (e.g. {dropped[:5]})")
        cfg = T3v3ChainConfig(**kept)
    else:
        cfg = cfg_obj
    chain = T3v3Chain(cfg).to(device)
    sd_raw = (ck.get("model_state") or ck.get("model_state_dict")
              or ck.get("state_dict") or ck)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd_raw.items()}
    missing, unexpected = chain.load_state_dict(sd, strict=False)
    print(f"loaded: ckpt keys={len(sd)} missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 50:
        print(f"  warning: many missing — first 5: {missing[:5]}")
    if len(unexpected) > 0:
        print(f"  unexpected: {unexpected[:5]}")
    # Restore ecology buffers from checkpoint if present (EMAs etc.)
    eco = ck.get("ecology_state")
    if eco and isinstance(eco, dict):
        loaded = 0
        for k, v in eco.items():
            try:
                obj = chain
                parts = k.split(".")
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                buf = getattr(obj, parts[-1], None)
                if buf is not None and torch.is_tensor(buf) and buf.shape == v.shape:
                    buf.data.copy_(v.to(buf.device))
                    loaded += 1
            except Exception:
                pass
        print(f"  ecology_state: restored {loaded}/{len(eco)} buffers")
    chain.eval()
    # Capability probe: derive what the viewer can/can't visualize from the
    # *raw* state dict keys. Drives whether we emit Ω triad, trivectors, etc.
    capabilities = {
        "has_coupling":   any("_coupling_params"  in k for k in sd_raw),
        "has_trivectors": any("_trivector_params" in k for k in sd_raw),
        "has_dyn_omega":  any("_omega_shadow"     in k for k in sd_raw),
        "has_inter_stage_pc": any("inter_stage_predictor" in k for k in sd_raw),
        "has_scratchpad": any("scratchpad" in k for k in sd_raw),
        "n_primitives":   getattr(cfg, "n_primitives", 6),
        "null_cone_strength": float(getattr(cfg, "null_cone_strength", 0.0)),
        "hamiltonian_coupling": float(getattr(cfg, "hamiltonian_coupling", 0.0)),
        "sigma_hidden":   int(getattr(cfg, "sigma_hidden", 16)),
        "scratchpad_inject_entropy":
            list(getattr(cfg, "scratchpad_inject_entropy", []) or []),
    }
    return chain, cfg, device, capabilities


TRACES_DIR = Path.cwd() / "traces"  # caller may override


def update_manifest(out_path: Path, meta: dict):
    """Append/refresh the entry for this trace in traces/index.json."""
    TRACES_DIR.mkdir(exist_ok=True)
    manifest_path = TRACES_DIR / "index.json"
    existing = []
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None,
                    help="Output path. Default: traces/<ckpt-stem>_<timestamp>[_t<N>].jsonl")
    ap.add_argument("--prompt", default="The toroidal tesseract transformer encodes")
    ap.add_argument("--tokenizer", default=None,
                    help="HuggingFace tokenizer name. Default: auto-detect from cfg.vocab_size "
                         "(50257→gpt2, 256000→google/gemma-3-1b-it, 151936→Qwen/Qwen2.5-0.5B).")
    ap.add_argument("--n_tokens", type=int, default=0,
                    help="If >0, run autoregressive generation for this many tokens "
                         "(one full chain pass per token = many frames). If 0, single forward.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--prompt_id", default=None,
                    help="Tag for the prompt (e.g. 'factual', 'banana'). Used in filename + manifest.")
    ap.add_argument("--lineage", default=None,
                    help="Lineage label override. Defaults to ckpt parent dir name minus 'checkpoints_'.")
    args = ap.parse_args()

    chain, cfg, device, capabilities = load_chain(args.ckpt)
    print(f"capabilities: {capabilities}")
    # Force live ecology so EMAs update through the forward — otherwise primitives
    # stay frozen at checkpoint state and the trace has no per-step variation.
    if hasattr(cfg, "eval_live_primitives"):
        cfg.eval_live_primitives = True
        for s in chain.stages:
            if hasattr(s.cfg, "eval_live_primitives"):
                s.cfg.eval_live_primitives = True
        print("eval_live_primitives = True (forced for trace)")

    from transformers import AutoTokenizer
    # Auto-detect tokenizer from vocab_size if not specified
    tokenizer_name = args.tokenizer
    if tokenizer_name is None:
        vocab = getattr(cfg, "vocab_size", 50257)
        VOCAB_TO_TOKENIZER = {
            50257:  "gpt2",
            50304:  "gpt2",                       # padded gpt2 variant
            256000: "google/gemma-3-1b-it",
            262144: "google/gemma-3-1b-it",
            151936: "Qwen/Qwen2.5-0.5B",
            152064: "Qwen/Qwen2.5-0.5B",
        }
        tokenizer_name = VOCAB_TO_TOKENIZER.get(vocab, "gpt2")
        print(f"tokenizer auto-detected from vocab_size={vocab}: {tokenizer_name}")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    ids = torch.tensor([tok.encode(args.prompt)], device=device)

    rec = TraceRecorder(tokenizer=tok)
    handles = []
    for i, stage in enumerate(chain.stages):
        handles.append(stage.register_forward_hook(rec.hook(i)))

    # capture geometry once (positions are fixed; blockade radius is calibrated at init)
    stage_geoms = []
    with torch.no_grad():
        # touch one tiny forward to ensure positions/distances are materialized
        _ = chain(ids[:, :2])
        for i, stage in enumerate(chain.stages):
            stage_geoms.append(snapshot_stage_geometry(stage, i))
        rec.frames.clear()
        rec._frame_idx = 0
        rec._stage_call_counts = {}

        def capture_chain_state(state_dict, token_idx):
            """Pull chain-level signals from the forward output's state dict."""
            if not isinstance(state_dict, dict):
                return None
            entry = {"token_idx": token_idx}
            # ACT signals
            for k in ("act_halt_probs", "act_strain_values", "act_ponder_steps",
                      "act_per_stage_steps", "act_per_stage_strains",
                      "act_ponder_cost"):
                v = state_dict.get(k)
                if v is None:
                    continue
                if torch.is_tensor(v):
                    entry[k] = v.detach().cpu().tolist()
                elif isinstance(v, list):
                    entry[k] = [x.detach().cpu().tolist() if torch.is_tensor(x) else x for x in v]
                else:
                    entry[k] = v
            # Difficulty + scratchpad predictors (cached on chain object)
            try:
                dp = getattr(chain, "_last_difficulty_pred", None)
                if dp is not None:
                    entry["difficulty_pred"] = dp.detach().cpu().tolist()
            except Exception:
                pass
            try:
                sp = getattr(chain, "_last_scratchpad_pred", None)
                if sp is not None:
                    arr = sp.detach().cpu()
                    if arr.dim() == 2:
                        entry["scratchpad_pred"] = arr[-1].tolist()
                    else:
                        entry["scratchpad_pred"] = arr.tolist()
            except Exception:
                pass
            # v3.7+ dynamic Ω drift signals — chain-level scalars
            for k in ("_omega_displacement_ema", "_omega_variance_ema"):
                try:
                    v = getattr(chain, k, None)
                    if v is not None:
                        entry[k.lstrip("_")] = float(v.item() if torch.is_tensor(v) else v)
                except Exception:
                    pass
            return entry

        if args.n_tokens > 0:
            print(f"generating {args.n_tokens} tokens (one chain pass each)...")
            cur = ids
            for t in range(args.n_tokens):
                out = chain(cur, return_state=True) if "return_state" in chain.forward.__code__.co_varnames else chain(cur)
                if isinstance(out, (tuple, list)):
                    logits = out[0]
                    state_dict = out[-1] if isinstance(out[-1], dict) else None
                else:
                    logits = out
                    state_dict = None
                if state_dict is not None:
                    cs = capture_chain_state(state_dict, t)
                    if cs:
                        rec.chain_states.append(cs)
                if logits.dim() == 3:
                    next_logits = logits[:, -1, :]
                else:
                    next_logits = logits
                if args.temperature > 0:
                    probs = torch.softmax(next_logits / args.temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)
                else:
                    next_tok = next_logits.argmax(dim=-1, keepdim=True)
                cur = torch.cat([cur, next_tok], dim=1)
                if (t + 1) % 8 == 0:
                    print(f"  step {t+1}/{args.n_tokens}: frames={len(rec.frames)} chain_states={len(rec.chain_states)}")
        else:
            out = chain(ids, return_state=True) if "return_state" in chain.forward.__code__.co_varnames else chain(ids)
            if isinstance(out, (tuple, list)):
                state_dict = out[-1] if isinstance(out[-1], dict) else None
                if state_dict:
                    cs = capture_chain_state(state_dict, 0)
                    if cs:
                        rec.chain_states.append(cs)

    for h in handles:
        h.remove()

    n_heads = cfg.n_heads
    d_head = getattr(cfg, "d_head", cfg.d_model // n_heads)
    from datetime import datetime
    created = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # Lineage = parent dir name minus 'checkpoints_' prefix, unless overridden
    lineage = args.lineage or Path(args.ckpt).parent.name.replace("checkpoints_", "")
    pid = args.prompt_id or "default"
    ckpt_stem = Path(args.ckpt).stem
    if args.out is None:
        TRACES_DIR.mkdir(exist_ok=True)
        out_path = TRACES_DIR / f"{lineage}__{ckpt_stem}__{pid}.jsonl"
    else:
        out_path = Path(args.out)
    meta = {
        "type": "meta",
        "n_stages": len(chain.stages),
        "n_heads": n_heads,
        "d_head": d_head,
        "n_layers_per_stage": [len(s.layers) for s in chain.stages],
        "primitive_names": ["E", "I", "F", "V", "C", "K"][:capabilities["n_primitives"]],
        "primitive_signature": [+1, +1, +1, -1, -1, -1][:capabilities["n_primitives"]],
        "prompt": args.prompt,
        "prompt_id": pid,
        "lineage": lineage,
        "ckpt": ckpt_stem + ".pt",
        "n_tokens": args.n_tokens,
        "n_frames": len(rec.frames),
        "created": created,
        "capabilities": capabilities,
    }
    # Update meta n_frames in case ACT skipped — also include chain_states count
    meta["n_chain_states"] = len(rec.chain_states)
    with open(out_path, "w") as f:
        f.write(json.dumps(meta) + "\n")
        for sg in stage_geoms:
            f.write(json.dumps(sg) + "\n")
        for cs in rec.chain_states:
            f.write(json.dumps({"type": "chain_state", **cs}) + "\n")
        for fr in rec.frames:
            f.write(json.dumps(fr) + "\n")
    print(f"wrote {len(rec.frames)} frames + meta → {out_path}")
    # Update manifest so the viewer dropdown picks this up automatically
    if out_path.parent.resolve() == TRACES_DIR.resolve():
        update_manifest(out_path, meta)
        print(f"updated {TRACES_DIR / 'index.json'}")


if __name__ == "__main__":
    main()
