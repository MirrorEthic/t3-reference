"""Vendored from t3v36/t3v3_lm_eval.py — lm-eval-harness wrapper.

T3v3LM extends lm_eval.api.LM with ACT-aware loglikelihood scoring and
per-stage ponder-depth instrumentation. The clean public API is in
t3.lm_eval; this file holds the proven implementation.

The eval_bos_continuation_bug fix (add_special_tokens=False on the
continuation, line 378) is preserved here. Without it, BoolQ pins to
~0.22 because BOS gets double-counted in the continuation logprob.
"""

#!/usr/bin/env python3
"""
T³ v3 wrapper for EleutherAI lm-evaluation-harness
====================================================

Wraps T3v3Chain in the lm_eval LM interface for standard benchmarks.
Adapted from t3_lm_eval.py (v2) with v3-native imports.

Usage:
  python t3v3_lm_eval.py --checkpoint checkpoints_gpt2_t3_act/best.pt \
                          --tasks boolq,arc_challenge,arc_easy,piqa,hellaswag,winogrande,copa,rte \
                          --batch_size 16
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import argparse
import math
import re
from typing import List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t3._legacy_model import T3v3Config, migrate_sigma_projections
from t3._legacy_chain import T3v3Chain, T3v3ChainConfig

from collections import defaultdict
from transformers import AutoTokenizer
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
import lm_eval


def strip_compiled_prefix(state_dict):
    """Strip _orig_mod. prefix from torch.compile'd model state dicts."""
    cleaned = {}
    for k, v in state_dict.items():
        new_k = re.sub(r'_orig_mod\.', '', k)
        cleaned[new_k] = v
    return cleaned


def build_v3_config(ckpt):
    """Build T3v3ChainConfig from checkpoint config dict + state dict inference.

    Strategy: pass through ALL checkpoint config fields that are valid
    T3v3ChainConfig fields, then override structural params from state dict
    inference (more reliable) and eval-specific settings.
    """
    import dataclasses

    model_state = strip_compiled_prefix(ckpt['model_state'])
    cfg = ckpt.get('config', {})
    if not isinstance(cfg, dict):
        cfg = {}

    # --- Phase 1: Start from checkpoint config, filtered to valid fields ---
    valid_fields = {f.name for f in dataclasses.fields(T3v3ChainConfig)}
    kwargs = {k: v for k, v in cfg.items() if k in valid_fields}

    # --- Phase 2: Infer structural params from state dict (more reliable) ---

    # vocab_size, d_model from embedding
    for k, v in model_state.items():
        if 'embed' in k and 'weight' in k and v.dim() == 2 and 'norm' not in k and 'pos' not in k:
            kwargs['vocab_size'], kwargs['d_model'] = v.shape
            break

    # max_seq_len from positional embedding (if present)
    for k, v in model_state.items():
        if 'pos_embed' in k and 'weight' in k and v.dim() == 2:
            kwargs['max_seq_len'] = v.shape[0]
            break

    # Stages + layers from state dict keys
    stage_ids = set()
    layer_ids = defaultdict(set)
    for k in model_state.keys():
        if 'stages.' in k:
            parts = k.split('.')
            si = int(parts[parts.index('stages') + 1])
            stage_ids.add(si)
            if 'layers.' in k:
                li = int(parts[parts.index('layers') + 1])
                layer_ids[si].add(li)
    if stage_ids:
        kwargs['n_stages'] = max(stage_ids) + 1
    if layer_ids:
        kwargs['n_layers'] = max(len(v) for v in layer_ids.values())
        # Infer layers_per_stage from actual layer counts
        inferred_lps = [len(layer_ids[s]) for s in range(kwargs['n_stages'])]
        if len(set(inferred_lps)) > 1:  # variable allocation
            kwargs['layers_per_stage'] = inferred_lps

    # n_heads from sigma_projections
    sigma_ids = set()
    for k in model_state.keys():
        if 'sigma_projections.' in k:
            parts = k.split('.')
            idx = parts.index('sigma_projections')
            sigma_ids.add(int(parts[idx + 1]))
    if sigma_ids:
        kwargs['n_heads'] = len(sigma_ids)

    # d_ff from FFN weights
    d_model = kwargs.get('d_model', 768)
    if 'd_ff' not in kwargs:
        kwargs['d_ff'] = d_model * 4
        for k, v in model_state.items():
            if 'ffn' in k and 'weight' in k and v.dim() == 2:
                if v.shape[0] > d_model:
                    kwargs['d_ff'] = v.shape[0]
                    break

    # n_kv_heads from K projection shape
    n_heads = kwargs.get('n_heads', 12)
    if not kwargs.get('n_kv_heads'):
        for k, v in model_state.items():
            if 'k_proj.weight' in k and v.dim() == 2:
                kv_dim = v.shape[0]
                head_dim = d_model // n_heads if n_heads else 64
                inferred_kv = kv_dim // head_dim
                if inferred_kv != n_heads:
                    kwargs['n_kv_heads'] = inferred_kv
                break

    # Grounded primitives from buffer presence
    kwargs['grounded_primitives'] = any('_entropy_ema' in k for k in model_state.keys())

    # eco_key_bias: only enable if checkpoint actually has key_bias_proj weights
    # (prevents random-init contamination on pre-v3.4 checkpoints)
    if not any('key_bias_proj' in k for k in model_state.keys()):
        kwargs['eco_key_bias'] = False

    # Blockade strength from learnable parameter
    for k, v in model_state.items():
        if 'blockade_strength' in k:
            kwargs['blockade_strength'] = v.item()
            break

    # --- Phase 3: Eval-specific overrides ---
    kwargs['dropout'] = 0.0
    kwargs['stage_surprise_grad_scale'] = 0.0  # no grad scaling at eval
    kwargs['blockade_warmup_steps'] = 0
    kwargs['act_gradient_checkpointing'] = False
    kwargs['layer_gradient_checkpointing'] = False

    # Ensure layers_per_stage is a list
    lps = kwargs.get('layers_per_stage')
    if lps is not None and not isinstance(lps, list):
        kwargs['layers_per_stage'] = list(lps)

    max_seq_len = kwargs.get('max_seq_len', 1024)
    chain_cfg = T3v3ChainConfig(**kwargs)
    return chain_cfg, max_seq_len


@register_model("t3v3")
class T3v3LM(LM):
    """lm-eval wrapper for T³v3 Chain model."""

    def __init__(
        self,
        pretrained: str = None,
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int = None,
        **kwargs,
    ):
        super().__init__()

        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._batch_size = int(batch_size)
        self._max_length_override = int(max_length) if max_length is not None else None
        # v3.5: Live ecology + null cone during eval
        self._eval_live_primitives = kwargs.pop('eval_live_primitives', False)
        self._tokenizer_override = kwargs.pop('tokenizer_override', None)
        self._null_cone_strength = kwargs.pop('null_cone_strength', 0.0)

        if pretrained is None:
            raise ValueError("Must specify --model_args pretrained=<path>")

        print(f"Loading T³v3 from: {os.path.basename(pretrained)}")
        print(f"Device: {self._device}")

        ckpt = torch.load(pretrained, map_location='cpu', weights_only=False)
        ckpt['model_state'] = strip_compiled_prefix(ckpt['model_state'])
        ckpt['model_state'] = migrate_sigma_projections(ckpt['model_state'])

        # Tokenizer — auto-detect from checkpoint metadata, fallback to inference
        cfg = ckpt.get('config', {})
        tok_name = ckpt.get('tokenizer', None)
        if tok_name is None:
            source = cfg.get('source', '') if isinstance(cfg, dict) else ''
            source_model = ckpt.get('source_model', '')
            if 'smollm' in (source + source_model).lower():
                tok_name = 'HuggingFaceTB/SmolLM2-360M'
            elif isinstance(cfg, dict) and cfg.get('vocab_size', 50257) == 49152:
                tok_name = 'HuggingFaceTB/SmolLM2-360M'
            elif 'qwen' in (source + source_model).lower():
                tok_name = source_model or 'Qwen/Qwen2.5-1.5B'
            elif 'gemma' in (source + source_model).lower() or (isinstance(cfg, dict) and cfg.get('vocab_size', 0) == 262144):
                tok_name = source_model or 'google/gemma-3-270m'
            else:
                tok_name = 'gpt2'
        if self._tokenizer_override:
            tok_name = self._tokenizer_override
        print(f"  Tokenizer: {tok_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ACT override from CLI
        if kwargs.get('act_enabled') is not None:
            if isinstance(cfg, dict):
                if kwargs['act_enabled'].lower() in ('false', '0', 'no'):
                    cfg['act_enabled'] = False
                    cfg['act_per_stage'] = False
                    print("  ACT disabled via CLI override")

        # FlexAttention override from CLI
        if kwargs.get('no_flex'):
            if isinstance(cfg, dict):
                cfg['use_flex_attention'] = False
                print("  FlexAttention disabled via --no_flex override")

        chain_cfg, seq_len = build_v3_config(ckpt)
        self._max_length = self._max_length_override or seq_len

        print(f"  Config: {chain_cfg.n_stages} stages, "
              f"layers={chain_cfg.layers_per_stage or chain_cfg.n_layers}, "
              f"d={chain_cfg.d_model}, {chain_cfg.n_heads}h"
              f"{f' (GQA {chain_cfg.n_kv_heads}kv)' if chain_cfg.n_kv_heads else ''}, "
              f"vocab={chain_cfg.vocab_size}, seq={self._max_length}")
        print(f"  Arch: rope={chain_cfg.use_rope}"
              f"{f' base={chain_cfg.rope_base:.0f}' if chain_cfg.use_rope else ''}, "
              f"ffn={chain_cfg.ffn_type}, norm={chain_cfg.norm_type}, "
              f"flex_attn={chain_cfg.use_flex_attention}")
        if getattr(chain_cfg, 'eco_key_bias', False):
            print(f"  v3.4 CAC: eco_key_bias=True, live_ecology={getattr(chain_cfg, 'act_live_ecology', False)}")
        if chain_cfg.act_enabled:
            print(f"  ACT: max_ponder={chain_cfg.act_per_stage_max}, "
                  f"entropy_halt={chain_cfg.act_entropy_halt}, "
                  f"threshold={chain_cfg.act_entropy_halt_threshold}")
            print(f"  v3.2: preponder={chain_cfg.act_preponder_baseline}, "
                  f"adaptive={chain_cfg.act_adaptive_threshold}, "
                  f"hard_halt_eval={chain_cfg.act_hard_halt_eval}")

        self.model = T3v3Chain(chain_cfg).to(self._device)
        missing, unexpected = self.model.load_state_dict(ckpt['model_state'], strict=False)

        if missing:
            non_critical = ['_strain_ema', '_step', 'halt_head', '_last_ponder',
                            '_coherence_ema', '_chronos_ema', '_attn_fast_ema', '_attn_slow_ema',
                            '_valence_init_count', 'world_trace', 'is_predictor', 'pc_predictor',
                            '_entropy_delta_ema', 'difficulty_head', '_loss_ema']
            critical_missing = [k for k in missing if not any(x in k for x in non_critical)]
            if critical_missing:
                print(f"  Missing keys: {len(missing)} ({len(critical_missing)} critical)")
                for k in critical_missing[:5]:
                    print(f"    {k}")
            else:
                print(f"  Missing keys: {len(missing)} (all non-critical buffers)")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
            for k in unexpected[:3]:
                print(f"    {k}")

        # Restore ecology state
        eco = ckpt.get('ecology_state', None)
        if eco and 'head_states' in eco:
            n_eco = len(eco['head_states'])
            n_model = len(self.model.stages)
            if n_eco == n_model:
                for i, stage in enumerate(self.model.stages):
                    hs = stage.head_state
                    hs_state = eco['head_states'][i]
                    dev = hs._entropy_ema.device
                    for buf_name in ['_entropy_ema', '_friction_ema', '_valence_ema', '_entropy_prev',
                                     '_coherence_ema', '_chronos_ema', '_attn_fast_ema', '_attn_slow_ema']:
                        if buf_name in hs_state and hasattr(hs, buf_name):
                            getattr(hs, buf_name).copy_(hs_state[buf_name].to(dev))
                # v3.2: Restore entropy delta EMA if present
                if '_entropy_delta_ema' in eco and hasattr(self.model, '_entropy_delta_ema'):
                    dev = self.model._entropy_delta_ema.device
                    self.model._entropy_delta_ema.copy_(eco['_entropy_delta_ema'].to(dev))
                    print(f"  Entropy delta EMA restored: {self.model._entropy_delta_ema.tolist()}")
                print(f"  Ecology restored: {n_model} stages")
            else:
                print(f"  WARNING: ecology_state has {n_eco} stages but model has {n_model}")
        else:
            print(f"  No ecology_state in checkpoint")

        self.model.eval()

        # Match ecology_strength + warmup_frac to checkpoint's training step.
        # Bug (discovered 2026-04-22, memory f18ad4695cf4c63a): these default
        # to 1.0 at eval time, but training ramps them over blockade_warmup_steps.
        # Pre-warmup checkpoints evaluated at default eco=1.0 produce garbage
        # (e.g. transfer_step0.pt cos=0.24 vs Gemma, goes to 0.9993 with eco=0).
        # For post-warmup checkpoints (step >= blockade_warmup_steps), this is
        # a no-op — default behavior unchanged.
        ckpt_step = int(ckpt.get('step', 0)) if isinstance(ckpt, dict) else 0
        warmup_steps = max(int(getattr(chain_cfg, 'blockade_warmup_steps', 2000)), 1)
        eco_frac = min(ckpt_step / warmup_steps, 1.0)
        for m in self.model.modules():
            m._ecology_strength = eco_frac
            m._warmup_frac = eco_frac
        print(f"  Warmup match: step={ckpt_step}, warmup_steps={warmup_steps}, "
              f"ecology_strength={eco_frac:.4f}, warmup_frac={eco_frac:.4f}")

        # v3.5: Enable live ecology + null cone during eval if requested
        if getattr(self, '_eval_live_primitives', False):
            self.model.cfg.eval_live_primitives = True
            for stage in self.model.stages:
                stage.cfg.eval_live_primitives = True
            print(f"  v3.5: eval_live_primitives=True")
        if getattr(self, '_null_cone_strength', 0.0) > 0:
            self.model.cfg.null_cone_strength = self._null_cone_strength
            for stage in self.model.stages:
                stage.head_state.cfg.null_cone_strength = self._null_cone_strength
            print(f"  v3.5: null_cone_strength={self._null_cone_strength}")

        # Per-stage sigma report
        for i, stage in enumerate(self.model.stages):
            s = stage.head_state._last_head_sigmas.cpu().numpy()
            print(f"  Stage {i} sigma: [{min(s):.3f}, {max(s):.3f}]")
        print(f"Batch size: {self._batch_size}")

        # Ponder depth tracking
        self._ponder_log = []  # list of per-stage ponder lists, one per forward call
        self._act_enabled = chain_cfg.act_enabled

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int], **kwargs) -> str:
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results = []
        for i in range(0, len(requests), self._batch_size):
            batch = requests[i:i + self._batch_size]
            results.extend(self._loglikelihood_batch(batch))
        return results

    def _loglikelihood_batch(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        all_full_ids = []
        all_cont_lens = []
        for req in requests:
            context, continuation = req.args
            ctx_ids = self.tokenizer.encode(context)
            # BoolQ/lm-eval: continuation must NOT include BOS (bug fix 2026-04-23)
            cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
            full_ids = ctx_ids + cont_ids
            if len(full_ids) > self._max_length:
                full_ids = full_ids[-(self._max_length):]
                cont_len = min(len(cont_ids), len(full_ids))
            else:
                cont_len = len(cont_ids)
            all_full_ids.append(full_ids)
            all_cont_lens.append(cont_len)

        max_len = max(len(ids) for ids in all_full_ids)
        pad_id = self.tokenizer.eos_token_id
        padded = []
        for ids in all_full_ids:
            padded.append([pad_id] * (max_len - len(ids)) + ids)
        input_ids = torch.tensor(padded, device=self._device)

        logits = self.model(input_ids[:, :-1])
        if isinstance(logits, tuple):
            logits = logits[0]

        # Capture ponder depths after forward pass
        if self._act_enabled and hasattr(self.model, '_last_per_stage_steps'):
            self._ponder_log.append(list(self.model._last_per_stage_steps))

        log_probs = F.log_softmax(logits, dim=-1)

        results = []
        for i, (full_ids, cont_len) in enumerate(zip(all_full_ids, all_cont_lens)):
            seq_len = len(full_ids)
            pad_offset = max_len - seq_len
            cont_start = seq_len - cont_len
            total_logprob = 0.0
            is_greedy = True
            for j in range(cont_len):
                pos = pad_offset + cont_start + j - 1
                if pos < 0 or pos >= log_probs.shape[1]:
                    continue
                target_id = full_ids[cont_start + j]
                token_logprob = log_probs[i, pos, target_id].item()
                total_logprob += token_logprob
                if log_probs[i, pos].argmax().item() != target_id:
                    is_greedy = False
            results.append((total_logprob, is_greedy))
        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        results = []
        for req in requests:
            (text,) = req.args
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) == 0:
                results.append(0.0)
                continue
            total_logprob = 0.0
            for start in range(0, len(token_ids), self._max_length):
                chunk = token_ids[start:start + self._max_length]
                if len(chunk) < 2:
                    continue
                input_ids = torch.tensor([chunk], device=self._device)
                logits = self.model(input_ids[:, :-1])
                if isinstance(logits, tuple):
                    logits = logits[0]
                if self._act_enabled and hasattr(self.model, '_last_per_stage_steps'):
                    self._ponder_log.append(list(self.model._last_per_stage_steps))
                log_probs = F.log_softmax(logits, dim=-1)
                for j in range(log_probs.shape[1]):
                    target = chunk[j + 1]
                    total_logprob += log_probs[0, j, target].item()
            results.append(total_logprob)
        return results

    @torch.no_grad()
    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            stop = gen_kwargs.get("until", [])
            if isinstance(stop, str):
                stop = [stop]
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)

            token_ids = self.tokenizer.encode(context)
            if len(token_ids) > self._max_length - max_gen:
                token_ids = token_ids[-(self._max_length - max_gen):]

            input_ids = torch.tensor([token_ids], device=self._device)
            for _ in range(max_gen):
                if input_ids.shape[1] >= self._max_length:
                    break
                logits = self.model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_logits = logits[0, -1, :]
                if temperature > 0:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = next_logits.argmax().unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                generated_text = self.tokenizer.decode(input_ids[0, len(token_ids):].tolist())
                if any(s in generated_text for s in stop):
                    break
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            generated = self.tokenizer.decode(input_ids[0, len(token_ids):].tolist())
            for s in stop:
                if s in generated:
                    generated = generated[:generated.index(s)]
            results.append(generated)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T³v3 lm-eval benchmark runner")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tasks", type=str,
                        default="boolq,arc_challenge,arc_easy,piqa,hellaswag,winogrande,copa,rte")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_act", action='store_true', help="Disable ACT for comparison")
    parser.add_argument("--no_flex", action='store_true', help="Force standard attention (ignore checkpoint flex_attention setting)")
    parser.add_argument("--eval_live_primitives", action='store_true', help="v3.5: Live ecology during eval")
    parser.add_argument("--null_cone_strength", type=float, default=0.0, help="v3.5: Null cone restoring force")
    parser.add_argument("--tokenizer", type=str, default=None, help="Override tokenizer (e.g. google/gemma-3-270m). Auto-detected by default.")
    args = parser.parse_args()

    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.checkpoint)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tasks: {args.tasks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Few-shot: {args.num_fewshot}")
    if args.limit:
        print(f"Limit: {args.limit} examples per task")

    task_list = [t.strip() for t in args.tasks.split(",")]

    # Load checkpoint metadata for run tracking (lightweight, no model weights)
    _meta_ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    _meta_cfg = _meta_ckpt.get('config', {})
    if not isinstance(_meta_cfg, dict):
        _meta_cfg = {}
    run_meta = {
        'checkpoint_path': args.checkpoint,
        'checkpoint_dir': os.path.basename(os.path.dirname(args.checkpoint)),
        'checkpoint_file': os.path.basename(args.checkpoint),
        'step': _meta_ckpt.get('step', None),
        'val_ppl': _meta_ckpt.get('val_ppl', None),
        'source_model': _meta_ckpt.get('source_model', _meta_cfg.get('source', None)),
        'run_id': _meta_ckpt.get('run_id', None),
        'tokenizer': _meta_ckpt.get('tokenizer', None),
        'd_model': _meta_cfg.get('d_model', None),
        'n_heads': _meta_cfg.get('n_heads', None),
        'n_stages': _meta_cfg.get('n_stages', None),
        'layers_per_stage': _meta_cfg.get('layers_per_stage', None),
        'vocab_size': _meta_cfg.get('vocab_size', None),
        'mix_weights': _meta_ckpt.get('mix_weights', None),
        'act_enabled': _meta_cfg.get('act_enabled', None),
        'eco_key_bias': _meta_cfg.get('eco_key_bias', None),
    }
    del _meta_ckpt  # free memory before model load

    # Instantiate model once, run tasks one-by-one to capture per-task ponder depths
    model_kwargs = dict(pretrained=args.checkpoint, batch_size=args.batch_size)
    if args.no_act:
        model_kwargs['act_enabled'] = 'false'
    if args.no_flex:
        model_kwargs['no_flex'] = True
    if args.eval_live_primitives:
        model_kwargs['eval_live_primitives'] = True
    if args.null_cone_strength > 0:
        model_kwargs['null_cone_strength'] = args.null_cone_strength
    if args.tokenizer:
        model_kwargs['tokenizer_override'] = args.tokenizer
    lm = T3v3LM(**model_kwargs)

    all_results = {}
    ponder_by_task = {}

    for task_name in task_list:
        lm._ponder_log.clear()
        print(f"\n--- Running: {task_name} ---")
        result = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task_name],
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )
        all_results.update(result.get('results', {}))
        if lm._ponder_log:
            ponder_by_task[task_name] = list(lm._ponder_log)
            n = len(lm._ponder_log)
            arr = torch.tensor(lm._ponder_log, dtype=torch.float)
            means = arr.mean(0).tolist()
            stds = arr.std(0).tolist()
            total_mean = arr.sum(1).mean().item()
            print(f"  Ponder samples: {n}")
            for si, (m, s) in enumerate(zip(means, stds)):
                print(f"  S{si}: {m:.2f} +/- {s:.2f}")
            print(f"  Total: {total_mean:.1f}")

    ckpt_name = os.path.basename(args.checkpoint)
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS -- T3v3 {ckpt_name}")
    print(f"{'='*70}")

    print(f"\n{'Task':<25} {'Metric':<15} {'Score':<10} {'Stderr':<10}")
    print("-" * 60)

    for task_name, task_results in all_results.items():
        for metric_name, value in task_results.items():
            if metric_name.startswith('alias'):
                continue
            if 'stderr' in metric_name:
                continue
            clean_name = metric_name.replace(',none', '')
            stderr_key = clean_name + '_stderr,none'
            stderr = task_results.get(stderr_key, '')
            if isinstance(stderr, float):
                stderr = f"+/-{stderr:.4f}"
            else:
                stderr = ''
            if isinstance(value, float):
                print(f"{task_name:<25} {clean_name:<15} {value:<10.4f} {stderr}")

    # Ponder depth summary
    if ponder_by_task:
        print(f"\n{'='*70}")
        print("PONDER DEPTH PER TASK (ACT)")
        print(f"{'='*70}")
        n_stages = len(next(iter(ponder_by_task.values()))[0]) if ponder_by_task else 0
        header = f"{'Task':<20}"
        for si in range(n_stages):
            header += f"  {'S'+str(si)+' mean':>8}"
        header += f"  {'Total':>8}  {'N':>6}"
        print(header)
        print("-" * len(header))
        for task_name in task_list:
            if task_name not in ponder_by_task:
                continue
            arr = torch.tensor(ponder_by_task[task_name], dtype=torch.float)
            means = arr.mean(0).tolist()
            total = arr.sum(1).mean().item()
            n = len(ponder_by_task[task_name])
            row = f"{task_name:<20}"
            for m in means:
                row += f"  {m:>8.2f}"
            row += f"  {total:>8.1f}  {n:>6}"
            print(row)

    # GPT-2 + v2.5 reference
    print(f"\n{'='*70}")
    print("Reference (zero-shot)")
    print(f"{'='*70}")
    refs = {
        'boolq':         ('acc',      0.487, 0.610),
        'arc_challenge': ('acc_norm', 0.227, 0.255),
        'arc_easy':      ('acc_norm', 0.395, 0.355),
        'piqa':          ('acc_norm', 0.625, 0.540),
        'hellaswag':     ('acc_norm', 0.311, 0.287),
        'winogrande':    ('acc',      0.516, 0.508),
        'copa':          ('acc',      0.660, 0.510),
        'rte':           ('acc',      0.534, 0.534),
    }
    print(f"\n{'Task':<25} {'Metric':<15} {'GPT-2 124M':<12} {'T3 v2.5 350M S1surg':<20}")
    print("-" * 72)
    for task, (metric, gpt2, v25) in refs.items():
        print(f"{task:<25} {metric:<15} {gpt2:<12.3f} {v25:<20.3f}")

    # Save — auto-name with checkpoint identity
    if args.output_path is None:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Build descriptive filename from checkpoint metadata
        _src = (run_meta.get('source_model') or '').lower()
        if 'qwen' in _src:
            _model_tag = 'qwen15b'
        elif 'medium' in _src or ('gpt2' in _src and '355' in str(run_meta.get('d_model', ''))):
            _model_tag = 'gpt2m'
        elif run_meta.get('d_model') == 1536:
            _model_tag = 'qwen15b'
        elif run_meta.get('d_model') == 1024:
            _model_tag = 'gpt2m'
        else:
            _model_tag = 'gpt2s'
        _step = run_meta.get('step')
        _step_tag = f"_step{_step // 1000}k" if _step and _step >= 1000 else f"_step{_step}" if _step else ''
        _dir_tag = run_meta.get('checkpoint_dir', '')
        # Extract variant from dir name (e.g. cac_baseline, cac_full)
        _variant = ''
        if 'cac_' in _dir_tag:
            _variant = '_' + _dir_tag.split('cac_')[-1]
        elif 'flex' in _dir_tag:
            _variant = '_flex'
        args.output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"bench_{_model_tag}{_step_tag}{_variant}_{ts}.json"
        )

    import json
    ponder_stats = {}
    for task_name, logs in ponder_by_task.items():
        arr = torch.tensor(logs, dtype=torch.float)
        ponder_stats[task_name] = {
            'n_samples': len(logs),
            'per_stage_mean': arr.mean(0).tolist(),
            'per_stage_std': arr.std(0).tolist(),
            'total_mean': arr.sum(1).mean().item(),
            'total_std': arr.sum(1).std().item(),
        }

    save_data = {
        'results': all_results,
        'ponder_stats': ponder_stats,
        'run': run_meta,
        'config': {
            'checkpoint': ckpt_name,
            'tasks': task_list,
            'num_fewshot': args.num_fewshot,
            'batch_size': args.batch_size,
            'limit': args.limit,
            'act_disabled': args.no_act,
        },
    }
    with open(args.output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output_path}")
