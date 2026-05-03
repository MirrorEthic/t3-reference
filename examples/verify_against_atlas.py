"""Verify a locally-generated trace against the schema and (optionally) a
canonical published trace.

Two modes:
    1. Schema-only:   verify the local trace conforms to schema v1 invariants.
    2. Diff-vs-atlas: load both the local and a canonical reference trace,
                      compare structural metadata (n_stages, n_heads, layer
                      counts, capabilities). Per-frame numerical comparison is
                      skipped because canonical traces from different lineages
                      use different weights.

Usage:
    python examples/verify_against_atlas.py path/to/local_trace.jsonl
    python examples/verify_against_atlas.py path/to/local_trace.jsonl path/to/atlas_trace.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

from t3.tracing import load_trace


def check_schema_v1(trace: dict, label: str) -> None:
    meta = trace["meta"]
    geoms = trace["geoms"]
    frames = trace["frames"]
    chain_states = trace["chain_states"]

    assert meta is not None and meta["type"] == "meta", f"[{label}] missing meta"
    assert isinstance(meta["n_stages"], int) and meta["n_stages"] >= 1
    assert isinstance(meta["n_heads"], int) and meta["n_heads"] >= 1
    assert isinstance(meta["primitive_names"], list)
    assert len(meta["primitive_names"]) == len(meta["primitive_signature"])

    assert len(geoms) == meta["n_stages"], (
        f"[{label}] geom count {len(geoms)} != n_stages {meta['n_stages']}")
    for i in range(meta["n_stages"]):
        g = geoms[i]
        n_h = meta["n_heads"]
        assert len(g["head_positions"]) == n_h
        assert all(len(p) == 3 for p in g["head_positions"]), \
            f"[{label}] stage {i} head_positions not 3-D (3-torus)"
        assert len(g["distances"]) == n_h
        assert len(g["blockade_kernel"]) == n_h

    assert len(frames) > 0, f"[{label}] no frames"
    n_prim = meta["capabilities"]["n_primitives"]
    for fr in frames[:3]:
        assert fr["type"] == "frame"
        assert "stage_idx" in fr
        assert len(fr["primitives"]) == meta["n_heads"]
        assert all(len(p) == n_prim for p in fr["primitives"])
        assert len(fr["sigma"]) == meta["n_heads"]
        assert len(fr["Q"]) == meta["n_heads"]

    print(f"[{label}] schema v1 OK  "
          f"frames={len(frames)} chain_states={len(chain_states)} "
          f"stages={meta['n_stages']} heads={meta['n_heads']}")


def diff_meta(local: dict, atlas: dict) -> None:
    lm, am = local["meta"], atlas["meta"]
    print()
    print(f"  field            local            atlas")
    print(f"  ---------------- ---------------- ----------------")
    for k in ("n_stages", "n_heads", "d_head", "n_layers_per_stage",
              "primitive_names", "primitive_signature", "lineage", "n_tokens"):
        same = "✓" if lm.get(k) == am.get(k) else "✗"
        print(f"  {k:16s} {str(lm.get(k))[:16]:16s} {str(am.get(k))[:16]:16s} {same}")
    print()
    lc, ac = lm.get("capabilities", {}), am.get("capabilities", {})
    print(f"  capability       local            atlas")
    print(f"  ---------------- ---------------- ----------------")
    for k in ("has_coupling", "has_trivectors", "has_dyn_omega",
              "has_inter_stage_pc", "has_scratchpad", "n_primitives"):
        same = "✓" if lc.get(k) == ac.get(k) else "✗ (lineage difference)"
        print(f"  {k:16s} {str(lc.get(k))[:16]:16s} {str(ac.get(k))[:16]:16s} {same}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    local_path = Path(sys.argv[1])
    local = load_trace(local_path)
    check_schema_v1(local, "local")

    if len(sys.argv) >= 3:
        atlas_path = Path(sys.argv[2])
        atlas = load_trace(atlas_path)
        check_schema_v1(atlas, "atlas")
        diff_meta(local, atlas)


if __name__ == "__main__":
    main()
