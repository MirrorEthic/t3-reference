"""Smoke tests: T3Config faithfully projects the run-3 checkpoint config dict."""

from t3.config import T3Config


def test_defaults_match_v36_run3_shape():
    cfg = T3Config()
    assert cfg.n_primitives == 6
    assert cfg.n_stages == 3
    assert cfg.layers_per_stage == (4, 3, 5)
    assert cfg.total_layers == 12
    assert cfg.n_layers == max(cfg.layers_per_stage) == 5
    assert cfg.head_dim == 64                 # 768 / 12
    assert cfg.hamiltonian_trivectors is False  # run-3 was trained without trivectors
    assert cfg.act_per_stage_max == 4
    assert cfg.act_max_steps == 8
    assert cfg.logit_softcap == 30.0


def test_from_checkpoint_dict_drops_unknown_keys():
    raw = {
        "vocab_size": 50257,
        "d_model": 768,
        "n_heads": 12,
        # training-only — must be dropped:
        "lr": 3e-4,
        "sigma_spread_weight": 0.01,
        "act_ponder_weight": 0.01,
        "cooperative_prediction_weight": 0.01,
        "stage_surprise_grad_scale": 0.5,
    }
    cfg = T3Config.from_checkpoint_dict(raw)
    assert cfg.vocab_size == 50257
    assert cfg.d_model == 768
    assert cfg.n_heads == 12


def test_from_checkpoint_dict_coerces_list_to_tuple():
    raw = {
        "layers_per_stage": [4, 3, 5],
        "scratchpad_inject_entropy": [0.0, 0.0, 0.03],
    }
    cfg = T3Config.from_checkpoint_dict(raw)
    assert isinstance(cfg.layers_per_stage, tuple)
    assert cfg.layers_per_stage == (4, 3, 5)
    assert isinstance(cfg.scratchpad_inject_entropy, tuple)


def test_from_actual_run3_checkpoint(tmp_path):
    """Round-trip the actual published checkpoint config through T3Config."""
    import torch
    from pathlib import Path

    ckpt_path = Path(
        "/home/garret-sutherland/CVMP/T3_sims/t3v2/t3v3/t3v36/"
        "checkpoints_v36_run3/best.pt"
    )
    if not ckpt_path.exists():
        import pytest
        pytest.skip("Local run-3 checkpoint not present.")
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = T3Config.from_checkpoint_dict(ck["config"])
    assert cfg.vocab_size == 50257
    assert cfg.d_model == 768
    assert cfg.layers_per_stage == (4, 3, 5)
    assert cfg.hamiltonian_trivectors is False
    # The checkpoint records `version='v3.3-act'` (training-script tag) — the
    # release name "v3.6" comes from the campaign, not the version field.
    assert cfg.version == "v3.3-act"
    # Total of layers_per_stage is the real transformer block count:
    assert cfg.total_layers == 12
