"""Smoke test the public benchmarks API surface (no actual eval run)."""

import pytest

import t3


def test_t3_benchmarks_module_imports():
    from t3 import benchmarks
    assert hasattr(benchmarks, "T3LM")
    assert hasattr(benchmarks, "DEFAULT_TASKS")
    assert hasattr(benchmarks, "run_benchmark_suite")


def test_default_tasks_match_t3atlas_headline():
    from t3.benchmarks import DEFAULT_TASKS
    expected = {"boolq", "arc_easy", "arc_challenge", "piqa",
                "hellaswag", "winogrande", "copa", "rte"}
    assert set(DEFAULT_TASKS) == expected


def test_no_circular_import_with_lm_eval():
    """t3.benchmarks must not shadow upstream lm_eval — they coexist."""
    import lm_eval as upstream
    from t3 import benchmarks
    assert upstream.__name__ == "lm_eval"
    assert benchmarks.__name__ == "t3.benchmarks"
