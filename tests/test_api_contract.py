"""
API contract tests.

These tests freeze the public signatures exported by src/api.py.
Any change to a public function's parameter list or name will cause a
loud failure here, rather than a silent breakage three weeks later in an
experiment script.
"""
from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest

import src.api as api


# ---------------------------------------------------------------------------
# Required public names
# ---------------------------------------------------------------------------

REQUIRED_NAMES = [
    # demand
    "DemandSimulationConfig",
    "generate_multi_firm_demand",
    # federated
    "FirmDataset",
    "build_firm_datasets",
    "aggregate_coalition_statistics",
    "fit_ridge_from_statistics",
    "fit_and_evaluate_coalition",
    "verify_exact_aggregation",
    # inventory
    "EchelonConfig",
    "simulate_serial_supply_chain",
    "default_three_echelon_config",
    # shapley
    "CachedGame",
    "exact_shapley",
    "monte_carlo_shapley",
    "check_efficiency",
    "check_null_player",
    "check_symmetry",
    # commitment
    "Commitment",
    "commit_statistics",
    "verify_opening",
    "verify_opening_or_raise",
    "CommitmentError",
    # proof_cost
    "FixedPointSpec",
    "relation_multiplication_count",
    "run_mb1_benchmark",
    "extrapolate_mb2",
    "extrapolate_mb3",
    "fit_scaling_model",
    # theory
    "noise_injection_risk_bound",
    "size_declaration_is_unused",
    "sample_size_baseline_payment",
    "sample_size_baseline_monotonicity_check",
    # mechanism
    "MechanismResult",
    "make_operational_value_function",
    "make_accuracy_value_function",
    "run_zk_shapley_mechanism",
    "run_accuracy_shapley_mechanism",
    "run_equal_split_mechanism",
    "run_proportional_mechanism",
    "run_no_sharing_baseline",
    "apply_participation_costs",
    "scale_statistics_attack",
    "inject_noise_dataset",
    "deviation_payoff_analysis",
    "compute_divergence_metrics",
]


def test_all_required_names_are_exported():
    missing = [name for name in REQUIRED_NAMES if not hasattr(api, name)]
    assert missing == [], f"Missing from src.api: {missing}"


def test_api_all_list_is_consistent():
    """Every name in __all__ must exist as an attribute."""
    for name in api.__all__:
        assert hasattr(
            api, name), f"{name} is in __all__ but missing from module"


# ---------------------------------------------------------------------------
# Signature freezes for the most critical entry points
# ---------------------------------------------------------------------------

def _param_names(fn) -> list[str]:
    return list(inspect.signature(fn).parameters.keys())


def test_run_zk_shapley_mechanism_signature():
    params = _param_names(api.run_zk_shapley_mechanism)
    # Required positional / keyword parameters (order may vary for defaults)
    assert "firm_names" in params
    assert "v_func" in params
    assert "exact" in params
    assert "n_permutations" in params
    assert "rng" in params


def test_run_accuracy_shapley_mechanism_signature():
    params = _param_names(api.run_accuracy_shapley_mechanism)
    assert "firm_names" in params
    assert "v_accuracy_func" in params or "v_func" in params


def test_run_proportional_mechanism_signature():
    params = _param_names(api.run_proportional_mechanism)
    assert "firm_names" in params
    assert "v_func" in params
    assert "size_proxy" in params


def test_make_operational_value_function_signature():
    params = _param_names(api.make_operational_value_function)
    assert "datasets" in params
    assert "demand_paths" in params
    assert "echelons" in params
    assert "lam" in params
    assert "rng_seed" in params


def test_make_accuracy_value_function_signature():
    params = _param_names(api.make_accuracy_value_function)
    assert "datasets" in params
    assert "lam" in params


def test_commit_statistics_signature():
    params = _param_names(api.commit_statistics)
    assert "firm_name" in params
    assert "A" in params
    assert "b" in params


def test_verify_opening_signature():
    params = _param_names(api.verify_opening)
    assert "commitment" in params
    assert "A_opened" in params
    assert "b_opened" in params


def test_size_declaration_is_unused_signature():
    params = _param_names(api.size_declaration_is_unused)
    assert "payment_function" in params
    assert "accepted_statistics" in params
    assert "declared_sizes_a" in params
    assert "declared_sizes_b" in params


def test_deviation_payoff_analysis_signature():
    params = _param_names(api.deviation_payoff_analysis)
    assert "firm_names" in params
    assert "v_func_factory" in params
    assert "datasets" in params
    assert "deviator" in params
    assert "sigma_eta_grid" in params


def test_exact_shapley_signature():
    params = _param_names(api.exact_shapley)
    assert "game" in params


def test_monte_carlo_shapley_signature():
    params = _param_names(api.monte_carlo_shapley)
    assert "game" in params
    assert "n_permutations" in params
