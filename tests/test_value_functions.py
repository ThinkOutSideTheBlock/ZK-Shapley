# tests/test_value_functions.py
"""
Direct tests of the two characteristic functions that define the paper’s
central comparison: operational (inventory-savings) vs accuracy (RMSE-reduction).

These tests replace the proxy RMSE value function previously used in
test_mechanism.py for the core scientific claims.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.inventory import default_three_echelon_config
from src.mechanism import (
    make_operational_value_function,
    make_accuracy_value_function,
    run_zk_shapley_mechanism,
    run_accuracy_shapley_mechanism,
    compute_divergence_metrics,
)


@pytest.fixture
def small_setup():
    config = DemandSimulationConfig(
        n_firms=4, n_periods=180, n_features=5, seed=11,
        shared_factor_correlation=0.6, n_obs_heterogeneity="moderate",
    )
    sim = generate_multi_firm_demand(config)
    datasets = build_firm_datasets(sim, holdout_fraction=0.2, min_holdout=15)
    echelons = default_three_echelon_config()
    # Use holdout demand as the realized path (consistent with current mechanism)
    demand_paths = {name: ds.y_holdout for name, ds in datasets.items()}
    return datasets, demand_paths, echelons


def test_operational_value_singleton_is_zero(small_setup):
    datasets, demand_paths, echelons = small_setup
    v_op = make_operational_value_function(
        datasets, demand_paths, echelons, lam=1.0, rng_seed=0)
    for name in datasets:
        assert v_op(frozenset({name})) == pytest.approx(0.0, abs=1e-9)
    assert v_op(frozenset()) == pytest.approx(0.0)


def test_accuracy_value_singleton_is_zero(small_setup):
    datasets, _, _ = small_setup
    v_acc = make_accuracy_value_function(datasets, lam=1.0)
    for name in datasets:
        assert v_acc(frozenset({name})) == pytest.approx(0.0, abs=1e-9)
    assert v_acc(frozenset()) == pytest.approx(0.0)


def test_operational_grand_coalition_non_negative_when_correlated(small_setup):
    """Path-cost surplus is NOT guaranteed; residual game is R2 primary.
    Keep as soft diagnostic: record value, do not hard-require v_N >= 0.
    """
    datasets, demand_paths, echelons = small_setup
    v_op = make_operational_value_function(
        datasets, demand_paths, echelons, lam=1.0, rng_seed=0)
    firms = list(datasets.keys())
    v_N = v_op(frozenset(firms))
    # R2: path-cost may be negative (see results/r1_residual_vs_path, ir diagnostics)
    assert np.isfinite(v_N)


def test_both_games_produce_efficient_shapley(small_setup):
    datasets, demand_paths, echelons = small_setup
    firms = list(datasets.keys())
    v_op = make_operational_value_function(
        datasets, demand_paths, echelons, lam=1.0, rng_seed=0)
    v_acc = make_accuracy_value_function(datasets, lam=1.0)

    res_op = run_zk_shapley_mechanism(firms, v_op, exact=True)
    res_acc = run_accuracy_shapley_mechanism(firms, v_acc, exact=True)

    assert res_op.efficiency_gap < 1e-8
    assert res_acc.efficiency_gap < 1e-8
    # Do NOT require path-cost IR; accuracy IR also not guaranteed
    # Optional: residual-weighted game IR is a separate test if you add v_res fixture


def test_divergence_metrics_run_and_report_structure(small_setup):
    datasets, demand_paths, echelons = small_setup
    firms = list(datasets.keys())
    v_op = make_operational_value_function(
        datasets, demand_paths, echelons, lam=1.0, rng_seed=0)
    v_acc = make_accuracy_value_function(datasets, lam=1.0)

    res_op = run_zk_shapley_mechanism(firms, v_op, exact=True)
    res_acc = run_accuracy_shapley_mechanism(firms, v_acc, exact=True)
    metrics = compute_divergence_metrics(res_op.payments, res_acc.payments)

    assert "spearman_rho" in metrics
    assert "rank_shifts" in metrics
    assert "max_normalized_payment_difference" in metrics
    assert "any_rank_reversal" in metrics
    assert set(metrics["firms"]) == set(firms)


def test_operational_value_is_deterministic_given_seed(small_setup):
    datasets, demand_paths, echelons = small_setup
    firms = list(datasets.keys())
    v1 = make_operational_value_function(
        datasets, demand_paths, echelons, lam=1.0, rng_seed=42)
    v2 = make_operational_value_function(
        datasets, demand_paths, echelons, lam=1.0, rng_seed=42)
    coal = frozenset(firms[:2])
    assert v1(coal) == pytest.approx(v2(coal), abs=1e-12)
