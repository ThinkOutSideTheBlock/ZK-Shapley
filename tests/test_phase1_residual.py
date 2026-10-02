"""
Phase-1 unit tests: firm-specific predictive residual and size-scaled λ.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import (
    effective_ridge_lambda,
    firm_predictive_residual_std,
    fit_ridge_for_coalition,
)


@pytest.fixture(scope="module")
def small_data():
    cfg = DemandSimulationConfig(
        n_firms=3, n_periods=200, n_features=4, seed=11,
        shared_factor_correlation=0.5,
    )
    sim = generate_multi_firm_demand(cfg)
    datasets = build_firm_datasets(sim, holdout_fraction=0.25, min_holdout=30)
    paths = {n: ds.y_holdout for n, ds in datasets.items()}
    firms = tuple(sorted(datasets.keys()))
    return datasets, paths, firms


def test_effective_lambda_size_scaled():
    assert effective_ridge_lambda(1.0, 100, mode="size_scaled") == 100.0
    assert effective_ridge_lambda(1.0, 100, mode="fixed") == 1.0


def test_firm_predictive_residual_positive(small_data):
    datasets, _, firms = small_data
    beta = fit_ridge_for_coalition(datasets, frozenset({firms[0]}), lam0=1.0)
    s = firm_predictive_residual_std(beta, datasets[firms[0]])
    assert s >= 1e-6
    assert np.isfinite(s)


def test_singleton_still_zero_under_phase1(small_data):
    datasets, paths, firms = small_data
    cfg = CoalitionConfig(
        firms=firms,
        lambda_mode="size_scaled",
        residual_mode="predictive",
        seed=0,
    )
    ev = CoalitionEvaluator(datasets, paths, cfg)
    for f in firms:
        assert ev.value(frozenset({f})) == pytest.approx(0.0, abs=1e-9)
        assert ev.value_accuracy(
            frozenset({f})) == pytest.approx(0.0, abs=1e-9)


def test_phase1_efficiency(small_data):
    datasets, paths, firms = small_data
    cfg = CoalitionConfig(
        firms=firms,
        lambda_mode="size_scaled",
        residual_mode="predictive",
        seed=0,
    )
    ev = CoalitionEvaluator(datasets, paths, cfg)
    from src.shapley import CachedGame, exact_shapley, check_efficiency
    game = CachedGame(players=list(firms), v_func=ev.operational_v_func())
    phi = exact_shapley(game)
    v_N = game.v(frozenset(firms))
    check = check_efficiency(phi, v_N, tol=1e-9)
    assert check["satisfied"] is True
