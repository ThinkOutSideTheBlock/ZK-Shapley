"""
Integration tests for CoalitionEvaluator.

Requirements from the plan (Stage 2 exit criterion):
  - n=4 firms
  - efficiency gap < 1e-9
  - deterministic across two independent evaluator instances
  - v({i}) = 0 for both operational and accuracy games
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.shapley import CachedGame, exact_shapley, check_efficiency


@pytest.fixture(scope="module")
def base_data():
    config = DemandSimulationConfig(
        n_firms=4,
        n_periods=160,
        n_features=5,
        seed=21,
        shared_factor_correlation=0.55,
        n_obs_heterogeneity="moderate",
    )
    sim = generate_multi_firm_demand(config)
    datasets = build_firm_datasets(sim, holdout_fraction=0.2, min_holdout=12)
    demand_paths = {n: ds.y_holdout for n, ds in datasets.items()}
    firms = tuple(sorted(datasets.keys()))
    return datasets, demand_paths, firms


def _make_evaluator(datasets, demand_paths, firms, cache_dir=None, seed=0):
    cfg = CoalitionConfig(firms=firms, seed=seed)
    return CoalitionEvaluator(
        datasets=datasets,
        demand_paths=demand_paths,
        config=cfg,
        cache_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# Basic invariants
# ---------------------------------------------------------------------------

def test_singleton_operational_value_is_zero(base_data):
    datasets, demand_paths, firms = base_data
    ev = _make_evaluator(datasets, demand_paths, firms)
    for f in firms:
        assert ev.value(frozenset({f})) == pytest.approx(0.0, abs=1e-9)
    assert ev.value(frozenset()) == pytest.approx(0.0)


def test_singleton_accuracy_value_is_zero(base_data):
    datasets, demand_paths, firms = base_data
    ev = _make_evaluator(datasets, demand_paths, firms)
    for f in firms:
        assert ev.value_accuracy(
            frozenset({f})) == pytest.approx(0.0, abs=1e-9)
    assert ev.value_accuracy(frozenset()) == pytest.approx(0.0)


def test_grand_coalition_operational_value_non_negative(base_data):
    datasets, demand_paths, firms = base_data
    ev = _make_evaluator(datasets, demand_paths, firms)
    v_N = ev.value(frozenset(firms))
    # Under moderate correlation we expect positive surplus
    assert v_N >= -1e-6


# ---------------------------------------------------------------------------
# Efficiency + determinism (plan exit criterion)
# ---------------------------------------------------------------------------

def test_efficiency_gap_below_1e9(base_data):
    datasets, demand_paths, firms = base_data
    ev = _make_evaluator(datasets, demand_paths, firms)

    game = CachedGame(players=list(firms), v_func=ev.operational_v_func())
    phi = exact_shapley(game)
    v_N = game.v(frozenset(firms))
    check = check_efficiency(phi, v_N, tol=1e-9)
    assert check["satisfied"] is True
    assert check["abs_gap"] < 1e-9


def test_determinism_across_two_evaluator_instances(base_data):
    """Two independent evaluators with the same seed must produce identical values."""
    datasets, demand_paths, firms = base_data
    ev1 = _make_evaluator(datasets, demand_paths, firms, seed=42)
    ev2 = _make_evaluator(datasets, demand_paths, firms, seed=42)

    coal = frozenset(firms[:3])
    assert ev1.value(coal) == pytest.approx(ev2.value(coal), abs=1e-12)
    assert ev1.value_accuracy(coal) == pytest.approx(
        ev2.value_accuracy(coal), abs=1e-12
    )


def test_disk_cache_roundtrip(base_data):
    datasets, demand_paths, firms = base_data
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        ev1 = _make_evaluator(datasets, demand_paths,
                              firms, cache_dir=cache_dir, seed=7)
        coal = frozenset(firms)
        v1 = ev1.value(coal)
        # Second evaluator loads from disk
        ev2 = _make_evaluator(datasets, demand_paths,
                              firms, cache_dir=cache_dir, seed=7)
        v2 = ev2.value(coal)
        assert v1 == pytest.approx(v2, abs=1e-12)
        # Cache should have been populated
        assert ev1.cache_info()["value_op"] >= 1


def test_fit_returns_consistent_coefficients(base_data):
    datasets, demand_paths, firms = base_data
    ev = _make_evaluator(datasets, demand_paths, firms)
    S = frozenset(firms[:2])
    beta1 = ev.fit(S)
    beta2 = ev.fit(S)          # cache hit
    np.testing.assert_array_equal(beta1, beta2)


def test_cost_breakdown_fields(base_data):
    datasets, demand_paths, firms = base_data
    ev = _make_evaluator(datasets, demand_paths, firms)
    bd = ev.cost(frozenset(firms))
    assert bd.total == pytest.approx(bd.holding + bd.shortage, abs=1e-8)
    assert 0.0 <= bd.fill_rate <= 1.0 + 1e-9
    assert set(bd.per_firm.keys()) == set(firms)
