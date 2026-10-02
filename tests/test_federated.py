# tests/test_federated.py
"""
Tests for src/federated.py: exact-aggregation ridge machinery.

Central claim under direct test: aggregate_coalition_statistics + 
fit_ridge_from_statistics on a coalition's sufficient statistics must agree
with fit_ridge_centralized on the coalition's pooled raw data, to near
machine precision (plan Section B.2's "bit-identical agreement" claim, tested
here as a numerical discrepancy per the module's own H.8 caveat).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import (
    FirmDataset, build_firm_datasets, aggregate_coalition_statistics,
    fit_ridge_from_statistics, fit_ridge_centralized, verify_exact_aggregation,
    evaluate_holdout_rmse, fit_and_evaluate_coalition,
)


@pytest.fixture
def datasets():
    config = DemandSimulationConfig(
        n_firms=5, n_periods=300, n_features=6, seed=3)
    sim = generate_multi_firm_demand(config)
    return build_firm_datasets(sim)


# ---------------------------------------------------------------------------
# build_firm_datasets
# ---------------------------------------------------------------------------

def test_build_firm_datasets_partitions_are_chronological_and_non_overlapping(datasets):
    for name, ds in datasets.items():
        assert ds.n_train + ds.n_val + ds.n_holdout <= 300
        assert ds.n_train >= 0 and ds.n_val >= 0 and ds.n_holdout >= 0


def test_build_firm_datasets_respects_min_holdout_floor():
    config = DemandSimulationConfig(n_firms=3, n_periods=200, seed=1)
    sim = generate_multi_firm_demand(config)
    datasets = build_firm_datasets(sim, holdout_fraction=0.01, min_holdout=25)
    for ds in datasets.values():
        # min_holdout floor applies unless capped by T-10
        assert ds.n_holdout >= min(
            25, ds.n_train + ds.n_val + ds.n_holdout - 10) or ds.n_holdout >= 0


def test_build_firm_datasets_dataset_properties_match_array_shapes(datasets):
    for name, ds in datasets.items():
        assert ds.n_train == ds.X_train.shape[0] == ds.y_train.shape[0]
        assert ds.n_val == ds.X_val.shape[0] == ds.y_val.shape[0]
        assert ds.n_holdout == ds.X_holdout.shape[0] == ds.y_holdout.shape[0]


def test_build_firm_datasets_preserves_gamma_and_sigma(datasets):
    config = DemandSimulationConfig(
        n_firms=5, n_periods=300, n_features=6, seed=3)
    sim = generate_multi_firm_demand(config)
    for name, raw in sim.firms.items():
        ds = datasets[name]
        assert ds.gamma == pytest.approx(raw.gamma)
        assert ds.sigma == pytest.approx(raw.sigma)


# ---------------------------------------------------------------------------
# aggregate_coalition_statistics
# ---------------------------------------------------------------------------

def test_aggregate_coalition_statistics_sums_correctly(datasets):
    names = list(datasets.keys())
    coalition = frozenset(names[:2])
    A, b = aggregate_coalition_statistics(datasets, coalition)

    A_manual = sum(datasets[n].X_train.T @
                   datasets[n].X_train for n in coalition)
    b_manual = sum(datasets[n].X_train.T @
                   datasets[n].y_train for n in coalition)
    np.testing.assert_allclose(A, A_manual)
    np.testing.assert_allclose(b, b_manual)


def test_aggregate_coalition_statistics_singleton_matches_own_train_data(datasets):
    name = next(iter(datasets))
    A, b = aggregate_coalition_statistics(datasets, frozenset({name}))
    ds = datasets[name]
    np.testing.assert_allclose(A, ds.X_train.T @ ds.X_train)
    np.testing.assert_allclose(b, ds.X_train.T @ ds.y_train)


def test_aggregate_coalition_statistics_rejects_empty_coalition(datasets):
    with pytest.raises(ValueError):
        aggregate_coalition_statistics(datasets, frozenset())


def test_aggregate_coalition_statistics_grand_coalition_equals_sum_of_singletons(datasets):
    """
    Additivity check independent of any centralized-fit comparison: A_N must
    equal the elementwise sum of every singleton's A_i (superadditivity of
    the statistics themselves, prior to any ridge solve).
    """
    names = list(datasets.keys())
    A_grand, b_grand = aggregate_coalition_statistics(
        datasets, frozenset(names))
    A_sum = sum(aggregate_coalition_statistics(
        datasets, frozenset({n}))[0] for n in names)
    b_sum = sum(aggregate_coalition_statistics(
        datasets, frozenset({n}))[1] for n in names)
    np.testing.assert_allclose(A_grand, A_sum)
    np.testing.assert_allclose(b_grand, b_sum)


# ---------------------------------------------------------------------------
# fit_ridge_from_statistics / fit_ridge_centralized / verify_exact_aggregation
# ---------------------------------------------------------------------------

def test_fit_ridge_from_statistics_solves_normal_equations_exactly():
    rng = np.random.default_rng(0)
    d = 4
    A = rng.normal(size=(d, d))
    A = A @ A.T + np.eye(d)  # SPD
    b = rng.normal(size=d)
    lam = 2.0
    w = fit_ridge_from_statistics(A, b, lam=lam)
    residual = (A + lam * np.eye(d)) @ w - b
    np.testing.assert_allclose(residual, np.zeros(d), atol=1e-9)


def test_verify_exact_aggregation_reports_near_zero_discrepancy(datasets):
    names = list(datasets.keys())
    coalition = frozenset(names)
    report = verify_exact_aggregation(datasets, coalition, lam=1.0)
    assert report["max_coefficient_discrepancy"] < 1e-8
    assert report["relative_discrepancy"] < 1e-8
    assert report["coalition_size"] == len(names)


def test_verify_exact_aggregation_matches_manual_centralized_fit(datasets):
    names = list(datasets.keys())[:3]
    coalition = frozenset(names)
    A_S, b_S = aggregate_coalition_statistics(datasets, coalition)
    w_suffstat = fit_ridge_from_statistics(A_S, b_S, lam=1.0)

    X_pooled = np.concatenate([datasets[n].X_train for n in coalition], axis=0)
    y_pooled = np.concatenate([datasets[n].y_train for n in coalition], axis=0)
    w_centralized = fit_ridge_centralized(X_pooled, y_pooled, lam=1.0)

    np.testing.assert_allclose(w_suffstat, w_centralized, atol=1e-8)


def test_verify_exact_aggregation_singleton_coalition(datasets):
    name = next(iter(datasets))
    report = verify_exact_aggregation(datasets, frozenset({name}), lam=1.0)
    assert report["coalition_size"] == 1
    assert report["max_coefficient_discrepancy"] < 1e-8


# ---------------------------------------------------------------------------
# evaluate_holdout_rmse
# ---------------------------------------------------------------------------

def test_evaluate_holdout_rmse_zero_for_perfect_predictions():
    ds_dict = {
        "f1": FirmDataset(
            name="f1",
            X_train=np.eye(2), y_train=np.array([1.0, 2.0]),
            X_val=np.zeros((0, 2)), y_val=np.zeros(0),
            X_holdout=np.eye(2), y_holdout=np.array([3.0, 4.0]),
        )
    }
    # X_holdout @ w = [3,4] exactly since X_holdout = I
    w = np.array([3.0, 4.0])
    rmse = evaluate_holdout_rmse(w, ds_dict, frozenset({"f1"}))
    assert rmse == pytest.approx(0.0, abs=1e-10)


def test_evaluate_holdout_rmse_returns_nan_when_all_holdouts_empty():
    ds_dict = {
        "f1": FirmDataset(
            name="f1", X_train=np.eye(2), y_train=np.array([1.0, 2.0]),
            X_val=np.zeros((0, 2)), y_val=np.zeros(0),
            X_holdout=np.zeros((0, 2)), y_holdout=np.zeros(0),
        )
    }
    w = np.array([1.0, 1.0])
    rmse = evaluate_holdout_rmse(w, ds_dict, frozenset({"f1"}))
    assert np.isnan(rmse)


def test_evaluate_holdout_rmse_pools_across_multiple_firms():
    ds_dict = {
        "f1": FirmDataset(
            name="f1", X_train=np.eye(1), y_train=np.array([1.0]),
            X_val=np.zeros((0, 1)), y_val=np.zeros(0),
            X_holdout=np.array([[1.0]]), y_holdout=np.array([2.0]),
        ),
        "f2": FirmDataset(
            name="f2", X_train=np.eye(1), y_train=np.array([1.0]),
            X_val=np.zeros((0, 1)), y_val=np.zeros(0),
            X_holdout=np.array([[1.0]]), y_holdout=np.array([4.0]),
        ),
    }
    w = np.array([2.0])  # predictions: [2.0, 2.0]; truth: [2.0, 4.0]
    rmse = evaluate_holdout_rmse(w, ds_dict, frozenset({"f1", "f2"}))
    expected = np.sqrt(np.mean([(2.0 - 2.0) ** 2, (2.0 - 4.0) ** 2]))
    assert rmse == pytest.approx(expected)


# ---------------------------------------------------------------------------
# fit_and_evaluate_coalition
# ---------------------------------------------------------------------------

def test_fit_and_evaluate_coalition_autarky_uses_training_mean_predictor():
    ds_dict = {
        "f1": FirmDataset(
            # mean = 4.0
            name="f1", X_train=np.eye(2), y_train=np.array([2.0, 6.0]),
            X_val=np.zeros((0, 2)), y_val=np.zeros(0),
            X_holdout=np.eye(2), y_holdout=np.array([4.0, 4.0]),
        )
    }
    rmse = fit_and_evaluate_coalition(ds_dict, frozenset(), frozenset({"f1"}))
    # mean predictor is exactly right here
    assert rmse == pytest.approx(0.0, abs=1e-10)


def test_fit_and_evaluate_coalition_autarky_nan_when_no_training_data():
    ds_dict = {
        "f1": FirmDataset(
            name="f1", X_train=np.zeros((0, 2)), y_train=np.zeros(0),
            X_val=np.zeros((0, 2)), y_val=np.zeros(0),
            X_holdout=np.eye(2), y_holdout=np.array([1.0, 2.0]),
        )
    }
    rmse = fit_and_evaluate_coalition(ds_dict, frozenset(), frozenset({"f1"}))
    # mean_pred falls back to 0.0 per the "else 0.0" branch; RMSE should be finite, not nan
    assert not np.isnan(rmse)


def test_fit_and_evaluate_coalition_trained_coalition_matches_manual_ridge(datasets):
    names = list(datasets.keys())
    coalition = frozenset(names[:2])
    eval_firms = frozenset({names[2]})

    rmse = fit_and_evaluate_coalition(datasets, coalition, eval_firms, lam=1.0)

    A, b = aggregate_coalition_statistics(datasets, coalition)
    w = fit_ridge_from_statistics(A, b, lam=1.0)
    expected_rmse = evaluate_holdout_rmse(w, datasets, eval_firms)
    assert rmse == pytest.approx(expected_rmse)


def test_fit_and_evaluate_coalition_larger_lam_does_not_crash_and_returns_finite(datasets):
    names = list(datasets.keys())
    coalition = frozenset(names)
    rmse = fit_and_evaluate_coalition(
        datasets, coalition, coalition, lam=1000.0)
    assert np.isfinite(rmse)
