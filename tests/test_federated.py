"""
Critical correctness test: federated sufficient-statistics aggregation must be
EXACTLY mathematically equivalent to centralized ridge regression on pooled
raw data. This is the load-bearing claim of the entire FL module -- if this
test fails, every downstream Shapley/mechanism result is meaningless.
"""
import numpy as np
import pytest

from src.demand import default_firm_configs, simulate_multi_firm_demand
from src.federated import (
    build_firm_datasets, sufficient_statistics, fit_ridge_from_statistics,
    aggregate_coalition_statistics, fit_and_evaluate_coalition, N_FEATURES,
)


@pytest.fixture
def datasets():
    firms = default_firm_configs(n_firms=4, seed=1)
    sim = simulate_multi_firm_demand(firms, seed=7)
    return build_firm_datasets(sim, holdout_fraction=0.2)


def test_federated_equals_centralized_ridge(datasets):
    """
    Pool ALL firms' raw training data manually, fit centralized ridge
    regression, and confirm the federated (sufficient-statistics-summed)
    weight vector is numerically identical.
    """
    lam = 1.0
    names = list(datasets.keys())
    coalition = frozenset(names)

    # Centralized: manually concatenate raw X, y across all firms.
    X_pooled = np.concatenate([datasets[n].X_train for n in names], axis=0)
    y_pooled = np.concatenate([datasets[n].y_train for n in names], axis=0)
    w_centralized = np.linalg.solve(
        X_pooled.T @ X_pooled + lam * np.eye(N_FEATURES), X_pooled.T @ y_pooled
    )

    # Federated: sum sufficient statistics, then solve.
    A, b = aggregate_coalition_statistics(datasets, coalition)
    w_federated = fit_ridge_from_statistics(A, b, lam=lam)

    np.testing.assert_allclose(w_federated, w_centralized, rtol=1e-10, atol=1e-10)


def test_single_firm_statistics_match_direct_fit(datasets):
    """Sufficient statistics for a single firm, fed through the federated path,
    must match directly fitting ridge regression on that firm's own data."""
    lam = 2.5
    name = list(datasets.keys())[0]
    ds = datasets[name]

    A, b = sufficient_statistics(ds)
    w_via_stats = fit_ridge_from_statistics(A, b, lam=lam)

    w_direct = np.linalg.solve(ds.X_train.T @ ds.X_train + lam * np.eye(N_FEATURES), ds.X_train.T @ ds.y_train)
    np.testing.assert_allclose(w_via_stats, w_direct, rtol=1e-10, atol=1e-10)


def test_aggregation_is_associative(datasets):
    """A_{S1 u S2} = A_{S1} + A_{S2} for disjoint S1, S2 -- the additive
    separability property the entire FL-without-raw-data-sharing design relies on."""
    names = list(datasets.keys())
    S1, S2 = frozenset(names[:2]), frozenset(names[2:])
    A1, b1 = aggregate_coalition_statistics(datasets, S1)
    A2, b2 = aggregate_coalition_statistics(datasets, S2)
    A_union, b_union = aggregate_coalition_statistics(datasets, S1 | S2)

    np.testing.assert_allclose(A1 + A2, A_union, rtol=1e-12)
    np.testing.assert_allclose(b1 + b2, b_union, rtol=1e-12)


def test_empty_coalition_returns_finite_rmse(datasets):
    """The autarky (empty-coalition) baseline must be well-defined and finite,
    since it anchors v(emptyset) = 0 for the Shapley game."""
    all_firms = frozenset(datasets.keys())
    rmse = fit_and_evaluate_coalition(datasets, frozenset(), all_firms, lam=1.0)
    assert np.isfinite(rmse)
    assert rmse > 0


def test_more_data_improves_or_maintains_grand_coalition_fit(datasets):
    """
    Sanity check (not a strict theorem -- finite-sample noise can violate
    pointwise monotonicity, as documented in mechanism.py): on this seeded
    instance, the grand coalition should outperform the empty-coalition
    baseline by a non-trivial margin, confirming the simulated data actually
    contains exploitable shared signal (a degenerate all-noise dataset would
    make the entire Shapley analysis vacuous).
    """
    all_firms = frozenset(datasets.keys())
    rmse_empty = fit_and_evaluate_coalition(datasets, frozenset(), all_firms, lam=1.0)
    rmse_grand = fit_and_evaluate_coalition(datasets, all_firms, all_firms, lam=1.0)
    assert rmse_grand < rmse_empty, (
        f"Grand coalition RMSE ({rmse_grand:.4f}) should beat autarky RMSE "
        f"({rmse_empty:.4f}); demand generator may lack shared signal."
    )


def test_reproducibility(datasets):
    """Same seed must produce bit-identical demand series."""
    firms_a = default_firm_configs(n_firms=4, seed=1)
    sim_a = simulate_multi_firm_demand(firms_a, seed=7)
    firms_b = default_firm_configs(n_firms=4, seed=1)
    sim_b = simulate_multi_firm_demand(firms_b, seed=7)
    for name in sim_a.series:
        np.testing.assert_array_equal(sim_a.series[name], sim_b.series[name])
