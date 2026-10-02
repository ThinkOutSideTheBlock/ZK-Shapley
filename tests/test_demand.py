# tests/test_demand.py
from __future__ import annotations

import numpy as np
import pytest

from src.demand import (
    DemandSimulationConfig, generate_multi_firm_demand,
)


def test_generate_multi_firm_demand_produces_correct_firm_count_and_shapes():
    config = DemandSimulationConfig(
        n_firms=5, n_periods=200, n_features=6, seed=0)
    result = generate_multi_firm_demand(config)
    assert len(result.firms) == 5
    assert result.beta_true.shape == (6,)
    assert result.z.shape == (200,)
    for name, raw in result.firms.items():
        assert raw.X.shape[1] == 6
        assert raw.X.shape[0] == raw.n_obs
        assert raw.y.shape[0] == raw.n_obs
        assert raw.n_obs <= 200


def test_generate_multi_firm_demand_is_deterministic_given_seed():
    config = DemandSimulationConfig(n_firms=4, n_periods=100, seed=42)
    r1 = generate_multi_firm_demand(config)
    r2 = generate_multi_firm_demand(config)
    np.testing.assert_array_equal(r1.beta_true, r2.beta_true)
    np.testing.assert_array_equal(r1.z, r2.z)
    for name in r1.firms:
        np.testing.assert_array_equal(r1.firms[name].X, r2.firms[name].X)
        np.testing.assert_array_equal(r1.firms[name].y, r2.firms[name].y)


def test_generate_multi_firm_demand_different_seeds_differ():
    config_a = DemandSimulationConfig(n_firms=4, n_periods=100, seed=1)
    config_b = DemandSimulationConfig(n_firms=4, n_periods=100, seed=2)
    r1 = generate_multi_firm_demand(config_a)
    r2 = generate_multi_firm_demand(config_b)
    assert not np.array_equal(r1.z, r2.z)


def test_all_response_values_are_nonnegative():
    """The np.maximum(mean_demand + eps, 0.0) floor must always hold."""
    config = DemandSimulationConfig(
        n_firms=6, n_periods=150, base_noise_std=3.0, seed=7)
    result = generate_multi_firm_demand(config)
    for raw in result.firms.values():
        assert np.all(raw.y >= 0.0)


def test_intercept_column_is_constant_one():
    config = DemandSimulationConfig(
        n_firms=3, n_periods=50, n_features=5, seed=3)
    result = generate_multi_firm_demand(config)
    for raw in result.firms.values():
        np.testing.assert_array_equal(raw.X[:, 0], np.ones(raw.n_obs))


def test_beta_true_first_coefficient_is_fixed_scale():
    config = DemandSimulationConfig(
        n_firms=3, n_periods=50, beta_scale=2.0, seed=3)
    result = generate_multi_firm_demand(config)
    assert result.beta_true[0] == pytest.approx(10.0)  # beta_scale * 5.0


def test_common_factor_is_standardized():
    config = DemandSimulationConfig(
        n_firms=3, n_periods=500, shared_factor_correlation=0.7, seed=5)
    result = generate_multi_firm_demand(config)
    assert result.z.mean() == pytest.approx(0.0, abs=1e-8)
    assert result.z.std() == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("profile,expected_common,expected_outlier", [
    ("homogeneous", 0.6, None),
    ("one_high_quality", 0.4, 1.2),
    ("one_noisy", 0.6, 0.1),
])
def test_signal_quality_profiles_produce_expected_gamma_values(profile, expected_common, expected_outlier):
    config = DemandSimulationConfig(
        n_firms=6, n_periods=60, signal_quality_profile=profile, seed=9)
    result = generate_multi_firm_demand(config)
    gammas = sorted(raw.gamma for raw in result.firms.values())
    if expected_outlier is None:
        assert all(g == pytest.approx(expected_common) for g in gammas)
    else:
        n_outliers = sum(1 for g in gammas if g ==
                         pytest.approx(expected_outlier))
        n_common = sum(1 for g in gammas if g ==
                       pytest.approx(expected_common))
        assert n_outliers == 1
        assert n_common == 5


@pytest.mark.parametrize("profile", ["balanced", "moderate", "severe"])
def test_n_obs_heterogeneity_profiles_respect_floor_of_30(profile):
    config = DemandSimulationConfig(
        n_firms=8, n_periods=1000, base_n_obs=300, n_obs_heterogeneity=profile, seed=11,
    )
    result = generate_multi_firm_demand(config)
    assert all(raw.n_obs >= 30 for raw in result.firms.values())


def test_balanced_profile_gives_identical_n_obs_across_firms():
    config = DemandSimulationConfig(
        n_firms=5, n_periods=500, base_n_obs=200, n_obs_heterogeneity="balanced", seed=0,
    )
    result = generate_multi_firm_demand(config)
    n_obs_vals = [raw.n_obs for raw in result.firms.values()]
    assert len(set(n_obs_vals)) == 1


def test_severe_profile_produces_wider_spread_than_moderate_on_average():
    """
    Not a single-seed coin-flip claim: averages the observed n_obs range
    over several independent seeds so the comparison reflects the DGP's
    designed uniform(0.15, 3.0) vs uniform(0.6, 1.6) multiplier spread
    rather than one lucky/unlucky draw.
    """
    def avg_range(profile, seeds):
        ranges = []
        for s in seeds:
            config = DemandSimulationConfig(
                n_firms=10, n_periods=2000, base_n_obs=300,
                n_obs_heterogeneity=profile, seed=s,
            )
            result = generate_multi_firm_demand(config)
            vals = [raw.n_obs for raw in result.firms.values()]
            ranges.append(max(vals) - min(vals))
        return np.mean(ranges)

    seeds = list(range(10))
    moderate_avg = avg_range("moderate", seeds)
    severe_avg = avg_range("severe", seeds)
    assert severe_avg > moderate_avg


def test_unknown_n_obs_heterogeneity_profile_raises():
    config = DemandSimulationConfig(
        n_firms=3, n_periods=50, n_obs_heterogeneity="bogus")
    with pytest.raises(ValueError):
        generate_multi_firm_demand(config)


def test_unknown_signal_quality_profile_raises():
    config = DemandSimulationConfig(
        n_firms=3, n_periods=50, signal_quality_profile="bogus")
    with pytest.raises(ValueError):
        generate_multi_firm_demand(config)


def test_sigma_draws_are_within_specified_multiplicative_range():
    config = DemandSimulationConfig(
        n_firms=20, n_periods=50, base_noise_std=2.0, seed=13)
    result = generate_multi_firm_demand(config)
    sigmas = [raw.sigma for raw in result.firms.values()]
    assert all(2.0 * 0.7 - 1e-9 <= s <= 2.0 * 1.5 + 1e-9 for s in sigmas)


def test_feature_matrix_has_no_nans_or_infs():
    config = DemandSimulationConfig(
        n_firms=4, n_periods=300, n_features=10, seed=17)
    result = generate_multi_firm_demand(config)
    for raw in result.firms.values():
        assert np.all(np.isfinite(raw.X))
        assert np.all(np.isfinite(raw.y))


def test_n_periods_caps_n_obs_even_under_severe_heterogeneity():
    """
    _draw_n_obs can draw counts above n_periods under the 'severe' profile
    (uniform up to 3.0x base_n_obs); generate_multi_firm_demand must clip
    n_obs_i to min(n_obs_vec[i], T) so no firm's X/y arrays exceed the
    generated series length T.
    """
    config = DemandSimulationConfig(
        n_firms=6, n_periods=100, base_n_obs=300, n_obs_heterogeneity="severe", seed=21,
    )
    result = generate_multi_firm_demand(config)
    for raw in result.firms.values():
        assert raw.n_obs <= 100
        assert raw.X.shape[0] <= 100
