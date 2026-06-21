"""
Inventory simulator tests. Core strategy: use a DETERMINISTIC, perfectly
forecasted demand stream where the correct steady-state behavior is
analytically known (zero backorders, stable on-hand inventory at the
base-stock target), so bugs in the period-by-period accounting logic cannot
hide behind stochastic noise.
"""
import numpy as np
import pytest

from src.inventory import EchelonConfig, simulate_serial_supply_chain, default_three_echelon_config


def test_perfect_forecast_zero_variance_yields_zero_backorders():
    """
    Constant demand, perfect forecast, zero residual std, single echelon,
    service level 0.5 (z=0): the base-stock target exactly covers expected
    lead-time demand with no safety margin, and since demand is deterministic
    and perfectly forecasted, realized demand never exceeds the target ->
    backorders should be exactly zero after the first period.
    """
    T = 50
    demand = np.full(T, 10.0)
    forecast = np.full(T, 10.0)
    echelons = [EchelonConfig(name="single", lead_time=1, holding_cost=1.0, backorder_cost=5.0, service_level=0.5)]

    out = simulate_serial_supply_chain(demand, forecast, forecast_resid_std=0.0, echelons=echelons)

    assert np.all(out.backorder[0, 1:] == pytest.approx(0.0, abs=1e-8)), (
        "Deterministic, perfectly forecasted demand with z=0 safety factor "
        "should never produce backorders after warm-up."
    )
    assert out.fill_rate == pytest.approx(1.0, abs=1e-6)


def test_on_hand_and_backorder_never_simultaneously_positive():
    """
    Accounting invariant: an echelon cannot simultaneously hold positive
    on-hand inventory AND have an outstanding backorder at the same echelon
    in the same period (that would mean unmet demand sitting next to unused
    stock, a contradiction of FIFO fulfillment logic).
    """
    rng = np.random.default_rng(0)
    T = 80
    demand = np.clip(rng.normal(10, 4, size=T), 0, None)
    forecast = np.full(T, 10.0)
    echelons = default_three_echelon_config(service_level=0.9)

    out = simulate_serial_supply_chain(demand, forecast, forecast_resid_std=4.0, echelons=echelons)

    for e in range(len(echelons)):
        violation = np.logical_and(out.on_hand[e] > 1e-6, out.backorder[e] > 1e-6)
        assert not violation.any(), f"Echelon {e} has simultaneous positive on-hand and backorder."


def test_costs_are_nonnegative():
    rng = np.random.default_rng(1)
    T = 60
    demand = np.clip(rng.normal(15, 5, size=T), 0, None)
    forecast = np.full(T, 15.0)
    echelons = default_three_echelon_config()

    out = simulate_serial_supply_chain(demand, forecast, forecast_resid_std=5.0, echelons=echelons)
    assert np.all(out.holding_cost_series >= 0)
    assert np.all(out.backorder_cost_series >= 0)
    assert out.total_cost >= 0


def test_fill_rate_bounded():
    rng = np.random.default_rng(2)
    T = 100
    demand = np.clip(rng.normal(10, 8, size=T), 0, None)  # high variance, low safety stock -> stress test
    forecast = np.full(T, 10.0)
    echelons = [EchelonConfig(name="single", lead_time=2, holding_cost=1.0, backorder_cost=5.0, service_level=0.5)]

    out = simulate_serial_supply_chain(demand, forecast, forecast_resid_std=1.0, echelons=echelons)
    assert 0.0 <= out.fill_rate <= 1.0


def test_higher_forecast_uncertainty_increases_total_cost():
    """
    Holding all else fixed, increasing forecast_resid_std should (weakly)
    increase total system cost -- this is the central empirical claim linking
    forecast/data quality to downstream economic outcomes, and must hold on
    a controlled synthetic instance before we trust it on the FL pipeline.
    """
    rng = np.random.default_rng(3)
    T = 150
    demand = np.clip(rng.normal(20, 6, size=T), 0, None)
    forecast = np.full(T, 20.0)
    echelons = default_three_echelon_config(service_level=0.95)

    out_low_noise = simulate_serial_supply_chain(demand, forecast, forecast_resid_std=1.0, echelons=echelons)
    out_high_noise = simulate_serial_supply_chain(demand, forecast, forecast_resid_std=8.0, echelons=echelons)

    assert out_high_noise.total_cost > out_low_noise.total_cost, (
        f"Expected higher forecast uncertainty to increase cost: "
        f"low={out_low_noise.total_cost:.2f}, high={out_high_noise.total_cost:.2f}"
    )


def test_bullwhip_ratio_finite_and_nonnegative():
    rng = np.random.default_rng(4)
    T = 100
    demand = np.clip(rng.normal(12, 3, size=T), 0, None)
    forecast = np.full(T, 12.0)
    echelons = default_three_echelon_config()

    out = simulate_serial_supply_chain(demand, forecast, forecast_resid_std=3.0, echelons=echelons)
    assert np.isfinite(out.bullwhip_ratio)
    assert out.bullwhip_ratio >= 0
