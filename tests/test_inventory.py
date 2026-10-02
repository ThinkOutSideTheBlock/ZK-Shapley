# tests/test_inventory.py
"""
Tests for src/inventory.py: multi-echelon base-stock simulator.

Key structural property under test: with stochastic_lead_time=False (the
default and the only setting used anywhere in the paper), the simulator is
FULLY DETERMINISTIC given its non-rng arguments -- rng is not consumed at
all. This is asserted directly, not assumed, since every common-random-
numbers claim in mechanism.py rests on it.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.inventory import (
    EchelonConfig,
    SimulationOutput,
    default_three_echelon_config,
    simulate_serial_supply_chain,
    _service_level_to_z,
    _base_stock_level,
)


@pytest.fixture
def echelons():
    return default_three_echelon_config()


@pytest.fixture
def simple_demand_and_forecast():
    rng = np.random.default_rng(0)
    T = 60
    demand = np.maximum(rng.normal(10.0, 2.0, size=T), 0.0)
    forecast = demand + rng.normal(0.0, 0.5, size=T)
    return demand, forecast


# ---------------------------------------------------------------------------
# _service_level_to_z
# ---------------------------------------------------------------------------

def test_service_level_to_z_monotonic_in_service_level():
    z_low = _service_level_to_z(0.5)
    z_mid = _service_level_to_z(0.9)
    z_high = _service_level_to_z(0.99)
    assert z_low < z_mid < z_high


def test_service_level_to_z_at_50_percent_is_approximately_zero():
    assert _service_level_to_z(0.5) == pytest.approx(0.0, abs=1e-9)


def test_service_level_to_z_clips_extreme_inputs_without_crashing():
    z_low = _service_level_to_z(-5.0)
    z_high = _service_level_to_z(5.0)
    assert np.isfinite(z_low)
    assert np.isfinite(z_high)


# ---------------------------------------------------------------------------
# _base_stock_level
# ---------------------------------------------------------------------------

def test_base_stock_level_matches_formula():
    level = _base_stock_level(
        forecast_mean=10.0, residual_std=2.0, lead_time=3, z=1.5
    )
    # L_prot = lead_time + 1 = 4
    expected = 4 * 10.0 + 1.5 * np.sqrt(4) * 2.0  # 40 + 1.5*2*2 = 46
    assert level == pytest.approx(expected)


def test_base_stock_level_floors_negative_forecast_mean_to_zero():
    level = _base_stock_level(
        forecast_mean=-5.0, residual_std=2.0, lead_time=2, z=1.0
    )
    # mu floored at 0; L_prot = 3
    expected = 0.0 + 1.0 * np.sqrt(3) * 2.0
    assert level == pytest.approx(expected)


def test_base_stock_level_floors_residual_std_at_tiny_positive_value():
    level = _base_stock_level(
        forecast_mean=5.0, residual_std=-1.0, lead_time=1, z=1.0
    )
    # residual_std floored at 1e-6; L_prot = 2
    expected = 2 * 5.0 + 1.0 * np.sqrt(2) * 1e-6
    assert level == pytest.approx(expected, abs=1e-8)

# ---------------------------------------------------------------------------
# default_three_echelon_config
# ---------------------------------------------------------------------------

def test_default_three_echelon_config_has_three_echelons_in_downstream_first_order():
    echelons = default_three_echelon_config()
    assert len(echelons) == 3
    # most downstream, shortest lead time by convention here
    assert echelons[0].lead_time == 1
    # customer-facing echelon has real shortage cost
    assert echelons[0].shortage_cost > 0.0
    assert echelons[1].shortage_cost == 0.0
    assert echelons[2].shortage_cost == 0.0


# ---------------------------------------------------------------------------
# simulate_serial_supply_chain: determinism / CRN properties
# ---------------------------------------------------------------------------

def test_simulation_is_deterministic_regardless_of_rng_when_stochastic_lead_time_false(
    echelons, simple_demand_and_forecast
):
    demand, forecast = simple_demand_and_forecast
    out1 = simulate_serial_supply_chain(
        demand,
        forecast,
        residual_std=0.5,
        echelons=echelons,
        rng=np.random.default_rng(1),
        stochastic_lead_time=False,
    )
    out2 = simulate_serial_supply_chain(
        demand,
        forecast,
        residual_std=0.5,
        echelons=echelons,
        rng=np.random.default_rng(999),
        stochastic_lead_time=False,
    )
    assert out1.total_cost == pytest.approx(out2.total_cost)
    assert out1.holding_cost == pytest.approx(out2.holding_cost)
    assert out1.shortage_cost == pytest.approx(out2.shortage_cost)
    assert out1.fill_rate == pytest.approx(out2.fill_rate)


def test_simulation_deterministic_even_with_no_rng_supplied(
    echelons, simple_demand_and_forecast
):
    demand, forecast = simple_demand_and_forecast
    out1 = simulate_serial_supply_chain(demand, forecast, 0.5, echelons)
    out2 = simulate_serial_supply_chain(demand, forecast, 0.5, echelons)
    assert out1.total_cost == pytest.approx(out2.total_cost)


def test_stochastic_lead_time_without_rng_raises(echelons, simple_demand_and_forecast):
    demand, forecast = simple_demand_and_forecast
    with pytest.raises(ValueError):
        simulate_serial_supply_chain(
            demand,
            forecast,
            0.5,
            echelons,
            rng=None,
            stochastic_lead_time=True,
        )


# ---------------------------------------------------------------------------
# Cost/fill-rate structural properties
# ---------------------------------------------------------------------------

def test_total_cost_equals_holding_plus_shortage(echelons, simple_demand_and_forecast):
    demand, forecast = simple_demand_and_forecast
    out = simulate_serial_supply_chain(demand, forecast, 0.5, echelons)
    assert out.total_cost == pytest.approx(
        out.holding_cost + out.shortage_cost)


def test_fill_rate_is_bounded_between_zero_and_one(echelons, simple_demand_and_forecast):
    demand, forecast = simple_demand_and_forecast
    out = simulate_serial_supply_chain(demand, forecast, 0.5, echelons)
    assert 0.0 <= out.fill_rate <= 1.0 + 1e-9


def test_fill_rate_is_one_when_demand_is_identically_zero(echelons):
    T = 30
    demand = np.zeros(T)
    forecast = np.zeros(T)
    out = simulate_serial_supply_chain(demand, forecast, 0.1, echelons)
    assert out.fill_rate == pytest.approx(1.0)


def test_higher_residual_std_increases_safety_stock_driven_holding_cost(echelons):
    """
    Holding demand and forecast fixed, larger residual_std only increases the
    base-stock safety term z * sigma * sqrt(L). That raises on-hand inventory
    and therefore holding cost.

    Do NOT assert total_cost monotonicity: with high shortage penalty, extra
    safety stock can cut shortage cost by more than it adds holding cost, so
    total_cost can fall (observed empirically on default echelon params).
    """
    rng = np.random.default_rng(5)
    T = 80
    demand = np.maximum(rng.normal(15.0, 3.0, size=T), 0.0)
    # perfect mean forecast so residual_std is the only safety-stock driver
    forecast = demand.copy()

    out_tight = simulate_serial_supply_chain(
        demand, forecast, residual_std=0.5, echelons=echelons
    )
    out_loose = simulate_serial_supply_chain(
        demand, forecast, residual_std=5.0, echelons=echelons
    )

    assert out_loose.holding_cost >= out_tight.holding_cost - 1e-6
    assert out_loose.avg_inventory >= out_tight.avg_inventory - 1e-6
    # operational channel: fill rate should not collapse when safety stock rises
    assert out_loose.fill_rate >= out_tight.fill_rate - 1e-9


def test_perfect_forecast_does_not_increase_cost_vs_noisy_forecast(echelons):
    """
    Compare perfect vs noisy forecast under a *common* residual_std calibrated
    to the noisy errors. Isolates forecast-path quality from the separate
    residual_std knob.
    """
    rng = np.random.default_rng(3)
    T = 200
    demand = np.maximum(rng.normal(20.0, 4.0, size=T), 0.0)
    perfect_forecast = demand.copy()
    noise = rng.normal(0.0, 8.0, size=T)
    noisy_forecast = np.maximum(demand + noise, 0.0)

    # common residual calibration: use empirical std of the noisy residual
    common_resid = float(np.std(demand - noisy_forecast))
    common_resid = max(common_resid, 1e-6)

    out_perfect = simulate_serial_supply_chain(
        demand, perfect_forecast, residual_std=common_resid, echelons=echelons
    )
    out_noisy = simulate_serial_supply_chain(
        demand, noisy_forecast, residual_std=common_resid, echelons=echelons
    )

    # Perfect forecast should not be operationally worse on the same residual scale.
    assert out_perfect.total_cost <= out_noisy.total_cost + 1e-6


def test_simulation_output_fields_are_all_finite(echelons, simple_demand_and_forecast):
    demand, forecast = simple_demand_and_forecast
    out = simulate_serial_supply_chain(demand, forecast, 0.5, echelons)
    for field_val in (
        out.total_cost,
        out.holding_cost,
        out.shortage_cost,
        out.fill_rate,
        out.avg_inventory,
        out.avg_backorder,
    ):
        assert np.isfinite(field_val)


def test_simulation_handles_single_period_horizon(echelons):
    demand = np.array([5.0])
    forecast = np.array([5.0])
    out = simulate_serial_supply_chain(demand, forecast, 0.2, echelons)
    assert np.isfinite(out.total_cost)


def test_custom_single_echelon_config_runs():
    echelons = [
        EchelonConfig(
            holding_cost=1.0,
            shortage_cost=5.0,
            lead_time=1,
            service_level=0.9,
        )
    ]
    rng = np.random.default_rng(0)
    demand = np.maximum(rng.normal(10.0, 2.0, size=40), 0.0)
    forecast = demand.copy()
    out = simulate_serial_supply_chain(demand, forecast, 1.0, echelons)
    assert isinstance(out, SimulationOutput)
    assert out.total_cost >= 0.0
