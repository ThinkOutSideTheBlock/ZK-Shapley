"""
Multi-echelon base-stock inventory simulator: converts forecast quality into
an OPERATIONAL, dollar-denominated payoff.

Policy: order-up-to (base-stock), periodic review, fixed lead time L per
echelon. Base-stock level (standard discrete-time formula):

    S = mu * (L + 1) + z * sigma * sqrt(L + 1)

covering lead time plus the current review period. See Zipkin (2000).

Stage 0 fixes:
  - optional initial_on_hand for warm-start probes (Prop B″)
  - demand floored at 0
  - protection horizon L+1 (was L; caused chronic understocking)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class EchelonConfig:
    """Cost and service parameters for a single echelon in a serial supply chain."""
    holding_cost: float
    shortage_cost: float
    lead_time: int
    service_level: float = 0.95


@dataclass
class SimulationOutput:
    total_cost: float
    holding_cost: float
    shortage_cost: float
    fill_rate: float
    avg_inventory: float
    avg_backorder: float


def _service_level_to_z(service_level: float) -> float:
    from scipy.stats import norm
    service_level = float(np.clip(service_level, 1e-4, 1 - 1e-4))
    return float(norm.ppf(service_level))


def default_three_echelon_config() -> list[EchelonConfig]:
    return [
        EchelonConfig(holding_cost=1.0, shortage_cost=9.0,
                      lead_time=1, service_level=0.95),
        EchelonConfig(holding_cost=0.6, shortage_cost=0.0,
                      lead_time=2, service_level=0.90),
        EchelonConfig(holding_cost=0.3, shortage_cost=0.0,
                      lead_time=3, service_level=0.90),
    ]


def _base_stock_level(forecast_mean: float, residual_std: float, lead_time: int, z: float) -> float:
    """
    Order-up-to level for discrete-time base-stock with lead time L.

    Protects L+1 periods of demand (lead time + review period):
        mu_L    = (L + 1) * forecast_mean
        sigma_L = sqrt(L + 1) * residual_std
        S       = mu_L + z * sigma_L
    """
    L_prot = max(int(lead_time), 0) + 1
    mu_L = L_prot * max(forecast_mean, 0.0)
    sigma_L = np.sqrt(L_prot) * max(residual_std, 1e-6)
    return mu_L + z * sigma_L


def simulate_serial_supply_chain(
    demand_path: np.ndarray,
    forecast_path: np.ndarray,
    residual_std: float,
    echelons: list[EchelonConfig],
    rng: np.random.Generator | None = None,
    stochastic_lead_time: bool = False,
    initial_on_hand: float | None = None,
) -> SimulationOutput:
    if stochastic_lead_time and rng is None:
        raise ValueError(
            "stochastic_lead_time=True requires an explicit rng for reproducible "
            "common random numbers across coalitions."
        )

    if rng is None:
        rng = np.random.default_rng()

    T = len(demand_path)
    n_ech = len(echelons)

    total_holding = 0.0
    total_shortage = 0.0
    fulfilled = 0.0
    total_demand = 0.0
    inv_trace = np.zeros(n_ech)
    backorder_trace = np.zeros(n_ech)

    downstream_order_stream = np.asarray(demand_path, dtype=float).copy()

    for ech_idx, ech in enumerate(echelons):
        z = _service_level_to_z(ech.service_level)
        L = max(ech.lead_time, 1)
        pipe = [0.0] * L
        oh = float(initial_on_hand) if initial_on_hand is not None else 0.0
        bo = 0.0
        ech_holding, ech_shortage, ech_fulfilled, ech_demand = 0.0, 0.0, 0.0, 0.0

        realized_demand_here = downstream_order_stream
        next_order_stream = np.zeros(T)

        for t in range(T):
            arriving = pipe.pop(0)
            oh += arriving

            if ech_idx == 0:
                f_mean = float(forecast_path[t])
            else:
                if t > 0:
                    f_mean = float(
                        np.mean(realized_demand_here[max(0, t - 4):t]))
                else:
                    f_mean = float(realized_demand_here[0])

            target = _base_stock_level(
                f_mean,
                residual_std if ech_idx == 0 else max(residual_std, 1e-6),
                L,
                z,
            )

            in_pipeline = sum(pipe)
            order_qty = max(target - (oh - bo) - in_pipeline, 0.0)
            pipe.append(order_qty)
            next_order_stream[t] = order_qty

            d_t = max(float(realized_demand_here[t]), 0.0)
            ech_demand += d_t
            available = oh - bo
            if available >= d_t:
                oh -= d_t
                served = d_t
            else:
                served = max(available, 0.0)
                shortfall = d_t - served
                bo += shortfall
                oh = max(oh - served, 0.0)
            ech_fulfilled += served

            backlog_clear = min(bo, oh)
            oh -= backlog_clear
            bo -= backlog_clear

            ech_holding += ech.holding_cost * max(oh, 0.0)
            ech_shortage += ech.shortage_cost * max(bo, 0.0)
            inv_trace[ech_idx] += max(oh, 0.0)
            backorder_trace[ech_idx] += max(bo, 0.0)

        total_holding += ech_holding
        total_shortage += ech_shortage
        if ech_idx == 0:
            fulfilled += ech_fulfilled
            total_demand += ech_demand
        downstream_order_stream = next_order_stream

    total_cost = total_holding + total_shortage
    fill_rate = fulfilled / total_demand if total_demand > 0 else 1.0
    avg_inventory = float(np.sum(inv_trace) / (n_ech * T))
    avg_backorder = float(np.sum(backorder_trace) / (n_ech * T))

    return SimulationOutput(
        total_cost=float(total_cost),
        holding_cost=float(total_holding),
        shortage_cost=float(total_shortage),
        fill_rate=float(fill_rate),
        avg_inventory=avg_inventory,
        avg_backorder=avg_backorder,
    )
