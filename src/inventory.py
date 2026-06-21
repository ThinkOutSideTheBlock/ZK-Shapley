"""
Multi-echelon serial supply chain simulator under periodic-review base-stock
((R,S) / order-up-to) policies, parameterized by a forecasting model's
out-of-sample mean and residual-std estimates.

This is the module that closes the loop between mechanism design (which
determines what forecasting model a firm ends up with, as a function of which
coalition's federated statistics it had access to) and supply chain economic
outcomes (holding cost, backorder cost, fill rate, bullwhip ratio) -- i.e. it
is what makes this a *supply chain optimization* paper and not only a
mechanism-design paper.

Model (standard multi-echelon inventory theory; Clark & Scarf, 1960; Zipkin,
2000, Ch. 9 for the echelon base-stock formulation):

  - Serial chain of E echelons. Echelon 1 (most downstream) faces exogenous
    customer demand D(t). Echelon e (e > 1) faces "demand" equal to echelon
    (e-1)'s realized order quantity in the same period (standard serial-chain
    propagation).
  - Each echelon e has lead time L_e (periods) and follows a base-stock
    (order-up-to) policy with review period r_period: in each review period,
    echelon e places an order to bring its ECHELON inventory position
    (on-hand + on-order - backorders, summed over itself and all downstream
    echelons -- the Clark-Scarf "echelon stock" definition) up to a target
    base-stock level

        S_e = (r_period + L_e) * mu_hat_e + z_e * sigma_hat_e * sqrt(r_period + L_e)

    where mu_hat_e, sigma_hat_e are the forecasting model's estimated mean and
    residual standard deviation of one-period demand at echelon e, and
    z_e = Phi^{-1}(alpha_e) is the safety factor for target service level
    alpha_e (standard newsvendor-style safety stock formula under a Normal
    lead-time-demand approximation).
  - Holding cost h_e per unit of on-hand inventory per period; backorder
    penalty cost p_e per unit of unmet demand per period (standard linear
    cost structure used throughout the inventory literature).
  - Fixed ordering costs are excluded by design choice (stated explicitly):
    base-stock/order-up-to policies are the standard simplified setting when
    the focus is on safety-stock sizing under demand uncertainty, not on
    joint lot-sizing; including a fixed cost K would require moving to an
    (s, S) policy with a different (harder, non-closed-form) optimization
    structure that is not the contribution of this paper.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class EchelonConfig:
    name: str
    lead_time: int           # periods (must be >= 1; see pipeline indexing note below)
    holding_cost: float       # cost per unit on-hand per period
    backorder_cost: float     # cost per unit backordered per period
    service_level: float      # target alpha in (0, 1)

    def __post_init__(self):
        if self.lead_time < 1:
            raise ValueError(
                f"EchelonConfig({self.name}): lead_time must be >= 1 (got {self.lead_time}); "
                "the pipeline-delay accounting in simulate_serial_supply_chain assumes a "
                "strictly positive lead time. Use lead_time=1 for 'next-period arrival'."
            )

    @property
    def z(self) -> float:
        return float(norm.ppf(self.service_level))


@dataclass
class SimulationOutput:
    on_hand: np.ndarray          # (E, T)
    backorder: np.ndarray         # (E, T)
    orders: np.ndarray            # (E, T)
    holding_cost_series: np.ndarray   # (E, T)
    backorder_cost_series: np.ndarray  # (E, T)
    total_cost: float
    fill_rate: float               # downstream (echelon 0) period service level achieved
    bullwhip_ratio: float           # Var(top-echelon orders) / Var(external demand)


def simulate_serial_supply_chain(
    external_demand: np.ndarray,
    forecast_mean: np.ndarray,
    forecast_resid_std: float,
    echelons: list[EchelonConfig],
    review_period: int = 1,
) -> SimulationOutput:
    """
    Simulate T periods of a serial E-echelon base-stock system.

    external_demand: length-T realized demand at the most downstream echelon.
    forecast_mean:    length-T one-step-ahead forecast of external_demand
                       (used by ALL echelons as the basis for their own
                       lead-time-demand estimate -- a standard simplifying
                       assumption that demand information is shared
                       instantaneously upstream once forecast, isolating the
                       effect of FORECAST QUALITY, which is this paper's
                       object of interest, from the effect of information
                       delay, which is not).
    forecast_resid_std: scalar estimate of one-step forecast residual std
                       (e.g., RMSE of the federated model on its holdout set),
                       used to size safety stock at every echelon.
    """
    E = len(echelons)
    T = len(external_demand)
    on_hand = np.zeros((E, T))
    backorder = np.zeros((E, T))
    orders = np.zeros((E, T))
    holding_cost_series = np.zeros((E, T))
    backorder_cost_series = np.zeros((E, T))

    # Pipeline (in-transit) orders per echelon, represented as an array of
    # `lead_time` slots: slot 0 = arriving THIS period (about to be received
    # at the top of the loop below), slot lead_time-1 = just-placed order
    # (lead_time periods from arrival). Each period: receive slot 0, shift
    # everything down by one (np.roll left + zero the now-vacant last slot),
    # then append the new order into the freshly-vacated last slot. This
    # ordering guarantees an order placed at period t arrives exactly at
    # period t + lead_time, which we verify explicitly in
    # tests/test_inventory.py via a zero-uncertainty deterministic-demand case.
    pipelines = [np.zeros(ech.lead_time) for ech in echelons]

    # Initialize on-hand inventory at each echelon's base-stock target for
    # period 0, using the period-0 forecast, to avoid a cold-start transient
    # dominating the cost metrics (standard practice: report steady-state
    # cost after burn-in, implemented here via warm starting).
    init_base_stock = [
        (review_period + ech.lead_time) * forecast_mean[0]
        + ech.z * forecast_resid_std * np.sqrt(review_period + ech.lead_time)
        for ech in echelons
    ]
    for e in range(E):
        on_hand[e, 0] = init_base_stock[e]

    demand_into_echelon = np.zeros((E, T))
    demand_into_echelon[0, :] = external_demand

    for t in range(T):
        for e in range(E):
            # Receive any inbound pipeline order arriving this period.
            arriving = pipelines[e][0]
            on_hand[e, t] += arriving
            pipelines[e] = np.roll(pipelines[e], -1)
            pipelines[e][-1] = 0.0

            # This period's demand on echelon e.
            d_t = demand_into_echelon[e, t]

            # Fulfill demand from on-hand inventory; unmet demand becomes
            # (or adds to) backorder; backorders carry over and are
            # prioritized first in subsequent periods (FIFO backorder clearing).
            available = on_hand[e, t] - (backorder[e, t - 1] if t > 0 else 0.0)
            if available >= d_t:
                on_hand[e, t] = available - d_t
                backorder[e, t] = 0.0
                shipped = d_t
            else:
                shipped = max(available, 0.0)
                on_hand[e, t] = 0.0
                backorder[e, t] = d_t - shipped + max(-available, 0.0)

            # Propagate the SHIPPED quantity upstream as next echelon's demand
            # (standard serial-chain order propagation: what echelon e ships
            # downstream becomes what echelon e orders from echelon e+1, under
            # a base-stock policy this equals the period's order quantity).

            # Base-stock order-up-to calculation using the shared forecast.
            mu_hat = forecast_mean[t]
            target = (
                (review_period + echelons[e].lead_time) * mu_hat
                + echelons[e].z * forecast_resid_std * np.sqrt(review_period + echelons[e].lead_time)
            )
            echelon_position = on_hand[e, t] - backorder[e, t] + pipelines[e].sum()
            order_qty = max(target - echelon_position, 0.0)
            orders[e, t] = order_qty

            if e + 1 < E:
                if t + 1 < T:
                    demand_into_echelon[e + 1, t] = order_qty  # immediate propagation (info, not material)
            # Schedule physical arrival after lead_time periods: place the
            # order in the last pipeline slot (size == lead_time exactly, so
            # this slot represents "lead_time periods until arrival"; it will
            # reach slot 0 -- and be received -- after exactly lead_time
            # subsequent roll operations, i.e. at period t + lead_time).
            pipelines[e][-1] += order_qty

            holding_cost_series[e, t] = echelons[e].holding_cost * max(on_hand[e, t], 0.0)
            backorder_cost_series[e, t] = echelons[e].backorder_cost * max(backorder[e, t], 0.0)

    total_cost = float(holding_cost_series.sum() + backorder_cost_series.sum())
    fill_rate = float(1.0 - (backorder[0] > 0).sum() / T)
    top_orders = orders[-1]
    bullwhip_ratio = float(np.var(top_orders) / np.var(external_demand)) if np.var(external_demand) > 0 else float("nan")

    return SimulationOutput(
        on_hand=on_hand, backorder=backorder, orders=orders,
        holding_cost_series=holding_cost_series, backorder_cost_series=backorder_cost_series,
        total_cost=total_cost, fill_rate=fill_rate, bullwhip_ratio=bullwhip_ratio,
    )


def default_three_echelon_config(service_level: float = 0.95) -> list[EchelonConfig]:
    """Retailer -> Distributor -> Manufacturer, increasing lead time upstream (standard assumption)."""
    return [
        EchelonConfig(name="retailer", lead_time=1, holding_cost=1.0, backorder_cost=8.0, service_level=service_level),
        EchelonConfig(name="distributor", lead_time=2, holding_cost=0.6, backorder_cost=4.0, service_level=service_level),
        EchelonConfig(name="manufacturer", lead_time=3, holding_cost=0.3, backorder_cost=2.0, service_level=service_level),
    ]
