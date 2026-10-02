"""
Phase-1 Gate 1 diagnostic — Option 1 (fixed λ)

Purpose: isolate whether firm-specific predictive residual recovers
alignment under homogeneous costs, without size-scaled λ confounding.

After this diagnostic we will restore a calibrated size-scaled (or CV) λ.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.shapley import CachedGame, exact_shapley
from src.mechanism import compute_divergence_metrics
from src.inventory import simulate_serial_supply_chain
from experiments._harness import run_sweep, ScenarioResult


# ---------------------------------------------------------------------------
# Inventory cost with K replications (CRN across the K draws)
# ---------------------------------------------------------------------------

def _mean_inventory_cost(
    demand: np.ndarray,
    forecast: np.ndarray,
    residual_std: float,
    echelons,
    base_seed: int,
    K: int = 50,
) -> float:
    """Average total cost over K independent noise draws."""
    costs = []
    for k in range(K):
        rng = np.random.default_rng([base_seed, k])
        out = simulate_serial_supply_chain(
            demand, forecast, residual_std, echelons, rng=rng
        )
        costs.append(out.total_cost)
    return float(np.mean(costs))


def negative_control_scenario(cfg: dict, seed: int) -> ScenarioResult:
    n_firms = int(cfg.get("n_firms", 4))
    rho = float(cfg.get("shared_factor_correlation", 0.6))
    n_periods = int(cfg.get("n_periods", 500))
    n_features = int(cfg.get("n_features", 5))
    cb_over_ch = float(cfg.get("cb_over_ch", 9.0))
    lam0 = float(cfg.get("ridge_lambda", 1.0))
    K = int(cfg.get("n_inventory_reps", 50))

    dem_cfg = DemandSimulationConfig(
        n_firms=n_firms,
        n_periods=n_periods,
        n_features=n_features,
        shared_factor_correlation=rho,
        seed=seed,
        n_obs_heterogeneity=cfg.get("n_obs_heterogeneity", "moderate"),
    )
    sim = generate_multi_firm_demand(dem_cfg)
    datasets = build_firm_datasets(
        sim, val_fraction=0.15, holdout_fraction=0.30, min_holdout=50
    )
    demand_paths = {n: ds.y_holdout for n, ds in datasets.items()}
    firms = tuple(sorted(datasets.keys()))

    # Homogeneous costs (κ = 1). Fixed λ for this diagnostic only.
    coal_cfg = CoalitionConfig(
        firms=firms,
        holding_costs=(1.0, 0.6, 0.3),
        shortage_costs=(cb_over_ch, 0.0, 0.0),
        lead_times=(1, 2, 3),
        service_levels=(0.95, 0.90, 0.90),
        ridge_lambda=lam0,
        lambda_mode="fixed",          # <-- Option 1 diagnostic
        residual_mode="predictive",   # firm-specific validation residual
        seed=seed,
    )
    evaluator = CoalitionEvaluator(
        datasets=datasets,
        demand_paths=demand_paths,
        config=coal_cfg,
        cache_dir=None,
    )

    # ---- Override cost evaluation with K-replication average ----
    # We still use evaluator.fit / residual_std; only the inventory average changes.
    from src.federated_phase1_patch import firm_predictive_residual_std

    echelons = coal_cfg.to_echelons()
    autarky_cost: dict[str, float] = {}

    def _autarky(name: str) -> float:
        if name in autarky_cost:
            return autarky_cost[name]
        ds = datasets[name]
        beta = evaluator.fit(frozenset({name}))
        forecast = ds.X_holdout @ beta
        resid_std = firm_predictive_residual_std(beta, ds)
        demand = demand_paths[name][-ds.X_holdout.shape[0]:]
        c = _mean_inventory_cost(
            demand, forecast, resid_std, echelons,
            base_seed=seed * 1009 + hash(name) % 10_000,
            K=K,
        )
        autarky_cost[name] = c
        return c

    def v_op(S: frozenset[str]) -> float:
        if not S:
            return 0.0
        beta = evaluator.fit(S)
        total = 0.0
        for name in S:
            ds = datasets[name]
            forecast = ds.X_holdout @ beta
            resid_std = firm_predictive_residual_std(beta, ds)
            demand = demand_paths[name][-ds.X_holdout.shape[0]:]
            c_S = _mean_inventory_cost(
                demand, forecast, resid_std, echelons,
                base_seed=seed * 1009 + hash(frozenset(S) | {name}) % 10_000,
                K=K,
            )
            total += _autarky(name) - c_S
        return total

    def v_acc(S: frozenset[str]) -> float:
        return evaluator.value_accuracy(S)

    game_op = CachedGame(players=list(firms), v_func=v_op)
    game_acc = CachedGame(players=list(firms), v_func=v_acc)

    phi_op = exact_shapley(game_op)
    phi_acc = exact_shapley(game_acc)
    v_op_N = game_op.v(frozenset(firms))
    v_acc_N = game_acc.v(frozenset(firms))
    div = compute_divergence_metrics(phi_op, phi_acc)

    result = ScenarioResult()
    for firm in firms:
        result.add(firm=firm, metric="phi_operational",
                   value=float(phi_op[firm]))
        result.add(firm=firm, metric="phi_accuracy",
                   value=float(phi_acc[firm]))
        result.add(firm=firm, metric="ir_operational",
                   value=float(phi_op[firm] >= -1e-9))
        result.add(firm=firm, metric="ir_accuracy",
                   value=float(phi_acc[firm] >= -1e-9))

    result.add(firm="_summary", metric="v_op_N", value=float(v_op_N))
    result.add(firm="_summary", metric="v_acc_N", value=float(v_acc_N))
    result.add(firm="_summary", metric="spearman_rho",
               value=float(div["spearman_rho"]))
    result.add(
        firm="_summary",
        metric="payment_displacement",
        value=float(div["mean_normalized_payment_difference"]),
    )
    result.add(
        firm="_summary",
        metric="any_rank_reversal",
        value=float(div["any_rank_reversal"]),
    )
    result.add(
        firm="_summary",
        metric="efficiency_gap_op",
        value=float(abs(sum(phi_op.values()) - v_op_N)),
    )
    return result


def make_negative_control_grid() -> dict[str, list]:
    return {
        "n_firms": [4],
        "shared_factor_correlation": [0.6],
        "cb_over_ch": [1.0, 9.0],
        "n_periods": [500],
        "n_features": [5],
        "n_obs_heterogeneity": ["moderate"],
        "ridge_lambda": [1.0],
        "n_inventory_reps": [50],
    }


def main():
    output_dir = Path("results/negative_control_gate1_opt1")
    grid = make_negative_control_grid()

    df = run_sweep(
        scenario_fn=negative_control_scenario,
        param_grid=grid,
        n_seeds=15,              # slightly fewer; each run is K× heavier
        master_seed=42,
        output_dir=output_dir,
        stop_on_error=False,
    )

    rho = df.query("firm == '_summary' and metric == 'spearman_rho'")[
        "value"].astype(float)
    disp = df.query("firm == '_summary' and metric == 'payment_displacement'")[
        "value"].astype(float)
    v_op = df.query("firm == '_summary' and metric == 'v_op_N'")[
        "value"].astype(float)
    gap = df.query("firm == '_summary' and metric == 'efficiency_gap_op'")[
        "value"].astype(float)
    rev = df.query("firm == '_summary' and metric == 'any_rank_reversal'")[
        "value"].astype(float)

    print("\n=== Gate 1 diagnostic (fixed λ, K=50) ===")
    print(f"n runs              : {len(rho)}")
    print(f"mean Spearman ρ     : {rho.mean():.3f}   (target ≳ 0.90)")
    print(f"median Spearman ρ   : {rho.median():.3f}")
    print(f"mean displacement   : {disp.mean():.3f}")
    print(f"rank-reversal rate  : {rev.mean():.3f}")
    print(f"mean v_op(N)        : {v_op.mean():.3f}   (must be > 0)")
    print(f"max efficiency gap  : {gap.max():.2e}")

    for cb, g in df.query("firm == '_summary'").groupby("cb_over_ch"):
        r = g.query("metric == 'spearman_rho'")["value"].astype(float)
        print(
            f"  cb/ch={cb}: mean ρ={r.mean():.3f}, median ρ={r.median():.3f}")

    passed = bool(rho.mean() >= 0.90 and v_op.mean() > 0)
    print(f"\nGATE 1 DIAGNOSTIC {'PASSED' if passed else 'FAILED'}")
    return df


if __name__ == "__main__":
    main()
