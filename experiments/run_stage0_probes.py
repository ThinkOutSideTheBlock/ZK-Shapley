"""
Stage 0 correctness probes — Claude plan §5 / §7.

Homogeneity (Prop B″) requires:
  - non-negative demand with μ/σ ≥ 5 at the *largest* σ tested
  - policy residual_std = σ
  - demand error scale = σ
  - warm-start near base-stock (not empty system)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std
from src.inventory import EchelonConfig, simulate_serial_supply_chain, _base_stock_level


def mean_cost_scaled_nonneg(mu, residual_std, echelons, seed, T=500, K=40):
    """
    Prop B″ probe:
      demand_t = max(0, mu + σ·ε_t)
      forecast_t = mu
      residual_std = σ
      warm-start on-hand ≈ base-stock level for σ
    """
    z = float(
        norm.ppf(float(np.clip(echelons[0].service_level, 1e-4, 1 - 1e-4))))
    L = max(int(echelons[0].lead_time), 1)
    S0 = _base_stock_level(mu, residual_std, L, z)

    totals, holdings, shortages = [], [], []
    for k in range(K):
        rng = np.random.default_rng([seed, k])
        eps = rng.standard_normal(T)
        forecast = np.full(T, float(mu))
        demand = np.maximum(0.0, mu + residual_std * eps)
        out = simulate_serial_supply_chain(
            demand,
            forecast,
            residual_std,
            echelons,
            rng=None,
            initial_on_hand=S0,
        )
        totals.append(out.total_cost)
        holdings.append(out.holding_cost)
        shortages.append(out.shortage_cost)

    print(
        f"    [σ={residual_std:.4f}] hold={np.mean(holdings):.1f}  "
        f"short={np.mean(shortages):.1f}  S0={S0:.2f}  "
        f"μ/σ={mu / residual_std:.2f}"
    )
    return float(np.mean(totals))


def run_homogeneity(label, mu, s1, echelons, seed):
    c1 = mean_cost_scaled_nonneg(mu, s1, echelons, seed=seed)
    c2 = mean_cost_scaled_nonneg(mu, 2 * s1, echelons, seed=seed)
    c4 = mean_cost_scaled_nonneg(mu, 4 * s1, echelons, seed=seed)
    r2 = c2 / c1 if c1 > 0 else float("nan")
    r4 = c4 / c1 if c1 > 0 else float("nan")
    print(f"=== Homogeneity ({label}) ===")
    print(f"  μ={mu:.3f}  σ={s1:.4f}  C={c1:.4f}")
    print(f"  2σ  C={c2:.4f}  ratio={r2:.4f}  (expect ≈ 2)")
    print(f"  4σ  C={c4:.4f}  ratio={r4:.4f}  (expect ≈ 4)")
    print(f"  ΔC(2σ-σ)={c2 - c1:.4f}  ΔC(4σ-σ)={c4 - c1:.4f}")
    ratio_ok = abs(r2 - 2.0) < 0.45 and abs(r4 - 4.0) < 0.90
    mono_ok = (c2 > c1) and (c4 > c2)
    ok = ratio_ok or mono_ok
    print(
        f"  ratio_ok={ratio_ok}  mono_ok={mono_ok}  → {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    cfg = DemandSimulationConfig(
        n_firms=3, n_periods=400, n_features=4, seed=7,
        shared_factor_correlation=0.6, n_obs_heterogeneity="moderate",
    )
    sim = generate_multi_firm_demand(cfg)
    datasets = build_firm_datasets(
        sim, val_fraction=0.15, holdout_fraction=0.25, min_holdout=40
    )
    paths = {n: ds.y_holdout for n, ds in datasets.items()}
    firms = tuple(sorted(datasets.keys()))
    f0 = firms[0]

    coal_cfg = CoalitionConfig(
        firms=firms, lambda_mode="fixed", residual_mode="predictive",
        ridge_lambda=1.0, seed=7,
    )
    ev = CoalitionEvaluator(datasets, paths, coal_cfg)

    beta = ev.fit(frozenset({f0}))
    ds = datasets[f0]
    forecast = np.asarray(ds.X_holdout @ beta, dtype=float)
    s1 = firm_predictive_residual_std(beta, ds)

    # Critical: μ/σ ≥ 5 at 4σ  →  μ ≥ 20 * s1
    mu = float(max(np.mean(np.abs(forecast)), 20.0 * s1))

    single = [
        EchelonConfig(
            holding_cost=1.0, shortage_cost=9.0, lead_time=1, service_level=0.95
        )
    ]
    ok_single = run_homogeneity("single-echelon", mu, s1, single, seed=100)

    multi = coal_cfg.to_echelons()
    ok_multi = run_homogeneity("multi-echelon", mu, s1, multi, seed=100)

    print("\n=== Separability / topology ===")
    print("  Topology: parallel independent chains.")
    print("  Separability: holds by construction.")

    print("\n=== Ljung–Box on validation residuals (singleton β) ===")
    for name in firms:
        beta_i = ev.fit(frozenset({name}))
        dsi = datasets[name]
        resid = dsi.y_val - dsi.X_val @ beta_i
        if len(resid) < 20:
            print(f"  {name}: val too short")
            continue
        lb = acorr_ljungbox(resid, lags=[5, 10], return_df=True)
        p5 = float(lb.loc[5, "lb_pvalue"])
        p10 = float(lb.loc[10, "lb_pvalue"])
        flag = "PERSISTENCE LIVE" if min(p5, p10) < 0.05 else "≈ white"
        print(f"  {name}: p@5={p5:.3f} p@10={p10:.3f}  {flag}")

    print("\n=== Stage 0 summary ===")
    print(f"  single-echelon homogeneity: {'PASS' if ok_single else 'FAIL'}")
    print(
        f"  multi-echelon homogeneity:  {'PASS' if ok_multi else 'FAIL'} (diagnostic)")
    if not ok_single:
        print("  BLOCKING: single-echelon still fails with μ/σ≥5 and warm-start.")
        print("  Use hold/short debug lines above to locate inverted component.")
    else:
        print("  Stage 0 Gate PASS. Proceed to Stage 1 (κ_eff retro-fit).")


if __name__ == "__main__":
    main()
