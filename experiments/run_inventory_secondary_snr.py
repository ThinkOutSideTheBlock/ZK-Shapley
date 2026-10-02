"""
Inventory secondary: at κ=1, compare
  (A) residual-scale Shapley ranks
  (B) inventory-path cost Shapley ranks
on the same fits. Longer T + more CRN reps.
Report Spearman(A,B) — expect moderate/low if path noise dominates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std
from src.inventory import EchelonConfig, simulate_serial_supply_chain
from src.shapley import exact_shapley, CachedGame


def default_echelons():
    return [
        EchelonConfig(1.0, 9.0, 1, 0.95),
        EchelonConfig(0.6, 0.0, 2, 0.90),
        EchelonConfig(0.3, 0.0, 3, 0.90),
    ]


def spearman(phi_a, phi_b):
    firms = sorted(phi_a.keys())
    xa = np.array([phi_a[f] for f in firms])
    xb = np.array([phi_b[f] for f in firms])
    ra = np.empty(len(firms))
    rb = np.empty(len(firms))
    ra[np.argsort(xa)] = np.arange(len(firms))
    rb[np.argsort(xb)] = np.arange(len(firms))
    n = len(firms)
    return float(1 - 6 * np.sum((ra - rb) ** 2) / (n * (n * n - 1)))


def run_seed(seed: int, n_periods=800, K_crn=20) -> dict:
    cfg = DemandSimulationConfig(
        n_firms=4,
        n_periods=n_periods,
        n_features=5,
        shared_factor_correlation=0.6,
        n_obs_heterogeneity="moderate",
        seed=seed,
    )
    sim = generate_multi_firm_demand(cfg)
    datasets = build_firm_datasets(
        sim, val_fraction=0.12, holdout_fraction=0.30, min_holdout=80
    )
    firms = tuple(sorted(datasets.keys()))
    paths = {n: datasets[n].y_holdout for n in firms}
    echs = default_echelons()

    coal = CoalitionConfig(
        firms=firms,
        lambda_mode="fixed",
        residual_mode="predictive",
        ridge_lambda=1.0,
        seed=seed,
    )
    ev = CoalitionEvaluator(datasets, paths, coal)

    sig_cache, cost_cache = {}, {}

    def sigma(i, S):
        key = (i, S)
        if key in sig_cache:
            return sig_cache[key]
        beta = ev.fit(S)
        s = firm_predictive_residual_std(beta, datasets[i])
        sig_cache[key] = s
        return s

    def path_cost(i, S):
        key = (i, S)
        if key in cost_cache:
            return cost_cache[key]
        beta = ev.fit(S)
        ds = datasets[i]
        resid = firm_predictive_residual_std(beta, ds)
        forecast = ds.X_holdout @ beta
        demand = np.asarray(ds.y_holdout, dtype=float)
        costs = []
        for k in range(K_crn):
            out = simulate_serial_supply_chain(
                demand, forecast, resid, echs, rng=None
            )
            costs.append(out.total_cost)
        c = float(np.mean(costs))
        cost_cache[key] = c
        return c

    def v_resid(S):
        if not S:
            return 0.0
        return float(sum(sigma(i, frozenset({i})) - sigma(i, S) for i in S))

    def v_path(S):
        if not S:
            return 0.0
        return float(sum(path_cost(i, frozenset({i})) - path_cost(i, S) for i in S))

    phi_r = exact_shapley(CachedGame(list(firms), v_resid))
    phi_p = exact_shapley(CachedGame(list(firms), v_path))
    return {
        "seed": seed,
        "spearman_resid_vs_path": spearman(phi_r, phi_p),
        "v_resid_N": v_resid(frozenset(firms)),
        "v_path_N": v_path(frozenset(firms)),
    }


def main():
    n_seeds = 15
    rng = np.random.default_rng(7)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_seeds)]
    out = Path("results/inventory_secondary_snr")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, seed in enumerate(seeds):
        r = run_seed(seed)
        rows.append(r)
        print(
            f"  seed {i+1}/{n_seeds}  ρ(resid,path)={r['spearman_resid_vs_path']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    print("\n=== Inventory secondary (κ=1) ===")
    print(
        f"mean Spearman(resid ranks, path ranks): {df['spearman_resid_vs_path'].mean():.3f}")
    print(f"median: {df['spearman_resid_vs_path'].median():.3f}")
    print("Interpretation: lower ρ ⇒ path cost adds ranking noise; residual channel is sharper.")
    print(f"Wrote {out / 'results.csv'}")


if __name__ == "__main__":
    main()
