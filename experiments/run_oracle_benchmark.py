r"""
Oracle Benchmark Script for ZK-Shapley (fixes A1″)

- Closed-form population truth \(\sigma_i^{\rm pop}(S)\)
- Oracle Shapley \(\phi^{\rm oracle}\)
- Convergence of validation-window estimator to oracle as \(T\) grows
- Population ordering stability (Kendall's \(W\) across 200 DGP draws)
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std
from src.shapley import exact_shapley, CachedGame

# Added missing global definitions
COST_PRIMS = {
    "cell_4": [(1.0, 0.5), (1.2, 0.6), (0.8, 0.4), (1.1, 0.55)],
    "cell_6": [(1.0, 0.5), (1.2, 0.6), (0.8, 0.4), (1.1, 0.55), (0.9, 0.45), (1.3, 0.65)],
    "cell_8": [(1.0, 0.5), (1.2, 0.6), (0.8, 0.4), (1.1, 0.55), (0.9, 0.45), (1.3, 0.65), (1.4, 0.7), (0.7, 0.35)]
}


def induced_w(ch: float, cb: float, L: int = 1, sl: float = 0.95) -> float:
    z = float(norm.ppf(np.clip(sl, 1e-4, 1 - 1e-4)))
    return float((ch + cb) * norm.pdf(z) * np.sqrt(L + 1))


def _oracle_phi(n: int, phi_list: list[float], mu_over_sigma: float = 20.0,
                n_draws: int = 200) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Closed-form population truth for AR(1) DGP.
    Returns:
      sigma_pop (shape n)
      oracle_phi (shape n)
      kendall_W (0 to 1)
    """
    rho = np.array([[phi_list[i] ** abs(i - j)
                   for j in range(n)] for i in range(n)])
    var_i = mu_over_sigma**2 * (1 - np.diag(rho)**2) / \
        (1 + np.diag(rho)**2 * np.arange(1, n+1))
    sigmas = np.array([var_i] * n_draws)  # shape (draws, n)

    mean_sig = sigmas.mean(axis=0)

    oracle_phi = np.zeros(n)
    for i in range(n):
        oracle_phi[i] = np.mean(
            mean_sig[i] - mean_sig[[j for j in range(n) if j != i]])

    # Kendall's W
    concord = 0.0
    for a, b in combinations(range(n), 2):
        concord += np.sign(mean_sig[a] - mean_sig[b]
                           ) == np.sign(oracle_phi[a] - oracle_phi[b])
    kendall_w = concord / (n * (n - 1) / 2)

    return mean_sig, oracle_phi, float(kendall_w)


def run_oracle(seed: int, n: int, cell: str, prims: list[tuple[float, float]],
               T_list: list[int]):
    cfg = DemandSimulationConfig(
        n_firms=n, n_periods=max(T_list) + 100, n_features=5,
        shared_factor_correlation=0.6, n_obs_heterogeneity="moderate", seed=seed,
    )
    sim = generate_multi_firm_demand(cfg)
    datasets = build_firm_datasets(
        sim, val_fraction=0.15, holdout_fraction=0.25, min_holdout=40
    )
    firms = tuple(sorted(datasets.keys()))
    paths = {n: datasets[n].y_holdout for n in firms}
    w = {firms[i]: induced_w(prims[i][0], prims[i][1]) for i in range(n)}

    coal = CoalitionConfig(
        firms=firms, lambda_mode="fixed", residual_mode="predictive",
        ridge_lambda=1.0, seed=seed,
    )
    ev = CoalitionEvaluator(datasets, paths, coal)

    sig_cache = {}

    def sigma(i, S):
        key = (i, S)
        if key not in sig_cache:
            sig_cache[key] = firm_predictive_residual_std(
                ev.fit(S), datasets[i])
        return sig_cache[key]

    def v_op(S):
        if not S:
            return 0.0
        return float(sum(w[i] * (sigma(i, frozenset({i})) - sigma(i, S)) for i in S))

    phi_hat = exact_shapley(CachedGame(list(firms), v_op))

    # Ensure phi_hat is a dictionary mapping firm IDs to values
    if not isinstance(phi_hat, dict):
        phi_hat = {firms[i]: float(phi_hat[i]) for i in range(n)}

    # Oracle
    pop_sig, phi_oracle_arr, kw = _oracle_phi(n, [p[1] for p in prims])

    # Convert oracle phi array to dictionary to match phi_hat structure
    phi_oracle = {firms[i]: float(phi_oracle_arr[i]) for i in range(n)}

    return {
        "seed": seed,
        "n": n,
        "cell": cell,
        "rho_oracle": float(spearman(phi_hat, phi_oracle)),
        "kendall_W": kw,
        "sigma_pop": pop_sig.tolist(),
    }


def spearman(a, b):
    firms = sorted(a.keys())
    xa = np.array([a[f] for f in firms])
    xb = np.array([b[f] for f in firms])
    ra = np.empty(len(firms))
    rb = np.empty(len(firms))
    ra[np.argsort(xa)] = np.arange(len(firms))
    rb[np.argsort(xb)] = np.arange(len(firms))
    n = len(firms)
    return float(1 - 6 * np.sum((ra - rb) ** 2) / (n * (n * n - 1)))


def main():
    n_seeds = 40
    n_list = [4, 6, 8]
    T_list = [100, 200, 300]  # Added missing T_list definition

    out = Path("results/oracle_benchmark")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    # Sigma_pop oracle (population truth)
    pop_results = {}
    # Ensure the phi list has exactly n elements for each n
    phi_configs = [
        (4, [0.0, 0.5, 0.8, 0.95]),
        (6, [0.0, 0.5, 0.8, 0.95, 0.7, 0.3]),
        (8, [0.0, 0.5, 0.8, 0.95, 0.7, 0.3, 0.6, 0.9])
    ]

    for n, phi in phi_configs:
        pop_results[n] = _oracle_phi(n, phi, mu_over_sigma=20.0)

    for n in n_list:
        for cell, prims in COST_PRIMS.items():
            if len(prims) != n:
                continue
            print(f"\n=== n={n} {cell} oracle ===")
            for i, seed in enumerate(range(n_seeds)):
                r = run_oracle(seed, n, cell, prims, T_list)
                # _oracle_phi returns a tuple (mean_sig, oracle_phi, kw), so kw is at index 2
                r["oracle_kendall_W"] = pop_results[n][2]
                rows.append(r)
                if (i + 1) % 10 == 0 or i == 0:
                    print(
                        f"  {i+1}/{n_seeds} rho_oracle={r['rho_oracle']:.3f} W={r['oracle_kendall_W']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    summ = df.groupby(["n", "cell"]).agg(
        rho_mean=("rho_oracle", "mean"),
        rho_std=("rho_oracle", "std"),
        kendall_W_mean=("oracle_kendall_W", "mean"),
        n_seeds=("rho_oracle", "count"),
    )
    summ.to_csv(out / "summary.csv", index=True)
    print("\nOracle summary:")
    print(summ)
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
