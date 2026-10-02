"""
Nested-T Monte-Carlo oracle (same realization, growing prefixes).

1. Generate one long series (T_pop) under fixed firm ladders.
2. φ_oracle from the full series.
3. For each T in T_grid: rebuild datasets from chronological prefixes
   of length T (same seed / same path) → φ_hat(T).
4. Report ρ(φ_hat, φ_oracle) vs T and Kendall W of oracle ranks
   across structural seeds.

If ρ rises with T, residual-scale ranks are a real finite-sample estimand
on a fixed path. If not, freeze P1 framing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.demand import (
    DemandSimulationConfig,
    DemandSimulationResult,
    FirmRawSeries,
    generate_multi_firm_demand,
)
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std
from src.shapley import exact_shapley, CachedGame


COST_PRIMS = {
    "k1": [(1.0, 9.0)] * 8,
    "k2": [
        (1.0, 4.0),
        (1.0, 6.0),
        (1.0, 9.0),
        (1.0, 12.0),
        (1.0, 5.0),
        (1.0, 8.0),
        (1.0, 10.0),
        (1.0, 14.0),
    ],
}


def induced_w(ch: float, cb: float, L: int = 1, sl: float = 0.95) -> float:
    z = float(norm.ppf(np.clip(sl, 1e-4, 1 - 1e-4)))
    return float((ch + cb) * float(norm.pdf(z)) * np.sqrt(L + 1))


def spearman(phi_a: dict, phi_b: dict) -> float:
    firms = sorted(phi_a.keys())
    xa = np.array([phi_a[f] for f in firms], dtype=float)
    xb = np.array([phi_b[f] for f in firms], dtype=float)
    ra = np.empty(len(firms))
    rb = np.empty(len(firms))
    ra[np.argsort(xa)] = np.arange(len(firms), dtype=float)
    rb[np.argsort(xb)] = np.arange(len(firms), dtype=float)
    n = len(firms)
    if n < 2:
        return 1.0
    return float(1.0 - 6.0 * np.sum((ra - rb) ** 2) / (n * (n * n - 1)))


def l1_mean(phi_a: dict, phi_b: dict) -> float:
    firms = sorted(phi_a.keys())
    return float(np.mean([abs(phi_a[f] - phi_b[f]) for f in firms]))


def kendall_w_rankings(rank_matrix: np.ndarray) -> float:
    m, n = rank_matrix.shape
    if m < 2 or n < 2:
        return float("nan")
    R = rank_matrix.sum(axis=0)
    Rbar = R.mean()
    S = float(np.sum((R - Rbar) ** 2))
    W = 12.0 * S / (m**2 * (n**3 - n))
    return float(np.clip(W, 0.0, 1.0))


def ranks_from_phi(phi: dict) -> np.ndarray:
    firms = sorted(phi.keys())
    vals = np.array([phi[f] for f in firms], dtype=float)
    order = np.argsort(vals)
    r = np.empty(len(firms), dtype=float)
    r[order] = np.arange(len(firms), dtype=float)
    return r


def make_weights(firms: tuple[str, ...], prims: list) -> dict[str, float]:
    n = len(firms)
    prims = prims[:n]
    return {firms[i]: induced_w(prims[i][0], prims[i][1]) for i in range(n)}


def prefix_sim(sim: DemandSimulationResult, T: int) -> DemandSimulationResult:
    """Chronological prefix of each firm series (nested-T)."""
    firms = {}
    for name, fr in sim.firms.items():
        t = min(T, fr.n_obs, fr.X.shape[0])
        firms[name] = FirmRawSeries(
            name=name,
            X=fr.X[:t].copy(),
            y=fr.y[:t].copy(),
            gamma=fr.gamma,
            sigma=fr.sigma,
            n_obs=t,
        )
    z = sim.z[:T].copy() if sim.z is not None else sim.z
    return DemandSimulationResult(
        firms=firms, beta_true=sim.beta_true, z=z, config=sim.config
    )


def phi_from_sim(
    sim: DemandSimulationResult,
    prims: list,
    seed: int,
) -> dict[str, float]:
    n_periods = max(fr.n_obs for fr in sim.firms.values())
    val_frac = 0.20 if n_periods >= 800 else 0.15
    hold_frac = 0.25
    min_hold = max(20, n_periods // 25)
    datasets = build_firm_datasets(
        sim,
        val_fraction=val_frac,
        holdout_fraction=hold_frac,
        min_holdout=min_hold,
    )
    firms = tuple(sorted(datasets.keys()))
    paths = {name: datasets[name].y_holdout for name in firms}
    w = make_weights(firms, prims)

    coal = CoalitionConfig(
        firms=firms,
        lambda_mode="fixed",
        residual_mode="predictive",
        ridge_lambda=1.0,
        seed=seed,
    )
    ev = CoalitionEvaluator(datasets, paths, coal)
    cache: dict = {}

    def sigma(i: str, S: frozenset) -> float:
        key = (i, S)
        if key in cache:
            return cache[key]
        s = float(firm_predictive_residual_std(ev.fit(S), datasets[i]))
        cache[key] = s
        return s

    def v_op(S: frozenset) -> float:
        if not S:
            return 0.0
        return float(
            sum(w[i] * (sigma(i, frozenset({i})) - sigma(i, S)) for i in S)
        )

    phi = exact_shapley(CachedGame(list(firms), v_op))
    if not isinstance(phi, dict):
        phi = {firms[i]: float(phi[i]) for i in range(len(firms))}
    return {k: float(v) for k, v in phi.items()}


def main():
    n_list = [4]
    cells = ["k1", "k2"]
    T_pop = 2400
    T_grid = [150, 300, 600, 1200]
    n_struct = 25

    out = Path("results/oracle_mc_nested")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    order_rows = []
    rng = np.random.default_rng(20260902)

    for n in n_list:
        for cell in cells:
            prims = COST_PRIMS[cell][:n]
            print(f"\n=== nested n={n} cell={cell} T_pop={T_pop} ===")
            rank_mat = []

            for s_idx in range(n_struct):
                seed = int(rng.integers(0, 2**31 - 1))
                cfg = DemandSimulationConfig(
                    n_firms=n,
                    n_periods=T_pop,
                    n_features=5,
                    shared_factor_correlation=0.6,
                    n_obs_heterogeneity="fixed",
                    signal_quality_profile="fixed",
                    noise_profile="fixed",
                    base_n_obs=max(400, T_pop // 2),
                    base_noise_std=1.0,
                    seed=seed,
                )
                sim_full = generate_multi_firm_demand(cfg)
                phi_oracle = phi_from_sim(sim_full, prims, seed=seed)
                rank_mat.append(ranks_from_phi(phi_oracle))

                for T in T_grid:
                    sim_T = prefix_sim(sim_full, T)
                    phi_hat = phi_from_sim(sim_T, prims, seed=seed)
                    # same firm labels firm_0.. on same path
                    rho = spearman(phi_hat, phi_oracle)
                    rows.append(
                        {
                            "n": n,
                            "cell": cell,
                            "struct_idx": s_idx,
                            "seed": seed,
                            "T": T,
                            "rho_oracle": rho,
                            "l1_mean": l1_mean(phi_hat, phi_oracle),
                        }
                    )

                if (s_idx + 1) % 5 == 0 or s_idx == 0:
                    sub = [
                        r
                        for r in rows
                        if r["n"] == n
                        and r["cell"] == cell
                        and r["T"] == T_grid[-1]
                    ]
                    m = float(np.mean([r["rho_oracle"]
                              for r in sub])) if sub else float("nan")
                    print(
                        f"  struct {s_idx+1}/{n_struct}  mean ρ@T={T_grid[-1]} = {m:.3f}")

            W = kendall_w_rankings(np.asarray(rank_mat, dtype=float))
            order_rows.append(
                {"n": n, "cell": cell, "kendall_W_oracle": W, "n_struct": n_struct}
            )
            print(f"  Kendall W (oracle ranks) = {W:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    ord_df = pd.DataFrame(order_rows)
    ord_df.to_csv(out / "ordering_stability.csv", index=False)

    summ = (
        df.groupby(["n", "cell", "T"], as_index=False)
        .agg(
            rho_mean=("rho_oracle", "mean"),
            rho_std=("rho_oracle", "std"),
            rho_median=("rho_oracle", "median"),
            l1_mean=("l1_mean", "mean"),
            n_obs=("rho_oracle", "count"),
        )
        .sort_values(["n", "cell", "T"])
    )
    summ.to_csv(out / "summary_by_T.csv", index=False)

    print("\n=== Nested convergence (mean ρ vs T) ===")
    print(summ.to_string(index=False))
    print("\n=== Ordering stability ===")
    print(ord_df.to_string(index=False))
    print(f"\nWrote {out}/")
    print(
        "SUCCESS if mean ρ rises clearly with T (e.g. >0.5 at T=1200) "
        "and preferably W≳0.4. Else FREEZE."
    )


if __name__ == "__main__":
    main()
