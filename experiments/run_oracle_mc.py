"""
Monte-Carlo oracle benchmark (DGP-matched).

Population truth:
  Long series from generate_multi_firm_demand + exact federated ridge
  + Phase-1 validation RMS residual scale → v_op_pop → exact Shapley φ_oracle.

Finite-T estimator:
  Independent shorter series, same pipeline → φ_hat(T).

Reports:
  - Spearman ρ(φ_hat, φ_oracle) vs T
  - mean |φ_hat - φ_oracle|_1 / n
  - Kendall W of firm ordering by φ_oracle across structural seeds (stability)

Primary: n=4 (exact 16 coalitions). Optional n=6.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std
from src.shapley import exact_shapley, CachedGame


# ---------------------------------------------------------------------------
# Weights (cost channel primitives; pad/truncate to n)
# ---------------------------------------------------------------------------
COST_PRIMS = {
    "k1": [(1.0, 9.0)] * 8,  # homogeneous → κ_eff = 1
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
    "k5": [
        (1.0, 2.0),
        (1.0, 5.0),
        (1.0, 12.0),
        (1.0, 25.0),
        (1.0, 3.0),
        (1.0, 8.0),
        (1.0, 15.0),
        (1.0, 20.0),
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


def l1_norm_diff(phi_a: dict, phi_b: dict) -> float:
    firms = sorted(phi_a.keys())
    return float(
        np.mean([abs(phi_a[f] - phi_b[f]) for f in firms])
    )


def kendall_w_rankings(rank_matrix: np.ndarray) -> float:
    """
    rank_matrix: shape (n_draws, n_firms), each row ranks in {0,...,n-1}.
    Classic Kendall's W.
    """
    m, n = rank_matrix.shape
    if m < 2 or n < 2:
        return float("nan")
    R = rank_matrix.sum(axis=0)
    Rbar = R.mean()
    S = float(np.sum((R - Rbar) ** 2))
    W = 12.0 * S / (m**2 * (n**3 - n))
    return float(np.clip(W, 0.0, 1.0))


def make_weights(firms: tuple[str, ...], prims: list[tuple[float, float]]) -> dict[str, float]:
    n = len(firms)
    prims = prims[:n]
    return {firms[i]: induced_w(prims[i][0], prims[i][1]) for i in range(n)}


def build_v_op(
    datasets: dict,
    paths: dict,
    firms: tuple[str, ...],
    w: dict[str, float],
    seed: int,
) -> tuple[Callable, CoalitionEvaluator]:
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
        beta = ev.fit(S)
        s = firm_predictive_residual_std(beta, datasets[i])
        cache[key] = float(s)
        return cache[key]

    def v_op(S: frozenset) -> float:
        if not S:
            return 0.0
        total = 0.0
        for i in S:
            auto = sigma(i, frozenset({i}))
            coal_s = sigma(i, S)
            total += w[i] * (auto - coal_s)
        return float(total)

    return v_op, ev


def phi_from_series(
    n_firms: int,
    n_periods: int,
    seed: int,
    prims: list[tuple[float, float]],
    shared_factor_correlation: float = 0.6,
) -> dict[str, float]:
    cfg = DemandSimulationConfig(
        n_firms=n_firms,  # was: n  → NameError / wrong binding
        n_periods=n_periods,
        n_features=5,
        shared_factor_correlation=shared_factor_correlation,
        n_obs_heterogeneity="fixed",
        signal_quality_profile="fixed",
        noise_profile="fixed",
        base_n_obs=max(200, n_periods // 2),
        seed=seed,
    )
    sim = generate_multi_firm_demand(cfg)
    val_frac = 0.20 if n_periods >= 800 else 0.15
    hold_frac = 0.25
    min_hold = max(30, n_periods // 20)
    datasets = build_firm_datasets(
        sim,
        val_fraction=val_frac,
        holdout_fraction=hold_frac,
        min_holdout=min_hold,
    )
    firms = tuple(sorted(datasets.keys()))
    paths = {name: datasets[name].y_holdout for name in firms}
    w = make_weights(firms, prims)
    v_op, _ = build_v_op(datasets, paths, firms, w, seed=seed)
    phi = exact_shapley(CachedGame(list(firms), v_op))
    if not isinstance(phi, dict):
        phi = {firms[i]: float(phi[i]) for i in range(n_firms)}
    return {k: float(v) for k, v in phi.items()}


def ranks_from_phi(phi: dict) -> np.ndarray:
    firms = sorted(phi.keys())
    vals = np.array([phi[f] for f in firms], dtype=float)
    order = np.argsort(vals)
    r = np.empty(len(firms), dtype=float)
    r[order] = np.arange(len(firms), dtype=float)
    return r


def main():
    # --- config (smoke: smaller; full: as below) ---
    n_list = [4]  # add 6 if budget allows (64 coalitions)
    cells = ["k1", "k2"]  # k1 = κ=1 boundary; k2 = mild cost heterogeneity
    T_pop = 2400  # population / oracle series length
    T_grid = [150, 300, 600, 1200]  # finite-T estimator lengths
    n_struct = 25  # independent structural seeds (oracle draws)
    n_est_per_struct = 4  # short-series replicates per structural seed

    out = Path("results/oracle_mc")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    order_rows = []

    rng = np.random.default_rng(20260901)

    for n in n_list:
        for cell in cells:
            prims = COST_PRIMS[cell][:n]
            print(f"\n=== n={n} cell={cell}  T_pop={T_pop} ===")

            oracle_phis: list[dict] = []
            rank_mat = []

            for s_idx in range(n_struct):
                struct_seed = int(rng.integers(0, 2**31 - 1))
                # Population oracle on long series
                phi_oracle = phi_from_series(
                    n_firms=n,
                    n_periods=T_pop,
                    seed=struct_seed,
                    prims=prims,
                )
                oracle_phis.append(phi_oracle)
                rank_mat.append(ranks_from_phi(phi_oracle))

                # Finite-T estimators (independent series, different seeds)
                for T in T_grid:
                    for e_idx in range(n_est_per_struct):
                        est_seed = int(rng.integers(0, 2**31 - 1))
                        phi_hat = phi_from_series(
                            n_firms=n,
                            n_periods=T,
                            seed=est_seed,
                            prims=prims,
                        )
                        # Align keys (firm names may differ by seed generation order —
                        # force canonical firm_0.. via sorted zip on values order)
                        # Safer: rebuild phi_hat/oracle on sorted firm name lists
                        firms_o = sorted(phi_oracle.keys())
                        firms_h = sorted(phi_hat.keys())
                        # Map by rank position after sorting firm labels
                        # Firm labels are firm_0.. consistent within a run; across runs
                        # structural parameters match but labels are always firm_0..n-1
                        po = {f"f{i}": phi_oracle[firms_o[i]]
                              for i in range(n)}
                        ph = {f"f{i}": phi_hat[firms_h[i]] for i in range(n)}
                        rho = spearman(ph, po)
                        l1 = l1_norm_diff(ph, po)
                        rows.append(
                            {
                                "n": n,
                                "cell": cell,
                                "struct_idx": s_idx,
                                "struct_seed": struct_seed,
                                "T": T,
                                "est_idx": e_idx,
                                "rho_oracle": rho,
                                "l1_mean": l1,
                                "kappa_cell": cell,
                            }
                        )

                if (s_idx + 1) % 5 == 0 or s_idx == 0:
                    # running mean rho at largest T
                    sub = [r for r in rows if r["n"] == n and r["cell"]
                           == cell and r["T"] == T_grid[-1]]
                    mean_r = float(np.mean([r["rho_oracle"]
                                   for r in sub])) if sub else float("nan")
                    print(
                        f"  struct {s_idx+1}/{n_struct}  mean ρ@T={T_grid[-1]} so far = {mean_r:.3f}")

            rank_mat = np.asarray(rank_mat, dtype=float)
            W = kendall_w_rankings(rank_mat)
            order_rows.append(
                {"n": n, "cell": cell, "kendall_W_oracle": W, "n_struct": n_struct})
            print(f"  Kendall W (oracle φ ranks across structs) = {W:.3f}")

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

    print("\n=== Convergence summary (mean ρ vs T) ===")
    print(summ.to_string(index=False))
    print("\n=== Oracle ordering stability ===")
    print(ord_df.to_string(index=False))
    print(f"\nWrote {out}/")
    print(
        "Interpretation:\n"
        "  ρ rising toward 1 with T  → residual-scale estimand is real; short-T noise.\n"
        "  ρ flat near 0            → estimator does not track population residual value.\n"
        "  Kendall W high           → population firm ranking stable across DGP draws."
    )


if __name__ == "__main__":
    main()
