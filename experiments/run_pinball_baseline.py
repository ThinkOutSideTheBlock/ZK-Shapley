"""
Accuracy-game variants vs residual operational ranks on cost channel (n=4).
Metrics: RMSE, MSE, relative MAE, pinball at τ*_i = cb/(ch+cb).
If divergence collapses under pinball, framing weakens toward cost-aware loss.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std
from src.shapley import exact_shapley, CachedGame


def induced_w(ch, cb, L=1, sl=0.95):
    z = float(norm.ppf(np.clip(sl, 1e-4, 1 - 1e-4)))
    return float((ch + cb) * norm.pdf(z) * np.sqrt(L + 1))


def spearman(a, b):
    firms = sorted(a.keys())
    xa, xb = np.array([a[f] for f in firms]), np.array([b[f] for f in firms])
    ra, rb = np.empty(len(firms)), np.empty(len(firms))
    ra[np.argsort(xa)] = np.arange(len(firms))
    rb[np.argsort(xb)] = np.arange(len(firms))
    n = len(firms)
    return float(1 - 6 * np.sum((ra - rb) ** 2) / (n * (n * n - 1)))


def holdout_scores(beta, ds, metric, tau=0.5):
    pred = ds.X_holdout @ beta
    y = ds.y_holdout
    err = y - pred
    if metric == "rmse":
        return float(np.sqrt(np.mean(err**2)))
    if metric == "mse":
        return float(np.mean(err**2))
    if metric == "rmae":
        denom = np.maximum(np.abs(y), 1e-6)
        return float(np.mean(np.abs(err) / denom))
    if metric == "pinball":
        # pinball loss mean
        return float(np.mean(np.where(err >= 0, tau * err, (tau - 1) * err)))
    raise ValueError(metric)


COST_PRIMS = {
    "cost_k1": [(1.0, 9.0)] * 4,
    "cost_k2": [(1.0, 4.0), (1.0, 6.0), (1.0, 9.0), (1.0, 12.0)],
    "cost_k5": [(1.0, 2.0), (1.0, 5.0), (1.0, 12.0), (1.0, 25.0)],
    "cost_k10": [(1.0, 1.0), (1.0, 5.0), (1.0, 15.0), (1.0, 40.0)],
}


def run_seed(seed, cell, prims):
    cfg = DemandSimulationConfig(
        n_firms=4, n_periods=500, n_features=5,
        shared_factor_correlation=0.6, n_obs_heterogeneity="moderate", seed=seed,
    )
    sim = generate_multi_firm_demand(cfg)
    datasets = build_firm_datasets(
        sim, val_fraction=0.15, holdout_fraction=0.25, min_holdout=50
    )
    firms = tuple(sorted(datasets.keys()))
    paths = {n: datasets[n].y_holdout for n in firms}
    w = {firms[i]: induced_w(prims[i][0], prims[i][1]) for i in range(4)}
    tau = {firms[i]: prims[i][1] / (prims[i][0] + prims[i][1]) for i in range(4)}
    kappa = max(w.values()) / max(min(w.values()), 1e-12)

    coal = CoalitionConfig(
        firms=firms, lambda_mode="fixed", residual_mode="predictive",
        ridge_lambda=1.0, seed=seed,
    )
    ev = CoalitionEvaluator(datasets, paths, coal)
    sig_cache, fit_cache = {}, {}

    def fit(S):
        if S not in fit_cache:
            fit_cache[S] = ev.fit(S)
        return fit_cache[S]

    def sigma(i, S):
        key = (i, S)
        if key not in sig_cache:
            sig_cache[key] = firm_predictive_residual_std(fit(S), datasets[i])
        return sig_cache[key]

    def v_op(S):
        if not S:
            return 0.0
        return float(sum(w[i] * (sigma(i, frozenset({i})) - sigma(i, S)) for i in S))

    def make_vacc(metric):
        def v(S):
            if not S:
                return 0.0
            total = 0.0
            for i in S:
                auto = holdout_scores(fit(frozenset({i})), datasets[i], metric, tau[i])
                coal_s = holdout_scores(fit(S), datasets[i], metric, tau[i])
                total += auto - coal_s  # reduction = improvement
            return float(total)
        return v

    phi_op = exact_shapley(CachedGame(list(firms), v_op))
    out = {"seed": seed, "cell": cell, "kappa_eff": float(kappa)}
    for metric in ("rmse", "mse", "rmae", "pinball"):
        phi_m = exact_shapley(CachedGame(list(firms), make_vacc(metric)))
        out[f"rho_op_vs_{metric}"] = spearman(phi_op, phi_m)
    return out


def main():
    n_seeds = 30
    rng = np.random.default_rng(7)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_seeds)]
    out = Path("results/pinball_baseline")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell, prims in COST_PRIMS.items():
        print(f"\n=== {cell} ===")
        for i, seed in enumerate(seeds):
            r = run_seed(seed, cell, prims)
            rows.append(r)
            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  {i+1}/{n_seeds}  ρ_rmse={r['rho_op_vs_rmse']:.2f}  "
                    f"ρ_pinball={r['rho_op_vs_pinball']:.2f}"
                )
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    summ = df.groupby("cell", as_index=False).agg(
        kappa_mean=("kappa_eff", "mean"),
        **{f"{m}_mean": (f"rho_op_vs_{m}", "mean")
           for m in ("rmse", "mse", "rmae", "pinball")}
    )
    print("\n", summ.to_string(index=False))
    print(f"Wrote {out}/")
    print(
        "Framing risk: if pinball column stays ≈1 across κ while rmse falls, "
        "prefer cost-aware loss narrative over operational Shapley divergence."
    )


if __name__ == "__main__":
    main()