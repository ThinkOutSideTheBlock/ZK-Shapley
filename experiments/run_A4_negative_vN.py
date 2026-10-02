"""
A4 diagnostic: negative grand-coalition surplus under path cost.

Reports, over seeds:
  - fraction with v_path(N) < 0
  - fraction with v_acc(N) < 0  (holdout RMSE reduction)
  - fraction with v_resid(N) < 0  (validation RMS reduction)
  - co-occurrence path∩acc, path∩resid
  - optional size-scaled λ comparison on the same seeds
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std, firm_holdout_rmse
from src.inventory import EchelonConfig, simulate_serial_supply_chain


def default_echelons():
    return [
        EchelonConfig(1.0, 9.0, 1, 0.95),
        EchelonConfig(0.6, 0.0, 2, 0.90),
        EchelonConfig(0.3, 0.0, 3, 0.90),
    ]


def run_seed(seed: int, lambda_mode: str = "fixed", lam0: float = 1.0, K_crn: int = 12):
    cfg = DemandSimulationConfig(
        n_firms=4,
        n_periods=600,
        n_features=5,
        shared_factor_correlation=0.6,
        n_obs_heterogeneity="moderate",
        seed=seed,
    )
    sim = generate_multi_firm_demand(cfg)
    datasets = build_firm_datasets(
        sim, val_fraction=0.12, holdout_fraction=0.28, min_holdout=70
    )
    firms = tuple(sorted(datasets.keys()))
    paths = {n: datasets[n].y_holdout for n in firms}
    echs = default_echelons()
    N = frozenset(firms)

    coal = CoalitionConfig(
        firms=firms,
        lambda_mode=lambda_mode,
        residual_mode="predictive",
        ridge_lambda=lam0,
        seed=seed,
    )
    ev = CoalitionEvaluator(datasets, paths, coal)

    # --- residual-scale v(N) ---
    def sigma(i, S):
        beta = ev.fit(S)
        return firm_predictive_residual_std(beta, datasets[i])

    v_resid = float(
        sum(sigma(i, frozenset({i})) - sigma(i, N) for i in firms)
    )

    # --- accuracy (holdout RMSE) v(N) ---
    def rmse(i, S):
        beta = ev.fit(S)
        return firm_holdout_rmse(beta, datasets[i])

    v_acc = float(sum(rmse(i, frozenset({i})) - rmse(i, N) for i in firms))

    # --- path cost v(N) ---
    def path_cost(i, S):
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
        return float(np.mean(costs))

    v_path = float(
        sum(path_cost(i, frozenset({i})) - path_cost(i, N) for i in firms)
    )

    return {
        "seed": seed,
        "lambda_mode": lambda_mode,
        "v_path_N": v_path,
        "v_acc_N": v_acc,
        "v_resid_N": v_resid,
        "neg_path": int(v_path < 0),
        "neg_acc": int(v_acc < 0),
        "neg_resid": int(v_resid < 0),
    }


def main():
    n_seeds = 30
    rng = np.random.default_rng(20260831)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_seeds)]
    out = Path("results/A4_negative_vN")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode in ("fixed", "size_scaled"):
        print(f"\n=== lambda_mode={mode} ===")
        for i, seed in enumerate(seeds):
            r = run_seed(seed, lambda_mode=mode, lam0=1.0)
            rows.append(r)
            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  {i+1}/{n_seeds}  v_path={r['v_path_N']:.2f}  "
                    f"v_acc={r['v_acc_N']:.4f}  v_resid={r['v_resid_N']:.4f}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)

    print("\n=== A4 summary ===")
    for mode in ("fixed", "size_scaled"):
        sub = df[df["lambda_mode"] == mode]
        n = len(sub)
        print(f"\n[{mode}] n={n}")
        print(f"  P(v_path<0)  = {sub['neg_path'].mean():.3f}")
        print(f"  P(v_acc<0)   = {sub['neg_acc'].mean():.3f}")
        print(f"  P(v_resid<0) = {sub['neg_resid'].mean():.3f}")
        print(
            f"  P(path<0 & acc<0)   = {((sub['neg_path'] == 1) & (sub['neg_acc'] == 1)).mean():.3f}"
        )
        print(
            f"  P(path<0 & resid<0) = {((sub['neg_path'] == 1) & (sub['neg_resid'] == 1)).mean():.3f}"
        )
        print(f"  mean v_path = {sub['v_path_N'].mean():.2f}")
        print(f"  mean v_acc  = {sub['v_acc_N'].mean():.4f}")
        print(f"  mean v_resid= {sub['v_resid_N'].mean():.4f}")

    print(f"\nWrote {out / 'results.csv'}")


if __name__ == "__main__":
    main()
