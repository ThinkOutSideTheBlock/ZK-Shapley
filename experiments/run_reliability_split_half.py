"""
Reliability: split-half correlation of φ_op under κ_eff = 1.

Split hold-out (or validation) into two halves; recompute residual-scale
Shapley on each half; report Spearman / Pearson of φ vectors across seeds.
Expect high correlation if residual channel is stable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets, FirmDataset
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std
from src.shapley import exact_shapley, CachedGame


def spearman_vec(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.empty(len(a))
    rb = np.empty(len(b))
    ra[np.argsort(a)] = np.arange(len(a))
    rb[np.argsort(b)] = np.arange(len(b))
    n = len(a)
    return float(1 - 6 * np.sum((ra - rb) ** 2) / (n * (n * n - 1)))


def phi_from_sigma(firms, sigma_fn, weights):
    def v(S):
        if not S:
            return 0.0
        return float(sum(weights[i] * (sigma_fn(i, frozenset({i})) - sigma_fn(i, S)) for i in S))

    game = CachedGame(list(firms), v)
    return exact_shapley(game)


def half_datasets(datasets: dict, which: str) -> dict:
    """Split each firm's val+holdout chronologically into two halves; use as 'val' for RMS."""
    out = {}
    for name, ds in datasets.items():
        # Use validation residual path: build a thin FirmDataset with half of holdout as val
        y = ds.y_holdout
        X = ds.X_holdout
        T = len(y)
        mid = T // 2
        if which == "A":
            Xv, yv = X[:mid], y[:mid]
        else:
            Xv, yv = X[mid:], y[mid:]
        # Keep train as original train; val = half holdout; holdout unused for sigma
        out[name] = FirmDataset(
            name=ds.name,
            X_train=ds.X_train,
            y_train=ds.y_train,
            X_val=Xv,
            y_val=yv,
            X_holdout=ds.X_holdout,  # unused for residual here
            y_holdout=ds.y_holdout,
        )
    return out


def run_seed(seed: int, n_firms=4, n_periods=500) -> dict:
    cfg = DemandSimulationConfig(
        n_firms=n_firms,
        n_periods=n_periods,
        n_features=5,
        shared_factor_correlation=0.6,
        n_obs_heterogeneity="moderate",
        seed=seed,
    )
    sim = generate_multi_firm_demand(cfg)
    full = build_firm_datasets(
        sim, val_fraction=0.15, holdout_fraction=0.30, min_holdout=60
    )
    firms = tuple(sorted(full.keys()))
    weights = {f: 1.0 for f in firms}  # κ = 1

    rows = {}
    for half in ("A", "B"):
        ds_h = half_datasets(full, half)
        paths = {n: ds_h[n].y_holdout for n in firms}
        coal = CoalitionConfig(
            firms=firms,
            lambda_mode="fixed",
            residual_mode="predictive",
            ridge_lambda=1.0,
            seed=seed,
        )
        ev = CoalitionEvaluator(ds_h, paths, coal)
        cache = {}

        def sigma(i, S, _ev=ev, _ds=ds_h, _cache=cache):
            key = (i, S)
            if key in _cache:
                return _cache[key]
            beta = _ev.fit(S)
            s = firm_predictive_residual_std(beta, _ds[i], min_val_rows=8)
            _cache[key] = s
            return s

        phi = phi_from_sigma(firms, sigma, weights)
        rows[half] = np.array([phi[f] for f in firms])

    return {
        "seed": seed,
        "spearman_AB": spearman_vec(rows["A"], rows["B"]),
        "pearson_AB": float(np.corrcoef(rows["A"], rows["B"])[0, 1]),
    }


def main():
    n_seeds = 30
    rng = np.random.default_rng(20260831)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_seeds)]
    out = Path("results/reliability_split_half")
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for i, seed in enumerate(seeds):
        try:
            r = run_seed(seed)
            records.append(r)
        except Exception as e:
            print(f"[warn] seed={seed}: {e}")
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_seeds}")

    df = pd.DataFrame(records)
    df.to_csv(out / "results.csv", index=False)
    print("\n=== Split-half reliability of φ_op (κ=1) ===")
    print(f"n seeds        : {len(df)}")
    print(f"mean Spearman  : {df['spearman_AB'].mean():.3f}")
    print(f"median Spearman: {df['spearman_AB'].median():.3f}")
    print(f"mean Pearson   : {df['pearson_AB'].mean():.3f}")
    print(f"Wrote {out / 'results.csv'}")


if __name__ == "__main__":
    main()
