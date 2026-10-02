"""
Persistence channel: identical costs & lead times; firm-level AR(1) φ_i
induces A(φ,H) and thus κ_eff > 1 with no cost asymmetry.

w_i ∝ φ(z) * A(φ_i, H), H = L+1
Rankings from residual-scale Δσ (semi-structural) with induced w.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std
from src.shapley import exact_shapley, CachedGame


def A_ar1(phi: float, H: int) -> float:
    """Lead-time aggregation factor for AR(1) errors, horizon H = L+1."""
    phi = float(np.clip(phi, -0.99, 0.99))
    H = int(max(H, 1))
    if abs(phi) < 1e-12:
        return float(np.sqrt(H))
    # Var(sum_{h=0}^{H-1} e_{t+h}) / σ^2 for AR(1)
    num = H * (1 - phi**2) - 2 * phi * (1 - phi**H)
    den = (1 - phi) ** 2
    return float(np.sqrt(max(num / den, 1e-12)))


def induced_w(phi: float, L: int = 1, ch: float = 1.0, cb: float = 9.0, sl: float = 0.95) -> float:
    z = float(norm.ppf(np.clip(sl, 1e-4, 1 - 1e-4)))
    pz = float(norm.pdf(z))
    H = L + 1
    return float((ch + cb) * pz * A_ar1(phi, H))


def spearman(phi_a: dict, phi_b: dict) -> float:
    firms = sorted(phi_a.keys())
    xa = np.array([phi_a[f] for f in firms])
    xb = np.array([phi_b[f] for f in firms])
    ra = np.empty(len(firms))
    rb = np.empty(len(firms))
    ra[np.argsort(xa)] = np.arange(len(firms))
    rb[np.argsort(xb)] = np.arange(len(firms))
    n = len(firms)
    return float(1 - 6 * np.sum((ra - rb) ** 2) / (n * (n * n - 1)))


def displacement(phi_a: dict, phi_b: dict) -> float:
    firms = sorted(phi_a.keys())
    xa = np.array([phi_a[f] for f in firms])
    xb = np.array([phi_b[f] for f in firms])
    ra = np.empty(len(firms))
    rb = np.empty(len(firms))
    ra[np.argsort(xa)] = np.arange(len(firms))
    rb[np.argsort(xb)] = np.arange(len(firms))
    return float(np.mean(np.abs(ra - rb)) / max(len(firms) - 1, 1))


# Firm-level φ schedules (identical ch, cb, L)
PERSIST_GRIDS = {
    "pers_k1": [0.0, 0.0, 0.0, 0.0],          # κ=1
    "pers_k2": [0.0, 0.2, 0.4, 0.6],          # moderate spread
    "pers_k3": [0.0, 0.3, 0.6, 0.85],         # stronger
    "pers_k5": [0.0, 0.4, 0.7, 0.95],         # high persistence on one firm
}


def run_one(seed: int, cell: str, phis: list[float], n_periods: int = 500):
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
        sim, val_fraction=0.15, holdout_fraction=0.25, min_holdout=50
    )
    firms = tuple(sorted(datasets.keys()))
    paths = {n: datasets[n].y_holdout for n in firms}

    # Identical costs/lead times; only φ differs → A(φ,H) drives κ
    L = 1
    w = {firms[i]: induced_w(phis[i], L=L) for i in range(4)}
    vals = np.array(list(w.values()))
    kappa = float(vals.max() / max(vals.min(), 1e-12))

    coal = CoalitionConfig(
        firms=firms,
        lambda_mode="fixed",
        residual_mode="predictive",
        ridge_lambda=1.0,
        seed=seed,
    )
    ev = CoalitionEvaluator(datasets, paths, coal)
    cache = {}

    def sigma(i, S):
        key = (i, S)
        if key in cache:
            return cache[key]
        beta = ev.fit(S)
        s = firm_predictive_residual_std(beta, datasets[i])
        cache[key] = s
        return s

    def v_acc(S):
        if not S:
            return 0.0
        return float(sum(sigma(i, frozenset({i})) - sigma(i, S) for i in S))

    def v_op(S):
        if not S:
            return 0.0
        return float(
            sum(w[i] * (sigma(i, frozenset({i})) - sigma(i, S)) for i in S)
        )

    phi_acc = exact_shapley(CachedGame(list(firms), v_acc))
    phi_op = exact_shapley(CachedGame(list(firms), v_op))
    N = frozenset(firms)
    return {
        "cell": cell,
        "channel": "persistence",
        "kappa_eff": kappa,
        "seed": seed,
        "phis": str(phis),
        "spearman_rho": spearman(phi_op, phi_acc),
        "displacement": displacement(phi_op, phi_acc),
        "rank_reversal": float(
            max(phi_op, key=phi_op.get) != max(phi_acc, key=phi_acc.get)
        ),
        "v_op_N": v_op(N),
        "eff_gap_op": abs(sum(phi_op.values()) - v_op(N)),
    }


def main():
    n_seeds = 25
    rng = np.random.default_rng(20260831)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_seeds)]
    out = Path("results/stage2_persistence_channel")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for cell, phis in PERSIST_GRIDS.items():
        print(f"\n=== {cell}  phis={phis} ===")
        for i, seed in enumerate(seeds):
            r = run_one(seed, cell, phis)
            rows.append(r)
            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  {i+1}/{n_seeds}  κ={r['kappa_eff']:.2f}  ρ={r['spearman_rho']:.3f}")
        sub = [r for r in rows if r["cell"] == cell]
        print(
            f"  mean ρ={np.mean([r['spearman_rho'] for r in sub]):.3f}  "
            f"mean κ={np.mean([r['kappa_eff'] for r in sub]):.2f}  "
            f"reversal={np.mean([r['rank_reversal'] for r in sub]):.2f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    summ = (
        df.groupby("cell", as_index=False)
        .agg(
            kappa_mean=("kappa_eff", "mean"),
            spearman_mean=("spearman_rho", "mean"),
            spearman_std=("spearman_rho", "std"),
            displacement_mean=("displacement", "mean"),
            rank_reversal_mean=("rank_reversal", "mean"),
            n_seeds=("spearman_rho", "count"),
        )
        .sort_values("kappa_mean")
    )
    summ.to_csv(out / "summary.csv", index=False)
    print("\n=== Persistence channel summary ===")
    print(summ.to_string(index=False))
    print(f"\nWrote {out}/")
    print(
        "Earn three-channel if: ρ=1 at pers_k1 and ρ declines as κ rises "
        "(with zero cost/lead asymmetry)."
    )


if __name__ == "__main__":
    main()
