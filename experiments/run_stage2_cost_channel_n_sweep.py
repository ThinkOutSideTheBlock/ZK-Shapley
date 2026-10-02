"""
Cost-channel κ sweep at n in {4,6,8}: median/IQR, bootstrap CIs,
Jonckheere–Terpstra monotonicity of Spearman ρ vs ordered κ cells.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std
from src.shapley import exact_shapley, CachedGame


def induced_w(ch, cb, L=1, sl=0.95):
    z = float(norm.ppf(np.clip(sl, 1e-4, 1 - 1e-4)))
    return float((ch + cb) * norm.pdf(z) * np.sqrt(L + 1))


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


def top1(phi_a, phi_b):
    return float(max(phi_a, key=phi_a.get) != max(phi_b, key=phi_b.get))


def displacement(phi_a, phi_b):
    firms = sorted(phi_a.keys())
    xa = np.array([phi_a[f] for f in firms])
    xb = np.array([phi_b[f] for f in firms])
    ra = np.empty(len(firms))
    rb = np.empty(len(firms))
    ra[np.argsort(xa)] = np.arange(len(firms))
    rb[np.argsort(xb)] = np.arange(len(firms))
    return float(np.mean(np.abs(ra - rb)) / max(len(firms) - 1, 1))


# Per-firm (ch, cb) schedules; pad/truncate to n_firms
COST_BASE = {
    "cost_k1": [(1.0, 9.0)] * 8,
    "cost_k2": [(1.0, 4.0), (1.0, 6.0), (1.0, 9.0), (1.0, 12.0),
                (1.0, 5.0), (1.0, 8.0), (1.0, 10.0), (1.0, 14.0)],
    "cost_k5": [(1.0, 2.0), (1.0, 5.0), (1.0, 12.0), (1.0, 25.0),
                (1.0, 3.0), (1.0, 8.0), (1.0, 15.0), (1.0, 20.0)],
    "cost_k10": [(1.0, 1.0), (1.0, 5.0), (1.0, 15.0), (1.0, 40.0),
                 (1.0, 2.0), (1.0, 8.0), (1.0, 20.0), (1.0, 35.0)],
}


def run_one(n_firms, cell, prims, seed, n_periods=400):
    prims = prims[:n_firms]
    cfg = DemandSimulationConfig(
        n_firms=n_firms,
        n_periods=n_periods,
        n_features=5,
        shared_factor_correlation=0.6,
        n_obs_heterogeneity="moderate",
        seed=seed,
    )
    sim = generate_multi_firm_demand(cfg)
    datasets = build_firm_datasets(
        sim, val_fraction=0.15, holdout_fraction=0.25, min_holdout=40
    )
    firms = tuple(sorted(datasets.keys()))
    paths = {n: datasets[n].y_holdout for n in firms}
    w = {firms[i]: induced_w(prims[i][0], prims[i][1]) for i in range(n_firms)}
    kappa = max(w.values()) / max(min(w.values()), 1e-12)

    coal = CoalitionConfig(
        firms=firms, lambda_mode="fixed", residual_mode="predictive",
        ridge_lambda=1.0, seed=seed,
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
        return float(sum(w[i] * (sigma(i, frozenset({i})) - sigma(i, S)) for i in S))

    # Exact Shapley: for n=8 use MC if needed — here exact (256 coalitions)
    phi_acc = exact_shapley(CachedGame(list(firms), v_acc))
    phi_op = exact_shapley(CachedGame(list(firms), v_op))
    rho = spearman(phi_op, phi_acc)
    return {
        "n_firms": n_firms,
        "cell": cell,
        "kappa_eff": float(kappa),
        "seed": seed,
        "spearman_rho": rho,
        "displacement": displacement(phi_op, phi_acc),
        "top1_disagreement": top1(phi_op, phi_acc),
        "pairwise_disagreement": float(rho < 1.0 - 1e-12),
    }


def bootstrap_ci(x, n_boot=2000, alpha=0.1, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    boots = [np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(x)), float(lo), float(hi)


def main():
    n_seeds = 40
    n_list = [4, 6, 8]
    # Smoke: n_list = [4]; n_seeds = 5
    rng = np.random.default_rng(20260901)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_seeds)]
    out = Path("results/stage2_cost_n_sweep")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in n_list:
        for cell, prims in COST_BASE.items():
            print(f"\n=== n={n} {cell} ===")
            for i, seed in enumerate(seeds):
                r = run_one(n, cell, prims, seed)
                rows.append(r)
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  {i+1}/{n_seeds} κ={r['kappa_eff']:.2f} ρ={r['spearman_rho']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)

    summary_rows = []
    for (n, cell), g in df.groupby(["n_firms", "cell"]):
        m, lo, hi = bootstrap_ci(g["spearman_rho"].to_numpy())
        summary_rows.append({
            "n_firms": n,
            "cell": cell,
            "kappa_mean": g["kappa_eff"].mean(),
            "rho_mean": m,
            "rho_ci_lo": lo,
            "rho_ci_hi": hi,
            "rho_median": g["spearman_rho"].median(),
            "rho_q10": g["spearman_rho"].quantile(0.10),
            "rho_q90": g["spearman_rho"].quantile(0.90),
            "top1_mean": g["top1_disagreement"].mean(),
            "pairwise_mean": g["pairwise_disagreement"].mean(),
            "n_seeds": len(g),
        })
    summ = pd.DataFrame(summary_rows).sort_values(["n_firms", "kappa_mean"])
    summ.to_csv(out / "summary.csv", index=False)

    print("\n=== Jonckheere–Terpstra (ρ decreases with cell order k1<k2<k5<k10) ===")
    order = ["cost_k1", "cost_k2", "cost_k5", "cost_k10"]
    for n in n_list:
        samples = [df[(df.n_firms == n) & (df.cell == c)]["spearman_rho"].to_numpy()
                   for c in order]
        # scipy.stats.jonckheere_terpstra may need scipy>=1.12; fallback manual note
        try:
            from scipy.stats import jonckheere_terpstra as jt
            # alternative API: combine with group labels
            vals, groups = [], []
            for gi, c in enumerate(order):
                v = df[(df.n_firms == n) & (df.cell == c)]["spearman_rho"].to_numpy()
                vals.extend(v.tolist())
                groups.extend([gi] * len(v))
            # Prefer permutation trend: correlate rank(cell) with rho
            cell_rank = {c: i for i, c in enumerate(order)}
            sub = df[df.n_firms == n].copy()
            sub["ord"] = sub["cell"].map(cell_rank)
            # Negative trend expected
            from scipy.stats import spearmanr
            jt_rho, jt_p = spearmanr(sub["ord"], sub["spearman_rho"])
            print(f"  n={n}: Spearman(ord, ρ)={jt_rho:.3f}  p={jt_p:.4g}")
        except Exception as e:
            print(f"  n={n}: trend test failed ({e})")

    print("\n", summ.to_string(index=False))
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()