"""
Three-channel primitive κ sweep (Claude tautology defense).

Channels (vary across firms; others fixed):
  cost      : shortage/holding multipliers → cb/ch heterogeneity
  leadtime  : L_i heterogeneity
  (optional) persistence: deferred if DGP lacks per-firm φ

Induced κ_eff = max w_i / min w_i
w_i = (ch_i + cb_i) * phi(z) * sqrt(L_i + 1)

Rankings from residual-scale Δσ (semi-structural), weighted by induced w.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std
from src.shapley import exact_shapley, CachedGame


def phi_z(service_level: float) -> float:
    z = float(norm.ppf(np.clip(service_level, 1e-4, 1 - 1e-4)))
    return float(norm.pdf(z)), z


def induced_w(ch: float, cb: float, L: int, service_level: float = 0.95) -> float:
    pz, _ = phi_z(service_level)
    A = np.sqrt(max(L, 0) + 1)
    return float((ch + cb) * pz * A)


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


# --- Channel grids: list of per-firm (ch, cb, L) at retail echelon ---
# κ≈1: identical primitives
# Higher κ: graded heterogeneity in one channel only

CHANNEL_GRIDS = {
    "cost_k1": [
        (1.0, 9.0, 1),
        (1.0, 9.0, 1),
        (1.0, 9.0, 1),
        (1.0, 9.0, 1),
    ],
    "cost_k2": [
        (1.0, 4.0, 1),
        (1.0, 6.0, 1),
        (1.0, 9.0, 1),
        (1.0, 12.0, 1),
    ],
    "cost_k5": [
        (1.0, 2.0, 1),
        (1.0, 5.0, 1),
        (1.0, 12.0, 1),
        (1.0, 25.0, 1),
    ],
    "cost_k10": [
        (1.0, 1.0, 1),
        (1.0, 5.0, 1),
        (1.0, 15.0, 1),
        (1.0, 40.0, 1),
    ],
    "lead_k1": [
        (1.0, 9.0, 1),
        (1.0, 9.0, 1),
        (1.0, 9.0, 1),
        (1.0, 9.0, 1),
    ],
    "lead_k2": [
        (1.0, 9.0, 1),
        (1.0, 9.0, 1),
        (1.0, 9.0, 2),
        (1.0, 9.0, 3),
    ],
    "lead_k5": [
        (1.0, 9.0, 1),
        (1.0, 9.0, 2),
        (1.0, 9.0, 4),
        (1.0, 9.0, 8),
    ],
}


@dataclass
class Cfg:
    n_firms: int = 4
    n_periods: int = 500
    n_features: int = 5
    shared_factor_correlation: float = 0.6
    n_obs_heterogeneity: str = "moderate"
    n_seeds: int = 25
    master_seed: int = 20260831
    ridge_lambda: float = 1.0


def run_one(cfg: Cfg, cell: str, prims: list[tuple], seed: int) -> dict:
    dcfg = DemandSimulationConfig(
        n_firms=cfg.n_firms,
        n_periods=cfg.n_periods,
        n_features=cfg.n_features,
        shared_factor_correlation=cfg.shared_factor_correlation,
        n_obs_heterogeneity=cfg.n_obs_heterogeneity,
        seed=seed,
    )
    sim = generate_multi_firm_demand(dcfg)
    datasets = build_firm_datasets(
        sim, val_fraction=0.15, holdout_fraction=0.25, min_holdout=50
    )
    firms = tuple(sorted(datasets.keys()))
    paths = {n: datasets[n].y_holdout for n in firms}

    # Induced weights from primitives
    w = {}
    for i, name in enumerate(firms):
        ch, cb, L = prims[i]
        w[name] = induced_w(ch, cb, L)
    vals = np.array(list(w.values()))
    kappa = float(vals.max() / max(vals.min(), 1e-12))

    coal = CoalitionConfig(
        firms=firms,
        lambda_mode="fixed",
        residual_mode="predictive",
        ridge_lambda=cfg.ridge_lambda,
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
    channel = cell.split("_")[0]
    return {
        "cell": cell,
        "channel": channel,
        "kappa_eff": kappa,
        "seed": seed,
        "spearman_rho": spearman(phi_op, phi_acc),
        "displacement": displacement(phi_op, phi_acc),
        "rank_reversal": float(
            max(phi_op, key=phi_op.get) != max(phi_acc, key=phi_acc.get)
        ),
        "v_op_N": v_op(N),
        "eff_gap_op": abs(sum(phi_op.values()) - v_op(N)),
    }


def main():
    cfg = Cfg()
    out = Path("results/stage2_three_channel")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.master_seed)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=cfg.n_seeds)]

    rows = []
    for cell, prims in CHANNEL_GRIDS.items():
        print(f"\n=== {cell} ===")
        for i, seed in enumerate(seeds):
            r = run_one(cfg, cell, prims, seed)
            rows.append(r)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  {i+1}/{cfg.n_seeds}  κ={r['kappa_eff']:.2f}  ρ={r['spearman_rho']:.3f}")
        sub = [r for r in rows if r["cell"] == cell]
        print(
            f"  mean ρ={np.mean([r['spearman_rho'] for r in sub]):.3f}  "
            f"mean κ={np.mean([r['kappa_eff'] for r in sub]):.2f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    summ = (
        df.groupby(["cell", "channel"], as_index=False)
        .agg(
            kappa_mean=("kappa_eff", "mean"),
            spearman_mean=("spearman_rho", "mean"),
            spearman_std=("spearman_rho", "std"),
            displacement_mean=("displacement", "mean"),
            rank_reversal_mean=("rank_reversal", "mean"),
            n_seeds=("spearman_rho", "count"),
        )
        .sort_values(["channel", "kappa_mean"])
    )
    summ.to_csv(out / "summary.csv", index=False)
    print("\n=== Three-channel summary ===")
    print(summ.to_string(index=False))
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()