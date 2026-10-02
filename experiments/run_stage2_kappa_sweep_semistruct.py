"""
Semi-structural κ sweep: both games from the same Phase-1 residual scales.

v_acc(S) = sum_i [σ_i({i}) - σ_i(S)]
v_op(S)  = sum_i w_i [σ_i({i}) - σ_i(S)]

w_i = cost multiplier for firm i (κ schedule).
σ_i(S) = firm_predictive_residual_std(β_S, firm i).

Expect: ρ(κ=1) ≈ 1; ρ decreases as κ rises.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std
from src.shapley import exact_shapley, CachedGame


KAPPA_SCHEDULES = {
    "kappa_1": [1.0, 1.0, 1.0, 1.0],
    "kappa_2": [1.0, 1.0, 1.5, 2.0],
    "kappa_5": [1.0, 1.5, 3.0, 5.0],
    "kappa_10": [1.0, 2.0, 5.0, 10.0],
}


@dataclass
class SweepConfig:
    n_firms: int = 4
    n_periods: int = 500
    n_features: int = 5
    shared_factor_correlation: float = 0.6
    n_obs_heterogeneity: str = "moderate"
    holdout_fraction: float = 0.25
    val_fraction: float = 0.15
    min_holdout: int = 50
    ridge_lambda: float = 1.0
    lambda_mode: str = "fixed"
    residual_mode: str = "predictive"
    n_seeds: int = 30
    master_seed: int = 20260831
    schedules: dict = field(default_factory=lambda: dict(KAPPA_SCHEDULES))


def kappa_of(mults):
    a = np.asarray(mults, dtype=float)
    return float(a.max() / max(a.min(), 1e-12))


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


def run_one(cfg: SweepConfig, schedule: str, mults: list[float], seed: int) -> dict:
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
        sim,
        val_fraction=cfg.val_fraction,
        holdout_fraction=cfg.holdout_fraction,
        min_holdout=cfg.min_holdout,
    )
    firms = tuple(sorted(datasets.keys()))
    paths = {n: datasets[n].y_holdout for n in firms}
    weights = {firms[i]: float(mults[i]) for i in range(cfg.n_firms)}

    coal_cfg = CoalitionConfig(
        firms=firms,
        lambda_mode=cfg.lambda_mode,
        residual_mode=cfg.residual_mode,
        ridge_lambda=cfg.ridge_lambda,
        seed=seed,
    )
    ev = CoalitionEvaluator(datasets, paths, coal_cfg)

    sigma_cache: dict[tuple[str, frozenset], float] = {}

    def sigma(i: str, S: frozenset) -> float:
        key = (i, S)
        if key in sigma_cache:
            return sigma_cache[key]
        beta = ev.fit(S)
        s = firm_predictive_residual_std(beta, datasets[i])
        sigma_cache[key] = s
        return s

    def v_acc(S: frozenset) -> float:
        if not S:
            return 0.0
        return float(sum(sigma(i, frozenset({i})) - sigma(i, S) for i in S))

    def v_op(S: frozenset) -> float:
        if not S:
            return 0.0
        return float(
            sum(weights[i] *
                (sigma(i, frozenset({i})) - sigma(i, S)) for i in S)
        )

    game_acc = CachedGame(list(firms), v_acc)
    game_op = CachedGame(list(firms), v_op)
    phi_acc = exact_shapley(game_acc)
    phi_op = exact_shapley(game_op)

    N = frozenset(firms)
    return {
        "schedule": schedule,
        "kappa_eff": kappa_of(mults),
        "seed": seed,
        "spearman_rho": spearman(phi_op, phi_acc),
        "displacement": displacement(phi_op, phi_acc),
        "rank_reversal": float(
            max(phi_op, key=phi_op.get) != max(phi_acc, key=phi_acc.get)
        ),
        "v_op_N": v_op(N),
        "v_acc_N": v_acc(N),
        "eff_gap_op": abs(sum(phi_op.values()) - v_op(N)),
        "eff_gap_acc": abs(sum(phi_acc.values()) - v_acc(N)),
    }


def main():
    cfg = SweepConfig()
    out = Path("results/stage2_kappa_semistruct")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.master_seed)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=cfg.n_seeds)]

    rows = []
    t0 = time.time()
    for name, mults in cfg.schedules.items():
        print(f"\n=== {name}  κ={kappa_of(mults):.2f} ===")
        for i, seed in enumerate(seeds):
            r = run_one(cfg, name, mults, seed)
            rows.append(r)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  seed {i+1}/{cfg.n_seeds}  ρ={r['spearman_rho']:.3f}")
        sub = [r for r in rows if r["schedule"] == name]
        print(
            f"  mean ρ={np.mean([r['spearman_rho'] for r in sub]):.3f}  "
            f"median={np.median([r['spearman_rho'] for r in sub]):.3f}  "
            f"reversal={np.mean([r['rank_reversal'] for r in sub]):.2f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)

    summ = (
        df.groupby(["schedule", "kappa_eff"], as_index=False)
        .agg(
            spearman_mean=("spearman_rho", "mean"),
            spearman_std=("spearman_rho", "std"),
            spearman_lo=("spearman_rho", lambda x: float(np.quantile(x, 0.1))),
            spearman_hi=("spearman_rho", lambda x: float(np.quantile(x, 0.9))),
            displacement_mean=("displacement", "mean"),
            rank_reversal_mean=("rank_reversal", "mean"),
            v_op_N_mean=("v_op_N", "mean"),
            eff_gap_op_mean=("eff_gap_op", "mean"),
            n_seeds=("spearman_rho", "count"),
        )
        .sort_values("kappa_eff")
    )
    summ.to_csv(out / "summary_by_kappa.csv", index=False)
    (out / "config.json").write_text(json.dumps(
        {"config": asdict(cfg)}, indent=2, default=str))

    print("\n=== Semi-structural κ-sweep summary ===")
    print(summ.to_string(index=False))
    print(f"\nelapsed {time.time()-t0:.1f}s → {out}/")

    rho1 = float(summ.loc[summ["schedule"] ==
                 "kappa_1", "spearman_mean"].iloc[0])
    if rho1 >= 0.85:
        print("PASS pattern: high ρ at κ=1 — residual channel is aligned.")
    else:
        print(
            f"κ=1 mean ρ={rho1:.3f} still low — increase n_periods / min_holdout "
            "or reduce residual noise before inventory-path claims."
        )


if __name__ == "__main__":
    main()
