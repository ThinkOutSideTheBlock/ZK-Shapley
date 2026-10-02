"""
IR / efficiency / superadditivity diagnostics on residual-scale v_op
and accuracy v_acc (member-relative, v({i})=0).

IR here: phi_i >= v({i}) = 0  (standard Shapley IR vs singleton).
Efficiency: sum phi = v(N) within tol.
Pairwise superadditivity probe: v({i,j}) >= v({i})+v({j}) = 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std
from src.shapley import exact_shapley, CachedGame
from src.mechanism import run_zk_shapley_mechanism, run_accuracy_shapley_mechanism
from scipy.stats import norm


def induced_w(ch: float, cb: float, L: int = 1, sl: float = 0.95) -> float:
    z = float(norm.ppf(np.clip(sl, 1e-4, 1 - 1e-4)))
    return float((ch + cb) * float(norm.pdf(z)) * np.sqrt(L + 1))


COST_CELLS = {
    "k1": [(1.0, 9.0)] * 4,
    "k2": [(1.0, 4.0), (1.0, 6.0), (1.0, 9.0), (1.0, 12.0)],
    "k5": [(1.0, 2.0), (1.0, 5.0), (1.0, 12.0), (1.0, 25.0)],
    "k10": [(1.0, 1.0), (1.0, 5.0), (1.0, 15.0), (1.0, 40.0)],
}


def make_v_op(datasets, paths, firms, prims, seed: int):
    w = {
        firms[i]: induced_w(prims[i][0], prims[i][1])
        for i in range(len(firms))
    }
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
        if key not in cache:
            cache[key] = float(
                firm_predictive_residual_std(ev.fit(S), datasets[i])
            )
        return cache[key]

    def v_op(S):
        if not S:
            return 0.0
        return float(
            sum(w[i] * (sigma(i, frozenset({i})) - sigma(i, S)) for i in S)
        )

    return v_op


def make_v_acc(datasets, firms, seed: int):
    coal = CoalitionConfig(
        firms=firms,
        lambda_mode="fixed",
        residual_mode="predictive",
        ridge_lambda=1.0,
        seed=seed,
    )
    paths = {n: datasets[n].y_holdout for n in firms}
    ev = CoalitionEvaluator(datasets, paths, coal)
    autarky = {}

    def rmse(i, S):
        from src.federated_phase1_patch import firm_holdout_rmse

        beta = ev.fit(S)
        return float(firm_holdout_rmse(beta, datasets[i]))

    def v_acc(S):
        if not S:
            return 0.0
        total = 0.0
        for i in S:
            if i not in autarky:
                autarky[i] = rmse(i, frozenset({i}))
            total += autarky[i] - rmse(i, S)
        return float(total)

    return v_acc


def pairwise_superadd_rate(v, firms) -> float:
    firms = list(firms)
    ok = 0
    tot = 0
    for a in range(len(firms)):
        for b in range(a + 1, len(firms)):
            i, j = firms[a], firms[b]
            vij = v(frozenset({i, j}))
            tot += 1
            if vij >= -1e-10:
                ok += 1
    return ok / tot if tot else float("nan")


def main():
    n_seeds = 30
    n_periods = 400
    out = Path("results/ir_core_diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for cell, prims in COST_CELLS.items():
        print(f"=== {cell} ===")
        for s in range(n_seeds):
            seed = 10_000 + s
            cfg = DemandSimulationConfig(
                n_firms=4,
                n_periods=n_periods,
                n_features=5,
                shared_factor_correlation=0.6,
                n_obs_heterogeneity="fixed",
                signal_quality_profile="fixed",
                noise_profile="fixed",
                seed=seed,
            )
            sim = generate_multi_firm_demand(cfg)
            datasets = build_firm_datasets(
                sim, val_fraction=0.15, holdout_fraction=0.25, min_holdout=30
            )
            firms = tuple(sorted(datasets.keys()))
            paths = {n: datasets[n].y_holdout for n in firms}

            v_op = make_v_op(datasets, paths, firms, prims, seed)
            v_acc = make_v_acc(datasets, firms, seed)

            res_op = run_zk_shapley_mechanism(
                list(firms), v_op, exact=True
            )
            res_acc = run_accuracy_shapley_mechanism(
                list(firms), v_acc, exact=True
            )

            phi_op = res_op.payments
            phi_acc = res_acc.payments
            vN_op = v_op(frozenset(firms))
            vN_acc = v_acc(frozenset(firms))

            rows.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "vN_op": vN_op,
                    "vN_acc": vN_acc,
                    "eff_gap_op": res_op.efficiency_gap,
                    "eff_gap_acc": res_acc.efficiency_gap,
                    "frac_ir_op": float(
                        np.mean([phi_op[f] >= -1e-12 for f in firms])
                    ),
                    "all_ir_op": bool(
                        all(phi_op[f] >= -1e-12 for f in firms)
                    ),
                    "frac_ir_acc": float(
                        np.mean([phi_acc[f] >= -1e-12 for f in firms])
                    ),
                    "all_ir_acc": bool(
                        all(phi_acc[f] >= -1e-12 for f in firms)
                    ),
                    "superadd_pair_op": pairwise_superadd_rate(v_op, firms),
                    "superadd_pair_acc": pairwise_superadd_rate(v_acc, firms),
                    "neg_vN_op": int(vN_op < 0),
                    "neg_vN_acc": int(vN_acc < 0),
                }
            )
            if (s + 1) % 10 == 0:
                print(f"  {s+1}/{n_seeds}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    summ = (
        df.groupby("cell", as_index=False)
        .agg(
            mean_frac_ir_op=("frac_ir_op", "mean"),
            rate_all_ir_op=("all_ir_op", "mean"),
            mean_frac_ir_acc=("frac_ir_acc", "mean"),
            rate_all_ir_acc=("all_ir_acc", "mean"),
            mean_superadd_op=("superadd_pair_op", "mean"),
            mean_superadd_acc=("superadd_pair_acc", "mean"),
            p_neg_vN_op=("neg_vN_op", "mean"),
            p_neg_vN_acc=("neg_vN_acc", "mean"),
            max_eff_gap_op=("eff_gap_op", "max"),
            max_eff_gap_acc=("eff_gap_acc", "max"),
            n=("seed", "count"),
        )
    )
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.to_string(index=False))
    print(f"Wrote {out}/")


if __name__ == "__main__":
    main()
