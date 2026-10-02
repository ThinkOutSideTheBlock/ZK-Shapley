"""
Stage 2 / Gate 1′ — E0 zero-noise structural check (Claude plan).

v_acc(S) = sum_{i in S} [σ_i({i}) - σ_i(S)]
v_op(S)  = sum_{i in S} w_i [σ_i({i}) - σ_i(S)]

Gate 1′ passes iff:
  - at κ_eff = 1 (w_i identical): Spearman ρ == 1
  - at κ_eff > 1: ρ < 1 and decreases as κ increases
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def all_coalitions(firms: list[str]):
    n = len(firms)
    for k in range(0, n + 1):
        for comb in combinations(firms, k):
            yield frozenset(comb)


def sigma_i_of_S(
    i: str,
    S: frozenset[str],
    sigma_singleton: dict[str, float],
    pool_gain: float = 0.35,
) -> float:
    """
    Deterministic residual scale: singleton level, reduced by pooling gain
    that grows with |S| and is larger when partners have smaller sigma
    (simple shared-factor style improvement).
    """
    if not S or i not in S:
        return float(sigma_singleton[i])
    if len(S) == 1:
        return float(sigma_singleton[i])
    own = sigma_singleton[i]
    partners = [sigma_singleton[j] for j in S if j != i]
    peer = float(np.mean(partners))
    # Pooling pulls toward a lower effective scale
    blended = 0.5 * own + 0.5 * min(own, peer)
    reduction = pool_gain * (len(S) - 1) / max(len(sigma_singleton) - 1, 1)
    return float(max(own * (1.0 - reduction) * (blended / own), 1e-6))


def characteristic_values(
    firms: list[str],
    sigma_singleton: dict[str, float],
    weights: dict[str, float],
    pool_gain: float = 0.35,
):
    """Return dicts S -> v_acc(S), S -> v_op(S) for all coalitions."""
    v_acc = {}
    v_op = {}
    for S in all_coalitions(firms):
        if not S:
            v_acc[S] = 0.0
            v_op[S] = 0.0
            continue
        acc = 0.0
        op = 0.0
        for i in S:
            s_auto = sigma_singleton[i]
            s_S = sigma_i_of_S(i, S, sigma_singleton, pool_gain=pool_gain)
            gain = s_auto - s_S
            acc += gain
            op += weights[i] * gain
        v_acc[S] = acc
        v_op[S] = op
    return v_acc, v_op


def exact_shapley_from_dict(firms: list[str], v: dict) -> dict[str, float]:
    """Exact Shapley for n<=10 from a fully tabulated v."""
    n = len(firms)
    from math import factorial

    phi = {i: 0.0 for i in firms}
    for i in firms:
        others = [j for j in firms if j != i]
        for k in range(len(others) + 1):
            for comb in combinations(others, k):
                S = frozenset(comb)
                S_i = S | {i}
                w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                phi[i] += w * (v[S_i] - v[S])
    return phi


def spearman(phi_a: dict[str, float], phi_b: dict[str, float]) -> float:
    firms = sorted(phi_a.keys())
    a = np.array([phi_a[f] for f in firms])
    b = np.array([phi_b[f] for f in firms])
    ra = stats_rank(a)
    rb = stats_rank(b)
    n = len(firms)
    d2 = np.sum((ra - rb) ** 2)
    return float(1 - 6 * d2 / (n * (n**2 - 1)))


def stats_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def kappa_of_weights(weights: dict[str, float]) -> float:
    vals = np.array(list(weights.values()), dtype=float)
    return float(np.max(vals) / max(np.min(vals), 1e-12))


def run_cell(
    firms: list[str],
    sigma_singleton: dict[str, float],
    weights: dict[str, float],
    pool_gain: float = 0.35,
) -> dict:
    v_acc, v_op = characteristic_values(
        firms, sigma_singleton, weights, pool_gain=pool_gain
    )
    phi_acc = exact_shapley_from_dict(firms, v_acc)
    phi_op = exact_shapley_from_dict(firms, v_op)
    rho = spearman(phi_op, phi_acc)
    # Efficiency checks
    N = frozenset(firms)
    eff_acc = abs(sum(phi_acc.values()) - v_acc[N])
    eff_op = abs(sum(phi_op.values()) - v_op[N])
    return {
        "kappa_eff": kappa_of_weights(weights),
        "spearman_rho": rho,
        "v_op_N": v_op[N],
        "v_acc_N": v_acc[N],
        "eff_gap_op": eff_op,
        "eff_gap_acc": eff_acc,
        "phi_op": phi_op,
        "phi_acc": phi_acc,
    }


def main():
    firms = [f"f{i}" for i in range(4)]
    # Heterogeneous singleton residual scales (always; weights carry κ)
    sigma_singleton = {
        "f0": 1.00,
        "f1": 1.15,
        "f2": 1.30,
        "f3": 1.50,
    }

    # Weight schedules: κ_eff = 1, 2, 5, 10
    schedules = {
        "kappa=1": {f: 1.0 for f in firms},
        "kappa=2": {"f0": 1.0, "f1": 1.0, "f2": 1.5, "f3": 2.0},
        "kappa=5": {"f0": 1.0, "f1": 1.5, "f2": 3.0, "f3": 5.0},
        "kappa=10": {"f0": 1.0, "f1": 2.0, "f2": 5.0, "f3": 10.0},
    }

    rows = []
    print("=== Stage 2 / Gate 1′ — E0 structural check ===\n")
    for name, w in schedules.items():
        res = run_cell(firms, sigma_singleton, w)
        rows.append(
            {
                "cell": name,
                "kappa_eff": res["kappa_eff"],
                "spearman_rho": res["spearman_rho"],
                "v_op_N": res["v_op_N"],
                "v_acc_N": res["v_acc_N"],
                "eff_gap_op": res["eff_gap_op"],
                "eff_gap_acc": res["eff_gap_acc"],
            }
        )
        print(
            f"{name:10s}  κ={res['kappa_eff']:.2f}  "
            f"ρ={res['spearman_rho']:.4f}  "
            f"v_op(N)={res['v_op_N']:.4f}  "
            f"eff_op={res['eff_gap_op']:.2e}"
        )

    df = pd.DataFrame(rows)
    out = Path("results/stage2_E0")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "E0_summary.csv", index=False)

    # Gate logic
    rho_at_1 = float(df.loc[df["cell"] == "kappa=1", "spearman_rho"].iloc[0])
    rhos = df["spearman_rho"].to_numpy()
    kappas = df["kappa_eff"].to_numpy()
    mono = all(rhos[i] >= rhos[i + 1] - 1e-9 for i in range(len(rhos) - 1))

    print("\n=== Gate 1′ decision ===")
    print(f"  ρ at κ=1     : {rho_at_1:.6f}  (need ≈ 1)")
    print(f"  monotone ↓   : {mono}")
    pass_align = abs(rho_at_1 - 1.0) < 1e-9
    pass_mono = mono and rhos[-1] < rhos[0] - 1e-9
    if pass_align and pass_mono:
        print("  GATE 1′ PASSED — theory+structure OK; proceed to redesigned κ sweep.")
    elif pass_align and not pass_mono:
        print("  ALIGNMENT OK; monotonicity weak — inspect weight schedule.")
    else:
        print("  GATE 1′ FAILED — structural bug in value mapping.")

    print(f"\nWrote {out / 'E0_summary.csv'}")


if __name__ == "__main__":
    main()