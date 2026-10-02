"""
R1 validation: residual-weighted Shapley vs path-cost Shapley.

Same firms / seeds / coalitions / (c_h, c_b):
  v_res(S)  = Σ_i w_i (σ_i({i}) − σ_i(S))     # Phase-1 predictive residual
  v_path(S) = Σ_i (C_i({i}) − C_i(S))         # inventory cost on hold-out
  φ_res  = Sh(v_res),  φ_path = Sh(v_path)

Path-cost policy (implementation-faithful):
  - single echelon per firm
  - service_level = critical fractile τ = c_b / (c_h + c_b)  (aligned with w)
  - base-stock: S = (L+1)*max(μ̂,0) + z * sqrt(L+1) * σ
  - simulator is deterministic when stochastic_lead_time=False
    (rng unused); one evaluation per (firm, coalition)

Usage:
  python -m experiments.run_r1_residual_vs_path
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import FirmDataset, build_firm_datasets
from src.federated_phase1_patch import (
    firm_predictive_residual_std,
    fit_ridge_for_coalition,
)
from src.inventory import EchelonConfig, simulate_serial_supply_chain
from src.shapley import CachedGame, exact_shapley

# ---------------------------------------------------------------------------
# Cost primitives
# ---------------------------------------------------------------------------
COST_CELLS: dict[str, list[tuple[float, float]]] = {
    "k1": [(1.0, 9.0)] * 4,
    "k2": [(1.0, 4.0), (1.0, 6.0), (1.0, 9.0), (1.0, 12.0)],
    "k5": [(1.0, 2.0), (1.0, 5.0), (1.0, 12.0), (1.0, 25.0)],
}

DEFAULT_L = 1


def inventory_weight(ch: float, cb: float, L: int = DEFAULT_L) -> float:
    """w = (c_h + c_b) φ(z_τ) √(L+1), τ = c_b/(c_h+c_b)."""
    tau = float(cb) / (float(ch) + float(cb))
    z = float(norm.ppf(np.clip(tau, 1e-6, 1.0 - 1e-6)))
    return float((ch + cb) * norm.pdf(z) * np.sqrt(max(L, 0) + 1))


def kappa_eff(weights: list[float]) -> float:
    w = np.asarray(weights, dtype=float)
    return float(np.max(w) / max(float(np.min(w)), 1e-15))


def spearman_dict(
    a: dict[str, float], b: dict[str, float], firms: list[str]
) -> float:
    xa = np.array([a[f] for f in firms], dtype=float)
    xb = np.array([b[f] for f in firms], dtype=float)
    if np.allclose(xa, xa[0]) or np.allclose(xb, xb[0]):
        return 0.0
    r, _ = spearmanr(xa, xb)
    return float(r) if np.isfinite(r) else 0.0


def l1_mean_dict(
    a: dict[str, float], b: dict[str, float], firms: list[str]
) -> float:
    xa = np.array([a[f] for f in firms], dtype=float)
    xb = np.array([b[f] for f in firms], dtype=float)
    return float(np.mean(np.abs(xa - xb)))


def firm_echelons(ch: float, cb: float) -> list[EchelonConfig]:
    """
    Single echelon; service level = critical fractile τ = cb/(ch+cb)
    so path-cost safety factor matches residual-weight construction.
    """
    tau = float(cb) / (float(ch) + float(cb))
    tau = float(np.clip(tau, 1e-4, 1.0 - 1e-4))
    return [
        EchelonConfig(
            holding_cost=float(ch),
            shortage_cost=float(cb),
            lead_time=DEFAULT_L,
            service_level=tau,
        )
    ]


def path_cost_once(
    demand: np.ndarray,
    forecast: np.ndarray,
    residual_std: float,
    echelons: list[EchelonConfig],
) -> float:
    """
    Single deterministic evaluation.
    Under stochastic_lead_time=False the simulator does not use rng;
    demand/forecast/residual_std fully determine the cost path.
    """
    out = simulate_serial_supply_chain(
        demand,
        forecast,
        float(residual_std),
        echelons,
        rng=None,
        stochastic_lead_time=False,
    )
    return float(out.total_cost)


def build_v_res(
    datasets: dict[str, FirmDataset],
    firms: list[str],
    weights: dict[str, float],
    lam0: float = 1.0,
    lambda_mode: str = "size_scaled",
):
    sigma_cache: dict[tuple[str, frozenset[str]], float] = {}

    def sigma(name: str, S: frozenset[str]) -> float:
        key = (name, S)
        if key in sigma_cache:
            return sigma_cache[key]
        beta = fit_ridge_for_coalition(
            datasets, S, lam0=lam0, lambda_mode=lambda_mode
        )
        s = float(firm_predictive_residual_std(beta, datasets[name]))
        sigma_cache[key] = s
        return s

    def v(S: frozenset[str]) -> float:
        if not S:
            return 0.0
        return float(
            sum(
                weights[name]
                * (sigma(name, frozenset({name})) - sigma(name, S))
                for name in S
            )
        )

    return v


def build_v_path(
    datasets: dict[str, FirmDataset],
    firms: list[str],
    prims: list[tuple[float, float]],
    seed: int,
    lam0: float = 1.0,
    lambda_mode: str = "size_scaled",
):
    """
    Path-cost savings vs singleton autarky, firm-specific (c_h, c_b).
    Residual scale = Phase-1 predictive RMS (same channel as v_res).
    seed is unused (deterministic simulator); kept for call-site compatibility.
    """
    del seed  # deterministic; no RNG stream
    cost_cache: dict[tuple[str, frozenset[str]], float] = {}
    firm_prim = {firms[i]: prims[i] for i in range(len(firms))}

    def cost_of(name: str, S: frozenset[str]) -> float:
        key = (name, S)
        if key in cost_cache:
            return cost_cache[key]
        ds = datasets[name]
        beta = fit_ridge_for_coalition(
            datasets, S, lam0=lam0, lambda_mode=lambda_mode
        )
        y = np.asarray(ds.y_holdout, dtype=float).ravel()
        X = np.asarray(ds.X_holdout, dtype=float)
        T = min(len(y), X.shape[0])
        y = y[:T]
        forecast = X[:T] @ beta
        residual_std = max(float(firm_predictive_residual_std(beta, ds)), 1e-8)
        ch, cb = firm_prim[name]
        echelons = firm_echelons(ch, cb)
        c = path_cost_once(y, forecast, residual_std, echelons)
        cost_cache[key] = c
        return c

    def v(S: frozenset[str]) -> float:
        if not S:
            return 0.0
        return float(
            sum(
                cost_of(name, frozenset({name})) - cost_of(name, S)
                for name in S
            )
        )

    return v


def shapley_payments(
    firms: list[str], v_func
) -> tuple[dict[str, float], float]:
    game = CachedGame(players=list(firms), v_func=v_func)
    phi = exact_shapley(game)
    vN = float(v_func(frozenset(firms)))
    return dict(phi), vN


def run_one_seed(
    seed: int,
    cell: str,
    prims: list[tuple[float, float]],
    n_firms: int = 4,
    n_periods: int = 400,
    lam0: float = 1.0,
    lambda_mode: str = "size_scaled",
) -> dict:
    cfg = DemandSimulationConfig(
        n_firms=n_firms,
        n_periods=n_periods,
        n_features=5,
        seed=int(seed),
        shared_factor_correlation=0.6,
        n_obs_heterogeneity="fixed",
        signal_quality_profile="fixed",
        noise_profile="fixed",
    )
    sim = generate_multi_firm_demand(cfg)
    datasets = build_firm_datasets(
        sim, val_fraction=0.15, holdout_fraction=0.25, min_holdout=30
    )
    firms = sorted(datasets.keys())
    assert len(firms) == n_firms
    prims = list(prims[:n_firms])

    w_list = [inventory_weight(ch, cb) for ch, cb in prims]
    weights = {firms[i]: w_list[i] for i in range(n_firms)}
    ke = kappa_eff(w_list)

    v_res = build_v_res(
        datasets, firms, weights, lam0=lam0, lambda_mode=lambda_mode
    )
    v_path = build_v_path(
        datasets,
        firms,
        prims,
        seed=seed,
        lam0=lam0,
        lambda_mode=lambda_mode,
    )

    phi_res, vN_res = shapley_payments(firms, v_res)
    phi_path, vN_path = shapley_payments(firms, v_path)

    rho = spearman_dict(phi_res, phi_path, firms)
    l1 = l1_mean_dict(phi_res, phi_path, firms)

    return {
        "seed": int(seed),
        "cell": cell,
        "kappa_eff": ke,
        "rho_phi": rho,
        "l1_phi": l1,
        "vN_path": vN_path,
        "vN_res": vN_res,
        "neg_vN_path": int(vN_path < 0),
        "neg_vN_res": int(vN_res < 0),
        "eff_gap_res": abs(sum(phi_res[f] for f in firms) - vN_res),
        "eff_gap_path": abs(sum(phi_path[f] for f in firms) - vN_path),
    }


def main():
    # Full paper: n_seeds=30, cells=all. Smoke: n_seeds=5, cells=["k1"].
    n_seeds = 30
    cells = list(COST_CELLS.keys())
    n_firms = 4
    n_periods = 400
    lambda_mode = "size_scaled"
    lam0 = 1.0

    out = Path("results/r1_residual_vs_path")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260831)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_seeds)]

    rows: list[dict] = []
    for cell in cells:
        prims = COST_CELLS[cell]
        ke = kappa_eff(
            [inventory_weight(ch, cb) for ch, cb in prims[:n_firms]]
        )
        print(f"=== cell={cell}  κ_eff≈{ke:.2f} ===")
        for i, seed in enumerate(seeds):
            r = run_one_seed(
                seed=seed,
                cell=cell,
                prims=prims,
                n_firms=n_firms,
                n_periods=n_periods,
                lam0=lam0,
                lambda_mode=lambda_mode,
            )
            rows.append(r)
            if (i + 1) % 10 == 0 or i == 0 or (i + 1) == n_seeds:
                print(
                    f"  {i+1}/{n_seeds}  ρ(φ)={r['rho_phi']:.3f}  "
                    f"ℓ1={r['l1_phi']:.4f}  "
                    f"vN_path={r['vN_path']:.2f}  vN_res={r['vN_res']:.4f}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)

    summ = (
        df.groupby("cell", as_index=False)
        .agg(
            kappa_eff=("kappa_eff", "mean"),
            rho_phi_mean=("rho_phi", "mean"),
            rho_phi_std=("rho_phi", "std"),
            rho_phi_median=("rho_phi", "median"),
            l1_mean=("l1_phi", "mean"),
            l1_median=("l1_phi", "median"),
            p_neg_vN_path=("neg_vN_path", "mean"),
            p_neg_vN_res=("neg_vN_res", "mean"),
            mean_vN_path=("vN_path", "mean"),
            mean_vN_res=("vN_res", "mean"),
            max_eff_gap_res=("eff_gap_res", "max"),
            max_eff_gap_path=("eff_gap_path", "max"),
            n_seeds=("seed", "count"),
        )
    )
    summ.to_csv(out / "summary.csv", index=False)
    print("\n=== R1 residual vs path-cost summary ===")
    print(summ.to_string(index=False))
    print(f"Wrote {out}/")
    print(
        "Gate:\n"
        "  k1 mean ρ(φ_res, φ_path) ≳ 0.7  → residual tracks path ranks under τ-aligned policy.\n"
        "  k1 ρ ≈ 0                 → residual ≠ path-cost ranks (boundary; expected possible)."
    )


if __name__ == "__main__":
    main()
