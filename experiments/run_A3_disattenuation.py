"""
A3′: same-seed reliability and disattenuated resid–path rank correlation.

For each seed:
  - Split hold-out into A/B halves
  - Residual φ_op on half A and half B  → spearman → r_resid (Spearman–Brown optional)
  - Path-cost φ_op on half A and half B → r_path
  - Full-window residual vs path ranks → rho_raw
  - rho_star = rho_raw / sqrt(r_resid * r_path)  (clipped)

Also reports bootstrap CI on mean rho_star.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets, FirmDataset
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std
from src.inventory import EchelonConfig, simulate_serial_supply_chain
from src.shapley import exact_shapley, CachedGame


def spearman_vec(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.empty(len(a))
    rb = np.empty(len(b))
    ra[np.argsort(a)] = np.arange(len(a), dtype=float)
    rb[np.argsort(b)] = np.arange(len(b), dtype=float)
    n = len(a)
    return float(1 - 6 * np.sum((ra - rb) ** 2) / (n * (n * n - 1)))


def spearman_dict(phi_a: dict, phi_b: dict) -> float:
    firms = sorted(phi_a.keys())
    return spearman_vec(
        np.array([phi_a[f] for f in firms]),
        np.array([phi_b[f] for f in firms]),
    )


def half_ds(datasets: dict, which: str) -> dict:
    out = {}
    for name, ds in datasets.items():
        X, y = ds.X_holdout, ds.y_holdout
        mid = len(y) // 2
        if which == "A":
            Xv, yv = X[:mid], y[:mid]
        else:
            Xv, yv = X[mid:], y[mid:]
        out[name] = FirmDataset(
            name=ds.name,
            X_train=ds.X_train,
            y_train=ds.y_train,
            X_val=Xv,
            y_val=yv,
            X_holdout=ds.X_holdout,
            y_holdout=ds.y_holdout,
        )
    return out


def default_echelons():
    return [
        EchelonConfig(1.0, 9.0, 1, 0.95),
        EchelonConfig(0.6, 0.0, 2, 0.90),
        EchelonConfig(0.3, 0.0, 3, 0.90),
    ]


def phi_resid(datasets, paths, firms, seed, lambda_mode="fixed"):
    coal = CoalitionConfig(
        firms=firms,
        lambda_mode=lambda_mode,
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
        s = firm_predictive_residual_std(beta, datasets[i], min_val_rows=8)
        cache[key] = s
        return s

    def v(S):
        if not S:
            return 0.0
        return float(sum(sigma(i, frozenset({i})) - sigma(i, S) for i in S))

    return exact_shapley(CachedGame(list(firms), v))


def phi_path(datasets, paths, firms, seed, K_crn=8):
    coal = CoalitionConfig(
        firms=firms,
        lambda_mode="fixed",
        residual_mode="predictive",
        ridge_lambda=1.0,
        seed=seed,
    )
    ev = CoalitionEvaluator(datasets, paths, coal)
    echs = default_echelons()
    cache = {}

    def cost(i, S):
        key = (i, S)
        if key in cache:
            return cache[key]
        beta = ev.fit(S)
        ds = datasets[i]
        resid = firm_predictive_residual_std(beta, ds, min_val_rows=8)
        # Use val as evaluation path when holdout was split into val
        X_eval = ds.X_val if len(ds.y_val) >= 8 else ds.X_holdout
        y_eval = ds.y_val if len(ds.y_val) >= 8 else ds.y_holdout
        forecast = X_eval @ beta
        demand = np.asarray(y_eval, dtype=float)
        costs = []
        for k in range(K_crn):
            out = simulate_serial_supply_chain(
                demand, forecast, resid, echs, rng=None
            )
            costs.append(out.total_cost)
        c = float(np.mean(costs))
        cache[key] = c
        return c

    def v(S):
        if not S:
            return 0.0
        return float(sum(cost(i, frozenset({i})) - cost(i, S) for i in S))

    return exact_shapley(CachedGame(list(firms), v))


def run_seed(seed: int) -> dict:
    cfg = DemandSimulationConfig(
        n_firms=4,
        n_periods=700,
        n_features=5,
        shared_factor_correlation=0.6,
        n_obs_heterogeneity="moderate",
        seed=seed,
    )
    sim = generate_multi_firm_demand(cfg)
    full = build_firm_datasets(
        sim, val_fraction=0.12, holdout_fraction=0.35, min_holdout=90
    )
    firms = tuple(sorted(full.keys()))
    paths_full = {n: full[n].y_holdout for n in firms}

    # Full-window residual vs path
    phi_r_full = phi_resid(full, paths_full, firms, seed)
    # Path on full holdout: temporarily point val to holdout for cost eval
    full_path_ds = {}
    for n, ds in full.items():
        full_path_ds[n] = FirmDataset(
            name=ds.name,
            X_train=ds.X_train,
            y_train=ds.y_train,
            X_val=ds.X_holdout,
            y_val=ds.y_holdout,
            X_holdout=ds.X_holdout,
            y_holdout=ds.y_holdout,
        )
    phi_p_full = phi_path(full_path_ds, paths_full, firms, seed)
    rho_raw = spearman_dict(phi_r_full, phi_p_full)

    # Split-half residual
    ds_A = half_ds(full, "A")
    ds_B = half_ds(full, "B")
    paths_A = {n: ds_A[n].y_holdout for n in firms}
    paths_B = {n: ds_B[n].y_holdout for n in firms}
    phi_r_A = phi_resid(ds_A, paths_A, firms, seed)
    phi_r_B = phi_resid(ds_B, paths_B, firms, seed)
    r_resid_half = spearman_dict(phi_r_A, phi_r_B)
    # Spearman–Brown full-window estimate
    r_resid = (
        2 * r_resid_half / (1 + r_resid_half)
        if r_resid_half > -0.999
        else float("nan")
    )

    # Split-half path
    phi_p_A = phi_path(ds_A, paths_A, firms, seed)
    phi_p_B = phi_path(ds_B, paths_B, firms, seed)
    r_path_half = spearman_dict(phi_p_A, phi_p_B)
    r_path = (
        2 * r_path_half / (1 + r_path_half) if r_path_half > -0.999 else float("nan")
    )

    denom = np.sqrt(max(r_resid, 1e-8) * max(r_path, 1e-8))
    rho_star = float(np.clip(rho_raw / denom, -1.5, 1.5)) if denom > 0 else float("nan")

    return {
        "seed": seed,
        "rho_raw": rho_raw,
        "r_resid_half": r_resid_half,
        "r_resid_SB": r_resid,
        "r_path_half": r_path_half,
        "r_path_SB": r_path,
        "rho_star": rho_star,
    }


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.1, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=len(x), replace=True)
        boots.append(np.mean(samp))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(x)), float(lo), float(hi)


def main():
    n_seeds = 30
    rng = np.random.default_rng(20260831)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_seeds)]
    out = Path("results/A3_disattenuation")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, seed in enumerate(seeds):
        try:
            r = run_seed(seed)
            rows.append(r)
            print(
                f"  {i+1}/{n_seeds}  rho={r['rho_raw']:.3f}  "
                f"r_res={r['r_resid_SB']:.3f}  r_path={r['r_path_SB']:.3f}  "
                f"rho*={r['rho_star']:.3f}"
            )
        except Exception as e:
            print(f"  [warn] seed={seed}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)

    print("\n=== A3′ summary ===")
    for col in ("rho_raw", "r_resid_half", "r_resid_SB", "r_path_half", "r_path_SB", "rho_star"):
        m, lo, hi = bootstrap_ci(df[col].to_numpy())
        print(f"  {col:14s}  mean={m:.3f}  90% CI [{lo:.3f}, {hi:.3f}]")

    print(
        "\nEstimands:\n"
        "  rho_raw   = full-window residual ranks vs path ranks\n"
        "  r_*_half  = split-half Spearman of φ on halved windows\n"
        "  r_*_SB    = Spearman–Brown full-window reliability estimate\n"
        "  rho_star  = rho_raw / sqrt(r_resid_SB * r_path_SB)\n"
    )
    print(f"Wrote {out / 'results.csv'}")


if __name__ == "__main__":
    main()