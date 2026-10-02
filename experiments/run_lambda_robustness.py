"""
λ-robustness for residual-fixed reweight experiments.

Holds coalition residual scales σ_i(S) fixed within each λ mode, then
compares Spearman(φ^{res}, φ^σ) under:
  - lambda_mode = "size_scaled"  (canonical: λ_S = λ_0 * n_S)
  - lambda_mode = "fixed"        (λ_S = λ_0)

Design matches Phase B: cache Δ from validation residual RMS; only w changes
across cost cells k1/k2/k5/k10.

Usage:
  python -m experiments.run_lambda_robustness
"""
from __future__ import annotations

from itertools import chain, combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.federated_phase1_patch import (
    fit_ridge_for_coalition,
    firm_predictive_residual_std,
)
from src.shapley import CachedGame, exact_shapley


def critical_tau(ch: float, cb: float) -> float:
    return float(cb) / (float(ch) + float(cb))


def induced_w(ch: float, cb: float, L: int = 1) -> float:
    tau = critical_tau(ch, cb)
    z = float(norm.ppf(np.clip(tau, 1e-6, 1.0 - 1e-6)))
    return float((ch + cb) * norm.pdf(z) * np.sqrt(L + 1))


COST_BASE = {
    "k1": [(1.0, 9.0)] * 8,
    "k2": [
        (1.0, 4.0), (1.0, 6.0), (1.0, 9.0), (1.0, 12.0),
        (1.0, 5.0), (1.0, 8.0), (1.0, 10.0), (1.0, 14.0),
    ],
    "k5": [
        (1.0, 2.0), (1.0, 5.0), (1.0, 12.0), (1.0, 25.0),
        (1.0, 3.0), (1.0, 8.0), (1.0, 15.0), (1.0, 20.0),
    ],
    "k10": [
        (1.0, 1.0), (1.0, 5.0), (1.0, 15.0), (1.0, 40.0),
        (1.0, 2.0), (1.0, 8.0), (1.0, 20.0), (1.0, 35.0),
    ],
}


def powerset(firms: tuple[str, ...]):
    s = list(firms)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def make_weights(firms: tuple[str, ...], prims: list[tuple[float, float]]) -> dict[str, float]:
    n = len(firms)
    prims = prims[:n]
    return {firms[i]: induced_w(prims[i][0], prims[i][1]) for i in range(n)}


def kappa_eff(w: dict[str, float]) -> float:
    vals = np.asarray(list(w.values()), dtype=float)
    return float(np.max(vals) / max(np.min(vals), 1e-15))


def cache_sigma(
    datasets: dict,
    firms: tuple[str, ...],
    lam0: float,
    lambda_mode: str,
) -> dict[tuple[str, frozenset[str]], float]:
    """σ_i(S) for all coalitions (validation residual RMS)."""
    cache: dict[tuple[str, frozenset[str]], float] = {}
    for S_t in powerset(firms):
        S = frozenset(S_t)
        if not S:
            continue
        beta = fit_ridge_for_coalition(
            datasets, S, lam0=lam0, lambda_mode=lambda_mode
        )
        for name in S:
            cache[(name, S)] = float(
                firm_predictive_residual_std(beta, datasets[name])
            )
    return cache


def build_v_from_cache(
    firms: tuple[str, ...],
    sigma_cache: dict,
    weights: dict[str, float] | None,
) -> Callable[[frozenset[str]], float]:
    def v(S: frozenset[str]) -> float:
        if not S:
            return 0.0
        total = 0.0
        for name in S:
            s_auto = sigma_cache[(name, frozenset({name}))]
            s_coal = sigma_cache[(name, S)]
            delta = s_auto - s_coal
            w = 1.0 if weights is None else float(weights[name])
            total += w * delta
        return float(total)

    return v


def shapley_phi(firms: tuple[str, ...], v_func) -> dict[str, float]:
    game = CachedGame(players=list(firms), v_func=v_func)
    return exact_shapley(game)


def spearman_dict(a: dict[str, float], b: dict[str, float], firms: tuple[str, ...]) -> float:
    xa = np.array([a[f] for f in firms], dtype=float)
    xb = np.array([b[f] for f in firms], dtype=float)
    if np.allclose(xa, xa[0]) or np.allclose(xb, xb[0]):
        return 1.0 if np.allclose(xa, xb) else 0.0
    r, _ = spearmanr(xa, xb)
    return float(r) if np.isfinite(r) else 0.0


def top1_disagree(a: dict[str, float], b: dict[str, float], firms: tuple[str, ...]) -> int:
    ia = max(firms, key=lambda f: a[f])
    ib = max(firms, key=lambda f: b[f])
    return int(ia != ib)


def run_seed(
    seed: int,
    n: int,
    lambda_mode: str,
    lam0: float = 1.0,
    n_periods: int = 400,
) -> list[dict]:
    cfg = DemandSimulationConfig(
        n_firms=n,
        n_periods=n_periods,
        n_features=5,
        seed=seed,
        shared_factor_correlation=0.6,
        n_obs_heterogeneity="fixed",
        signal_quality_profile="fixed",
        noise_profile="fixed",
    )
    sim = generate_multi_firm_demand(cfg)
    datasets = build_firm_datasets(
        sim, val_fraction=0.15, holdout_fraction=0.25, min_holdout=30
    )
    firms = tuple(sorted(datasets.keys()))

    sigma_cache = cache_sigma(
        datasets, firms, lam0=lam0, lambda_mode=lambda_mode)
    v_unit = build_v_from_cache(firms, sigma_cache, weights=None)
    phi_unit = shapley_phi(firms, v_unit)

    rows = []
    for cell, prims in COST_BASE.items():
        w = make_weights(firms, prims)
        v_w = build_v_from_cache(firms, sigma_cache, weights=w)
        phi_w = shapley_phi(firms, v_w)
        rho = spearman_dict(phi_w, phi_unit, firms)
        rows.append(
            {
                "seed": seed,
                "n": n,
                "lambda_mode": lambda_mode,
                "cell": cell,
                "kappa_eff": kappa_eff(w),
                "rho": rho,
                "top1_dis": top1_disagree(phi_w, phi_unit, firms),
                "vN_unit": float(v_unit(frozenset(firms))),
                "vN_w": float(v_w(frozenset(firms))),
            }
        )
    return rows


def main():
    # Full paper core: n=4, 30 seeds, both λ modes
    n = 4
    n_seeds = 30
    lam0 = 1.0
    modes = ("size_scaled", "fixed")

    out = Path("results/lambda_robustness")
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for mode in modes:
        print(f"\n=== lambda_mode={mode}  n={n} ===")
        for i, seed in enumerate(range(20_000, 20_000 + n_seeds)):
            rows.extend(run_seed(seed, n=n, lambda_mode=mode, lam0=lam0))
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  {i+1}/{n_seeds}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)

    summ = (
        df.groupby(["lambda_mode", "cell"], as_index=False)
        .agg(
            kappa_mean=("kappa_eff", "mean"),
            rho_mean=("rho", "mean"),
            rho_std=("rho", "std"),
            rho_median=("rho", "median"),
            top1_mean=("top1_dis", "mean"),
            n_seeds=("seed", "count"),
        )
        .sort_values(["lambda_mode", "kappa_mean"])
    )
    summ.to_csv(out / "summary.csv", index=False)
    print("\n=== λ-robustness summary ===")
    print(summ.to_string(index=False))

    # Within-mode trend: Spearman(kappa, rho) across cells per seed
    trend_rows = []
    for mode in modes:
        sub = df[df["lambda_mode"] == mode]
        for seed, g in sub.groupby("seed"):
            if g["kappa_eff"].nunique() < 2:
                continue
            r, p = spearmanr(g["kappa_eff"], g["rho"])
            trend_rows.append(
                {
                    "lambda_mode": mode,
                    "seed": seed,
                    "spearman_kappa_rho": float(r) if np.isfinite(r) else np.nan,
                }
            )
    tr = pd.DataFrame(trend_rows)
    tr.to_csv(out / "within_seed_trend.csv", index=False)
    print("\n=== Within-seed Spearman(κ, ρ) by λ mode ===")
    for mode in modes:
        s = tr[tr["lambda_mode"] == mode]["spearman_kappa_rho"].dropna()
        frac_neg = float((s < 0).mean()) if len(s) else float("nan")
        print(
            f"{mode}: mean r={s.mean():.3f}  median r={s.median():.3f}  "
            f"frac r<0={frac_neg:.3f}  n={len(s)}"
        )

    lines = [
        "λ-robustness",
        f"n={n} n_seeds={n_seeds} lam0={lam0}",
        "Canonical: size_scaled. Comparator: fixed.",
        "",
        summ.to_string(index=False),
        "",
        "Expect: qualitative decline of rho with kappa under BOTH modes.",
        "Do not require identical rho levels across modes.",
    ]
    (out / "report.txt").write_text("\n".join(lines))
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
