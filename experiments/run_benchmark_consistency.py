"""
Benchmark-consistency validation for inventory-motivated residual weights.

Regime A (ideal newsvendor-style / Lemma assumptions):
  - i.i.d. Gaussian demand errors (no forecast misspecification)
  - known residual scale σ
  - single-echelon base-stock with service_level = τ = cb/(ch+cb)
  - fixed mean demand μ, horizon T, costs (ch, cb), L=1
  - demand = max(μ + ε, 0) optional floor; primary = untruncated Gaussian path
  Shows: mean path cost scales ~linearly in σ; C / (w σ) approximately stable.

Regime B pointer:
  Finite-horizon federated residual vs path-cost ranks are reported by
  experiments/run_r1_residual_vs_path.py (not re-run here).

Usage:
  python -m experiments.run_benchmark_consistency
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.inventory import EchelonConfig, simulate_serial_supply_chain


def critical_tau(ch: float, cb: float) -> float:
    return float(cb) / (float(ch) + float(cb))


def inventory_weight(ch: float, cb: float, L: int = 1) -> float:
    tau = critical_tau(ch, cb)
    z = float(norm.ppf(np.clip(tau, 1e-6, 1.0 - 1e-6)))
    return float((ch + cb) * norm.pdf(z) * np.sqrt(L + 1))


def echelons_tau(ch: float, cb: float, L: int = 1) -> list[EchelonConfig]:
    tau = float(np.clip(critical_tau(ch, cb), 1e-4, 1.0 - 1e-4))
    return [
        EchelonConfig(
            holding_cost=float(ch),
            shortage_cost=float(cb),
            lead_time=L,
            service_level=tau,
        )
    ]


def mean_cost_sigma(
    sigma: float,
    ch: float,
    cb: float,
    mu: float = 50.0,
    T: int = 500,
    n_paths: int = 40,
    seed: int = 0,
    floor_demand: bool = False,
) -> float:
    """
    Ideal regime: forecast = μ (perfect mean), residual_std = σ (known).
    Demand paths: μ + σ Z_t, Z~N(0,1), optional nonneg floor.
    Simulator deterministic given path → average over independent paths.
    """
    echs = echelons_tau(ch, cb, L=1)
    forecast = np.full(T, float(mu))
    costs = []
    rng = np.random.default_rng(int(seed))
    for _ in range(n_paths):
        eps = rng.normal(0.0, float(sigma), size=T)
        demand = mu + eps
        if floor_demand:
            demand = np.maximum(demand, 0.0)
        out = simulate_serial_supply_chain(
            demand,
            forecast,
            float(sigma),
            echs,
            rng=None,
            stochastic_lead_time=False,
        )
        costs.append(float(out.total_cost))
    return float(np.mean(costs))


def regime_a_scaling(
    ch: float = 1.0,
    cb: float = 9.0,
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    mu: float = 50.0,
    T: int = 500,
    n_paths: int = 40,
    seed: int = 7,
) -> pd.DataFrame:
    w = inventory_weight(ch, cb, L=1)
    rows = []
    for s in sigmas:
        c = mean_cost_sigma(
            sigma=s, ch=ch, cb=cb, mu=mu, T=T, n_paths=n_paths, seed=seed
        )
        rows.append(
            {
                "ch": ch,
                "cb": cb,
                "tau": critical_tau(ch, cb),
                "w": w,
                "sigma": s,
                "mean_cost": c,
                "cost_per_period": c / T,
                "w_sigma": w * s,
                "ratio_C_over_w_sigma": (c / T) / max(w * s, 1e-15),
            }
        )
    return pd.DataFrame(rows)


def regime_a_ranking_alignment(
    ch: float = 1.0,
    cb: float = 9.0,
    firm_sigmas: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
    mu: float = 50.0,
    T: int = 500,
    n_paths: int = 40,
    seed: int = 11,
) -> pd.DataFrame:
    """
    Four synthetic firms differ only in known σ.
    Autarky cost C_i; 'coalition' oracle gives σ_coal = min_i σ_i for all
    (perfect pooling lower bound) — only for ranking check of w*Δσ vs ΔC.
    Simpler check: rank by w*(σ_ref - σ_i) with σ_ref = max σ should
    match rank by cost reduction vs worst firm, and rank by -C_i.
    """
    w = inventory_weight(ch, cb, L=1)
    costs = []
    for j, s in enumerate(firm_sigmas):
        c = mean_cost_sigma(
            sigma=s,
            ch=ch,
            cb=cb,
            mu=mu,
            T=T,
            n_paths=n_paths,
            seed=seed + 100 * j,
        )
        costs.append(c)
    costs = np.asarray(costs, dtype=float)
    sig = np.asarray(firm_sigmas, dtype=float)
    # value vs worst (largest) sigma autarky
    sigma_worst = float(np.max(sig))
    delta_sigma = sigma_worst - sig
    score_w = w * delta_sigma
    delta_cost = float(np.max(costs)) - costs  # reduction vs costliest

    # higher reduction → rank 0
    rank_cost = np.argsort(np.argsort(-delta_cost))
    rank_w = np.argsort(np.argsort(-score_w))
    # Spearman between score_w and delta_cost
    from scipy.stats import spearmanr

    rho, _ = spearmanr(score_w, delta_cost)
    rows = []
    for i, s in enumerate(firm_sigmas):
        rows.append(
            {
                "firm": i,
                "sigma": s,
                "mean_cost": costs[i],
                "delta_sigma_vs_worst": delta_sigma[i],
                "w_delta_sigma": score_w[i],
                "delta_cost_vs_worst": delta_cost[i],
                "rank_w": int(rank_w[i]),
                "rank_cost": int(rank_cost[i]),
            }
        )
    meta = {
        "spearman_w_vs_delta_cost": float(rho) if np.isfinite(rho) else 0.0,
        "ranks_identical": bool(np.array_equal(rank_w, rank_cost)),
    }
    return pd.DataFrame(rows), meta


def main():
    out = Path("results/benchmark_consistency")
    out.mkdir(parents=True, exist_ok=True)

    print("=== Regime A: cost scaling in σ (ch=1, cb=9, τ=0.9) ===")
    df_scale = regime_a_scaling()
    df_scale.to_csv(out / "regime_a_scaling.csv", index=False)
    print(df_scale.to_string(index=False))
    ratios = df_scale["ratio_C_over_w_sigma"].to_numpy()
    print(
        f"\nratio C/(wσ) range: [{ratios.min():.3f}, {ratios.max():.3f}]  "
        f"(stable ⇒ near-linear scaling under Lemma assumptions)"
    )

    print("\n=== Regime A: ranking alignment wΔσ vs ΔC ===")
    df_rank, meta = regime_a_ranking_alignment()
    df_rank.to_csv(out / "regime_a_ranking.csv", index=False)
    print(df_rank.to_string(index=False))
    print(
        f"Spearman(wΔσ, ΔC) = {meta['spearman_w_vs_delta_cost']:.3f}  "
        f"ranks_identical={meta['ranks_identical']}"
    )

    # Point to empirical boundary (Regime B / R1)
    r1 = Path("results/r1_residual_vs_path/summary.csv")
    lines = [
        "Benchmark consistency",
        "",
        "Regime A (ideal): see regime_a_scaling.csv, regime_a_ranking.csv",
        "  Expect: ratio C/(wσ) roughly stable across σ; rank(wΔσ)=rank(ΔC).",
        "",
        "Regime B (federated residual vs path-cost ranks):",
    ]
    if r1.exists():
        s = pd.read_csv(r1)
        lines.append(s.to_string(index=False))
        lines.append(
            "\nInterpretation: under Lemma assumptions weighting matches cost "
            "scaling/ranks; under the finite-horizon forecast-driven DGP, "
            "residual-weighted Shapley ranks need not match path-cost ranks "
            f"(R1 k1 mean ρ ≈ {float(s.loc[s['cell'] == 'k1', 'rho_phi_mean'].values[0]):.2f})."
        )
    else:
        lines.append(
            "  Run: python -m experiments.run_r1_residual_vs_path"
        )
    (out / "report.txt").write_text("\n".join(lines))
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
