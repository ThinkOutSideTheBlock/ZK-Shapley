"""Lemma-1 vs path-cost sanity: i.i.d. Gaussian demand, known mean, single echelon."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
# adjust import if needed
from src.inventory import EchelonConfig, simulate_serial_supply_chain


def w_coeff(ch, cb, L=1):
    tau = cb / (ch + cb)
    z = float(norm.ppf(np.clip(tau, 1e-6, 1 - 1e-6)))
    return float((ch + cb) * norm.pdf(z) * np.sqrt(L + 1))


def mean_path_cost(mu, sigma, ch, cb, L, T=500, K=40, seed=0):
    """Demand ~ N(mu, sigma^2) iid; forecast = mu; residual_std = sigma."""
    echelons = [EchelonConfig(
        holding_cost=ch, shortage_cost=cb, lead_time=L, service_level=cb/(ch+cb))]
    costs = []
    for k in range(K):
        rng = np.random.default_rng([seed, k])
        demand = np.maximum(rng.normal(mu, sigma, size=T), 0.0)
        forecast = np.full(T, float(mu))
        out = simulate_serial_supply_chain(
            demand, forecast, float(sigma), echelons, rng=rng)
        costs.append(out.total_cost)
    return float(np.mean(costs))


def main():
    ch, cb, L, mu, T, K = 1.0, 9.0, 1, 50.0, 500, 50
    w = w_coeff(ch, cb, L)
    sigmas = [1.0, 2.0, 3.0, 4.0]
    rows = []
    c0 = mean_path_cost(mu, sigmas[0], ch, cb, L, T=T, K=K, seed=1)
    for s in sigmas:
        c = mean_path_cost(mu, s, ch, cb, L, T=T, K=K, seed=1)
        # savings vs sigma0: C(s0)-C(s) vs w*(s0-s)  [if homogeneous in uncertainty]
        # rough scale: per-period * T if cost ~ w*sigma per period
        bench = w * (sigmas[0] - s) * T
        # Better: report C/T and w*sigma
        rows.append({
            "sigma": s,
            "mean_cost": c,
            "cost_per_period": c / T,
            "w_sigma": w * s,
            "ratio_Cpp_over_w_sigma": (c / T) / (w * s) if s > 0 else np.nan,
        })
        print(
            f"sigma={s:.1f}  C/T={c/T:.4f}  w*sigma={w*s:.4f}  ratio={(c/T)/(w*s):.4f}")
    Path("results/lemma1_sanity").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv("results/lemma1_sanity/summary.csv", index=False)
    print("Wrote results/lemma1_sanity/summary.csv")
    print("PASS if cost_per_period roughly proportional to sigma (ratio stable across sigma).")
    print("FAIL if ratio explodes/collapses → inspect inventory accounting.")


if __name__ == "__main__":
    main()
