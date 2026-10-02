"""R2 smoke: residual-weighted exact Shapley + efficiency on n=4."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.federated_phase1_patch import fit_ridge_for_coalition, firm_predictive_residual_std
from src.shapley import CachedGame, exact_shapley


def inv_w(ch, cb, L=1, sl=0.95):
    tau = cb / (ch + cb)
    z = float(norm.ppf(np.clip(tau, 1e-6, 1 - 1e-6)))
    return float((ch + cb) * norm.pdf(z) * np.sqrt(L + 1))


def main():
    cfg = DemandSimulationConfig(
        n_firms=4, n_periods=200, n_features=5, seed=0,
        shared_factor_correlation=0.6,
        n_obs_heterogeneity="fixed",
        signal_quality_profile="fixed",
        noise_profile="fixed",
    )
    sim = generate_multi_firm_demand(cfg)
    ds = build_firm_datasets(sim, val_fraction=0.15,
                             holdout_fraction=0.25, min_holdout=30)
    firms = [f"firm_{i}" for i in range(4)]
    prims = [(1.0, 9.0)] * 4
    weights = {firms[i]: inv_w(*prims[i]) for i in range(4)}
    cache = {}

    def sigma(name, S):
        key = (name, S)
        if key not in cache:
            beta = fit_ridge_for_coalition(
                ds, S, lam0=1.0, lambda_mode="size_scaled")
            cache[key] = float(firm_predictive_residual_std(beta, ds[name]))
        return cache[key]

    def v_res(S):
        if not S:
            return 0.0
        return float(sum(weights[i] * (sigma(i, frozenset({i})) - sigma(i, S)) for i in S))

    game = CachedGame(players=firms, v_func=v_res)
    phi = exact_shapley(game)
    vN = game.v(frozenset(firms))
    gap = abs(sum(phi.values()) - vN)
    print("phi", {k: round(v, 6) for k, v in phi.items()})
    print("vN", round(vN, 6), "eff_gap", gap)
    assert gap < 1e-8
    assert abs(v_res(frozenset({firms[0]}))) < 1e-12
    print("R2 smoke PASSED")


if __name__ == "__main__":
    main()
