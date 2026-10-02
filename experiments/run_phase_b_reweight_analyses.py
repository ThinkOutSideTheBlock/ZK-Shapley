"""
Phase B: deepen cost-channel claim without new thesis.

Per seed:
  1) Fit all coalitions once; cache sigma_i(S) (validation residual scale).
  2) Unit-weight game v^sigma and phi^sigma.
  3) Cost-channel weights (k1/k2/k5/k10): rho, top1, D_phi.
  4) Alpha path Option B: w(alpha) = (1-a)*w_k1 + a*w_k10
     (true linear path from empirical k1 weights to empirical k10 weights);
     first rank / top1 flip vs equal-weight (alpha=0) ordering.
  5) Weight-assignment permutations at fixed multiset (k10).

Outputs under results/phase_b_reweight/
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std
from src.shapley import exact_shapley


# ---------------------------------------------------------------------------
# Weights (normal-loss; match paper COST lists)
# ---------------------------------------------------------------------------
def induced_w(ch: float, cb: float, L: int = 1) -> float:
    tau = cb / (ch + cb)
    z = float(norm.ppf(np.clip(tau, 1e-6, 1 - 1e-6)))
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


def weights_for(n: int, cell: str) -> np.ndarray:
    prims = COST_BASE[cell][:n]
    return np.array([induced_w(ch, cb) for ch, cb in prims], dtype=float)


def kappa_of(w: np.ndarray) -> float:
    return float(np.max(w) / np.min(w))


def mix_weights(w_k1: np.ndarray, w_k10: np.ndarray, alpha: float) -> np.ndarray:
    """Option B: w(α) = (1-α) w^(k1) + α w^(k10)."""
    a = float(alpha)
    return (1.0 - a) * w_k1 + a * w_k10


# ---------------------------------------------------------------------------
# Residual cache: fit once
# ---------------------------------------------------------------------------
def powerset_nonempty(firms: tuple[str, ...]):
    n = len(firms)
    for r in range(1, n + 1):
        for comb in combinations(firms, r):
            yield frozenset(comb)


def build_sigma_cache(
    datasets: dict,
    firms: tuple[str, ...],
    seed: int,
    lambda_mode: str = "size_scaled",
    ridge_lambda: float = 1.0,
) -> dict[tuple[str, frozenset], float]:
    """sigma[(firm, S)] = residual scale of firm under coalition S."""
    paths = {n: datasets[n].y_holdout for n in firms}
    coal = CoalitionConfig(
        firms=firms,
        lambda_mode=lambda_mode,
        residual_mode="predictive",
        ridge_lambda=ridge_lambda,
        seed=seed,
    )
    ev = CoalitionEvaluator(datasets, paths, coal)
    cache: dict[tuple[str, frozenset], float] = {}
    for S in powerset_nonempty(firms):
        beta = ev.fit(S)
        for i in S:
            cache[(i, S)] = float(
                firm_predictive_residual_std(beta, datasets[i])
            )
    return cache


def make_v_from_sigma(
    firms: tuple[str, ...],
    sigma: dict[tuple[str, frozenset], float],
    w: dict[str, float],
) -> Callable[[frozenset], float]:
    def v(S: frozenset) -> float:
        if not S:
            return 0.0
        total = 0.0
        for i in S:
            s_auto = sigma[(i, frozenset({i}))]
            s_coal = sigma[(i, S)]
            total += w[i] * (s_auto - s_coal)
        return float(total)

    return v


def shapley_phi(firms: tuple[str, ...], v) -> dict[str, float]:
    """Return dict firm -> phi_i. Tries several API shapes."""
    fl = list(firms)
    try:
        from src.mechanism import run_zk_shapley_mechanism

        res = run_zk_shapley_mechanism(fl, v, exact=True)
        return {k: float(res.payments[k]) for k in fl}
    except Exception:
        pass
    try:
        phi, _meta = exact_shapley(firms, v)
        if isinstance(phi, dict):
            return {k: float(phi[k]) for k in fl}
        return {firms[i]: float(phi[i]) for i in range(len(firms))}
    except TypeError:
        phi = exact_shapley(fl, v)
        if isinstance(phi, dict):
            return {k: float(phi[k]) for k in fl}
        return {firms[i]: float(phi[i]) for i in range(len(firms))}


def rank_tuple(phi: dict[str, float], firms: tuple[str, ...]) -> tuple:
    """Higher phi => better rank (rank 0 = top). Deterministic firm-index tie-break."""
    order = sorted(firms, key=lambda f: (-phi[f], firms.index(f)))
    pos = {f: r for r, f in enumerate(order)}
    return tuple(pos[f] for f in firms)


def top1_firm(phi: dict[str, float], firms: tuple[str, ...]) -> str:
    return max(firms, key=lambda f: (phi[f], -firms.index(f)))


def spearman_phi(
    phi_a: dict[str, float],
    phi_b: dict[str, float],
    firms: tuple[str, ...],
) -> float:
    a = [phi_a[f] for f in firms]
    b = [phi_b[f] for f in firms]
    if len(set(np.round(a, 12))) == 1 and len(set(np.round(b, 12))) == 1:
        return 1.0
    r, _ = spearmanr(a, b)
    return float(r) if np.isfinite(r) else float("nan")


def payment_share_divergence(
    phi_res: dict[str, float],
    phi_sig: dict[str, float],
    vN_res: float,
    vN_sig: float,
    firms: tuple[str, ...],
    eps: float = 1e-12,
) -> float:
    """D_phi = 0.5 * sum |s_res - s_sig|; only meaningful if both vN > 0."""
    if vN_res <= eps or vN_sig <= eps:
        return float("nan")
    s_res = np.array([phi_res[f] / vN_res for f in firms])
    s_sig = np.array([phi_sig[f] / vN_sig for f in firms])
    return float(0.5 * np.sum(np.abs(s_res - s_sig)))


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------
def run_seed(
    seed: int,
    n: int,
    n_periods: int = 400,
    n_perm: int = 24,
    alpha_grid: np.ndarray | None = None,
    lambda_mode: str = "size_scaled",
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 1.0, 21)

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
    sigma = build_sigma_cache(datasets, firms, seed, lambda_mode=lambda_mode)

    # unit weights (unweighted residual game)
    w_unit = {f: 1.0 for f in firms}
    v_sig = make_v_from_sigma(firms, sigma, w_unit)
    phi_sig = shapley_phi(firms, v_sig)
    vN_sig = float(v_sig(frozenset(firms)))
    ranks_sig = rank_tuple(phi_sig, firms)
    top_sig = top1_firm(phi_sig, firms)

    cell_rows = []
    for cell in ("k1", "k2", "k5", "k10"):
        wv = weights_for(n, cell)
        w = {firms[i]: float(wv[i]) for i in range(n)}
        v_res = make_v_from_sigma(firms, sigma, w)
        phi_res = shapley_phi(firms, v_res)
        vN_res = float(v_res(frozenset(firms)))
        cell_rows.append(
            {
                "seed": seed,
                "n": n,
                "cell": cell,
                "kappa": kappa_of(wv),
                "rho": spearman_phi(phi_res, phi_sig, firms),
                "top1_dis": int(top1_firm(phi_res, firms) != top_sig),
                "rank_changed": int(rank_tuple(phi_res, firms) != ranks_sig),
                "D_phi": payment_share_divergence(
                    phi_res, phi_sig, vN_res, vN_sig, firms
                ),
                "vN_res": vN_res,
                "vN_sig": vN_sig,
                "pos_both": int(vN_res > 0 and vN_sig > 0),
            }
        )

    # ------------------------------------------------------------------
    # Option B alpha path: (1-α) w^(k1) + α w^(k10)
    # Baseline ranking = ranking at α=0 (empirical k1), which matches
    # equal-weight ranks by linearity when k1 is constant across firms.
    # ------------------------------------------------------------------
    w_k1 = weights_for(n, "k1")
    w_k10 = weights_for(n, "k10")

    # Explicit α=0 baseline from empirical k1 (not unit vector)
    w0 = {firms[i]: float(w_k1[i]) for i in range(n)}
    v0 = make_v_from_sigma(firms, sigma, w0)
    phi0 = shapley_phi(firms, v0)
    ranks0 = rank_tuple(phi0, firms)
    top0 = top1_firm(phi0, firms)

    alpha_rows = []
    first_rank_flip = None
    first_top1_flip = None
    for a in alpha_grid:
        wv = mix_weights(w_k1, w_k10, a)
        wv = np.maximum(wv, 1e-9)
        w = {firms[i]: float(wv[i]) for i in range(n)}
        v_res = make_v_from_sigma(firms, sigma, w)
        phi_res = shapley_phi(firms, v_res)
        vN_res = float(v_res(frozenset(firms)))
        rc = int(rank_tuple(phi_res, firms) != ranks0)
        t1 = int(top1_firm(phi_res, firms) != top0)
        if rc and first_rank_flip is None:
            first_rank_flip = float(a)
        if t1 and first_top1_flip is None:
            first_top1_flip = float(a)
        alpha_rows.append(
            {
                "seed": seed,
                "n": n,
                "alpha": float(a),
                "kappa": kappa_of(wv),
                "rho": spearman_phi(phi_res, phi_sig, firms),
                "top1_dis": t1,
                "rank_changed": rc,
                "D_phi": payment_share_divergence(
                    phi_res, phi_sig, vN_res, vN_sig, firms
                ),
                "vN_res": vN_res,
                "pos_both": int(vN_res > 0 and vN_sig > 0),
                "path": "k1_to_k10",
            }
        )

    def _kappa_at(a: float | None) -> float | None:
        if a is None:
            return None
        return kappa_of(mix_weights(w_k1, w_k10, a))

    thresh_rows = [
        {
            "seed": seed,
            "n": n,
            "path": "k1_to_k10",
            "alpha_first_rank_flip": first_rank_flip,
            "alpha_first_top1_flip": first_top1_flip,
            "kappa_at_first_rank_flip": _kappa_at(first_rank_flip),
            "kappa_at_first_top1_flip": _kappa_at(first_top1_flip),
            "ever_rank_flip": int(first_rank_flip is not None),
            "ever_top1_flip": int(first_top1_flip is not None),
            # sanity: α=0 ranking vs unit-weight ranking (should match)
            "rank0_matches_unit": int(ranks0 == ranks_sig),
        }
    ]

    # permutations of k10 weight multiset
    rng = np.random.default_rng(seed + 17)
    base = weights_for(n, "k10")
    perm_rows = []
    for p in range(n_perm):
        wv = base.copy()
        rng.shuffle(wv)
        w = {firms[i]: float(wv[i]) for i in range(n)}
        v_res = make_v_from_sigma(firms, sigma, w)
        phi_res = shapley_phi(firms, v_res)
        vN_res = float(v_res(frozenset(firms)))
        perm_rows.append(
            {
                "seed": seed,
                "n": n,
                "perm": p,
                "kappa": kappa_of(wv),
                "rho": spearman_phi(phi_res, phi_sig, firms),
                "top1_dis": int(top1_firm(phi_res, firms) != top_sig),
                "D_phi": payment_share_divergence(
                    phi_res, phi_sig, vN_res, vN_sig, firms
                ),
                "pos_both": int(vN_res > 0 and vN_sig > 0),
            }
        )

    return cell_rows, alpha_rows, thresh_rows, perm_rows


def main():
    # Smoke: n_list=[4], n_seeds=5, n_perm=8
    # Full paper: n_list=[4, 6, 8], n_seeds=40, n_perm=24
    n_list = [4, 6, 8]
    n_seeds = 40
    n_perm = 24
    n_periods = 400
    lambda_mode = "size_scaled"

    out = Path("results/phase_b_reweight")
    out.mkdir(parents=True, exist_ok=True)

    all_cell, all_alpha, all_thresh, all_perm = [], [], [], []
    for n in n_list:
        print(f"=== n={n} ===")
        for s in range(n_seeds):
            seed = 20000 + s
            cell_rows, alpha_rows, thresh_rows, perm_rows = run_seed(
                seed=seed,
                n=n,
                n_periods=n_periods,
                n_perm=n_perm,
                lambda_mode=lambda_mode,
            )
            all_cell.extend(cell_rows)
            all_alpha.extend(alpha_rows)
            all_thresh.extend(thresh_rows)
            all_perm.extend(perm_rows)
            if (s + 1) % 5 == 0 or s == 0:
                print(f"  seed {s + 1}/{n_seeds}")

    df_c = pd.DataFrame(all_cell)
    df_a = pd.DataFrame(all_alpha)
    df_t = pd.DataFrame(all_thresh)
    df_p = pd.DataFrame(all_perm)
    df_c.to_csv(out / "cell_metrics.csv", index=False)
    df_a.to_csv(out / "alpha_path.csv", index=False)
    df_t.to_csv(out / "thresholds.csv", index=False)
    df_p.to_csv(out / "permutations.csv", index=False)

    print("\n=== Cell summary (mean rho, top1, D_phi | pos_both) ===")
    for (n, cell), g in df_c.groupby(["n", "cell"]):
        gpos = g[g["pos_both"] == 1]
        dphi = gpos["D_phi"].mean() if len(gpos) else float("nan")
        print(
            f"n={n} {cell}: rho={g['rho'].mean():.3f}  "
            f"top1={g['top1_dis'].mean():.3f}  "
            f"rank_chg={g['rank_changed'].mean():.3f}  "
            f"D_phi(pos)={dphi:.3f}  n_pos={len(gpos)}/{len(g)}"
        )

    print("\n=== Threshold summary (Option B: k1 → k10 path) ===")
    for n, g in df_t.groupby("n"):
        print(
            f"n={n}: frac ever rank flip={g['ever_rank_flip'].mean():.3f}  "
            f"frac ever top1 flip={g['ever_top1_flip'].mean():.3f}  "
            f"median alpha rank flip={g['alpha_first_rank_flip'].median(skipna=True)}  "
            f"median alpha top1 flip={g['alpha_first_top1_flip'].median(skipna=True)}  "
            f"rank0_matches_unit={g['rank0_matches_unit'].mean():.3f}"
        )

    print("\n=== Permutation summary (k10 multiset) ===")
    for n, g in df_p.groupby("n"):
        print(
            f"n={n}: mean rho={g['rho'].mean():.3f}  "
            f"std rho={g['rho'].std():.3f}  "
            f"mean top1={g['top1_dis'].mean():.3f}"
        )

    lines = [
        "Phase B reweight analyses (Option B alpha path: k1 → k10)",
        f"n_list={n_list} n_seeds={n_seeds} n_perm={n_perm} lambda_mode={lambda_mode}",
        "alpha path: w(a)=(1-a)*w_k1 + a*w_k10",
        "",
        df_c.groupby(["n", "cell"])[["rho", "top1_dis", "rank_changed"]]
        .mean()
        .to_string(),
        "",
        "Thresholds:",
        df_t.groupby("n")[
            ["ever_rank_flip", "ever_top1_flip"]
        ].mean().to_string(),
    ]
    (out / "report.txt").write_text("\n".join(lines))
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
