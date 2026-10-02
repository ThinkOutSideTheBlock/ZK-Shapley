"""
Phase-0 diagnostic: at equal weights, residual-weighted and unweighted
residual games MUST share sign(v(N)) and IR pattern if they use the same Δσ.

Prints per-seed: v_res, v_acc, ratio, IR flags.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.shapley import exact_shapley, CachedGame


def induced_w(ch: float, cb: float, L: int = 1, sl: float = 0.95) -> float:
    tau = cb / (ch + cb)
    z = float(norm.ppf(np.clip(tau, 1e-6, 1 - 1e-6)))
    return float((ch + cb) * norm.pdf(z) * np.sqrt(L + 1))


def main():
    n_seeds = 30
    n_firms = 4
    # equal-weight cell (k1)
    prims = [(1.0, 9.0)] * n_firms
    rows = []

    for s in range(n_seeds):
        seed = 10000 + s
        cfg = DemandSimulationConfig(
            n_firms=n_firms,
            n_periods=400,
            n_features=5,
            seed=seed,
            shared_factor_correlation=0.6,
            n_obs_heterogeneity="moderate",
        )
        sim = generate_multi_firm_demand(cfg)
        datasets = build_firm_datasets(
            sim, val_fraction=0.15, holdout_fraction=0.25, min_holdout=40
        )
        firms = tuple(sorted(datasets.keys()))
        w = {firms[i]: induced_w(*prims[i]) for i in range(n_firms)}
        # constant weight
        w0 = next(iter(w.values()))

        coal_cfg = CoalitionConfig(
            lam=1.0,
            lambda_mode="size_scaled",  # change if IR script uses fixed
            residual_mode="phase1",
        )
        # adapt kwargs if your CoalitionEvaluator signature differs
        try:
            ev = CoalitionEvaluator(
                datasets=datasets,
                demand_paths={n: datasets[n].y_holdout for n in firms},
                echelons=coal_cfg.to_echelons() if hasattr(coal_cfg, "to_echelons") else None,
                config=coal_cfg,
            )
        except TypeError:
            ev = CoalitionEvaluator(datasets, coal_cfg)

        # Prefer residual-scale API if present
        if hasattr(ev, "value") and hasattr(ev, "value_accuracy"):
            v_acc = lambda S, _ev=ev: float(_ev.value_accuracy(S))
            # weighted residual: sum w_i * (sigma_i({i}) - sigma_i(S))
            def v_res(S, _ev=ev, _w=w):
                if not S:
                    return 0.0
                total = 0.0
                for name in S:
                    # fallback: scale accuracy contribution if no sigma API
                    if hasattr(_ev, "firm_residual"):
                        s_auto = _ev.firm_residual(frozenset({name}), name)
                        s_coal = _ev.firm_residual(S, name)
                        total += _w[name] * (s_auto - s_coal)
                    else:
                        # approximate via value_accuracy member terms if exposed
                        return float(_w0 * _ev.value_accuracy(S))
                return float(total)
        else:
            raise SystemExit("Adapt: expose value_accuracy / residual on CoalitionEvaluator")

        game_acc = CachedGame(firms, v_acc) if True else None
        # CachedGame(firms, v_func) — match your API
        try:
            phi_acc, meta_acc = exact_shapley(firms, v_acc)
            phi_res, meta_res = exact_shapley(firms, v_res)
        except TypeError:
            from src.shapley import exact_shapley as es
            # try dict-style
            phi_acc = es(list(firms), v_acc)
            phi_res = es(list(firms), v_res)

        N = frozenset(firms)
        va = float(v_acc(N))
        vr = float(v_res(N))
        ratio = vr / va if abs(va) > 1e-12 else float("nan")

        def ir_ok(phi, vN):
            if vN < 0:
                return False  # all-IR impossible under efficiency
            return all(phi.get(i, phi[i] if isinstance(phi, dict) else 0) >= -1e-9 for i in firms)

        # normalize phi to dict
        if not isinstance(phi_acc, dict):
            phi_acc = {firms[i]: float(phi_acc[i]) for i in range(n_firms)}
            phi_res = {firms[i]: float(phi_res[i]) for i in range(n_firms)}

        row = {
            "seed": seed,
            "v_acc": va,
            "v_res": vr,
            "ratio": ratio,
            "w0": w0,
            "neg_acc": int(va < 0),
            "neg_res": int(vr < 0),
            "all_ir_acc": int(ir_ok(phi_acc, va)),
            "all_ir_res": int(ir_ok(phi_res, vr)),
            "rho": float(
                np.corrcoef(
                    [phi_acc[i] for i in firms],
                    [phi_res[i] for i in firms],
                )[0, 1]
            )
            if len(set(phi_acc.values())) > 1
            else 1.0,
        }
        rows.append(row)
        if (s + 1) % 10 == 0:
            print(
                f"  {s+1}/{n_seeds}  v_acc={va:.4f} v_res={vr:.4f} "
                f"ratio={ratio:.4f} (expect ~{w0:.4f})"
            )

    neg_acc = np.mean([r["neg_acc"] for r in rows])
    neg_res = np.mean([r["neg_res"] for r in rows])
    ratios = [r["ratio"] for r in rows if np.isfinite(r["ratio"])]
    print("\n=== k1 corollary check ===")
    print(f"P(v_acc<0)={neg_acc:.3f}  P(v_res<0)={neg_res:.3f}  (must match)")
    print(f"mean ratio v_res/v_acc={np.mean(ratios):.4f}  w0={w0:.4f}")
    print(f"all-IR acc={np.mean([r['all_ir_acc'] for r in rows]):.3f}  "
          f"res={np.mean([r['all_ir_res'] for r in rows]):.3f}")
    print(
        "PASS if P(neg) match and ratio ≈ w0 and all-IR match.\n"
        "FAIL → IR script used different objects (e.g. path cost vs residual)."
    )


if __name__ == "__main__":
    main()