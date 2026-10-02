"""
Stage 2 — redesigned κ_eff sweep on corrected instrument (L+1 base-stock,
Phase-1 predictive residual, parallel topology).

Cells
-----
  κ=1   : identical cost multipliers for all firms  → expect high Spearman ρ
  κ=2,5,10 : graded firm-level cost multipliers     → expect ρ to fall

Outputs
-------
  results/stage2_kappa_sweep/
    results.csv          tidy long
    summary_by_kappa.csv
    config.json
    run_log.txt
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.inventory import EchelonConfig
from src.shapley import exact_shapley, CachedGame
from src.mechanism import compute_divergence_metrics


# ---------------------------------------------------------------------------
# κ schedules: firm-level multipliers on (holding, shortage) of every echelon
# ---------------------------------------------------------------------------

KAPPA_SCHEDULES: dict[str, list[float]] = {
    # multipliers applied to firm index 0..n-1 (n_firms must match length)
    "kappa_1": [1.0, 1.0, 1.0, 1.0],
    "kappa_2": [1.0, 1.0, 1.5, 2.0],
    "kappa_5": [1.0, 1.5, 3.0, 5.0],
    "kappa_10": [1.0, 2.0, 5.0, 10.0],
}


def scale_echelons(
    base: list[EchelonConfig], mult: float
) -> list[EchelonConfig]:
    """Scale holding and shortage costs of every echelon by mult."""
    return [
        EchelonConfig(
            holding_cost=ech.holding_cost * mult,
            shortage_cost=ech.shortage_cost * mult,
            lead_time=ech.lead_time,
            service_level=ech.service_level,
        )
        for ech in base
    ]


def default_base_echelons() -> list[EchelonConfig]:
    return [
        EchelonConfig(holding_cost=1.0, shortage_cost=9.0,
                      lead_time=1, service_level=0.95),
        EchelonConfig(holding_cost=0.6, shortage_cost=0.0,
                      lead_time=2, service_level=0.90),
        EchelonConfig(holding_cost=0.3, shortage_cost=0.0,
                      lead_time=3, service_level=0.90),
    ]


@dataclass
class SweepConfig:
    n_firms: int = 4
    n_periods: int = 400
    n_features: int = 5
    shared_factor_correlation: float = 0.6
    n_obs_heterogeneity: str = "moderate"
    holdout_fraction: float = 0.25
    val_fraction: float = 0.15
    min_holdout: int = 40
    ridge_lambda: float = 1.0
    lambda_mode: str = "fixed"
    residual_mode: str = "predictive"
    n_seeds: int = 20
    master_seed: int = 20260831
    schedules: dict[str, list[float]] = field(
        default_factory=lambda: dict(KAPPA_SCHEDULES)
    )


def kappa_from_mults(mults: list[float]) -> float:
    a = np.asarray(mults, dtype=float)
    return float(np.max(a) / max(np.min(a), 1e-12))


def run_one_seed(
    cfg: SweepConfig,
    schedule_name: str,
    mults: list[float],
    seed: int,
) -> list[dict]:
    """One (schedule × seed) → tidy rows for payments + summary metrics."""
    assert len(mults) == cfg.n_firms

    dcfg = DemandSimulationConfig(
        n_firms=cfg.n_firms,
        n_periods=cfg.n_periods,
        n_features=cfg.n_features,
        shared_factor_correlation=cfg.shared_factor_correlation,
        n_obs_heterogeneity=cfg.n_obs_heterogeneity,
        seed=seed,
    )
    sim = generate_multi_firm_demand(dcfg)
    datasets = build_firm_datasets(
        sim,
        val_fraction=cfg.val_fraction,
        holdout_fraction=cfg.holdout_fraction,
        min_holdout=cfg.min_holdout,
    )
    firms = tuple(sorted(datasets.keys()))
    paths = {n: ds.y_holdout for n, ds in datasets.items()}

    # Per-firm echelon lists (parallel chains; cost mult implements w_i)
    base = default_base_echelons()
    firm_echelons = {
        name: scale_echelons(base, mults[i])
        for i, name in enumerate(firms)
    }

    coal_cfg = CoalitionConfig(
        firms=firms,
        lambda_mode=cfg.lambda_mode,
        residual_mode=cfg.residual_mode,
        ridge_lambda=cfg.ridge_lambda,
        seed=seed,
    )
    # Evaluator uses a single echelon list for the "default" path; we override
    # cost() per firm via a thin wrapper below.
    ev = CoalitionEvaluator(datasets, paths, coal_cfg)

    # --- operational value with firm-specific cost scales ---
    # Cache autarky and coalition costs per firm under its own echelons.
    _cost_cache: dict[tuple[str, frozenset], float] = {}

    def firm_cost(name: str, S: frozenset[str]) -> float:
        key = (name, S)
        if key in _cost_cache:
            return _cost_cache[key]
        beta = ev.fit(S)
        from src.federated_phase1_patch import firm_predictive_residual_std
        from src.inventory import simulate_serial_supply_chain

        ds = datasets[name]
        residual = firm_predictive_residual_std(beta, ds)
        forecast = ds.X_holdout @ beta
        demand = ds.y_holdout
        echs = firm_echelons[name]
        # Mean over a few CRN paths for stability
        costs = []
        for k in range(8):
            rng = np.random.default_rng([seed, hash(name) % 10_000, k])
            out = simulate_serial_supply_chain(
                demand, forecast, residual, echs, rng=None
            )
            costs.append(out.total_cost)
        c = float(np.mean(costs))
        _cost_cache[key] = c
        return c

    def v_op(S: frozenset[str]) -> float:
        if not S:
            return 0.0
        total = 0.0
        for name in S:
            c_auto = firm_cost(name, frozenset({name}))
            c_S = firm_cost(name, S)
            total += c_auto - c_S
        return total

    def v_acc(S: frozenset[str]) -> float:
        return ev.value_accuracy(S)

    # Exact Shapley
    game_op = CachedGame(list(firms), v_op)
    game_acc = CachedGame(list(firms), v_acc)
    phi_op = exact_shapley(game_op)
    phi_acc = exact_shapley(game_acc)

    # Divergence metrics (if available); else local Spearman
    try:
        div = compute_divergence_metrics(phi_op, phi_acc)
        spearman = float(div.get("spearman_rho", np.nan))
        displacement = float(div.get("displacement", np.nan))
        rank_rev = float(div.get("rank_reversal", np.nan))
    except Exception:
        spearman = _spearman(phi_op, phi_acc)
        displacement = _displacement(phi_op, phi_acc)
        rank_rev = float(
            max(phi_op, key=phi_op.get) != max(phi_acc, key=phi_acc.get)
        )

    N = frozenset(firms)
    vN_op = v_op(N)
    vN_acc = v_acc(N)
    eff_op = abs(sum(phi_op.values()) - vN_op)
    eff_acc = abs(sum(phi_acc.values()) - vN_acc)

    kappa = kappa_from_mults(mults)
    rows = []
    base_meta = {
        "schedule": schedule_name,
        "kappa_eff": kappa,
        "seed": seed,
        "n_firms": cfg.n_firms,
        "shared_factor_correlation": cfg.shared_factor_correlation,
        "n_periods": cfg.n_periods,
        "v_op_N": vN_op,
        "v_acc_N": vN_acc,
        "eff_gap_op": eff_op,
        "eff_gap_acc": eff_acc,
        "spearman_rho": spearman,
        "displacement": displacement,
        "rank_reversal": rank_rev,
    }

    # Summary row
    rows.append({**base_meta, "firm": "_summary",
                "metric": "spearman_rho", "value": spearman})
    rows.append({**base_meta, "firm": "_summary",
                "metric": "displacement", "value": displacement})
    rows.append({**base_meta, "firm": "_summary",
                "metric": "rank_reversal", "value": rank_rev})
    rows.append({**base_meta, "firm": "_summary",
                "metric": "v_op_N", "value": vN_op})
    rows.append({**base_meta, "firm": "_summary",
                "metric": "eff_gap_op", "value": eff_op})

    for f in firms:
        rows.append({**base_meta, "firm": f,
                    "metric": "phi_op", "value": phi_op[f]})
        rows.append({**base_meta, "firm": f,
                    "metric": "phi_acc", "value": phi_acc[f]})

    return rows


def _spearman(a: dict, b: dict) -> float:
    firms = sorted(a.keys())
    xa = np.array([a[f] for f in firms])
    xb = np.array([b[f] for f in firms])
    ra = np.empty(len(firms))
    rb = np.empty(len(firms))
    ra[np.argsort(xa)] = np.arange(len(firms))
    rb[np.argsort(xb)] = np.arange(len(firms))
    n = len(firms)
    d2 = np.sum((ra - rb) ** 2)
    return float(1 - 6 * d2 / (n * (n * n - 1)))


def _displacement(a: dict, b: dict) -> float:
    firms = sorted(a.keys())
    xa = np.array([a[f] for f in firms])
    xb = np.array([b[f] for f in firms])
    ra = np.empty(len(firms))
    rb = np.empty(len(firms))
    ra[np.argsort(xa)] = np.arange(len(firms))
    rb[np.argsort(xb)] = np.arange(len(firms))
    return float(np.mean(np.abs(ra - rb)) / max(len(firms) - 1, 1))


def main():
    cfg = SweepConfig()
    out = Path("results/stage2_kappa_sweep")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.master_seed)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=cfg.n_seeds)]

    t0 = time.time()
    all_rows: list[dict] = []
    log_lines = []

    for sched_name, mults in cfg.schedules.items():
        if len(mults) != cfg.n_firms:
            raise ValueError(
                f"{sched_name}: expected {cfg.n_firms} multipliers, got {len(mults)}"
            )
        k = kappa_from_mults(mults)
        print(
            f"\n=== schedule {sched_name}  κ_eff={k:.2f}  ({cfg.n_seeds} seeds) ===")
        for i, seed in enumerate(seeds):
            rows = run_one_seed(cfg, sched_name, mults, seed)
            all_rows.extend(rows)
            if (i + 1) % 5 == 0 or i == 0:
                sp = rows[0]["spearman_rho"]
                print(f"  seed {i+1}/{cfg.n_seeds}  ρ={sp:.3f}")
        # schedule-level snapshot
        sub = [r for r in all_rows if r["schedule"] ==
               sched_name and r["metric"] == "spearman_rho"]
        rhos = [r["value"] for r in sub]
        msg = (
            f"{sched_name}: mean ρ={np.mean(rhos):.3f}  "
            f"median={np.median(rhos):.3f}  "
            f"reversal={np.mean([r['value'] for r in all_rows if r['schedule'] == sched_name and r['metric'] == 'rank_reversal']):.2f}"
        )
        print("  " + msg)
        log_lines.append(msg)

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "results.csv", index=False)

    # Summary by schedule
    summ = (
        df.query("firm == '_summary' and metric == 'spearman_rho'")
        .groupby(["schedule", "kappa_eff"], as_index=False)
        .agg(
            spearman_mean=("value", "mean"),
            spearman_std=("value", "std"),
            spearman_lo=("value", lambda x: float(np.quantile(x, 0.1))),
            spearman_hi=("value", lambda x: float(np.quantile(x, 0.9))),
            n_seeds=("value", "count"),
        )
    )
    # attach displacement / reversal
    for metric in ("displacement", "rank_reversal", "v_op_N", "eff_gap_op"):
        tmp = (
            df.query(f"firm == '_summary' and metric == '{metric}'")
            .groupby("schedule", as_index=False)
            .agg(**{f"{metric}_mean": ("value", "mean")})
        )
        summ = summ.merge(tmp, on="schedule", how="left")

    summ = summ.sort_values("kappa_eff")
    summ.to_csv(out / "summary_by_kappa.csv", index=False)

    elapsed = time.time() - t0
    meta = {
        "config": asdict(cfg),
        "elapsed_sec": elapsed,
        "n_rows": len(df),
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2, default=str))
    (out / "run_log.txt").write_text("\n".join(log_lines) +
                                     f"\n\nelapsed={elapsed:.1f}s\n")

    print("\n=== Stage 2 κ-sweep summary ===")
    print(summ.to_string(index=False))
    print(f"\nArtifacts → {out}/")
    print(
        "Gate reading: mean ρ at κ=1 should be high (≳0.7–0.9 under residual noise); "
        "mean ρ should fall as κ rises."
    )


if __name__ == "__main__":
    main()
