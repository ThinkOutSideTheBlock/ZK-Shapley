"""
Statistical experiment harness for ZK-Shapley.

Design goals (plan Stage 3):
  - Cartesian (or explicit) parameter grid × independent seeds
  - Paired comparisons: operational and accuracy Shapley are computed on
    the *same* seed, the *same* datasets and the *same* fitted models
  - Tidy long-format parquet output
  - Full provenance: config.json, git_sha.txt, env.json, wall-time
  - Deterministic given the master seed

Usage
-----
    from experiments._harness import run_sweep, make_base_grid

    results = run_sweep(
        scenario_fn=my_scenario,
        param_grid=make_base_grid(),
        n_seeds=5,
        output_dir="results/smoke",
    )
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional dependencies (parquet)
# ---------------------------------------------------------------------------
try:
    import pyarrow  # noqa: F401
    _PARQUET_ENGINE = "pyarrow"
except ImportError:
    try:
        import fastparquet  # noqa: F401
        _PARQUET_ENGINE = "fastparquet"
    except ImportError:
        _PARQUET_ENGINE = None


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[1],
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _environment_snapshot() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "hostname": platform.node(),
        "cwd": str(Path.cwd()),
    }


def _config_hash(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------

def expand_grid(param_grid: dict[str, Iterable]) -> list[dict[str, Any]]:
    """
    Cartesian product of a parameter grid.
    Example:
        expand_grid({"n_firms": [3, 4], "rho": [0.3, 0.6]})
        → four dictionaries
    """
    keys = list(param_grid.keys())
    if not keys:
        return [{}]
    arrays = [list(param_grid[k]) for k in keys]
    from itertools import product
    return [dict(zip(keys, vals)) for vals in product(*arrays)]


def make_base_grid() -> dict[str, list]:
    """
    Minimal grid used for smoke tests and as a template for larger sweeps.
    Keep this small; full experimental grids live in experiment scripts.
    """
    return {
        "n_firms": [3, 4],
        "shared_factor_correlation": [0.3, 0.6],
        "cb_over_ch": [4.0, 9.0],          # shortage / holding ratio (retail)
        "n_periods": [120],
        "n_features": [4],
    }


def make_smoke_grid() -> dict[str, list]:
    """Two configs only — used by the smoke test."""
    return {
        "n_firms": [3],
        "shared_factor_correlation": [0.6],
        "cb_over_ch": [9.0],
        "n_periods": [100],
        "n_features": [4],
    }


# ---------------------------------------------------------------------------
# Scenario result container
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """
    One row-group of tidy output for a single (config, seed) pair.
    `records` is a list of flat dictionaries that become rows in the parquet.
    """
    records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, **kwargs) -> None:
        self.records.append(kwargs)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_sweep(
    scenario_fn: Callable[[dict[str, Any], int], ScenarioResult],
    param_grid: dict[str, Iterable],
    n_seeds: int = 5,
    master_seed: int = 0,
    output_dir: str | Path = "results/sweep",
    n_jobs: int = 1,
    stop_on_error: bool = False,
) -> pd.DataFrame:
    """
    Execute scenario_fn for every combination of parameters × seeds.

    Parameters
    ----------
    scenario_fn
        Callable(config_dict, seed) → ScenarioResult.
        Must be pure given (config, seed); the harness guarantees that
        operational and accuracy valuations inside the scenario share the
        same seed / data / models (paired design).
    param_grid
        Dictionary of parameter name → list of values.
    n_seeds
        Number of independent Monte-Carlo replications per config.
    master_seed
        Root seed from which per-run seeds are derived.
    output_dir
        Directory that will receive results.parquet, config.json,
        git_sha.txt, env.json and run_log.txt.
    n_jobs
        Currently sequential (n_jobs=1). Parallelism can be added later
        with a deterministic SeedSequence spawn; keep sequential for
        reproducibility until the parallel path is hardened.
    stop_on_error
        If True, the first exception aborts the whole sweep.

    Returns
    -------
    tidy DataFrame with one row per (config × seed × firm × metric).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = expand_grid(param_grid)
    seed_seq = np.random.SeedSequence(master_seed)
    child_seeds = seed_seq.spawn(len(configs) * n_seeds)

    all_records: list[dict[str, Any]] = []
    errors: list[str] = []
    t0 = time.perf_counter()

    idx = 0
    for cfg in configs:
        for local_seed_idx in range(n_seeds):
            run_seed = int(child_seeds[idx].generate_state(1)[0])
            idx += 1

            # Attach provenance fields that every row will carry
            row_meta = {
                **cfg,
                "seed": run_seed,
                "local_seed_idx": local_seed_idx,
                "master_seed": master_seed,
            }

            try:
                result = scenario_fn(cfg, run_seed)
                for rec in result.records:
                    all_records.append({**row_meta, **rec})
            except Exception as exc:
                msg = (
                    f"FAILED config={cfg} seed={run_seed}\n"
                    f"{traceback.format_exc()}"
                )
                errors.append(msg)
                if stop_on_error:
                    raise
                print(msg, file=sys.stderr)

    wall = time.perf_counter() - t0
    df = pd.DataFrame(all_records)

    # ------------------------------------------------------------------
    # Persist artefacts
    # ------------------------------------------------------------------
    if _PARQUET_ENGINE and len(df) > 0:
        df.to_parquet(output_dir / "results.parquet",
                      index=False, engine=_PARQUET_ENGINE)
    else:
        df.to_csv(output_dir / "results.csv", index=False)

    provenance = {
        "param_grid": {k: list(v) for k, v in param_grid.items()},
        "n_seeds": n_seeds,
        "master_seed": master_seed,
        "n_configs": len(configs),
        "n_runs_attempted": len(configs) * n_seeds,
        "n_rows": len(df),
        "wall_time_seconds": wall,
        "git_sha": _git_sha(),
        "config_hash": _config_hash(param_grid),
        "errors": errors,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)

    with open(output_dir / "git_sha.txt", "w") as f:
        f.write(_git_sha() + "\n")

    with open(output_dir / "env.json", "w") as f:
        json.dump(_environment_snapshot(), f, indent=2)

    with open(output_dir / "run_log.txt", "w") as f:
        f.write(f"wall_time_seconds: {wall:.2f}\n")
        f.write(f"rows: {len(df)}\n")
        f.write(f"errors: {len(errors)}\n")
        for e in errors:
            f.write(e + "\n---\n")

    print(
        f"[harness] finished {len(configs)} configs × {n_seeds} seeds → "
        f"{len(df)} rows in {wall:.1f}s  ({output_dir})"
    )
    return df


# ---------------------------------------------------------------------------
# Built-in paired scenario (operational vs accuracy Shapley)
# ---------------------------------------------------------------------------

def paired_shapley_scenario(cfg: dict[str, Any], seed: int) -> ScenarioResult:
    """
    Canonical paired scenario used by the paper’s core experiment.

    For a single (config, seed):
      1. Generate demand
      2. Build firm datasets
      3. Construct CoalitionEvaluator (same models for both games)
      4. Compute exact operational Shapley and exact accuracy Shapley
      5. Emit tidy rows for every firm and every metric
    """
    from src.demand import DemandSimulationConfig, generate_multi_firm_demand
    from src.federated import build_firm_datasets
    from src.coalition import CoalitionConfig, CoalitionEvaluator
    from src.shapley import CachedGame, exact_shapley
    from src.mechanism import compute_divergence_metrics
    from src.inventory import EchelonConfig

    # ----- data -----
    n_firms = int(cfg.get("n_firms", 4))
    rho = float(cfg.get("shared_factor_correlation", 0.5))
    n_periods = int(cfg.get("n_periods", 120))
    n_features = int(cfg.get("n_features", 4))
    cb_over_ch = float(cfg.get("cb_over_ch", 9.0))

    dem_cfg = DemandSimulationConfig(
        n_firms=n_firms,
        n_periods=n_periods,
        n_features=n_features,
        shared_factor_correlation=rho,
        seed=seed,
        n_obs_heterogeneity=cfg.get("n_obs_heterogeneity", "moderate"),
    )
    sim = generate_multi_firm_demand(dem_cfg)
    datasets = build_firm_datasets(sim, holdout_fraction=0.2, min_holdout=10)
    demand_paths = {n: ds.y_holdout for n, ds in datasets.items()}
    firms = tuple(sorted(datasets.keys()))

    # ----- inventory cost ratio (retail echelon) -----
    # holding fixed at 1.0, shortage = cb_over_ch
    coal_cfg = CoalitionConfig(
        firms=firms,
        holding_costs=(1.0, 0.6, 0.3),
        shortage_costs=(cb_over_ch, 0.0, 0.0),
        lead_times=(1, 2, 3),
        service_levels=(0.95, 0.90, 0.90),
        ridge_lambda=float(cfg.get("ridge_lambda", 1.0)),
        seed=seed,
    )
    evaluator = CoalitionEvaluator(
        datasets=datasets,
        demand_paths=demand_paths,
        config=coal_cfg,
        cache_dir=None,          # in-memory only inside a single run
    )

    # ----- paired Shapley (same evaluator → same models) -----
    game_op = CachedGame(players=list(
        firms), v_func=evaluator.operational_v_func())
    game_acc = CachedGame(players=list(
        firms), v_func=evaluator.accuracy_v_func())

    phi_op = exact_shapley(game_op)
    phi_acc = exact_shapley(game_acc)
    v_op_N = game_op.v(frozenset(firms))
    v_acc_N = game_acc.v(frozenset(firms))

    div = compute_divergence_metrics(phi_op, phi_acc)

    # ----- tidy records -----
    result = ScenarioResult()
    result.metadata = {
        "v_op_N": v_op_N,
        "v_acc_N": v_acc_N,
        "spearman_rho": div["spearman_rho"],
        "any_rank_reversal": div["any_rank_reversal"],
    }

    for firm in firms:
        result.add(
            firm=firm,
            metric="phi_operational",
            value=float(phi_op[firm]),
        )
        result.add(
            firm=firm,
            metric="phi_accuracy",
            value=float(phi_acc[firm]),
        )
        result.add(
            firm=firm,
            metric="ir_operational",
            value=float(phi_op[firm] >= -1e-9),
        )
        result.add(
            firm=firm,
            metric="ir_accuracy",
            value=float(phi_acc[firm] >= -1e-9),
        )

    # scalar summary rows (firm = "_summary")
    result.add(firm="_summary", metric="v_op_N", value=float(v_op_N))
    result.add(firm="_summary", metric="v_acc_N", value=float(v_acc_N))
    result.add(firm="_summary", metric="spearman_rho",
               value=float(div["spearman_rho"]))
    result.add(
        firm="_summary",
        metric="payment_displacement",
        value=float(div["mean_normalized_payment_difference"]),
    )
    result.add(
        firm="_summary",
        metric="any_rank_reversal",
        value=float(div["any_rank_reversal"]),
    )
    result.add(
        firm="_summary",
        metric="efficiency_gap_op",
        value=float(abs(sum(phi_op.values()) - v_op_N)),
    )

    return result
