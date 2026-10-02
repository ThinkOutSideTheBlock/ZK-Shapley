"""
Experiment 2 — Accuracy Shapley vs Inventory Shapley

The central empirical claim of the paper (claims C1, C16):
  inventory-cost valuation can rank firms differently from accuracy valuation,
  and the divergence grows with cost asymmetry (cb/ch).

Design
------
- Paired comparisons: same seed, same data, same fitted models for both games.
- Primary sweep axis: cb/ch ∈ {1, 4, 9, 19, 49} (critical fractile moves into the tail).
- Secondary axes: correlation ρ and a modest firm-count check.
- 30 seeds per cell for stable bootstrap CIs.
- Exact Shapley (n ≤ 6).

Outputs
-------
results/exp2_accuracy_vs_inventory/
    results.parquet (or .csv)
    config.json
    git_sha.txt
    env.json
    run_log.txt
    summary_by_cb.csv          # ready for Table 5 / Figure 3
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._harness import run_sweep, paired_shapley_scenario


# ---------------------------------------------------------------------------
# Grid (plan Section F.2 / Exp. 2)
# ---------------------------------------------------------------------------

def make_exp2_grid() -> dict[str, list]:
    return {
        "n_firms": [4],
        "shared_factor_correlation": [0.3, 0.6, 0.9],
        "cb_over_ch": [1.0, 4.0, 9.0, 19.0, 49.0],
        "n_periods": [150],
        "n_features": [5],
        "n_obs_heterogeneity": ["moderate"],
        "ridge_lambda": [1.0],
    }


# ---------------------------------------------------------------------------
# Post-processing → publication-ready summary
# ---------------------------------------------------------------------------

def build_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Collapse to one row per (cb_over_ch, rho) with mean / CI of the
    headline metrics.  Written to summary_by_cb.csv for Table 5 / Fig. 3.
    """
    # Keep only summary rows
    sub = df.query("firm == '_summary'").copy()

    def _agg(g: pd.DataFrame) -> pd.Series:
        def mean_ci(metric: str):
            vals = g.loc[g["metric"] == metric, "value"].astype(float)
            if len(vals) == 0:
                return np.nan, np.nan, np.nan
            m = vals.mean()
            se = vals.std(ddof=1) / np.sqrt(len(vals)
                                            ) if len(vals) > 1 else 0.0
            # normal approx 95 % CI (sufficient for n_seeds=30)
            return m, m - 1.96 * se, m + 1.96 * se

        rho_m, rho_lo, rho_hi = mean_ci("spearman_rho")
        disp_m, disp_lo, disp_hi = mean_ci("payment_displacement")
        rev_m, _, _ = mean_ci("any_rank_reversal")
        vop_m, _, _ = mean_ci("v_op_N")

        return pd.Series({
            "spearman_rho_mean": rho_m,
            "spearman_rho_lo": rho_lo,
            "spearman_rho_hi": rho_hi,
            "displacement_mean": disp_m,
            "displacement_lo": disp_lo,
            "displacement_hi": disp_hi,
            "rank_reversal_rate": rev_m,
            "mean_v_op_N": vop_m,
            "n_seeds": g["seed"].nunique(),
        })

    summary = (
        sub.groupby(["cb_over_ch", "shared_factor_correlation"], sort=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    summary.to_csv(output_dir / "summary_by_cb.csv", index=False)
    print("\n=== Exp 2 summary (mean Spearman ρ & displacement by cb/ch × ρ) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = Path("results/exp2_accuracy_vs_inventory")
    grid = make_exp2_grid()

    # 3 ρ × 5 cb/ch = 15 configs × 30 seeds = 450 runs
    # Each run is a full exact Shapley on n=4 (16 coalitions) — expect
    # a few minutes to low tens of minutes depending on hardware.
    df = run_sweep(
        scenario_fn=paired_shapley_scenario,
        param_grid=grid,
        n_seeds=30,
        master_seed=2026,
        output_dir=output_dir,
        stop_on_error=False,          # continue on rare numerical glitches
    )

    # Sanity gates
    v_op = df.query("metric == 'v_op_N'")["value"]
    gap = df.query("metric == 'efficiency_gap_op'")["value"]
    print(f"\nmean v_op(N)     : {v_op.mean():.3f}  (must be > 0)")
    print(f"max efficiency gap: {gap.max():.2e}  (must be < 1e-8)")
    assert v_op.mean() > 0, "collaboration surplus vanished"
    assert gap.max() < 1e-8, "efficiency broken"

    summary = build_summary(df, output_dir)

    # Quick headline numbers for the decision gate (plan Week 5–6)
    overall_rho = df.query(
        "firm == '_summary' and metric == 'spearman_rho'")["value"]
    overall_disp = df.query(
        "firm == '_summary' and metric == 'payment_displacement'")["value"]
    print("\n=== Decision-gate snapshot ===")
    print(f"overall mean Spearman ρ     : {overall_rho.mean():.3f}")
    print(f"overall mean displacement   : {overall_disp.mean():.3f}")
    print(f"fraction of runs with rank reversal: "
          f"{df.query('firm == \"_summary\" and metric == \"any_rank_reversal\"')['value'].mean():.2f}")

    print(f"\nAll artefacts written to {output_dir}/")
    return df, summary


if __name__ == "__main__":
    main()
