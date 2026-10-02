"""
Smoke test for the statistical harness.

Runs a tiny grid (2 logical configs × 5 seeds) and verifies that:
  - parquet / csv is written
  - provenance files exist
  - operational surplus is positive on average
  - efficiency gap stays tiny
"""
from __future__ import annotations

from pathlib import Path

from experiments._harness import (
    run_sweep,
    make_smoke_grid,
    paired_shapley_scenario,
)


def main():
    out = Path("results/smoke")
    df = run_sweep(
        scenario_fn=paired_shapley_scenario,
        param_grid=make_smoke_grid(),
        n_seeds=5,
        master_seed=0,
        output_dir=out,
        stop_on_error=True,
    )

    print("\n=== Smoke-test summary ===")
    print(f"rows          : {len(df)}")
    print(f"columns       : {list(df.columns)}")

    # basic sanity
    v_op = df.query("metric == 'v_op_N'")["value"]
    print(f"mean v_op(N)  : {v_op.mean():.3f}  (should be > 0)")
    assert v_op.mean() > 0, "collaboration surplus vanished — check DGP"

    gap = df.query("metric == 'efficiency_gap_op'")["value"]
    print(f"max eff. gap  : {gap.max():.2e}  (should be < 1e-8)")
    assert gap.max() < 1e-8

    assert (out / "config.json").exists()
    assert (out / "git_sha.txt").exists()
    assert (out / "env.json").exists()
    assert (out / "results.parquet").exists() or (out / "results.csv").exists()

    print("Smoke test PASSED")
    return df


if __name__ == "__main__":
    main()
