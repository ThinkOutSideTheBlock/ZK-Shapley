"""
Stage 1 — Claude plan: retro-fit κ_eff on archived Exp-2 results.

Uses results.csv (tidy long format). No parquet required.
κ_eff proxy = max_i σ̂_i / min_i σ̂_i from firm-specific predictive residuals
(singleton β). Cost structure was homogeneous across firms in Exp-2, so
cost-based κ ≡ 1; residual-scale heterogeneity is the live channel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std


def load_exp2_summary(path: Path) -> pd.DataFrame:
    """Load tidy results.csv and keep one row per (config × seed) with spearman_rho."""
    df = pd.read_csv(path)
    # Expected columns from harness: firm, metric, value, seed, + config fields
    needed = {"firm", "metric", "value", "seed"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"results.csv missing columns: {missing}")

    sub = df.query("firm == '_summary' and metric == 'spearman_rho'").copy()
    sub["spearman_rho"] = sub["value"].astype(float)
    sub["seed"] = sub["seed"].astype(int)

    # Config columns that may be present
    config_cols = [
        c
        for c in [
            "n_firms",
            "shared_factor_correlation",
            "cb_over_ch",
            "n_periods",
            "n_features",
            "n_obs_heterogeneity",
            "ridge_lambda",
        ]
        if c in sub.columns
    ]
    keep = ["seed", "spearman_rho"] + config_cols
    return sub[keep].drop_duplicates().reset_index(drop=True)


def kappa_eff_from_residuals(datasets, firms, lam0: float = 1.0) -> float:
    """κ_eff = max σ̂_i / min σ̂_i using singleton predictive RMS residual."""
    sigmas = []
    for name in firms:
        S = frozenset({name})
        # Lightweight fit via CoalitionEvaluator path
        paths = {n: datasets[n].y_holdout for n in firms}
        cfg = CoalitionConfig(
            firms=tuple(firms),
            lambda_mode="fixed",
            residual_mode="predictive",
            ridge_lambda=lam0,
            seed=0,
        )
        ev = CoalitionEvaluator(datasets, paths, cfg)
        beta = ev.fit(S)
        s = firm_predictive_residual_std(beta, datasets[name])
        sigmas.append(s)
    sigmas = np.asarray(sigmas, dtype=float)
    return float(np.max(sigmas) / max(np.min(sigmas), 1e-12))


def rebuild_kappa_for_row(row: pd.Series) -> float:
    n_firms = int(row.get("n_firms", 4))
    rho = float(row.get("shared_factor_correlation", 0.6))
    n_periods = int(row.get("n_periods", 150))
    n_features = int(row.get("n_features", 5))
    seed = int(row["seed"])
    het = row.get("n_obs_heterogeneity", "moderate")
    if pd.isna(het):
        het = "moderate"
    lam0 = float(row.get("ridge_lambda", 1.0)
                 ) if "ridge_lambda" in row.index else 1.0

    cfg = DemandSimulationConfig(
        n_firms=n_firms,
        n_periods=n_periods,
        n_features=n_features,
        shared_factor_correlation=rho,
        seed=seed,
        n_obs_heterogeneity=str(het),
    )
    sim = generate_multi_firm_demand(cfg)
    datasets = build_firm_datasets(
        sim, val_fraction=0.15, holdout_fraction=0.20, min_holdout=20
    )
    firms = tuple(sorted(datasets.keys()))
    return kappa_eff_from_residuals(datasets, firms, lam0=lam0)


def main():
    results_path = Path("results/exp2_accuracy_vs_inventory/results.csv")
    if not results_path.exists():
        # fallback: try smoke or other known locations
        alts = list(Path("results").rglob("results.csv"))
        if not alts:
            raise FileNotFoundError(
                "No results.csv found under results/. Run Exp-2 first."
            )
        results_path = alts[0]
        print(f"[info] using {results_path}")

    out_dir = Path("results/stage1_kappaeff_retrofit")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_exp2_summary(results_path)
    print(f"Loaded {len(summary)} summary rows from {results_path}")

    # Optionally subsample for speed while debugging (set to None for full)
    max_rows = None  # e.g. 60 for a quick pass
    if max_rows is not None and len(summary) > max_rows:
        summary = summary.sample(
            n=max_rows, random_state=0).reset_index(drop=True)
        print(f"[info] subsampled to {len(summary)} rows")

    kappas = []
    for i, row in summary.iterrows():
        try:
            k = rebuild_kappa_for_row(row)
        except Exception as e:
            print(f"[warn] seed={row['seed']} failed: {e}")
            k = np.nan
        kappas.append(k)
        if (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{len(summary)}")

    summary = summary.copy()
    summary["kappa_eff"] = kappas
    summary = summary.dropna(subset=["kappa_eff", "spearman_rho"])
    summary["log_kappa"] = np.log(summary["kappa_eff"].clip(lower=1e-12))

    # Simple OLS: rho ~ log_kappa
    x = summary["log_kappa"].to_numpy()
    y = summary["spearman_rho"].to_numpy()
    if len(x) < 5:
        raise RuntimeError("Too few rows for regression.")

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    print("\n=== Stage 1 — κ_eff retro-fit ===")
    print(f"n runs              : {len(summary)}")
    print(f"mean κ_eff          : {summary['kappa_eff'].mean():.3f}")
    print(f"median κ_eff        : {summary['kappa_eff'].median():.3f}")
    print(f"mean Spearman ρ     : {summary['spearman_rho'].mean():.3f}")
    print(f"OLS: ρ = {intercept:.3f} + {slope:.3f} * log(κ_eff)")
    print(f"  SE(slope)         : {std_err:.3f}")
    print(f"  p-value           : {p_value:.4g}")
    print(f"  R²                : {r_value**2:.3f}")

    # Claude decision rule (informal)
    if slope < 0 and p_value < 0.05:
        print("\nReading: ρ decreases in log(κ_eff) — supports C19/C20 direction.")
    elif slope < 0:
        print("\nReading: negative slope but not significant — need more runs / better κ.")
    else:
        print("\nReading: slope not negative — check residual channel / κ definition.")

    summary.to_csv(out_dir / "kappa_eff_retrofit.csv", index=False)
    print(f"\nWrote {out_dir / 'kappa_eff_retrofit.csv'}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(summary["kappa_eff"],
                   summary["spearman_rho"], alpha=0.5, s=20)
        xx = np.linspace(summary["log_kappa"].min(),
                         summary["log_kappa"].max(), 50)
        ax.plot(np.exp(xx), intercept + slope * xx, color="crimson", lw=2)
        ax.set_xlabel(r"$\kappa_{\mathrm{eff}}$")
        ax.set_ylabel(r"Spearman $\rho$")
        ax.set_title("Stage 1: Spearman ρ vs κ_eff (Exp-2 archive)")
        fig.tight_layout()
        fig.savefig(out_dir / "kappa_eff_spearman.png", dpi=150)
        plt.close()
        print(f"Wrote {out_dir / 'kappa_eff_spearman.png'}")
    except Exception as e:
        print(f"[warn] plot skipped: {e}")


if __name__ == "__main__":
    main()
