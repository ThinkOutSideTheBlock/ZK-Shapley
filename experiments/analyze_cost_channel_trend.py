"""
R2 trend hygiene (M3–M4).

Primary unit: seed (and seed × n when n is available).
Does NOT treat pooled seed×cell rows as iid for significance.

Outputs under results/r2_cost_channel_trend/:
  trend_report.txt
  within_seed_by_n.csv   (if n column present)
  within_seed_pooled_n.csv  (descriptive only if n mixed)
  cell_means.csv
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest

CANDIDATES = [
    Path("results/stage2_cost_n_sweep/results.csv"),
    Path("results/stage2_three_channel/results.csv"),
    Path("results/stage2_kappa_semistruct/results.csv"),
    Path("results/stage2_kappa_sweep/results.csv"),
]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for key, opts in {
        "cell": ["cell", "schedule"],
        "kappa": ["kappa_eff", "kappa_mean", "kappa"],
        "rho": ["spearman_rho", "rho", "rho_mean", "spearman_mean", "rho_phi"],
        "seed": ["seed", "struct_seed", "master_seed"],
        "n": ["n_firms", "n"],
    }.items():
        for o in opts:
            if o in cols:
                rename[cols[o]] = key
                break
    d = df.rename(columns=rename)
    return d


def load_seed_level() -> pd.DataFrame | None:
    for p in CANDIDATES:
        if not p.exists():
            continue
        d = _normalize(pd.read_csv(p))
        if {"kappa", "rho"}.issubset(d.columns):
            if "seed" not in d.columns:
                d["seed"] = np.arange(len(d))
            d["kappa"] = d["kappa"].astype(float)
            d["rho"] = d["rho"].astype(float)
            if "n" in d.columns:
                d["n"] = d["n"].astype(int)
            d["_source"] = str(p)
            return d.dropna(subset=["kappa", "rho"])
    return None


def within_group_spearman(g: pd.DataFrame) -> dict | None:
    if g["kappa"].nunique() < 3:
        return None
    r, p = spearmanr(g["kappa"], g["rho"])
    return {
        "r": float(r),
        "p_naive": float(p),  # naive; not used for headline inference
        "n_cells": int(g["kappa"].nunique()),
        "kappa_min": float(g["kappa"].min()),
        "kappa_max": float(g["kappa"].max()),
        "rho_at_min_kappa": float(g.loc[g["kappa"].idxmin(), "rho"]),
        "rho_at_max_kappa": float(g.loc[g["kappa"].idxmax(), "rho"]),
    }


def sign_test_negative(rs: np.ndarray) -> dict:
    """Two-sided sign test: H0 = P(r<0)=1/2 (ignore exact zeros)."""
    rs = np.asarray(rs, dtype=float)
    rs = rs[np.isfinite(rs)]
    n_pos = int(np.sum(rs > 0))
    n_neg = int(np.sum(rs < 0))
    n = n_pos + n_neg
    if n == 0:
        return {"n": 0, "n_neg": 0, "n_pos": 0, "p_sign": np.nan}
    # binomtest two-sided under p=0.5
    res = binomtest(n_neg, n, p=0.5, alternative="two-sided")
    return {
        "n": n,
        "n_neg": n_neg,
        "n_pos": n_pos,
        "frac_neg": n_neg / n,
        "p_sign": float(res.pvalue),
    }


def main():
    out = Path("results/r2_cost_channel_trend")
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    d = load_seed_level()
    if d is None:
        msg = (
            "No seed-level CSV found.\n"
            "Point CANDIDATES to a long results file with kappa, rho, seed "
            "(and ideally n_firms).\n"
        )
        (out / "trend_report.txt").write_text(msg)
        print(msg)
        return

    lines.append(f"Source: {d['_source'].iloc[0]}\n")
    lines.append(f"Rows: {len(d)}\n\n")

    # --- Cell means (descriptive) ---
    if "cell" in d.columns:
        summ = (
            d.groupby("cell", as_index=False)
            .agg(
                kappa=("kappa", "mean"),
                rho_mean=("rho", "mean"),
                rho_std=("rho", "std"),
                n_obs=("rho", "count"),
            )
            .sort_values("kappa")
        )
        if "n" in d.columns:
            # prefer cell means within each n if possible
            summ_n = (
                d.groupby(["n", "cell"], as_index=False)
                .agg(
                    kappa=("kappa", "mean"),
                    rho_mean=("rho", "mean"),
                    rho_std=("rho", "std"),
                    n_obs=("rho", "count"),
                )
                .sort_values(["n", "kappa"])
            )
            summ_n.to_csv(out / "cell_means_by_n.csv", index=False)
        summ.to_csv(out / "cell_means.csv", index=False)
        lines.append("Cell means (descriptive only):\n")
        lines.append(summ.to_string(index=False) + "\n\n")

    # --- Primary: within-seed × n (preferred) ---
    rows_by_n = []
    if "n" in d.columns:
        for (seed, n), g in d.groupby(["seed", "n"]):
            st = within_group_spearman(g)
            if st is None:
                continue
            rows_by_n.append({"seed": seed, "n": int(n), **st})
        if rows_by_n:
            pdf = pd.DataFrame(rows_by_n)
            pdf.to_csv(out / "within_seed_by_n.csv", index=False)
            lines.append(
                "=== Within-seed Spearman(kappa, rho) CONDITIONAL ON n ===\n")
            for n, g in pdf.groupby("n"):
                st = sign_test_negative(g["r"].values)
                lines.append(
                    f"n={n}: mean r={g['r'].mean():.4f}, median r={g['r'].median():.4f}, "
                    f"frac r<0={st['frac_neg']:.3f} ({st['n_neg']}/{st['n']}), "
                    f"sign-test p={st['p_sign']:.3e}, n_units={len(g)}\n"
                )
            st_all = sign_test_negative(pdf["r"].values)
            lines.append(
                f"All seed×n units: mean r={pdf['r'].mean():.4f}, "
                f"frac r<0={st_all['frac_neg']:.3f} ({st_all['n_neg']}/{st_all['n']}), "
                f"sign-test p={st_all['p_sign']:.3e}\n\n"
            )
            lines.append(
                "HEADLINE: use frac r<0 and sign-test on seed×n units; "
                "do not cite pooled row-count Spearman p-values.\n\n"
            )

    # --- Fallback: within-seed pooling all n (descriptive; may confound n and kappa) ---
    rows_seed = []
    for seed, g in d.groupby("seed"):
        st = within_group_spearman(g)
        if st is None:
            continue
        rows_seed.append({"seed": seed, **st})
    if rows_seed:
        sdf = pd.DataFrame(rows_seed)
        sdf.to_csv(out / "within_seed_pooled_n.csv", index=False)
        st = sign_test_negative(sdf["r"].values)
        lines.append(
            "=== Within-seed Spearman pooling n (DESCRIPTIVE; may mix n and kappa) ===\n"
            f"mean r={sdf['r'].mean():.4f}, median r={sdf['r'].median():.4f}, "
            f"frac r<0={st['frac_neg']:.3f} ({st['n_neg']}/{st['n']}), "
            f"sign-test p={st['p_sign']:.3e}, n_seeds={len(sdf)}\n\n"
        )

    # --- Explicit non-claim on pooled iid Spearman ---
    r_all, p_all = spearmanr(d["kappa"], d["rho"])
    lines.append(
        "=== Pooled seed×cell Spearman (NOT for significance claims) ===\n"
        f"N_rows={len(d)}, r={r_all:.4f}, naive_p={p_all:.4e}\n"
        "Dependent across cells sharing a seed; do not report naive_p in the paper.\n"
    )

    report = "".join(lines)
    (out / "trend_report.txt").write_text(report)
    print(report)
    print(f"Wrote {out}/")


if __name__ == "__main__":
    main()
