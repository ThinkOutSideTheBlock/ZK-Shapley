"""
Cost-calibration table for inventory-motivated residual weights.

Maps operational primitives (ch, cb, L) → critical fractile τ, z_τ, weight w,
and κ_eff for the paper's cost cells (k1/k2/k5/k10).

Also reports a small retail/industrial reference ladder (illustrative ranges
commonly used in inventory teaching examples; not claimed as field estimates).

Usage:
  python -m experiments.run_cost_calibration
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def critical_tau(ch: float, cb: float) -> float:
    return float(cb) / (float(ch) + float(cb))


def inventory_weight(ch: float, cb: float, L: int = 1) -> float:
    tau = critical_tau(ch, cb)
    z = float(norm.ppf(np.clip(tau, 1e-6, 1.0 - 1e-6)))
    return float((ch + cb) * norm.pdf(z) * np.sqrt(L + 1))


def kappa_eff(ws: list[float]) -> float:
    a = np.asarray(ws, dtype=float)
    return float(np.max(a) / max(np.min(a), 1e-15))


# Paper cost cells (same primitives as Phase B / R1; pad to 8 for n-sweep)
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


def cell_table(n: int = 4, L: int = 1) -> pd.DataFrame:
    rows = []
    for cell, prims in COST_BASE.items():
        prims_n = prims[:n]
        ws = [inventory_weight(ch, cb, L=L) for ch, cb in prims_n]
        ke = kappa_eff(ws)
        for i, ((ch, cb), w) in enumerate(zip(prims_n, ws)):
            tau = critical_tau(ch, cb)
            z = float(norm.ppf(np.clip(tau, 1e-6, 1.0 - 1e-6)))
            rows.append(
                {
                    "n": n,
                    "cell": cell,
                    "firm": f"firm_{i}",
                    "ch": ch,
                    "cb": cb,
                    "cb_over_ch": cb / ch,
                    "L": L,
                    "tau": tau,
                    "z_tau": z,
                    "w": w,
                    "kappa_eff": ke,
                }
            )
    return pd.DataFrame(rows)


def reference_ladder(L: int = 1) -> pd.DataFrame:
    """
    Illustrative (ch, cb) pairs spanning low-to-high service critical fractiles.
    Labels are pedagogical, not econometric estimates.
    """
    pairs = [
        ("balanced", 1.0, 1.0),
        ("moderate_shortage", 1.0, 4.0),
        ("high_service", 1.0, 9.0),
        ("very_high_service", 1.0, 19.0),
        ("extreme", 1.0, 49.0),
    ]
    rows = []
    for label, ch, cb in pairs:
        tau = critical_tau(ch, cb)
        z = float(norm.ppf(np.clip(tau, 1e-6, 1.0 - 1e-6)))
        w = inventory_weight(ch, cb, L=L)
        rows.append(
            {
                "label": label,
                "ch": ch,
                "cb": cb,
                "cb_over_ch": cb / ch,
                "L": L,
                "tau": tau,
                "z_tau": z,
                "w": w,
            }
        )
    return pd.DataFrame(rows)


def main():
    out = Path("results/cost_calibration")
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for n in (4, 6, 8):
        df = cell_table(n=n, L=1)
        frames.append(df)
        print(f"\n=== Paper cells n={n} (L=1) ===")
        summ = (
            df.groupby("cell", as_index=False)
            .agg(
                kappa_eff=("kappa_eff", "first"),
                tau_min=("tau", "min"),
                tau_max=("tau", "max"),
                w_min=("w", "min"),
                w_max=("w", "max"),
                cb_ch_min=("cb_over_ch", "min"),
                cb_ch_max=("cb_over_ch", "max"),
            )
            .sort_values("kappa_eff")
        )
        print(summ.to_string(index=False))
        df.to_csv(out / f"weights_n{n}.csv", index=False)
        summ.to_csv(out / f"summary_n{n}.csv", index=False)

    ref = reference_ladder(L=1)
    ref.to_csv(out / "reference_ladder.csv", index=False)
    print("\n=== Reference τ ladder (illustrative) ===")
    print(ref.to_string(index=False))

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out / "weights_all_n.csv", index=False)

    lines = [
        "Cost calibration",
        "Weights use τ = cb/(ch+cb), w = (ch+cb) φ(z_τ) √(L+1), L=1.",
        "Paper cells k1–k10 induce moderate κ_eff at n=4 (~1–3); larger n may change κ slightly via extra primitives.",
        "Reference ladder is pedagogical only.",
        "",
        "Locked Regime A: rank(wΔσ)=rank(ΔC) under ideal newsvendor-style probe.",
        "Locked R1 boundary: residual-weighted vs path-cost ranks not equivalent (k1 mean ρ ≈ -0.59).",
    ]
    (out / "report.txt").write_text("\n".join(lines))
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
