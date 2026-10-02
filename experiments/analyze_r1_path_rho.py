"""Descriptive CI for residual vs path Spearman (limitation section)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

p = Path("results/r1_residual_vs_path/results.csv")
out = Path("results/r2_path_rho_summary")
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(p)
# column name from R1 script
col = "rho_phi" if "rho_phi" in df.columns else "rho"
rows = []
rng = np.random.default_rng(0)
for cell, g in df.groupby("cell"):
    x = g[col].astype(float).values
    n = len(x)
    boots = []
    for _ in range(2000):
        boots.append(float(np.mean(rng.choice(x, size=n, replace=True))))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    rows.append(
        dict(
            cell=cell,
            n=n,
            mean_rho=float(np.mean(x)),
            std=float(np.std(x, ddof=1)),
            ci95_lo=float(lo),
            ci95_hi=float(hi),
        )
    )
s = pd.DataFrame(rows)
s.to_csv(out / "path_rho_by_cell.csv", index=False)
print(s.to_string(index=False))
print(f"Wrote {out}/")
