"""Export inventory-motivated weights for R2 paper tables."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

CELLS = {
    "k1": [(1.0, 9.0)] * 4,
    "k2": [(1.0, 4.0), (1.0, 6.0), (1.0, 9.0), (1.0, 12.0)],
    "k5": [(1.0, 2.0), (1.0, 5.0), (1.0, 12.0), (1.0, 25.0)],
}
L, SL = 1, 0.95


def w(ch, cb):
    tau = cb / (ch + cb)
    z = float(norm.ppf(np.clip(tau, 1e-6, 1 - 1e-6)))
    return float((ch + cb) * norm.pdf(z) * np.sqrt(L + 1))


def main():
    out = Path("results/r2_weight_grid")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell, prims in CELLS.items():
        ws = [w(ch, cb) for ch, cb in prims]
        ke = max(ws) / min(ws)
        for i, ((ch, cb), wi) in enumerate(zip(prims, ws)):
            rows.append(
                dict(cell=cell, firm=f"firm_{i}", ch=ch, cb=cb,
                     L=L, service_level=SL, w=wi, kappa_eff=ke)
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "weights_by_cell.csv", index=False)
    print(df.to_string(index=False))
    print(f"Wrote {out}/weights_by_cell.csv")


if __name__ == "__main__":
    main()
