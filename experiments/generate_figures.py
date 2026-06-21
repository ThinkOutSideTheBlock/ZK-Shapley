"""
Generates every figure referenced in the paper, reading from results/*.csv
produced by run_main_experiment.py. Each figure is saved as both PDF (vector,
for LaTeX \\includegraphics) and PNG (for quick inspection).

Run: python3 -m experiments.generate_figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# --- Publication style ---
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "legend.frameon": False,
})

COLOR_ZK = "#1b6ca8"
COLOR_EQUAL = "#9b9b9b"
COLOR_PROP = "#d4843c"
COLOR_NOSHARE = "#b94a4a"
COLOR_ACCENT = "#3f8f5f"


def _save(fig, name: str):
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def fig_payment_comparison():
    df = pd.read_csv(RESULTS_DIR / "exp1_payments.csv")
    df = df.sort_values("zk_shapley_payment", ascending=False).reset_index(drop=True)
    n = len(df)
    x = np.arange(n)
    width = 0.26

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x - width, df["zk_shapley_payment"], width, label="ZK-Shapley (proposed)", color=COLOR_ZK)
    ax.bar(x, df["equal_split_payment"], width, label="Equal Split", color=COLOR_EQUAL)
    ax.bar(x + width, df["proportional_payment"], width, label="Proportional-to-Size", color=COLOR_PROP)

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("firm_", "Firm ") for f in df["firm"]], fontsize=9)
    ax.set_ylabel("Payment (forecast-utility units)")
    ax.set_title("Payment Allocation by Mechanism")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, "fig2_payment_comparison")


def fig_fairness_correlation():
    df = pd.read_csv(RESULTS_DIR / "exp1_payments.csv")
    n = len(df)

    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    sizes = 220 * (df["n_train_rows"] / df["n_train_rows"].max()) + 30
    ax.scatter(df["singleton_value"], df["zk_shapley_payment"], s=sizes,
               c=COLOR_ZK, alpha=0.75, edgecolor="white", linewidth=1.0)
    for _, row in df.iterrows():
        ax.annotate(row["firm"].replace("firm_", "F"), (row["singleton_value"], row["zk_shapley_payment"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    coeffs = np.polyfit(df["singleton_value"], df["zk_shapley_payment"], 1)
    xs = np.linspace(df["singleton_value"].min(), df["singleton_value"].max(), 50)
    ax.plot(xs, np.polyval(coeffs, xs), "--", color="gray", linewidth=1.2, alpha=0.8)
    corr = np.corrcoef(df["singleton_value"], df["zk_shapley_payment"])[0, 1]

    ax.set_xlabel(r"Standalone value $v(\{i\})$ (firm acting alone, no federation)")
    ax.set_ylabel("ZK-Shapley payment")
    ax.set_title(f"Payment vs. Standalone Marginal Value ($n$={n} firms, r = {corr:.2f})")
    ax.text(0.03, 0.95, "Marker size = training rows", transform=ax.transAxes,
            fontsize=8, va="top", color="gray")
    ax.text(0.03, 0.04,
            r"$v(\{i\})$ is the weight-$1/n$ term in the exact Shapley sum,"
            "\nso this relationship is structural, not merely empirical.",
            transform=ax.transAxes, fontsize=7.5, va="bottom", color="gray", style="italic")
    _save(fig, "fig3_fairness_correlation")


def fig_mc_convergence():
    df = pd.read_csv(RESULTS_DIR / "exp2_mc_convergence.csv")
    with open(RESULTS_DIR / "exp2_exact_reference.json") as f:
        ref = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    ax = axes[0]
    ax.errorbar(df["n_permutations"], df["l1_error_mean"], yerr=df["l1_error_std"],
                marker="o", color=COLOR_ZK, capsize=3, linewidth=1.5, markersize=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Monte Carlo permutation budget")
    ax.set_ylabel(r"$L_1$ error vs. exact Shapley")
    ax.set_title(f"(a) Convergence ($n$={ref['n_firms']} firms, 8 seeds/point)")

    ax = axes[1]
    ax.plot(df["n_unique_evaluations_avg"], df["wall_time_sec_mean"],
            marker="s", color=COLOR_ACCENT, linewidth=1.5, markersize=5,
            label="Monte Carlo")
    ax.axhline(ref["exact_time_sec"], color=COLOR_NOSHARE, linestyle="--", linewidth=1.3,
               label=f"Exact ($2^{{{ref['n_firms']}}}$={ref['exact_evaluations']} evals)")
    ax.set_xlabel("Characteristic function evaluations")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("(b) Computational cost")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "fig4_mc_convergence")


def fig_deviation_analysis():
    df = pd.read_csv(RESULTS_DIR / "exp3_deviation_analysis.csv")
    deviations = ["honest", "free_ride", "noise_injection", "data_inflation"]
    dev_labels = ["Honest", "Free-Ride", "Noise\nInjection", "Data Size\nInflation"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=False)
    for ax, role, title in zip(axes, ["smallest_data", "largest_data"],
                                 ["(a) Smallest-data firm", "(b) Largest-data firm"]):
        sub = df[df.target_firm_role == role].set_index("deviation").reindex(deviations)
        x = np.arange(len(deviations))
        width = 0.32

        zk_vals = sub["zk_shapley_payment"].values
        prop_vals = sub["proportional_payment"].values

        bars_zk = ax.bar(x - width / 2, np.nan_to_num(zk_vals, nan=0.0), width,
                          color=COLOR_ZK, label="ZK-Shapley")
        bars_prop = ax.bar(x + width / 2, prop_vals, width,
                            color=COLOR_PROP, label="Proportional")

        # Mark the structurally-inapplicable ZK-Shapley/data-inflation bar explicitly.
        if np.isnan(zk_vals[-1]):
            ax.text(x[-1] - width / 2, 0.02, "N/A\n(no size\ninput)", ha="center", va="bottom",
                    fontsize=7, color=COLOR_ZK, style="italic")

        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(dev_labels, fontsize=8.5)
        ax.set_title(title, fontsize=10)
        if role == "smallest_data":
            ax.set_ylabel("Payment")
        ax.legend(fontsize=8)

    fig.suptitle("Incentive Stress Test: Payment Under Deviation Strategies", fontsize=12, y=1.03)
    fig.tight_layout()
    _save(fig, "fig5_deviation_analysis")


def fig_inventory_impact():
    df = pd.read_csv(RESULTS_DIR / "exp4_inventory_impact.csv").sort_values("n_contributing_firms")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    ax.plot(df["n_contributing_firms"], df["total_inventory_cost"], marker="o",
            color=COLOR_ZK, linewidth=1.8, markersize=6)
    ax.scatter(df.loc[df.n_contributing_firms == 0, "n_contributing_firms"],
               df.loc[df.n_contributing_firms == 0, "total_inventory_cost"],
               color=COLOR_NOSHARE, s=80, zorder=5, label="No Sharing")
    ax.scatter(df.loc[df.n_contributing_firms == df.n_contributing_firms.max(), "n_contributing_firms"],
               df.loc[df.n_contributing_firms == df.n_contributing_firms.max(), "total_inventory_cost"],
               color=COLOR_ACCENT, s=80, zorder=5, label="Full Participation")
    ax.set_xlabel("Number of contributing firms (largest data first)")
    ax.set_ylabel("Total 3-echelon inventory cost")
    ax.set_title("(a) System cost vs. participation level")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(df["n_contributing_firms"], df["bullwhip_ratio"], marker="^",
            color=COLOR_ACCENT, linewidth=1.8, markersize=6, label="Bullwhip ratio")
    ax2 = ax.twinx()
    ax2.plot(df["n_contributing_firms"], df["forecast_rmse"], marker="d",
             color=COLOR_PROP, linewidth=1.4, markersize=5, linestyle="--", label="Forecast RMSE")
    ax2.set_ylabel("Forecast RMSE", color=COLOR_PROP)
    ax2.tick_params(axis="y", colors=COLOR_PROP)
    ax.set_ylabel("Bullwhip ratio", color=COLOR_ACCENT)
    ax.tick_params(axis="y", colors=COLOR_ACCENT)
    ax.set_xlabel("Number of contributing firms")
    ax.set_title("(b) Bullwhip amplification vs. forecast quality")

    fig.tight_layout()
    _save(fig, "fig6_inventory_impact")


def fig_zk_cost_scaling():
    from src.commitment import estimate_zk_cost
    row_counts = np.array([50, 100, 500, 1000, 5000, 1e4, 5e4, 1e5, 5e5, 2e6])
    proving = [estimate_zk_cost(int(n), n_features=6).proving_time_sec for n in row_counts]
    verifying = [estimate_zk_cost(int(n), n_features=6).verification_time_sec for n in row_counts]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(row_counts, proving, marker="o", color=COLOR_NOSHARE, label="Proving time (est.)")
    ax.plot(row_counts, verifying, marker="s", color=COLOR_ACCENT, label="Verification time (est.)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Firm training rows per round")
    ax.set_ylabel("Time (s, estimated)")
    ax.set_title("Estimated ZK Proof Cost\n(calibrated to Chen et al., EuroSys 2024)")

    # Shade the realistic operating range used in this paper's experiments
    # (n_train ~ 75-700 rows per firm) and annotate that proving cost is
    # FLOOR-DOMINATED (fixed proof-system overhead) in this regime, not
    # data-size-dominated -- the linear gate-count scaling only starts to
    # matter at row counts roughly 3-4 orders of magnitude larger.
    ax.axvspan(75, 700, color=COLOR_ZK, alpha=0.08)
    ax.text(180, ax.get_ylim()[0] * 2.0, "this paper's\nfirm sizes", fontsize=7.5,
            color=COLOR_ZK, ha="center", style="italic")
    ax.legend(fontsize=9, loc="upper left")
    _save(fig, "fig7_zk_cost_scaling")


def fig_protocol_diagram():
    """Simple, reproducible schematic of the ZK-Shapley protocol (code-generated, not hand-drawn)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                            linewidth=1.1, edgecolor=color, facecolor=color, alpha=0.16)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="black")
        return (x + w / 2, y), (x + w / 2, y + h)

    def arrow(p1, p2, color="black"):
        a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color=color)
        ax.add_patch(a)

    firm_centers = []
    labels = ["Retailer\n(private $D_A$)", "Distributor\n(private $D_B$)", "Manufacturer\n(private $D_C$)"]
    for i, label in enumerate(labels):
        cx = 1.0 + i * 3.2
        bot, top = box(cx, 4.9, 2.4, 1.0, label, COLOR_ZK)
        firm_centers.append(bot)

    fl_bot, fl_top = box(0.6, 3.0, 8.8, 1.2,
                          "Federated ridge regression: each firm submits\n"
                          r"sufficient statistics $(A_i, b_i)$ + commit-reveal binding",
                          COLOR_ACCENT, fontsize=9.5)
    for c in firm_centers:
        arrow(c, (c[0], 4.2))

    sc_bot, sc_top = box(1.6, 1.5, 6.8, 1.1,
                          "Smart contract: verify commitments,\ncompute Shapley payoffs " r"$\phi_i(v)$",
                          COLOR_PROP, fontsize=9.5)
    arrow((5.0, 3.0), (5.0, 2.6))

    th_bot, th_top = box(1.0, 0.1, 8.0, 1.05,
                          "Theorem 1 (Efficiency): " r"$\sum_i \phi_i = v(N)$" "   |   "
                          "Theorem 2 (Fairness axioms)   |   structural robustness to size misreport",
                          "#444444", fontsize=8.3)
    arrow((5.0, 1.5), (5.0, 1.15))

    ax.set_title("ZK-Shapley Protocol Overview", fontsize=12, pad=14)
    _save(fig, "fig1_protocol_diagram")


if __name__ == "__main__":
    print("Generating figures...")
    fig_protocol_diagram()
    fig_payment_comparison()
    fig_fairness_correlation()
    fig_mc_convergence()
    fig_deviation_analysis()
    fig_inventory_impact()
    fig_zk_cost_scaling()
    print(f"\nAll figures written to {FIGURES_DIR}/")
