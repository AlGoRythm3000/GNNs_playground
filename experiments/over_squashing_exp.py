"""Experiment: Visualise and compare over-squashing metrics across graph types.

Run with:
    python experiments/over_squashing_exp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from gnns_playground.metrics.over_squashing import (
    compute_effective_resistance_matrix,
    compute_total_effective_resistance,
    forman_ricci_curvature,
    mean_forman_curvature,
    most_negatively_curved_edges,
    spectral_gap,
)
from gnns_playground.utils.graph_utils import (
    barabasi_albert_graph,
    complete_graph,
    cycle_graph,
    path_graph,
    random_graph,
)


def run_experiment(n: int = 20) -> None:
    """Compare over-squashing metrics on graphs with N nodes."""
    graphs = {
        "Path P_N": path_graph(n),
        "Cycle C_N": cycle_graph(n),
        "Erdos-Renyi G(N,0.2)": random_graph(n, p=0.2, seed=42),
        "Barabasi-Albert (m=2)": barabasi_albert_graph(n, m=2, seed=42),
        "Complete K_N": complete_graph(n),
    }

    print(f"\n{'='*70}")
    print(f"Over-Squashing Metrics  (N={n} nodes)")
    print(f"{'='*70}")
    header = f"{'Graph':<26} {'TER':>10} {'λ₂ (gap)':>12} {'Mean κ_F':>12}"
    print(header)
    print("-" * 70)

    results: dict[str, dict] = {}
    for name, g in graphs.items():
        ter = compute_total_effective_resistance(g)
        gap = spectral_gap(g)
        mkappa = mean_forman_curvature(g)
        results[name] = {"TER": ter, "spectral_gap": gap, "mean_kappa": mkappa}
        print(f"{name:<26} {ter:>10.2f} {gap:>12.4f} {mkappa:>12.4f}")

    # Show most bottlenecked edges for the path graph
    path_g = graphs["Path P_N"]
    print(f"\nTop-5 most negatively curved edges in Path P_{n}:")
    for edge, kappa in most_negatively_curved_edges(path_g, top_k=5):
        print(f"  {edge}  κ_F = {kappa:.2f}")

    # Plot TER and spectral gap
    _plot_results(results, n)


def _plot_results(results: dict, n: int) -> None:
    names = list(results.keys())
    ter_vals = [results[k]["TER"] for k in names]
    gap_vals = [results[k]["spectral_gap"] for k in names]
    kappa_vals = [results[k]["mean_kappa"] for k in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Over-Squashing Metrics (N={n})", fontsize=14)

    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for ax, vals, title, ylabel in zip(
        axes,
        [ter_vals, gap_vals, kappa_vals],
        ["Total Effective Resistance", "Spectral Gap λ₂", "Mean Forman Curvature κ_F"],
        ["TER (↓ better connectivity)", "λ₂ (↑ better connectivity)", "κ_F (↑ less bottleneck)"],
    ):
        bars = ax.bar(range(len(names)), vals, color=colours)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split(" ")[0] for n in names], rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() * 1.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    out_path = Path(__file__).parent / "over_squashing_results.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    run_experiment(n=20)
