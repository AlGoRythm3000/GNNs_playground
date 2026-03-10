"""Experiment: Visualise over-smoothing as depth increases in a GCN.

Run with:
    python experiments/over_smoothing_exp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from gnns_playground.metrics.over_smoothing import (
    dirichlet_energy,
    mean_average_distance,
    smoothness_ratio,
)
from gnns_playground.models.gcn import GCN
from gnns_playground.utils.graph_utils import (
    barabasi_albert_graph,
    nx_to_adj,
    nx_to_torch,
    random_graph,
)


def run_experiment(n: int = 50, max_depth: int = 10, seed: int = 0) -> None:
    """Study Dirichlet energy decay as GCN depth increases."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    graph = random_graph(n, p=0.15, seed=seed)
    _, x, adj = nx_to_torch(graph)

    print(f"\n{'='*60}")
    print(f"Over-Smoothing Experiment  (N={n} nodes, Erdos-Renyi p=0.15)")
    print(f"{'='*60}")

    depths = list(range(1, max_depth + 1))
    energies: list[float] = []
    mad_global: list[float] = []
    smooth_ratios: list[float] = []

    for depth in depths:
        hidden = [64] * (depth - 1) if depth > 1 else []
        model = GCN(in_features=x.shape[1], hidden_dims=hidden, out_features=16, dropout=0.0)
        model.eval()
        with torch.no_grad():
            h = model(x, adj)
        e = dirichlet_energy(h, adj)
        mad = mean_average_distance(h, mode="global")
        sr = smoothness_ratio(h, adj)

        energies.append(e)
        mad_global.append(mad)
        smooth_ratios.append(sr)

        print(
            f"  depth={depth:2d}  Dirichlet E={e:.4f}  MAD={mad:.4f}  smoothness_ratio={sr:.4f}"
        )

    _plot_results(depths, energies, mad_global, smooth_ratios, n)


def _plot_results(
    depths: list,
    energies: list,
    mad_global: list,
    smooth_ratios: list,
    n: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Over-Smoothing vs. GCN Depth  (N={n})", fontsize=14)

    for ax, vals, title, ylabel in zip(
        axes,
        [energies, mad_global, smooth_ratios],
        ["Dirichlet Energy", "Mean Average Distance (Global)", "Smoothness Ratio"],
        ["E(h) (↓ = more smooth)", "MAD (↓ = more smooth)", "MAD_local/MAD_global"],
    ):
        ax.plot(depths, vals, marker="o", linewidth=2)
        ax.set_xlabel("GCN depth (# layers)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / "over_smoothing_results.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    run_experiment(n=50, max_depth=10)
