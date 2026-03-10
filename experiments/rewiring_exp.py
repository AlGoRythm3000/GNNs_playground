"""Experiment: Compare SDRF and FOSR rewiring algorithms.

Run with:
    python experiments/rewiring_exp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from gnns_playground.metrics.over_squashing import (
    compute_total_effective_resistance,
    mean_forman_curvature,
    spectral_gap,
)
from gnns_playground.rewiring.fosr import fosr, random_edge_addition
from gnns_playground.rewiring.sdrf import sdrf
from gnns_playground.utils.graph_utils import (
    barabasi_albert_graph,
    cycle_graph,
    path_graph,
    random_graph,
)


def run_experiment(n: int = 30, seed: int = 42) -> None:
    """Compare rewiring algorithms on a path graph and a random graph."""
    graphs = {
        "Path": path_graph(n),
        "Random (p=0.1)": random_graph(n, p=0.1, seed=seed),
    }

    num_edges = max(1, n // 5)

    print(f"\n{'='*70}")
    print(f"Rewiring Experiment  (N={n} nodes, adding {num_edges} edges)")
    print(f"{'='*70}")

    for gname, base_graph in graphs.items():
        print(f"\n--- Graph: {gname} ---")
        _compare_rewirings(base_graph, gname, num_edges, seed)


def _compare_rewirings(
    base_graph: nx.Graph,
    graph_name: str,
    num_edges: int,
    seed: int,
) -> None:
    n = base_graph.number_of_nodes()

    rewired_sdrf, stats_sdrf = sdrf(
        base_graph,
        num_iterations=num_edges * 5,
        tau=0.0,
        max_edges_to_add=num_edges,
        seed=seed,
    )
    rewired_fosr, stats_fosr = fosr(
        base_graph,
        num_edges_to_add=num_edges,
    )
    rewired_rand, stats_rand = random_edge_addition(
        base_graph,
        num_edges_to_add=num_edges,
        seed=seed,
    )

    results = {
        "Original": base_graph,
        f"SDRF (+{stats_sdrf['edges_added']} edges)": rewired_sdrf,
        f"FOSR (+{stats_fosr['edges_added']} edges)": rewired_fosr,
        f"Random (+{stats_rand['edges_added']} edges)": rewired_rand,
    }

    print(f"\n{'Method':<30} {'TER':>10} {'λ₂':>10} {'Mean κ_F':>12}")
    print("-" * 65)
    for method, g in results.items():
        ter = compute_total_effective_resistance(g)
        gap = spectral_gap(g)
        mkappa = mean_forman_curvature(g)
        print(f"{method:<30} {ter:>10.2f} {gap:>10.4f} {mkappa:>12.4f}")

    _plot_curvature_histories(stats_sdrf, graph_name)


def _plot_curvature_histories(stats_sdrf: dict, graph_name: str) -> None:
    history = stats_sdrf.get("curvature_history", [])
    if not history:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, marker="o", linewidth=2, label="SDRF mean κ_F")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="κ_F = 0")
    ax.set_xlabel("SDRF iteration")
    ax.set_ylabel("Mean Forman-Ricci Curvature κ_F")
    ax.set_title(f"Curvature Evolution during SDRF  ({graph_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_name = graph_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
    out_path = Path(__file__).parent / f"rewiring_{safe_name}_curvature.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  Plot saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    run_experiment(n=30, seed=42)
