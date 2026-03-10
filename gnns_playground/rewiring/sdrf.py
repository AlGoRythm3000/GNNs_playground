"""Stochastic Discrete Ricci Flow (SDRF) graph rewiring.

SDRF iteratively removes edges with negative curvature and adds edges
between 2-hop neighbours to alleviate over-squashing.

Reference
---------
Topping et al. (2022) "Understanding over-squashing and bottlenecks on
graphs via curvature."  ICLR 2022.  https://arxiv.org/abs/2111.14522
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Curvature computation (Forman-Ricci)
# ---------------------------------------------------------------------------


def _forman_curvature(graph: nx.Graph, u: int, v: int) -> float:
    """Forman-Ricci curvature of edge (u, v) in an unweighted graph.

        κ_F(u, v) = 4 - deg(u) - deg(v)

    In a weighted generalisation each node weight equals its degree and each
    edge weight equals 1 (unit weights).
    """
    return 4.0 - graph.degree(u) - graph.degree(v)


def _all_curvatures(graph: nx.Graph) -> dict[tuple, float]:
    return {(u, v): _forman_curvature(graph, u, v) for u, v in graph.edges()}


# ---------------------------------------------------------------------------
# Edge addition candidate
# ---------------------------------------------------------------------------


def _candidate_edges_to_add(
    graph: nx.Graph,
    u: int,
    v: int,
) -> List[Tuple[int, int]]:
    """Return edges between 2-hop neighbours of u and v that don't yet exist.

    These are the candidates that increase curvature of (u, v).
    """
    nbrs_u = set(graph.neighbors(u)) - {v}
    nbrs_v = set(graph.neighbors(v)) - {u}
    candidates = []
    for a in nbrs_u:
        for b in nbrs_v:
            if a != b and not graph.has_edge(a, b):
                candidates.append((a, b))
    return candidates


# ---------------------------------------------------------------------------
# Main SDRF algorithm
# ---------------------------------------------------------------------------


def sdrf(
    graph: nx.Graph,
    num_iterations: int = 50,
    tau: float = 0.0,
    max_edges_to_add: Optional[int] = None,
    remove_edges: bool = False,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, dict]:
    """Apply Stochastic Discrete Ricci Flow to rewire a graph.

    Each iteration:

    1. Compute Forman-Ricci curvature κ_F for all edges.
    2. Pick the most negatively curved edge (u, v).
    3. Stochastically add one edge between 2-hop neighbours of u and v,
       weighted by the curvature improvement it would provide.
    4. Optionally remove the original edge if ``remove_edges=True`` and
       κ_F(u, v) < τ.

    Parameters
    ----------
    graph:
        Input undirected graph (will *not* be modified in place).
    num_iterations:
        Number of rewiring steps.
    tau:
        Curvature threshold; only edges with κ_F < τ are rewired.
    max_edges_to_add:
        Hard cap on total edges added.  ``None`` means no cap.
    remove_edges:
        If *True*, remove the target edge after adding the new one.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    rewired : nx.Graph
        Rewired graph.
    stats : dict
        ``{'edges_added': int, 'edges_removed': int,
           'iterations_run': int,
           'curvature_history': list[float]}``
    """
    rng = np.random.default_rng(seed)
    g = copy.deepcopy(graph)

    edges_added = 0
    edges_removed = 0
    curvature_history: List[float] = []

    for iteration in range(num_iterations):
        kappas = _all_curvatures(g)
        if not kappas:
            break

        mean_kappa = float(np.mean(list(kappas.values())))
        curvature_history.append(mean_kappa)

        # Find most negatively curved edge
        min_edge, min_kappa = min(kappas.items(), key=lambda x: x[1])

        if min_kappa >= tau:
            break  # No more edges below threshold

        u, v = min_edge
        candidates = _candidate_edges_to_add(g, u, v)

        if candidates:
            # Compute curvature improvement for each candidate
            # Prefer candidates connecting high-degree nodes (heuristic)
            improvements = np.array(
                [g.degree(a) + g.degree(b) for a, b in candidates], dtype=float
            )
            if improvements.sum() > 0:
                probs = improvements / improvements.sum()
            else:
                probs = np.ones(len(candidates)) / len(candidates)

            idx = rng.choice(len(candidates), p=probs)
            a, b = candidates[idx]
            g.add_edge(a, b)
            edges_added += 1

            if max_edges_to_add is not None and edges_added >= max_edges_to_add:
                break

        if remove_edges and g.has_edge(u, v):
            g.remove_edge(u, v)
            edges_removed += 1

    return g, {
        "edges_added": edges_added,
        "edges_removed": edges_removed,
        "iterations_run": len(curvature_history),
        "curvature_history": curvature_history,
    }
