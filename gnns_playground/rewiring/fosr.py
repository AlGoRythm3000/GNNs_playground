"""First-Order Spectral Rewiring (FOSR).

FOSR greedily adds edges that maximise the increase in the spectral gap
(Fiedler value λ₂) of the graph Laplacian using a first-order approximation.

Reference
---------
Karhadkar et al. (2023) "FoSR: First-Order Spectral Rewiring for Addressing
Over-Squashing in GNNs."  ICLR 2023.  https://arxiv.org/abs/2210.11790
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp

from gnns_playground.utils.graph_utils import graph_laplacian, nx_to_adj


# ---------------------------------------------------------------------------
# Spectral gap helpers
# ---------------------------------------------------------------------------


def _fiedler_vector(L: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return (λ₂, v₂) – Fiedler value and vector – of the Laplacian matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    return float(eigenvalues[idx[1]]), eigenvectors[:, idx[1]]


def _lambda2_gradient(fiedler_vec: np.ndarray, u: int, v: int) -> float:
    """First-order change in λ₂ when adding edge (u, v).

    By the matrix perturbation formula:

        Δλ₂ ≈ (v₂[u] - v₂[v])²
    """
    return float((fiedler_vec[u] - fiedler_vec[v]) ** 2)


# ---------------------------------------------------------------------------
# Main FOSR algorithm
# ---------------------------------------------------------------------------


def fosr(
    graph: nx.Graph,
    num_edges_to_add: int = 10,
    batch_size: int = 1,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, dict]:
    """Apply First-Order Spectral Rewiring to a graph.

    Each step computes the Fiedler vector of the current Laplacian, then adds
    the non-existing edge (u, v) that maximises (v₂[u] - v₂[v])².

    Parameters
    ----------
    graph:
        Input undirected graph (will *not* be modified in place).
    num_edges_to_add:
        Total number of edges to add.
    batch_size:
        Number of edges added before recomputing the Fiedler vector.
        ``1`` (default) is most accurate; larger values are faster.
    seed:
        Unused (kept for API consistency with SDRF).

    Returns
    -------
    rewired : nx.Graph
        Rewired graph.
    stats : dict
        ``{'edges_added': int, 'spectral_gap_history': list[float]}``
    """
    g = copy.deepcopy(graph)
    nodes = sorted(g.nodes())
    n = len(nodes)
    node_index = {v: i for i, v in enumerate(nodes)}

    edges_added = 0
    spectral_gap_history: list[float] = []

    while edges_added < num_edges_to_add:
        adj = nx_to_adj(g)
        L = graph_laplacian(adj, normalised=False).toarray()
        lambda2, fiedler_vec = _fiedler_vector(L)
        spectral_gap_history.append(lambda2)

        # Evaluate all non-existing edges in the current batch
        best_score = -1.0
        best_edges: list[tuple] = []

        for u in nodes:
            for v in nodes:
                if u >= v:
                    continue
                if g.has_edge(u, v):
                    continue
                i, j = node_index[u], node_index[v]
                score = _lambda2_gradient(fiedler_vec, i, j)
                if score > best_score:
                    best_score = score
                    best_edges = [(u, v)]
                elif score == best_score:
                    best_edges.append((u, v))

        if not best_edges:
            break  # Graph is complete

        # Add up to batch_size edges from the best candidates
        for u, v in best_edges[:batch_size]:
            if edges_added >= num_edges_to_add:
                break
            g.add_edge(u, v)
            edges_added += 1

    return g, {
        "edges_added": edges_added,
        "spectral_gap_history": spectral_gap_history,
    }


# ---------------------------------------------------------------------------
# Random rewiring baseline
# ---------------------------------------------------------------------------


def random_edge_addition(
    graph: nx.Graph,
    num_edges_to_add: int = 10,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, dict]:
    """Add random non-existing edges as a baseline rewiring strategy.

    Parameters
    ----------
    graph:
        Input undirected graph.
    num_edges_to_add:
        Number of edges to add.
    seed:
        Random seed.

    Returns
    -------
    rewired : nx.Graph
    stats : dict  with key ``'edges_added'``
    """
    rng = np.random.default_rng(seed)
    g = copy.deepcopy(graph)
    nodes = list(g.nodes())
    n = len(nodes)

    edges_added = 0
    attempts = 0
    max_attempts = num_edges_to_add * 100

    while edges_added < num_edges_to_add and attempts < max_attempts:
        u, v = rng.choice(n, size=2, replace=False)
        u_node, v_node = nodes[u], nodes[v]
        if not g.has_edge(u_node, v_node):
            g.add_edge(u_node, v_node)
            edges_added += 1
        attempts += 1

    return g, {"edges_added": edges_added}
