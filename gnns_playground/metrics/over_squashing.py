"""Over-squashing metrics for Graph Neural Networks.

Over-squashing arises when information from an exponentially growing
neighbourhood must be compressed into a fixed-size vector.  The metrics here
quantify how well the graph structure supports long-range information flow.

Key references
--------------
- Alon & Yahav (2021) "On the Bottleneck of Graph Neural Networks and its
  Practical Implications."  https://arxiv.org/abs/2006.05205
- Topping et al. (2022) "Understanding over-squashing and bottlenecks on
  graphs via curvature."  https://arxiv.org/abs/2111.14522
- Di Giovanni et al. (2023) "On Over-Squashing in Message Passing Neural
  Networks: The Impact of Width, Depth, and Topology."
  https://arxiv.org/abs/2302.02941
"""

from __future__ import annotations

from typing import Optional, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from gnns_playground.utils.graph_utils import graph_laplacian, nx_to_adj


# ---------------------------------------------------------------------------
# Effective resistance
# ---------------------------------------------------------------------------


def compute_effective_resistance_matrix(
    adj: sp.spmatrix | nx.Graph,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute the effective resistance matrix R where R_{st} = (e_s - e_t)^T L^+ (e_s - e_t).

    Equivalently, if L = U Λ U^T is the eigendecomposition and Λ^+ contains
    the pseudo-inverses of non-zero eigenvalues, then

        R = diag(L^+) 1^T  +  1 diag(L^+)^T  -  2 L^+

    Parameters
    ----------
    adj:
        Adjacency matrix (scipy sparse) or NetworkX graph.
    eps:
        Threshold below which eigenvalues are treated as zero.

    Returns
    -------
    np.ndarray of shape (N, N)
        Symmetric effective-resistance matrix.
    """
    if isinstance(adj, nx.Graph):
        adj = nx_to_adj(adj)
    L = graph_laplacian(adj, normalised=False)
    L_dense = L.toarray()
    # Moore-Penrose pseudoinverse of the Laplacian
    L_pinv = np.linalg.pinv(L_dense)
    diag = np.diag(L_pinv)
    # R[i,j] = L+[i,i] + L+[j,j] - 2 L+[i,j]
    R = diag[:, None] + diag[None, :] - 2 * L_pinv
    return np.maximum(R, 0.0)


def compute_total_effective_resistance(
    adj: sp.spmatrix | nx.Graph,
) -> float:
    """Compute the Total Effective Resistance (TER) of the graph.

    TER = (1/2) Σ_{i≠j} R_{ij}  =  N Σ_{k≥2} λ_k^{-1}

    A lower TER indicates better connectivity (less over-squashing).

    Parameters
    ----------
    adj:
        Adjacency matrix or NetworkX graph.

    Returns
    -------
    float
        Total effective resistance.
    """
    if isinstance(adj, nx.Graph):
        adj = nx_to_adj(adj)
    L = graph_laplacian(adj, normalised=False)
    eigenvalues = np.linalg.eigvalsh(L.toarray())
    # Sum inverses of non-trivial eigenvalues
    n = adj.shape[0]
    nontrivial = eigenvalues[eigenvalues > 1e-10]
    if len(nontrivial) == 0:
        return float("inf")
    return float(n * np.sum(1.0 / nontrivial))


def compute_pairwise_effective_resistance(
    adj: sp.spmatrix | nx.Graph,
    source: int,
    target: int,
) -> float:
    """Compute the effective resistance between two specific nodes.

    Parameters
    ----------
    adj:
        Adjacency matrix or NetworkX graph.
    source, target:
        Node indices.

    Returns
    -------
    float
    """
    R = compute_effective_resistance_matrix(adj)
    return float(R[source, target])


# ---------------------------------------------------------------------------
# Sensitivity / Jacobian-based over-squashing
# ---------------------------------------------------------------------------


def compute_sensitivity_matrix(
    adj: sp.spmatrix | nx.Graph,
    num_layers: int,
) -> np.ndarray:
    """Approximate the Jacobian sensitivity of GCN node embeddings.

    For a depth-L GCN without non-linearities:

        (∂ h_v^L / ∂ x_u) ≈ Ã^L[v, u]

    where Ã is the symmetrically normalised adjacency (with self-loops).
    A large (L, v, u) entry means node u can strongly influence node v's
    representation after L message-passing steps.

    Parameters
    ----------
    adj:
        Adjacency matrix or NetworkX graph.
    num_layers:
        Number of message-passing layers L.

    Returns
    -------
    np.ndarray of shape (N, N)
        Row v, column u = sensitivity of v to u after *num_layers* steps.
    """
    from gnns_playground.utils.graph_utils import normalized_adjacency

    if isinstance(adj, nx.Graph):
        adj = nx_to_adj(adj)
    adj_norm = normalized_adjacency(adj, add_self_loops=True)
    result = adj_norm.toarray()
    for _ in range(num_layers - 1):
        result = result @ adj_norm.toarray()
    return result


# ---------------------------------------------------------------------------
# Spectral measures
# ---------------------------------------------------------------------------


def spectral_gap(adj: sp.spmatrix | nx.Graph) -> float:
    """Return the spectral gap (algebraic connectivity, Fiedler value) λ_2.

    A larger spectral gap indicates better graph connectivity and less
    susceptibility to over-squashing.

    Parameters
    ----------
    adj:
        Adjacency matrix or NetworkX graph.

    Returns
    -------
    float
    """
    if isinstance(adj, nx.Graph):
        adj = nx_to_adj(adj)
    L = graph_laplacian(adj, normalised=True)
    eigenvalues = np.linalg.eigvalsh(L.toarray())
    eigenvalues_sorted = np.sort(eigenvalues)
    return float(max(0.0, eigenvalues_sorted[1])) if len(eigenvalues_sorted) > 1 else 0.0


def cheeger_constant_approx(adj: sp.spmatrix | nx.Graph) -> float:
    """Return a spectral lower bound on the Cheeger constant h(G).

    By the Cheeger inequality:  λ_2 / 2  ≤  h(G)  ≤  √(2 λ_2)

    This function returns λ_2 / 2 as a lower bound.

    Parameters
    ----------
    adj:
        Adjacency matrix or NetworkX graph.

    Returns
    -------
    float
    """
    return spectral_gap(adj) / 2.0


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------


def forman_ricci_curvature(
    graph: nx.Graph,
) -> dict[tuple[int, int], float]:
    """Compute the Forman-Ricci curvature for every edge.

    For an unweighted graph:

        κ_F(u, v) = 4 - deg(u) - deg(v)

    This simplified version treats all edge weights as 1.

    Parameters
    ----------
    graph:
        Undirected NetworkX graph.

    Returns
    -------
    dict
        Mapping (u, v) -> κ_F(u, v).
    """
    curvature = {}
    for u, v in graph.edges():
        ku = graph.degree(u)
        kv = graph.degree(v)
        curvature[(u, v)] = float(4 - ku - kv)
    return curvature


def mean_forman_curvature(graph: nx.Graph) -> float:
    """Average Forman-Ricci curvature across all edges."""
    kappa = forman_ricci_curvature(graph)
    if not kappa:
        return 0.0
    return float(np.mean(list(kappa.values())))


def most_negatively_curved_edges(
    graph: nx.Graph,
    top_k: int = 10,
) -> list[tuple[tuple[int, int], float]]:
    """Return the *top_k* most negatively curved edges.

    Parameters
    ----------
    graph:
        Undirected NetworkX graph.
    top_k:
        Number of edges to return.

    Returns
    -------
    list of ((u, v), curvature) sorted by ascending curvature.
    """
    kappa = forman_ricci_curvature(graph)
    sorted_edges = sorted(kappa.items(), key=lambda x: x[1])
    return sorted_edges[:top_k]
