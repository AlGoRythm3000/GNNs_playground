"""Graph utility functions: conversions, generators, and normalisation helpers."""

from __future__ import annotations

from typing import Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


# ---------------------------------------------------------------------------
# Adjacency / Laplacian helpers
# ---------------------------------------------------------------------------


def normalized_adjacency(adj: sp.spmatrix, add_self_loops: bool = True) -> sp.spmatrix:
    """Return the symmetrically normalised adjacency matrix Ã = D^{-1/2} A D^{-1/2}.

    Parameters
    ----------
    adj:
        Unweighted or weighted sparse adjacency matrix (N × N).
    add_self_loops:
        If *True*, add the identity before normalising (Kipf & Welling 2017).

    Returns
    -------
    scipy.sparse.csr_matrix
        Normalised adjacency Ã.
    """
    adj = sp.csr_matrix(adj, dtype=float)
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0])
    degree = np.array(adj.sum(axis=1)).flatten()
    degree_safe = np.where(degree > 0, degree, 1.0)  # replace 0 with 1 to avoid division
    degree_inv_sqrt = np.where(degree > 0, degree_safe ** -0.5, 0.0)
    D_inv_sqrt = sp.diags(degree_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


def graph_laplacian(adj: sp.spmatrix, normalised: bool = True) -> sp.spmatrix:
    """Compute the (normalised) graph Laplacian.

    Parameters
    ----------
    adj:
        Sparse adjacency matrix (N × N).
    normalised:
        If *True* return the normalised Laplacian L = I - D^{-1/2} A D^{-1/2};
        otherwise return the combinatorial Laplacian L = D - A.

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    adj = sp.csr_matrix(adj, dtype=float)
    n = adj.shape[0]
    degree = np.array(adj.sum(axis=1)).flatten()
    if normalised:
        degree_safe = np.where(degree > 0, degree, 1.0)
        degree_inv_sqrt = np.where(degree > 0, degree_safe ** -0.5, 0.0)
        D_inv_sqrt = sp.diags(degree_inv_sqrt)
        return sp.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt
    else:
        D = sp.diags(degree)
        return D - adj


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def nx_to_torch(
    graph: nx.Graph,
    node_features: np.ndarray | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, sp.csr_matrix]:
    """Convert a NetworkX graph to torch tensors.

    Parameters
    ----------
    graph:
        Input NetworkX graph.
    node_features:
        Optional array of shape (N, F) with per-node features.
        If *None*, the identity matrix is used (one-hot encoding).

    Returns
    -------
    edge_index : torch.LongTensor of shape (2, E)
        COO-format edge indices.
    x : torch.FloatTensor of shape (N, F)
        Node features.
    adj : scipy.sparse.csr_matrix of shape (N, N)
        Adjacency matrix.
    """
    nodes = sorted(graph.nodes())
    n = len(nodes)
    node_index = {v: i for i, v in enumerate(nodes)}

    rows, cols = [], []
    for u, v in graph.edges():
        i, j = node_index[u], node_index[v]
        rows += [i, j]
        cols += [j, i]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    adj = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n), dtype=float
    )

    if node_features is None:
        x = torch.eye(n)
    else:
        x = torch.tensor(node_features, dtype=torch.float32)

    return edge_index, x, adj


def adj_to_edge_index(adj: sp.spmatrix) -> torch.Tensor:
    """Convert a sparse adjacency matrix to a COO edge_index tensor.

    Parameters
    ----------
    adj:
        Sparse adjacency matrix (N × N).

    Returns
    -------
    torch.LongTensor of shape (2, E)
    """
    adj = sp.coo_matrix(adj)
    rows = torch.tensor(adj.row, dtype=torch.long)
    cols = torch.tensor(adj.col, dtype=torch.long)
    return torch.stack([rows, cols], dim=0)


def edge_index_to_adj(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    """Convert a COO edge_index tensor to a sparse adjacency matrix.

    Parameters
    ----------
    edge_index:
        LongTensor of shape (2, E).
    num_nodes:
        Total number of nodes N.

    Returns
    -------
    scipy.sparse.csr_matrix of shape (N, N)
    """
    rows = edge_index[0].numpy()
    cols = edge_index[1].numpy()
    data = np.ones(len(rows))
    return sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))


def adj_to_nx(adj: sp.spmatrix) -> nx.Graph:
    """Convert a sparse adjacency matrix to a NetworkX graph."""
    return nx.from_scipy_sparse_array(adj)


def nx_to_adj(graph: nx.Graph) -> sp.csr_matrix:
    """Convert a NetworkX graph to a sparse adjacency matrix."""
    return nx.to_scipy_sparse_array(graph, format="csr", dtype=float)


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------


def path_graph(n: int) -> nx.Graph:
    """Return a path graph P_n with *n* nodes."""
    return nx.path_graph(n)


def grid_graph(rows: int, cols: int) -> nx.Graph:
    """Return a 2-D grid graph with *rows* × *cols* nodes."""
    return nx.grid_2d_graph(rows, cols)


def random_graph(n: int, p: float, seed: int | None = None) -> nx.Graph:
    """Return an Erdős–Rényi random graph G(n, p)."""
    return nx.erdos_renyi_graph(n, p, seed=seed)


def barabasi_albert_graph(n: int, m: int, seed: int | None = None) -> nx.Graph:
    """Return a Barabási–Albert preferential-attachment graph."""
    return nx.barabasi_albert_graph(n, m, seed=seed)


def cycle_graph(n: int) -> nx.Graph:
    """Return a cycle graph C_n with *n* nodes."""
    return nx.cycle_graph(n)


def complete_graph(n: int) -> nx.Graph:
    """Return a complete graph K_n with *n* nodes."""
    return nx.complete_graph(n)
