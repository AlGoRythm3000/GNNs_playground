"""Over-smoothing metrics for Graph Neural Networks.

Over-smoothing occurs when repeated message-passing causes node
representations to converge to indistinguishable values.  The metrics here
quantify how much discriminative information is preserved after L layers.

Key references
--------------
- Li et al. (2018) "Deeper Insights into Graph Convolutional Networks for
  Semi-Supervised Classification."  https://arxiv.org/abs/1801.07606
- Cai & Wang (2020) "A Note on Over-Smoothing for Graph Neural Networks."
  https://arxiv.org/abs/2006.13318
- Rusch et al. (2023) "A Survey on Oversmoothing in Graph Neural Networks."
  https://arxiv.org/abs/2303.10993
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch

from gnns_playground.utils.graph_utils import graph_laplacian


# ---------------------------------------------------------------------------
# Dirichlet energy
# ---------------------------------------------------------------------------


def dirichlet_energy(
    features: torch.Tensor | np.ndarray,
    adj: sp.spmatrix | np.ndarray,
    normalised: bool = False,
) -> float:
    """Compute the Dirichlet energy of node features on the graph.

    E(f) = Σ_{(i,j)∈E} ‖ f_i - f_j ‖²  =  tr( F^T L F )

    where F is the N×d feature matrix and L is the (normalised) Laplacian.

    A low Dirichlet energy indicates that adjacent nodes have similar
    features (over-smoothing).  As depth increases, E(f) → 0.

    Parameters
    ----------
    features:
        Node feature matrix (N, d).  Torch tensor or NumPy array.
    adj:
        Adjacency matrix (N, N), scipy sparse or NumPy dense.
    normalised:
        Whether to use the normalised Laplacian.

    Returns
    -------
    float
        Dirichlet energy.
    """
    if isinstance(features, torch.Tensor):
        F_np = features.detach().cpu().numpy()
    else:
        F_np = np.asarray(features, dtype=float)

    if isinstance(adj, np.ndarray):
        adj = sp.csr_matrix(adj)

    L = graph_laplacian(adj, normalised=normalised)
    energy = np.trace(F_np.T @ L.toarray() @ F_np)
    return float(energy)


# ---------------------------------------------------------------------------
# Mean Average Distance (MAD)
# ---------------------------------------------------------------------------


def mean_average_distance(
    features: torch.Tensor | np.ndarray,
    adj: sp.spmatrix | np.ndarray | None = None,
    mode: str = "global",
) -> float:
    """Compute the Mean Average Distance (MAD) of node features.

    MAD quantifies how similar, on average, adjacent (or all) node
    representations are.  Lower MAD → more over-smoothing.

    Parameters
    ----------
    features:
        Node feature matrix (N, d).
    adj:
        Adjacency matrix.  Required when ``mode='local'``.
    mode:
        ``'global'``: average cosine distance over *all* pairs.
        ``'local'``: average cosine distance over *neighbouring* pairs only.

    Returns
    -------
    float
        MAD value in [0, 1].
    """
    if isinstance(features, torch.Tensor):
        F_np = features.detach().cpu().numpy().astype(float)
    else:
        F_np = np.asarray(features, dtype=float)

    # Normalise rows
    norms = np.linalg.norm(F_np, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    F_norm = F_np / norms

    # Cosine similarity matrix
    sim = F_norm @ F_norm.T  # (N, N) in [-1, 1]
    dist = 1.0 - np.clip(sim, -1.0, 1.0)  # cosine distance in [0, 2]

    if mode == "global":
        n = F_np.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return float(dist[mask].mean())

    elif mode == "local":
        if adj is None:
            raise ValueError("adj must be provided for mode='local'")
        if isinstance(adj, np.ndarray):
            adj_dense = adj
        else:
            adj_dense = adj.toarray()
        mask = adj_dense.astype(bool)
        np.fill_diagonal(mask, False)
        if mask.sum() == 0:
            return 0.0
        return float(dist[mask].mean())

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'global' or 'local'.")


# ---------------------------------------------------------------------------
# Smoothness ratio
# ---------------------------------------------------------------------------


def smoothness_ratio(
    features: torch.Tensor | np.ndarray,
    adj: sp.spmatrix | np.ndarray,
) -> float:
    """Ratio of intra-neighbour distance to inter-non-neighbour distance.

    smoothness_ratio = MAD_local / MAD_global

    Values close to 1 mean that neighbours are as dissimilar as non-neighbours
    (no over-smoothing).  Values close to 0 indicate severe over-smoothing.

    Parameters
    ----------
    features:
        Node features (N, d).
    adj:
        Adjacency matrix (N, N).

    Returns
    -------
    float
    """
    mad_local = mean_average_distance(features, adj=adj, mode="local")
    mad_global = mean_average_distance(features, mode="global")
    if mad_global < 1e-10:
        return 0.0
    return float(mad_local / mad_global)


# ---------------------------------------------------------------------------
# Layer-wise energy decay
# ---------------------------------------------------------------------------


def energy_decay_profile(
    gcn_model: "GCN",  # noqa: F821
    x: torch.Tensor,
    adj: sp.spmatrix | np.ndarray,
) -> list[float]:
    """Compute the Dirichlet energy after each layer of a GCN.

    Useful for detecting at which depth over-smoothing sets in.

    Parameters
    ----------
    gcn_model:
        A ``GCN`` instance (from ``gnns_playground.models``).
    x:
        Initial node features (N, F).
    adj:
        Adjacency matrix.

    Returns
    -------
    list[float]
        Dirichlet energies [E_0, E_1, ..., E_L] where E_0 is the input energy.
    """
    import scipy.sparse as sp_ 

    if isinstance(adj, np.ndarray):
        adj_sp = sp_.csr_matrix(adj)
    else:
        adj_sp = adj

    energies = [dirichlet_energy(x, adj_sp)]

    gcn_model.eval()
    with torch.no_grad():
        for layer_idx in range(len(gcn_model.convs)):
            h = gcn_model.get_embeddings(x, adj, layer=layer_idx)
            energies.append(dirichlet_energy(h, adj_sp))

    return energies
