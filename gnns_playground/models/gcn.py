"""Graph Convolutional Network (Kipf & Welling, ICLR 2017).

Reference: https://arxiv.org/abs/1609.02907
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnns_playground.utils.graph_utils import normalized_adjacency


class GCNConv(nn.Module):
    """A single Graph Convolutional layer.

    Computes  H' = σ( Ã H W )  where  Ã = D^{-1/2}(A+I)D^{-1/2}.

    Parameters
    ----------
    in_features:
        Dimension of input node features.
    out_features:
        Dimension of output node features.
    bias:
        Whether to add a learnable bias term.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Node feature matrix of shape (N, in_features).
        adj_norm:
            Normalised (dense) adjacency matrix of shape (N, N).

        Returns
        -------
        torch.Tensor of shape (N, out_features)
        """
        support = x @ self.weight          # (N, out_features)
        out = adj_norm @ support            # (N, out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        in_f = self.weight.shape[0]
        out_f = self.weight.shape[1]
        return f"in_features={in_f}, out_features={out_f}, bias={self.bias is not None}"


class GCN(nn.Module):
    """Multi-layer Graph Convolutional Network.

    Parameters
    ----------
    in_features:
        Dimension of input node features.
    hidden_dims:
        List of hidden layer sizes.  Each entry adds one GCN layer.
    out_features:
        Dimension of output (e.g. number of classes for node classification).
    dropout:
        Dropout probability applied between layers.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int],
        out_features: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        dims = [in_features] + hidden_dims + [out_features]
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        adj: sp.spmatrix | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through all GCN layers.

        Parameters
        ----------
        x:
            Node feature matrix (N, in_features).
        adj:
            Raw (un-normalised) adjacency matrix.  May be a scipy sparse
            matrix, a NumPy dense array, or a torch tensor.  Self-loops are
            added and symmetric normalisation is applied automatically.

        Returns
        -------
        torch.Tensor of shape (N, out_features)
            Raw logits (no final activation).
        """
        adj_norm = self._prepare_adj(adj, x.device)

        for i, conv in enumerate(self.convs[:-1]):
            x = F.relu(conv(x, adj_norm))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj_norm)
        return x

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_adj(
        adj: sp.spmatrix | np.ndarray | torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalise and convert *adj* to a dense torch tensor."""
        if isinstance(adj, torch.Tensor):
            adj_np = adj.cpu().numpy()
            adj_sp = sp.csr_matrix(adj_np)
        elif isinstance(adj, np.ndarray):
            adj_sp = sp.csr_matrix(adj)
        else:
            adj_sp = adj

        adj_norm_sp = normalized_adjacency(adj_sp, add_self_loops=True)
        adj_norm_dense = torch.tensor(adj_norm_sp.toarray(), dtype=torch.float32)
        return adj_norm_dense.to(device)

    def get_embeddings(
        self,
        x: torch.Tensor,
        adj: sp.spmatrix | np.ndarray | torch.Tensor,
        layer: int = -1,
    ) -> torch.Tensor:
        """Return node embeddings from an intermediate layer.

        Parameters
        ----------
        x:
            Node features.
        adj:
            Adjacency matrix (any supported format).
        layer:
            Index of the layer whose *output* to return.  ``-1`` means the
            last hidden layer (before the final projection).

        Returns
        -------
        torch.Tensor
        """
        adj_norm = self._prepare_adj(adj, x.device)
        target = len(self.convs) + layer if layer < 0 else layer

        for i, conv in enumerate(self.convs):
            is_last = i == len(self.convs) - 1
            x = conv(x, adj_norm)
            if i == target:
                return x
            if not is_last:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
