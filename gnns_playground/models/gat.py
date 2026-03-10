"""Graph Attention Network (Veličković et al., ICLR 2018).

Reference: https://arxiv.org/abs/1710.10903
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATConv(nn.Module):
    """A single multi-head Graph Attention layer.

    For each node i:

        e_{ij} = LeakyReLU( a^T [ W h_i ‖ W h_j ] )
        α_{ij} = softmax_j(e_{ij})
        h_i'   = σ( Σ_j α_{ij} W h_j )

    Parameters
    ----------
    in_features:
        Input feature dimension.
    out_features:
        Output feature dimension *per head*.
    num_heads:
        Number of independent attention heads.
    concat:
        If *True* concatenate head outputs (final dim = out_features*num_heads);
        otherwise average them (final dim = out_features).
    dropout:
        Dropout applied to attention coefficients.
    negative_slope:
        Negative slope for LeakyReLU.
    bias:
        Whether to add a learnable bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        concat: bool = True,
        dropout: float = 0.6,
        negative_slope: float = 0.2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout

        # Shared linear transformation W for all heads
        self.W = nn.Parameter(torch.empty(num_heads, in_features, out_features))
        # Attention parameters: a_src and a_dst (decomposed attention vector)
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features, 1))
        self.bias_param = nn.Parameter(torch.zeros(num_heads * out_features if concat else out_features)) if bias else None
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self,
        x: torch.Tensor,
        adj: sp.spmatrix | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Node features (N, in_features).
        adj:
            Adjacency matrix (any supported format).  The mask is used to
            restrict attention to existing edges.

        Returns
        -------
        torch.Tensor
            Shape (N, out_features * num_heads) if ``concat=True`` else
            (N, out_features).
        """
        n = x.shape[0]
        device = x.device
        mask = self._adj_to_mask(adj, n, device)  # (N, N) bool

        # Linear transform: (H, N, F_out) where H = num_heads
        # W: (H, F_in, F_out), x: (N, F_in) → Wh: (H, N, F_out)
        Wh = torch.einsum("hif,ni->hnf", self.W, x)  # (H, N, F_out)

        # Attention scores: decomposed attention vector
        # a_src/a_dst: (H, F_out, 1) → e_src/e_dst: (H, N)
        e_src = (Wh * self.a_src.permute(0, 2, 1)).sum(dim=-1)  # (H, N)
        e_dst = (Wh * self.a_dst.permute(0, 2, 1)).sum(dim=-1)  # (H, N)
        # e[h, i, j] = e_src[h, i] + e_dst[h, j]
        e = self.leaky_relu(e_src.unsqueeze(-1) + e_dst.unsqueeze(-2))  # (H, N, N)

        # Mask out non-edges
        e = e.masked_fill(~mask.unsqueeze(0), float("-inf"))

        alpha = F.softmax(e, dim=-1)  # (H, N, N)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Aggregate
        out = torch.einsum("hnm,hmf->hnf", alpha, Wh)  # (H, N, F_out)

        if self.concat:
            out = out.permute(1, 0, 2).reshape(n, -1)  # (N, H * F_out)
        else:
            out = out.mean(dim=0)  # (N, F_out)

        if self.bias_param is not None:
            out = out + self.bias_param
        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"heads={self.num_heads}, concat={self.concat}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adj_to_mask(
        adj: sp.spmatrix | np.ndarray | torch.Tensor,
        n: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a boolean adjacency mask of shape (N, N) including self-loops."""
        if isinstance(adj, torch.Tensor):
            dense = adj.to(device).bool()
        elif isinstance(adj, np.ndarray):
            dense = torch.tensor(adj, dtype=torch.bool, device=device)
        else:  # scipy sparse
            dense = torch.tensor(adj.toarray(), dtype=torch.bool, device=device)
        # Add self-loops to the mask
        dense = dense | torch.eye(n, dtype=torch.bool, device=device)
        return dense


class GAT(nn.Module):
    """Multi-layer Graph Attention Network.

    Parameters
    ----------
    in_features:
        Input feature dimension.
    hidden_dim:
        Feature dimension of each hidden attention head.
    out_features:
        Output dimension (e.g. number of classes).
    num_layers:
        Total number of GAT layers (>= 1).
    num_heads:
        Number of attention heads in *hidden* layers.
    dropout:
        Dropout applied both to features and attention coefficients.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        assert num_layers >= 1

        layers: List[nn.Module] = []
        if num_layers == 1:
            layers.append(
                GATConv(in_features, out_features, num_heads=1, concat=False, dropout=dropout)
            )
        else:
            # First layer: concat heads
            layers.append(
                GATConv(in_features, hidden_dim, num_heads=num_heads, concat=True, dropout=dropout)
            )
            # Intermediate layers
            for _ in range(num_layers - 2):
                layers.append(
                    GATConv(hidden_dim * num_heads, hidden_dim, num_heads=num_heads, concat=True, dropout=dropout)
                )
            # Final layer: average heads
            layers.append(
                GATConv(hidden_dim * num_heads, out_features, num_heads=1, concat=False, dropout=dropout)
            )

        self.layers = nn.ModuleList(layers)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        adj: sp.spmatrix | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Node features (N, in_features).
        adj:
            Adjacency matrix (any supported format).

        Returns
        -------
        torch.Tensor of shape (N, out_features)
            Raw logits.
        """
        for layer in self.layers[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(layer(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, adj)
        return x
