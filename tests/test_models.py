"""Tests for GNN model implementations (GCN and GAT)."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from gnns_playground.models.gcn import GCN, GCNConv
from gnns_playground.models.gat import GAT, GATConv
from gnns_playground.utils.graph_utils import nx_to_adj, random_graph


def _make_simple_graph(n: int = 10):
    """Return adj and feature matrix for a simple random graph."""
    g = random_graph(n, p=0.4, seed=1)
    adj = nx_to_adj(g)
    x = torch.randn(n, 8)
    return adj, x


class TestGCNConv:
    def test_output_shape(self):
        n, fin, fout = 10, 8, 4
        adj, x = _make_simple_graph(n)
        conv = GCNConv(fin, fout)
        adj_norm = torch.tensor(
            sp.csr_matrix(adj).toarray(), dtype=torch.float32
        )
        out = conv(x, adj_norm)
        assert out.shape == (n, fout)

    def test_no_bias(self):
        conv = GCNConv(4, 4, bias=False)
        assert conv.bias is None


class TestGCN:
    def setup_method(self):
        self.n = 15
        self.adj, self.x = _make_simple_graph(self.n)

    def test_output_shape(self):
        model = GCN(in_features=8, hidden_dims=[16, 16], out_features=3, dropout=0.0)
        model.eval()
        with torch.no_grad():
            out = model(self.x, self.adj)
        assert out.shape == (self.n, 3)

    def test_single_layer(self):
        model = GCN(in_features=8, hidden_dims=[], out_features=4, dropout=0.0)
        model.eval()
        with torch.no_grad():
            out = model(self.x, self.adj)
        assert out.shape == (self.n, 4)

    def test_accepts_dense_numpy_adj(self):
        model = GCN(in_features=8, hidden_dims=[8], out_features=2, dropout=0.0)
        model.eval()
        adj_dense = self.adj.toarray()
        with torch.no_grad():
            out = model(self.x, adj_dense)
        assert out.shape == (self.n, 2)

    def test_accepts_torch_tensor_adj(self):
        model = GCN(in_features=8, hidden_dims=[8], out_features=2, dropout=0.0)
        model.eval()
        adj_t = torch.tensor(self.adj.toarray(), dtype=torch.float32)
        with torch.no_grad():
            out = model(self.x, adj_t)
        assert out.shape == (self.n, 2)

    def test_get_embeddings(self):
        model = GCN(in_features=8, hidden_dims=[16], out_features=4, dropout=0.0)
        model.eval()
        with torch.no_grad():
            emb = model.get_embeddings(self.x, self.adj, layer=0)
        assert emb.shape == (self.n, 16)

    def test_training_mode_runs(self):
        model = GCN(in_features=8, hidden_dims=[16], out_features=3, dropout=0.5)
        out = model(self.x, self.adj)
        assert out.shape == (self.n, 3)


class TestGATConv:
    def test_output_shape_concat(self):
        n, fin, fout, heads = 10, 8, 4, 2
        adj, x = _make_simple_graph(n)
        conv = GATConv(fin, fout, num_heads=heads, concat=True, dropout=0.0)
        out = conv(x, adj)
        assert out.shape == (n, fout * heads)

    def test_output_shape_average(self):
        n, fin, fout, heads = 10, 8, 4, 2
        adj, x = _make_simple_graph(n)
        conv = GATConv(fin, fout, num_heads=heads, concat=False, dropout=0.0)
        out = conv(x, adj)
        assert out.shape == (n, fout)


class TestGAT:
    def setup_method(self):
        self.n = 12
        self.adj, self.x = _make_simple_graph(self.n)

    def test_two_layer(self):
        model = GAT(
            in_features=8,
            hidden_dim=4,
            out_features=3,
            num_layers=2,
            num_heads=2,
            dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            out = model(self.x, self.adj)
        assert out.shape == (self.n, 3)

    def test_single_layer(self):
        model = GAT(
            in_features=8,
            hidden_dim=4,
            out_features=3,
            num_layers=1,
            dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            out = model(self.x, self.adj)
        assert out.shape == (self.n, 3)
