"""Tests for graph utility functions."""

import numpy as np
import networkx as nx
import pytest
import scipy.sparse as sp
import torch

from gnns_playground.utils.graph_utils import (
    adj_to_edge_index,
    adj_to_nx,
    barabasi_albert_graph,
    complete_graph,
    cycle_graph,
    edge_index_to_adj,
    graph_laplacian,
    normalized_adjacency,
    nx_to_adj,
    nx_to_torch,
    path_graph,
    random_graph,
)


class TestGraphGenerators:
    def test_path_graph(self):
        g = path_graph(5)
        assert g.number_of_nodes() == 5
        assert g.number_of_edges() == 4

    def test_cycle_graph(self):
        g = cycle_graph(6)
        assert g.number_of_nodes() == 6
        assert g.number_of_edges() == 6

    def test_random_graph(self):
        g = random_graph(10, p=0.5, seed=0)
        assert g.number_of_nodes() == 10

    def test_barabasi_albert(self):
        g = barabasi_albert_graph(20, m=2, seed=0)
        assert g.number_of_nodes() == 20
        assert g.number_of_edges() >= 2 * (20 - 2)

    def test_complete_graph(self):
        g = complete_graph(5)
        assert g.number_of_nodes() == 5
        assert g.number_of_edges() == 10  # C(5,2)


class TestAdjacencyHelpers:
    def setup_method(self):
        self.g = path_graph(4)  # 0-1-2-3
        self.adj = nx_to_adj(self.g)

    def test_normalized_adjacency_shape(self):
        adj_norm = normalized_adjacency(self.adj)
        assert adj_norm.shape == (4, 4)

    def test_normalized_adjacency_symmetry(self):
        adj_norm = normalized_adjacency(self.adj)
        diff = np.abs(adj_norm.toarray() - adj_norm.toarray().T).max()
        assert diff < 1e-10

    def test_graph_laplacian_combinatorial(self):
        L = graph_laplacian(self.adj, normalised=False)
        # Row sums of L should be zero
        row_sums = np.abs(np.array(L.sum(axis=1)).flatten())
        assert row_sums.max() < 1e-10

    def test_graph_laplacian_normalised(self):
        L = graph_laplacian(self.adj, normalised=True)
        # Diagonal of normalised Laplacian is 1 for non-isolated nodes
        diag = np.diag(L.toarray())
        assert np.all(diag >= 0)

    def test_nx_to_adj_roundtrip(self):
        adj2 = nx_to_adj(adj_to_nx(self.adj))
        diff = np.abs(self.adj.toarray() - adj2.toarray()).max()
        assert diff < 1e-10

    def test_adj_to_edge_index(self):
        ei = adj_to_edge_index(self.adj)
        assert ei.shape[0] == 2
        assert ei.dtype == torch.long

    def test_edge_index_to_adj_roundtrip(self):
        ei = adj_to_edge_index(self.adj)
        adj2 = edge_index_to_adj(ei, num_nodes=4)
        diff = np.abs(self.adj.toarray() - adj2.toarray()).max()
        assert diff < 1e-10

    def test_nx_to_torch_default_features(self):
        ei, x, adj = nx_to_torch(self.g)
        assert x.shape == (4, 4)  # identity features

    def test_nx_to_torch_custom_features(self):
        features = np.random.randn(4, 8)
        ei, x, adj = nx_to_torch(self.g, node_features=features)
        assert x.shape == (4, 8)
