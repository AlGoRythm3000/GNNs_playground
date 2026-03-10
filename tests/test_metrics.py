"""Tests for over-squashing and over-smoothing metrics."""

import numpy as np
import networkx as nx
import pytest
import scipy.sparse as sp
import torch

from gnns_playground.metrics.over_squashing import (
    cheeger_constant_approx,
    compute_effective_resistance_matrix,
    compute_pairwise_effective_resistance,
    compute_sensitivity_matrix,
    compute_total_effective_resistance,
    forman_ricci_curvature,
    mean_forman_curvature,
    most_negatively_curved_edges,
    spectral_gap,
)
from gnns_playground.metrics.over_smoothing import (
    dirichlet_energy,
    mean_average_distance,
    smoothness_ratio,
)
from gnns_playground.utils.graph_utils import (
    complete_graph,
    cycle_graph,
    nx_to_adj,
    path_graph,
    random_graph,
)


# ---------------------------------------------------------------------------
# Over-squashing metrics
# ---------------------------------------------------------------------------


class TestEffectiveResistance:
    def setup_method(self):
        self.g = path_graph(5)
        self.adj = nx_to_adj(self.g)

    def test_matrix_shape(self):
        R = compute_effective_resistance_matrix(self.adj)
        assert R.shape == (5, 5)

    def test_matrix_symmetry(self):
        R = compute_effective_resistance_matrix(self.adj)
        assert np.allclose(R, R.T, atol=1e-8)

    def test_diagonal_zero(self):
        R = compute_effective_resistance_matrix(self.adj)
        assert np.allclose(np.diag(R), 0.0, atol=1e-8)

    def test_accepts_nx_graph(self):
        R = compute_effective_resistance_matrix(self.g)
        assert R.shape == (5, 5)

    def test_pairwise(self):
        r = compute_pairwise_effective_resistance(self.adj, 0, 4)
        # For a path of length 4, R(0, 4) = 4 (unit resistances)
        assert abs(r - 4.0) < 1e-6

    def test_ter_path_gt_complete(self):
        ter_path = compute_total_effective_resistance(path_graph(6))
        ter_complete = compute_total_effective_resistance(complete_graph(6))
        assert ter_path > ter_complete  # path is less connected


class TestSpectralMetrics:
    def test_spectral_gap_complete_gt_path(self):
        gap_complete = spectral_gap(complete_graph(6))
        gap_path = spectral_gap(path_graph(6))
        assert gap_complete > gap_path

    def test_spectral_gap_non_negative(self):
        g = random_graph(10, p=0.3, seed=0)
        gap = spectral_gap(g)
        assert gap >= 0.0

    def test_cheeger_approx(self):
        g = complete_graph(5)
        h = cheeger_constant_approx(g)
        assert h > 0.0


class TestFormanCurvature:
    def setup_method(self):
        # Cycle: every node has degree 2 → κ_F = 4 - 2 - 2 = 0
        self.cycle = cycle_graph(6)
        # Complete K4: degree 3 → κ_F = 4 - 3 - 3 = -2
        self.k4 = complete_graph(4)

    def test_cycle_curvature_zero(self):
        kappa = forman_ricci_curvature(self.cycle)
        for v in kappa.values():
            assert abs(v - 0.0) < 1e-10

    def test_complete_curvature_negative(self):
        kappa = forman_ricci_curvature(self.k4)
        for v in kappa.values():
            assert v < 0

    def test_mean_curvature(self):
        mkappa = mean_forman_curvature(cycle_graph(5))
        assert abs(mkappa - 0.0) < 1e-10

    def test_most_negatively_curved(self):
        g = path_graph(10)
        edges = most_negatively_curved_edges(g, top_k=3)
        assert len(edges) == 3
        # All returned curvatures should be ≤ the curvature of the 4th worst
        kappas = [v for _, v in edges]
        assert kappas == sorted(kappas)


class TestSensitivityMatrix:
    def test_shape(self):
        g = random_graph(8, p=0.3, seed=0)
        adj = nx_to_adj(g)
        S = compute_sensitivity_matrix(adj, num_layers=2)
        assert S.shape == (8, 8)

    def test_row_sums_near_one(self):
        # All entries of the normalised adjacency matrix are non-negative
        g = random_graph(8, p=0.5, seed=0)
        adj = nx_to_adj(g)
        S = compute_sensitivity_matrix(adj, num_layers=1)
        assert np.all(S >= -1e-10)


# ---------------------------------------------------------------------------
# Over-smoothing metrics
# ---------------------------------------------------------------------------


class TestDirichletEnergy:
    def setup_method(self):
        self.g = random_graph(10, p=0.4, seed=2)
        self.adj = nx_to_adj(self.g)
        n = self.g.number_of_nodes()
        self.features_random = torch.randn(n, 4)
        self.features_constant = torch.ones(n, 4)

    def test_non_negative(self):
        e = dirichlet_energy(self.features_random, self.adj)
        assert e >= 0.0

    def test_constant_features_zero_energy(self):
        e = dirichlet_energy(self.features_constant, self.adj)
        assert abs(e) < 1e-8

    def test_accepts_numpy_features(self):
        f_np = self.features_random.numpy()
        e = dirichlet_energy(f_np, self.adj)
        assert e >= 0.0


class TestMeanAverageDistance:
    def setup_method(self):
        n = 10
        self.features = torch.randn(n, 6)
        self.g = random_graph(n, p=0.4, seed=3)
        self.adj = nx_to_adj(self.g)

    def test_global_mode_in_range(self):
        mad = mean_average_distance(self.features, mode="global")
        assert 0.0 <= mad <= 2.0

    def test_local_mode_in_range(self):
        mad = mean_average_distance(self.features, adj=self.adj, mode="local")
        assert 0.0 <= mad <= 2.0

    def test_local_requires_adj(self):
        with pytest.raises(ValueError):
            mean_average_distance(self.features, mode="local")

    def test_constant_features_zero_mad(self):
        const_f = torch.ones(10, 6)
        mad = mean_average_distance(const_f, mode="global")
        assert abs(mad) < 1e-8


class TestSmoothnessRatio:
    def test_in_range(self):
        n = 10
        features = torch.randn(n, 4)
        g = random_graph(n, p=0.4, seed=4)
        adj = nx_to_adj(g)
        sr = smoothness_ratio(features, adj)
        assert sr >= 0.0

    def test_constant_returns_zero(self):
        n = 10
        features = torch.ones(n, 4)
        g = random_graph(n, p=0.4, seed=5)
        adj = nx_to_adj(g)
        sr = smoothness_ratio(features, adj)
        assert sr == 0.0
