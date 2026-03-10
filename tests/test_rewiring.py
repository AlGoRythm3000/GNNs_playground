"""Tests for graph rewiring algorithms (SDRF and FOSR)."""

import networkx as nx
import numpy as np
import pytest

from gnns_playground.rewiring.fosr import fosr, random_edge_addition
from gnns_playground.rewiring.sdrf import sdrf
from gnns_playground.metrics.over_squashing import spectral_gap
from gnns_playground.utils.graph_utils import cycle_graph, path_graph, random_graph


class TestSDRF:
    def setup_method(self):
        self.g = path_graph(10)

    def test_returns_graph_and_stats(self):
        g_new, stats = sdrf(self.g, num_iterations=5, seed=0)
        assert isinstance(g_new, nx.Graph)
        assert "edges_added" in stats
        assert "edges_removed" in stats
        assert "curvature_history" in stats

    def test_does_not_modify_original(self):
        original_edges = set(self.g.edges())
        sdrf(self.g, num_iterations=5, seed=0)
        assert set(self.g.edges()) == original_edges

    def test_nodes_preserved(self):
        g_new, _ = sdrf(self.g, num_iterations=10, seed=0)
        assert g_new.number_of_nodes() == self.g.number_of_nodes()

    def test_edges_added(self):
        g_new, stats = sdrf(self.g, num_iterations=20, max_edges_to_add=5, seed=0)
        assert stats["edges_added"] <= 5
        assert g_new.number_of_edges() >= self.g.number_of_edges()

    def test_remove_edges_option(self):
        g_new, stats = sdrf(
            self.g,
            num_iterations=5,
            tau=0.0,
            remove_edges=True,
            seed=0,
        )
        # Some edges might be removed
        assert stats["edges_removed"] >= 0

    def test_tau_threshold(self):
        # tau=100 means all edges qualify → should rewire aggressively
        g_pos, stats_pos = sdrf(self.g, num_iterations=5, tau=100.0, seed=0)
        # tau=-100 means no edge qualifies → no changes
        g_neg, stats_neg = sdrf(self.g, num_iterations=5, tau=-100.0, seed=0)
        assert stats_neg["edges_added"] == 0

    def test_curvature_history_length(self):
        _, stats = sdrf(self.g, num_iterations=10, seed=0)
        assert len(stats["curvature_history"]) <= 10

    def test_on_cycle(self):
        g = cycle_graph(8)
        g_new, stats = sdrf(g, num_iterations=10, seed=1)
        assert g_new.number_of_nodes() == g.number_of_nodes()

    def test_reproducible_with_seed(self):
        g1, s1 = sdrf(self.g, num_iterations=10, seed=42)
        g2, s2 = sdrf(self.g, num_iterations=10, seed=42)
        assert s1["edges_added"] == s2["edges_added"]


class TestFOSR:
    def setup_method(self):
        self.g = path_graph(8)

    def test_returns_graph_and_stats(self):
        g_new, stats = fosr(self.g, num_edges_to_add=3)
        assert isinstance(g_new, nx.Graph)
        assert "edges_added" in stats
        assert "spectral_gap_history" in stats

    def test_does_not_modify_original(self):
        original_edges = set(self.g.edges())
        fosr(self.g, num_edges_to_add=3)
        assert set(self.g.edges()) == original_edges

    def test_edges_added(self):
        k = 4
        g_new, stats = fosr(self.g, num_edges_to_add=k)
        assert stats["edges_added"] == k
        assert g_new.number_of_edges() == self.g.number_of_edges() + k

    def test_spectral_gap_non_decreasing(self):
        # FOSR should not decrease the spectral gap
        gap_before = spectral_gap(self.g)
        g_new, _ = fosr(self.g, num_edges_to_add=3)
        gap_after = spectral_gap(g_new)
        assert gap_after >= gap_before - 1e-8


class TestRandomEdgeAddition:
    def setup_method(self):
        self.g = path_graph(10)

    def test_returns_graph_and_stats(self):
        g_new, stats = random_edge_addition(self.g, num_edges_to_add=3, seed=0)
        assert isinstance(g_new, nx.Graph)
        assert "edges_added" in stats

    def test_does_not_modify_original(self):
        original_edges = set(self.g.edges())
        random_edge_addition(self.g, num_edges_to_add=3, seed=0)
        assert set(self.g.edges()) == original_edges

    def test_correct_number_of_edges_added(self):
        k = 5
        g_new, stats = random_edge_addition(self.g, num_edges_to_add=k, seed=7)
        assert stats["edges_added"] == k
        assert g_new.number_of_edges() == self.g.number_of_edges() + k
