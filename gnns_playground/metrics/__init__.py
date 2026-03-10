"""Metrics sub-package – over-squashing and over-smoothing."""

from gnns_playground.metrics.over_smoothing import (
    dirichlet_energy,
    energy_decay_profile,
    mean_average_distance,
    smoothness_ratio,
)
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

__all__ = [
    # over-squashing
    "cheeger_constant_approx",
    "compute_effective_resistance_matrix",
    "compute_pairwise_effective_resistance",
    "compute_sensitivity_matrix",
    "compute_total_effective_resistance",
    "forman_ricci_curvature",
    "mean_forman_curvature",
    "most_negatively_curved_edges",
    "spectral_gap",
    # over-smoothing
    "dirichlet_energy",
    "energy_decay_profile",
    "mean_average_distance",
    "smoothness_ratio",
]
