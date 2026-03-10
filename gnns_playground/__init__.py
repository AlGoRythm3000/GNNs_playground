"""GNNs Playground – experiment toolkit for foundational GNN research.

Modules
-------
gnns_playground.models    – GCN and GAT implementations
gnns_playground.metrics   – over-squashing and over-smoothing metrics
gnns_playground.rewiring  – graph rewiring algorithms (SDRF, FOSR)
gnns_playground.utils     – graph utilities and converters
"""

__version__ = "0.1.0"

from gnns_playground import metrics, models, rewiring, utils

__all__ = ["metrics", "models", "rewiring", "utils", "__version__"]
