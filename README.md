# GNNs Playground

Experiment toolkit developed for my internship at **INRIA × CentraleSupélec** on
foundational models of Graph Neural Networks.  The library provides ready-to-use
tools for studying three core GNN challenges:

- **Over-squashing** – information bottlenecks caused by exponentially growing
  message-passing neighbourhoods
- **Over-smoothing** – convergence of node representations to indistinguishable
  values as network depth increases
- **Graph rewiring** – topology-modification strategies (SDRF, FOSR) that
  alleviate over-squashing by improving graph connectivity

---

## Project structure

```
GNNs_playground/
├── gnns_playground/          # Main Python package
│   ├── models/
│   │   ├── gcn.py            # Graph Convolutional Network (Kipf & Welling 2017)
│   │   └── gat.py            # Graph Attention Network (Veličković et al. 2018)
│   ├── metrics/
│   │   ├── over_squashing.py # Effective resistance, spectral gap, Forman curvature
│   │   └── over_smoothing.py # Dirichlet energy, MAD, smoothness ratio
│   ├── rewiring/
│   │   ├── sdrf.py           # Stochastic Discrete Ricci Flow (Topping et al. 2022)
│   │   └── fosr.py           # First-Order Spectral Rewiring (Karhadkar et al. 2023)
│   └── utils/
│       └── graph_utils.py    # Graph generators, adjacency helpers, conversions
├── experiments/
│   ├── over_squashing_exp.py # Compare over-squashing metrics across graph types
│   ├── over_smoothing_exp.py # Visualise Dirichlet energy decay vs. GCN depth
│   └── rewiring_exp.py       # Compare SDRF / FOSR / random rewiring strategies
└── tests/                    # pytest unit tests (66 tests)
```

---

## Setup

```bash
# Clone and install dependencies
git clone <repo-url>
cd GNNs_playground
pip install -r requirements.txt

# (Optional) install as editable package
pip install -e .
```

**Dependencies:** `torch`, `scipy`, `networkx`, `numpy`, `matplotlib`, `pytest`

---

## Quick start

### Over-squashing metrics

```python
from gnns_playground.utils.graph_utils import path_graph, complete_graph
from gnns_playground.metrics.over_squashing import (
    compute_total_effective_resistance,
    spectral_gap,
    forman_ricci_curvature,
)

g_path = path_graph(10)
g_complete = complete_graph(10)

# Total Effective Resistance (lower = less over-squashing)
print(compute_total_effective_resistance(g_path))      # large
print(compute_total_effective_resistance(g_complete))  # small

# Spectral gap λ₂ (higher = better connectivity)
print(spectral_gap(g_path))      # small
print(spectral_gap(g_complete))  # large

# Forman-Ricci curvature per edge (more negative = more bottleneck)
kappa = forman_ricci_curvature(g_path)
```

### Over-smoothing metrics

```python
import torch
from gnns_playground.metrics.over_smoothing import dirichlet_energy, mean_average_distance
from gnns_playground.utils.graph_utils import random_graph, nx_to_adj

g = random_graph(50, p=0.15, seed=0)
adj = nx_to_adj(g)
features = torch.randn(50, 16)

# Dirichlet energy (lower → more over-smoothed)
print(dirichlet_energy(features, adj))

# Mean Average Distance (lower → more over-smoothed)
print(mean_average_distance(features, adj=adj, mode="local"))
```

### Graph rewiring

```python
from gnns_playground.rewiring.sdrf import sdrf
from gnns_playground.rewiring.fosr import fosr
from gnns_playground.utils.graph_utils import path_graph

g_path = path_graph(20)

# SDRF – curvature-based rewiring
g_rewired, stats = sdrf(g_path, num_iterations=50, max_edges_to_add=5)
print(stats)  # {'edges_added': 5, 'curvature_history': [...], ...}

# FOSR – spectral gap maximisation
g_rewired, stats = fosr(g_path, num_edges_to_add=5)
print(stats)  # {'edges_added': 5, 'spectral_gap_history': [...]}
```

### GNN models

```python
import torch
from gnns_playground.models.gcn import GCN
from gnns_playground.models.gat import GAT

features = torch.randn(50, 16)

# GCN: 2-layer with hidden dim 64
gcn = GCN(in_features=16, hidden_dims=[64, 64], out_features=7, dropout=0.5)
logits = gcn(features, adj)  # (N, 7)

# GAT: 2-layer with 4 attention heads
gat = GAT(in_features=16, hidden_dim=8, out_features=7, num_heads=4)
logits = gat(features, adj)  # (N, 7)
```

---

## Experiments

```bash
# Compare over-squashing metrics across graph topologies
python experiments/over_squashing_exp.py

# Study over-smoothing as GCN depth increases
python experiments/over_smoothing_exp.py

# Benchmark rewiring algorithms (SDRF vs FOSR vs random)
python experiments/rewiring_exp.py
```

Each experiment prints a summary table and saves plots to the `experiments/` directory.

---

## Running tests

```bash
pytest tests/ -v
# 66 tests – models, metrics, rewiring, utilities
```

---

## Key references

| Topic | Paper |
|-------|-------|
| GCN | Kipf & Welling (2017) *Semi-Supervised Classification with GCNs* |
| GAT | Veličković et al. (2018) *Graph Attention Networks* |
| Over-squashing | Alon & Yahav (2021) *On the Bottleneck of GNNs* |
| Curvature & over-squashing | Topping et al. (2022) *Understanding over-squashing via curvature* |
| SDRF | Topping et al. (2022) |
| FOSR | Karhadkar et al. (2023) *FoSR: First-Order Spectral Rewiring* |
| Over-smoothing | Rusch et al. (2023) *A Survey on Oversmoothing in GNNs* |
