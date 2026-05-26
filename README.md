# Spliceosome-A Kinetic Modelling Framework

> *A computational framework for modelling drug-induced perturbations of spliceosome A complex assembly — built on stochastic kinetics, graph theory, and biologically grounded parameter sampling.*

---

## What this project is about

Pre-mRNA splicing is one of the most tightly regulated steps in gene expression. The spliceosome A complex — the first stable RNA–protein assembly on the pre-mRNA — is a critical decision point: once it forms, the splice site is committed. Disrupting this step is therefore a high-value pharmacological target, particularly in diseases driven by aberrant splicing (cancer, neurodegeneration, rare genetic disorders).

This framework models the **kinetics of A complex formation** under the influence of small-molecule drugs. The core question it answers is:

> *Given a drug that acts on binding specificity — reducing correct-site association, accelerating dissociation, or stabilising decoy sites — how does the steady-state probability of each molecular state change?*

The model is designed to be **mechanistically grounded** (rates follow real kinetic constraints, including proofreading-like selectivity), **mathematically exact** (steady-state probabilities via the Matrix-Tree Theorem, no numerical ODE integration), and **modular** (new drug mechanisms and topologies can be added in minutes).

---

## Why it matters for drug discovery

Most splicing-targeted drugs discovered to date were found empirically. This framework offers something different: a **first-principles kinetic lens** through which to reason about drug mechanism before committing to expensive assays.

Concretely, it lets you:

- **Screen drug mechanisms in silico** — compare 10+ mechanistic hypotheses (e.g. "does blocking association work better than destabilising the closed complex?") across thousands of random parameter sets in seconds
- **Identify robust mechanisms** — filter for models where the desired state shift holds regardless of the specific kinetic constants, not just at a single tuned parameter point
- **Run sensitivity analyses** — understand which kinetic parameters the outcome depends on most, informing which molecular feature to optimise in a lead compound
- **Work with uncertainty honestly** — parameter sets are sampled from biologically plausible ranges, so results reflect population-level behaviour rather than cherry-picked values

---

## How the model works

### Graph topology

The molecular system is represented as a **directed weighted graph** where nodes are molecular states and edges are kinetic transitions.

**3-node topology** (binding competition):

```
Node 1 (free RNA)  ⇌  Node 2 (correct site, open)    k1 / k_1
Node 1 (free RNA)  ⇌  Node 3 (decoy site)             k2 / k_2
```

**4-node topology** (with conformational closing step):

```
Node 1 (free)     ⇌  Node 2 (correct, open)    k1  / k_1
Node 1 (free)     ⇌  Node 3 (decoy)             k2  / k_2
Node 2 (open)     ⇌  Node 4 (correct, closed)   k3  / k4
```

### Steady-state computation

Steady-state probabilities are computed **analytically** using the **Matrix-Tree Theorem** (Kirchhoff's theorem on the reversed weighted DiGraph, via `networkx`). This avoids numerical integration entirely and gives exact solutions for any set of rate constants.

### Drug modulation

Each drug model modifies one or more kinetic rates as a function of drug concentration. Four primitive functions are available:

| Function | Direction | Applied to |
|---|---|---|
| `hill_decay` | ↓ decreasing | `k1`, `k_2` |
| `linear_decrease` | ↓ decreasing | `k1`, `k_2` |
| `linear_increase` | ↑ increasing | `k_1`, `k2` |
| `hill_activation` | ↑ increasing | `k_1`, `k2` |

The constraint that drugs act on **binding specificity** (not catalytic activity) is enforced: association to the correct site can only decrease, decoy association can only increase, and so on.

### Parameter sampling

Kinetic parameters are sampled from biologically motivated log-uniform distributions ($10^{-3}$ to $10^3$), with optional constraints enforcing:

- $k_1 \approx k_2$ (similar on-rates for correct and decoy sites)
- $k_{-2} \gg k_{-1}$ (longer dwell time at the correct site — proofreading-like selectivity)
- $k_{-1} \approx k_4$ (symmetry in the 4-node topology)

---

## Repository structure

```
.
├── simulation_core.py       # Simulation class, steady-state engine, run methods
├── model_database.py        # Catalogue of drug–mechanism models
├── parameter_function.py    # Parameter sampling and trend filtering
├── visualization.py         # Plotting: single model, grid, perturbation
├── sensitivity_analysis.py  # SimulationSensitivity class + analysis tools
└── README.md
```

---

## Quickstart

```python
import numpy as np
from sensitivity_analysis import SimulationSensitivity, plot_perturbation_lines, sensitivity_summary

# 1. Configure the simulation
sim = SimulationSensitivity(
    drug_conc   = np.linspace(0, 10, 30).tolist(),
    epochs      = 50,
    drug_params = {"K_hill": 2.0, "Vmax": 5.0},
    k_2_COST    = 100,      # enforce k_2 >= 100 * k_1
    nodes_num   = 3,
    assumption  = True,
)

# 2. Run a perturbation scan over alpha (drug potency coefficient)
results = sim.run_perturbation(
    param_name   = "alpha",
    param_values = np.logspace(-1, 2, 20).tolist(),
)

# 3. Summarise and plot
print(sensitivity_summary(results, "alpha", nodes_num=3))
plot_perturbation_lines(results, "alpha", nodes_num=3)
```

---

## Dependencies

```
python >= 3.10
networkx
numpy
pandas
matplotlib
scipy
```

---

## Background

This project is part of a master's thesis in Quantitative and Computational Biosciences (University of Padua) made under the supervision of Dott. Rosa Martinez - Corral at the Theoretical Regulatory Biology Group - PRBB, Barcelona. The modelling approach draws on:

- **Kinetic proofreading theory** (Hopfield 1974; Ninio 1975) — the biological intuition behind selectivity constraints
- **Kirchhoff's Matrix-Tree Theorem** — for exact steady-state computation on arbitrary graph topologies
- **RNA splicing biochemistry** — spliceosome assembly kinetics and the role of the A complex as a commitment step

---

*Questions or collaborations: open an issue or reach out directly.*
