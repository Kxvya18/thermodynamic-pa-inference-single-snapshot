# Thermodynamic Inference for Preferential-Attachment Networks

A research-oriented network-science project for studying preferential-attachment growth through statistical inference, reversibility diagnostics, and thermodynamic-style identities. The project treats the attachment exponent `beta` as an inverse-temperature-like parameter that controls preferential bias in network growth, then asks whether this latent growth parameter can be recovered from observed growth histories and final network structure.

The repository is packaged as a standalone research codebase. It contains the core preferential-attachment inference module, experiment runner, generated figures, finite-size tables, and lightweight validation tests needed to inspect the thermodynamic-inference workflow without relying on external local project folders.

## Research Motivation

Preferential-attachment models explain how heterogeneous degree distributions can emerge from cumulative advantage: high-degree nodes attract future links with greater probability. In practice, however, a network is often observed only after growth has already occurred, and the original attachment mechanism is partially hidden.

This project studies the inverse problem behind that setting:

> Given growth evidence or a final preferential-attachment network, what can be inferred about the attachment exponent that shaped the network?

The analysis frames `beta` as a control parameter analogous to inverse temperature. Lower and higher values of `beta` alter the concentration of attachment probability, the degree of hub dominance, the apparent arrow of network growth, and the statistical recoverability of the underlying dynamics.

## Technical Contributions

- Maximum-likelihood inference for the preferential-attachment exponent `beta`.
- Finite-sample bias analysis for beta estimation across multiple network sizes.
- Likelihood-ratio testing around the classical Barabasi-Albert baseline `beta = 1`.
- Detailed-balance and reversibility diagnostics for growth trajectories.
- Arrow-of-time style tests that distinguish forward growth from time-reversed alternatives.
- Thermodynamic-style identity checks involving susceptibility, heat-capacity-like quantities, and Fisher-information-style behavior.
- Compact figure/table generation for reproducible scientific analysis.

## Methodological Pipeline

```text
Preferential-attachment growth simulation
        |
        v
Degree-history and likelihood construction
        |
        v
Maximum-likelihood beta estimation
        |
        v
Finite-size bias and confidence diagnostics
        |
        v
Likelihood-ratio testing around beta = 1
        |
        v
Detailed-balance and arrow-of-time analysis
        |
        v
Thermodynamic identity validation
        |
        v
Figures, CSV tables, and reference checks
```

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── ba_inference.py
├── run_thermodynamic_figures.py
├── shared/
│   └── ba_experiments_full.py
├── tests/
│   └── ba_test.py
└── outputs/
    ├── main/
    └── extended/
```

## Main Components

| File or Folder | Purpose |
|---|---|
| `ba_inference.py` | Core graph generation, likelihood evaluation, MLE, LRT, and thermodynamic diagnostic utilities |
| `run_thermodynamic_figures.py` | Main script for regenerating thermodynamic-inference figures and result tables |
| `shared/ba_experiments_full.py` | Supporting experiment functions used by the figure runner |
| `tests/ba_test.py` | Lightweight reference run for reconstruction/collision sanity checks |
| `outputs/main/` | Main retained figures and CSV tables |
| `outputs/extended/` | Extended outputs for multiple attachment settings |

## Key Outputs

| Output | Description |
|---|---|
| `outputs/main/fig1_boltzmann.png` | Boltzmann-style likelihood behavior, MLE stability, and finite-size inference validation |
| `outputs/main/fig2_arrow_of_time.png` | Detailed-balance, reversibility, entropy-production, and LRT diagnostics |
| `outputs/main/fig4_identities.png` | Susceptibility and thermodynamic-identity validation |
| `outputs/main/table_mle_bias.csv` | Finite-sample beta-estimation bias table |
| `outputs/main/table_lrt_type1.csv` | Type-I error table for the likelihood-ratio test |
| `outputs/extended/` | Additional outputs for multiple attachment parameters |

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a lightweight reference check:

```bash
python tests/ba_test.py
```

Run the main figure workflow:

```bash
python run_thermodynamic_figures.py
```

The full figure workflow can be computationally heavier than the smoke test because it recomputes likelihood, finite-size, and diagnostic experiments.

## Scope

This repository focuses only on thermodynamic-style inference for preferential-attachment networks. Duplicate paper-facing output copies, raw scratch files, caches, and duplicate source folders are excluded so the project remains compact and reproducible.
