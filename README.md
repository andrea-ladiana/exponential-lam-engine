# Exponential LAM: Multilayer Hetero-Associative Engine

[![arXiv](https://img.shields.io/badge/arXiv-2607.29554-b31b1b.svg)](https://arxiv.org/abs/2607.29554)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2607.29554-0077c2.svg)](https://doi.org/10.48550/arXiv.2607.29554)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-13%20passed-brightgreen.svg)](test_lam.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> This repository provides the official implementation, Monte Carlo simulation engine, and analytical theory routines for the research paper:
>
> **Exponential Capacity in Multilayer Hetero-Associative Neural Networks**  
> Elena Agliari, Adriano Barra, Andrea Ladiana, Andrea Lepre  
> *arXiv preprint* [arXiv:2607.29554](https://arxiv.org/abs/2607.29554) (2026).  
> **Preprint:** [https://arxiv.org/abs/2607.29554](https://arxiv.org/abs/2607.29554) | **DOI:** [10.48550/arXiv.2607.29554](https://doi.org/10.48550/arXiv.2607.29554)

---

## Overview

Exponential Hopfield networks store a number of patterns that grows exponentially with the number of neurons. In their classical formulation, however, they are auto-associative: they complete a corrupted copy of a memory into the memory itself.

The **Multilayer Hetero-Associative Layered Associative Memory (LAM)** extends exponential networks to multi-layer architectures capable of storing and retrieving associations across distinct sensory channels or data domains (e.g. mapping T-cell receptor chains to target epitopes, or text cues to intents).

### Key Architectural Highlights

- **Multi-layer exponential energy**: Composed of $L$ layers of $N$ binary neurons, each layer carrying its own dataset of $P$ patterns. The energy is an exponential function of the product of per-layer Mattis overlaps:
  $$\mathcal{H} = - \sum_{\mu=1}^P \exp\left( \beta \prod_{l=1}^L m_l^\mu \right)$$
- **Exponential storage capacity**: Exact cavity / signal-to-noise large-deviation analysis proves that the aligned hetero-associative state remains a fixed point of zero-temperature dynamics up to:
  $$P_c \sim e^{N \rho_L}, \quad \text{with } \rho_L \sim L \log 2$$
- **Domain-universal retrieval**: Content-addressable memory capabilities hold across synthetic manifolds, correlated real-world biological sequences (VDJdb TCR-epitope triples), and natural-language intent data.

---

## Repository Structure

| File | Purpose |
| :--- | :--- |
| `lam_engine.py` | Reference NumPy engine containing exact 1-step local field dynamics, saddle-point solutions, critical capacity rate $\rho_L$, and finite-$N$ predictions. |
| `lam_engine_torch.py` | Vectorized, GPU-accelerated PyTorch engine designed for batched Monte Carlo retrieval trials, cue-seeding, and cue-clamping. |
| `config.py` | Device selector (`cuda` / `cpu`) and chunking configurations for large-scale evaluations. |
| `test_lam.py` | Comprehensive test suite covering theoretical limits, saddle-point solvers, and CPU/GPU engine consistency. |
| `CITATION.cff` | Citation metadata enabling GitHub's native **"Cite this repository"** button. |
| `pyproject.toml` | Packaging configuration and dependency definitions. |

---

## Installation

Install in editable development mode:

```bash
pip install -e .
```

To install with PyTorch and test dependencies:

```bash
pip install -e .[torch,dev]
```

Core dependencies: `numpy>=1.24`, `scipy>=1.10`.  
GPU acceleration requires `torch>=2.0`.

---

## Usage Examples

### 1. Theoretical Predictions

Evaluate the analytical capacity edge and the expected 1-step overlap:

```python
import lam_engine

L, N = 3, 64
sp = lam_engine.saddle_point(L)
print(f"Critical rate rho_L: {sp['rho_L']:.4f}")
print(f"Log-capacity (N * rho_L): {N * sp['rho_L']:.2f}")

# Expected 1-step overlap for P=20 patterns
m1 = lam_engine.theory_m1_one_step(L, P=20, N=N, r=1.0)
print(f"Theory m_1: {m1:.4f}")
```

### 2. NumPy Single-State Dynamics

Simulate retrieval of a single state using the reference CPU engine:

```python
import numpy as np
import lam_engine

P, L, N = 20, 2, 64
rng = np.random.default_rng(42)

# Generate P random patterns across L layers
xi = rng.choice([-1, 1], size=(P, L, N)).astype(np.int8)

# Initialize a state corrupted with noise (overlap r = 0.7) and perform one update step
sigma_init = lam_engine.initial_state(xi, mu0=0, r=0.7, rng=rng)
sigma_next = lam_engine.one_step_parallel(xi, sigma_init)

# Check Mattis overlap with the target archetype
print("Target overlap:", lam_engine.mattis_overlap(xi, sigma_next, mu0=0))
```

### 3. Batched GPU Dynamics with Cue Seed

Use the vectorized PyTorch engine to process batches of queries with clamped cue layers:

```python
import numpy as np
import lam_engine_torch

P, L, N, B = 50, 3, 128, 10
rng = np.random.default_rng(42)

xi = rng.choice([-1, 1], size=(P, L, N)).astype(np.int8)

# Batch of noisy queries (cues)
cues = rng.choice([-1, 1], size=(B, L, N)).astype(np.int8)

# Deterministically seed the unknown layers (layers 1 and 2) using cue layer 0
seeded_states = lam_engine_torch.cue_seed_auto(xi, cues, known_layers=[0])

# Evolve the network while clamping cue layer 0
final_states = lam_engine_torch.iterate_parallel_auto(
    xi, seeded_states, n_steps=3, clamp_layers=[0]
)
```

---

## Verification & Testing

Run the test suite with `pytest`:

```bash
pytest test_lam.py
```

To run with coverage reporting:

```bash
pytest test_lam.py --cov=lam_engine --cov=lam_engine_torch --cov-report=term-missing
```

---

## Citation

If you use this software or the exponential LAM framework in your research, please cite our preprint:

```bibtex
@article{agliari2026exponential,
  title={Exponential Capacity in Multilayer Hetero-Associative Neural Networks},
  author={Agliari, Elena and Barra, Adriano and Ladiana, Andrea and Lepre, Andrea},
  journal={arXiv preprint arXiv:2607.29554},
  year={2026},
  doi={10.48550/arXiv.2607.29554},
  url={https://arxiv.org/abs/2607.29554}
}
```

You can also use GitHub's **"Cite this repository"** button in the sidebar.

---

## License

MIT License — Copyright (c) 2026 Andrea Ladiana. See [`LICENSE`](LICENSE) for details.
