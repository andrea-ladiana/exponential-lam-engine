# Exponential LAM Monte Carlo Engine

Reproducibility repository for the Monte Carlo simulations discussed in "Exponential Capacity in Multilayer Hetero-Associative Hopfield Networks".

## Overview

This repository provides the core code to reproduce the numerical experiments and simulations presented in the paper *Exponential Capacity in Multilayer Hetero-Associative Hopfield Networks*. 

The paper introduces an exponential neural network composed of $L$ layers of binary neurons. Each layer stores its own dataset of $P$ patterns, and the energy function is structured as an exponential of the product of the per-layer Mattis overlaps. This formulation natively supports hetero-associative tasks (e.g., mapping T-cell receptor chains to an epitope) while provably retaining an exponential storage capacity $P_c \sim e^{N\rho_L}$.

The repository contains the LAM (Layered Associative Memory) engines that implement the exact zero-temperature synchronous update rules, evaluate the cavity/signal-to-noise analytical predictions, and probe the basins of attraction.

- `lam_engine.py`: Reference NumPy implementation including the exact one-step local field computation, as well as the closed-form theory for the capacity rate and finite-N predictions.
- `lam_engine_torch.py`: A batched, GPU-accelerated PyTorch implementation designed to rapidly run Monte Carlo retrieval trials with cue-clamping across large datasets (such as the Hidden Manifold Model and VDJdb experiments discussed in the paper).
- `config.py`: Basic device and chunking configuration for PyTorch operations.

## Requirements

The core dependencies are:
- `numpy`
- `scipy`

For GPU acceleration, `torch` is highly recommended.

## Usage Examples

Below are a few essential examples to illustrate the usage of the LAM engine.

### 1. Theoretical Predictions
Evaluate the theoretical capacity edge and the analytical 1-step overlap limit:

```python
import lam_engine

L, N = 3, 64
print("Log-capacity (N rho_L):", N * lam_engine.saddle_point(L)["rho_L"])

# Expected 1-step overlap for P=20 patterns
m1 = lam_engine.theory_m1_one_step(L, P=20, N=N, r=1.0)
print(f"Theory m_1: {m1:.4f}")
```

### 2. NumPy Single-State Dynamics
Simulate the retrieval of a single state using the reference CPU engine:

```python
import numpy as np
import lam_engine

P, L, N = 20, 2, 64
rng = np.random.default_rng(42)

# Store P random patterns
xi = rng.choice([-1, 1], size=(P, L, N)).astype(np.int8)

# Initialize a corrupted state and run a single update
sigma_init = lam_engine.initial_state(xi, mu0=0, r=0.7, rng=rng)
sigma_next = lam_engine.one_step_parallel(xi, sigma_init)

# Check Mattis overlap with the target pattern
print("Overlap:", lam_engine.mattis_overlap(xi, sigma_next, mu0=0))
```

### 3. Batched GPU Dynamics with Cue Seed
Use the vectorized PyTorch engine to process a batch of queries, clamping known layers:

```python
import numpy as np
import lam_engine_torch

P, L, N, B = 50, 3, 128, 10
rng = np.random.default_rng(42)

xi = rng.choice([-1, 1], size=(P, L, N)).astype(np.int8)

# Batch of queries (cues)
cues = rng.choice([-1, 1], size=(B, L, N)).astype(np.int8)

# Deterministically seed the unknown layers (e.g., layers 1 and 2) using cue layer 0
seeded_states = lam_engine_torch.cue_seed_auto(xi, cues, known_layers=[0])

# Evolve the system while clamping the cue layer
final_states = lam_engine_torch.iterate_parallel_auto(
    xi, seeded_states, n_steps=3, clamp_layers=[0]
)
```

## Testing

The codebase includes a comprehensive test suite targeting >= 95% coverage. To run the tests and verify coverage, install `pytest` and `pytest-cov`, then run:

```bash
pytest test_lam.py --cov=lam_engine --cov=lam_engine_torch --cov-report=term-missing
```
