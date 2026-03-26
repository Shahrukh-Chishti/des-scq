# des-scq

**Designing Superconducting Quantum Circuits** — a PyTorch-powered simulation and optimization library for lumped-element superconducting circuit models.

---

## Overview

`des-scq` provides a differentiable Hamiltonian simulation stack for superconducting qubit circuits. Circuits are described as a network of lumped elements (Josephson junctions, capacitors, inductors), the circuit Hamiltonian is assembled automatically via nodal analysis, and eigenenergies are obtained by exact diagonalization. Because the entire pipeline is built on PyTorch, gradients flow back through the spectrum to the circuit parameters — enabling gradient-based optimization of qubit properties such as transition frequencies, anharmonicity, and flux sensitivity.

**Key capabilities:**

- Automated Hamiltonian assembly from a graph of lumped elements
- Two fully supported basis representations: **Charge** and **Kerman** (mixed oscillator / charge / Josephson modes)
- Exact diagonalization over a configurable Hilbert space truncation
- External flux and charge-offset control with differentiable loop-flux threading
- Gradient-descent optimization of circuit parameters against arbitrary spectral targets
- Stochastic search over circuit parameter spaces for global exploration
- Built-in library of standard qubit models: transmon, fluxonium, zero-pi, c-shunted qubit, prismon, and more

---

## Architecture

```
des_scq/
├── components.py   Element classes (J, C, L, Control) and unit conversions
├── dense.py        Operator algebra — basis operators, tensor products, Fourier transforms
├── circuit.py      Circuit graph, Hamiltonian assembly, diagonalization
│   ├── Circuit     Base class: nodal analysis, loop-flux threading
│   ├── Charge      Subclass: pure charge-basis representation
│   └── Kerman      Subclass: mixed O/I/J basis (Kerman decomposition)
├── models.py       Pre-built circuit constructors (transmon, fluxonium, zeroPi, …)
├── optimization.py Gradient-descent optimizer wrapping circuit + loss function
├── discovery.py    Loss functions and parameter-space sampling utilities
└── utils.py        Plotting helpers (Plotly-based)
```

Data flows as follows:

```
models.py  ──►  circuit.py  ──►  dense.py
   │               │
   │         spectrumManifold()
   │               │
   ▼               ▼
components.py   eigenenergies
                    │
              discovery.py (loss)
                    │
              optimization.py (∂loss/∂params → Adam / RMSprop / LBFGS)
```

---

## Installation

```bash
pip install des-scq
```

**Dependencies:** `torch`, `numpy`, `scipy`, `networkx`, `pandas`

---

## Energy Units Convention

All energies inside the library are expressed in **GHz** (i.e., divided by Planck's constant *h*). Physical SI values must be converted before being passed to element constructors. Helper functions for this are provided in `components.py`:

| Quantity    | SI unit | Natural unit         | Energy unit (GHz) | Converter (SI → GHz) |
|-------------|---------|----------------------|-------------------|----------------------|
| Capacitance | F       | *e*²/*h* · 10⁹       | *Ec = e²/2Ch*     | `capE(C_SI)`         |
| Inductance  | H       | *Φ₀*²/*h* · 10⁹      | *El = Φ₀²/4π²Lh*  | `indE(L_SI)`         |
| Junction    | —       | —                    | *Ej* (given)      | direct (GHz)         |

```python
from des_scq.components import capE, indE

Ec = capE(45e-15)   # 45 fF  →  GHz
El = indE(10e-9)    # 10 nH  →  GHz
```

Flux values are expressed as **reduced flux** Φ/Φ₀ ∈ [0, 1], where Φ₀ = *h*/2*e* is the flux quantum.

---

## Quickstart

### 1 — Transmon spectrum in Charge basis

```python
from des_scq import models
from des_scq.circuit import Charge, hamiltonianEnergy
from torch import tensor

# Build circuit: Ej = 30 GHz, Ec = 0.3 GHz, 256 charge states
circuit = models.transmon(Charge, basis=[256], Ej=30.0, Ec=0.3)

# Assemble and diagonalize
H = circuit.hamiltonianLC() + circuit.hamiltonianJosephson()
energies = hamiltonianEnergy(H)
E10 = energies[1] - energies[0]
print(f"E₁₀ = {E10:.4f} GHz")
```

### 2 — Flux profile over a manifold of external flux values

```python
from numpy import linspace
from torch import tensor

flux_range  = tensor(linspace(0, 1, 51))
flux_profile = [[phi] for phi in flux_range]   # list of control points

Spectrum = circuit.spectrumManifold(flux_profile)

for phi, (energies, _) in zip(flux_range, Spectrum):
    E10 = (energies[1] - energies[0]).item()
    print(f"Φ/Φ₀ = {phi:.2f}   E₁₀ = {E10:.4f} GHz")
```

### 3 — Charge offset sensitivity (Charge basis)

```python
from torch import tensor

offset = {1: tensor(0.5)}   # 0.5 Cooper pairs on node 1
H = (circuit.hamiltonianLC()
     + circuit.hamiltonianJosephson()
     + circuit.hamiltonianChargeOffset(offset))
energies = hamiltonianEnergy(H)
```

### 4 — Zero-pi qubit in Kerman basis

```python
from des_scq import models
from des_scq.circuit import Kerman

basis   = {'O': [32], 'I': [], 'J': [8, 8]}
circuit = models.zeroPi(Kerman, basis, Ej=10., Ec=50., El=0.5)

No, Ni, Nj = circuit.kermanDistribution()
print(f"Oscillator modes: {No}, Island modes: {Ni}, Josephson modes: {Nj}")

flux_profile = [[tensor(phi)] for phi in linspace(0, 1, 21)]
Spectrum     = circuit.spectrumManifold(flux_profile)
```

### 5 — Gradient-descent optimization

```python
from des_scq.optimization import Optimization
from des_scq.discovery import lossTransition
from torch import tensor, float64 as double

# Target transition energies across the flux profile (in GHz)
E10_target = tensor([5.0] * 21, dtype=double)
E21_target = tensor([4.8] * 21, dtype=double)

loss_fn = lossTransition(E10_target, E21_target)
optim   = Optimization(circuit, flux_profile, loss_fn)
optim.initAlgo(lr=0.05)

dLogs, dParams, dCircuit = optim.optimization(iterations=200)
print(dLogs[['loss']].tail())
print(dCircuit.iloc[-1])   # final circuit parameters (GHz)
```

---

## Example Notebooks

| Notebook | Circuit | Demonstrates |
|---|---|---|
| `C-shunted_qubit.ipynb` | C-shunted qubit | Kerman basis, pre/post-optimization spectrum, loss landscape |
| `Flux_profile.ipynb` | Fluxonium | Target-guided flux-profile optimization |
| `Insensitive_Flux_profile.ipynb` | Fluxonium | Anharmonicity-flatness optimization |
| `Charge_Sensitivity_-_Transmon.ipynb` | Transmon | Charge dispersion vs. *Ej/Ec* ratio |
| `CPR_-_Modeling.ipynb` | Transmon | Parameter estimation from experimental CPR data |

---

## Module Reference

| Module | Key symbols |
|---|---|
| `components` | `J`, `C`, `L`, `Control`, `capE`, `indE`, `capSINat`, `indSINat` |
| `dense` | `basisQq`, `basisFq`, `basisQo`, `basisFo`, `chargeDisplacePlus/Minus`, `modeTensorProduct`, `transformationMatrix` |
| `circuit` | `Circuit`, `Charge`, `Kerman`, `hamiltonianEnergy` |
| `models` | `transmon`, `zeroPi`, `fluxonium`, `shuntedQubit`, `prismon`, `fluxoniumArray`, `oscillatorLC` |
| `optimization` | `Optimization` |
| `discovery` | `lossTransition`, `lossAnharmonicity`, `lossTransitionFlatness`, `lossDegeneracyWeighted`, `uniformParameters`, `truncNormalParameters`, `domainParameters` |

---

## Design Notes

**Sigmoid reparametrization.** Every circuit element parameter is stored as an unconstrained real number that maps to a bounded physical range through a sigmoid transform. This keeps optimization stable and prevents parameters from wandering outside physically meaningful bounds without requiring projected-gradient methods.

**Symmetry constraints.** The `pairs` dictionary passed to circuit constructors enforces hard equality between named components (e.g., `pairs={'Jy': 'Jx'}` ties the two junctions in a zero-pi qubit together). Symmetric components share the same `base` tensor, so gradients accumulate correctly.

**Basis truncation.** Hilbert-space dimension grows as the product of per-mode sizes. Typical stable values: `n = 32–512` for single-mode charge/oscillator circuits; `n = 8–16` per mode in multi-mode circuits. Verify convergence by checking that eigenenergies are stable as `n` increases.

**GPU support.** Pass `device='cuda'` to any circuit constructor. All PyTorch tensors are placed on that device; `numpy` calls in graph analysis run on CPU automatically.
