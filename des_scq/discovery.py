"""
des_scq.discovery
=================
Loss functions and parameter-space sampling utilities for circuit optimization.

This module provides two complementary tools:

**Loss functions**
    Callable factories that return ``(loss_tensor, metrics_dict)`` given a
    spectrum manifold and control profile.  Pass the returned callable to
    :class:`~des_scq.optimization.Optimization` as ``loss_function``.

**Parameter-space samplers**
    Functions that generate lists of circuit parameter dictionaries
    (``{ID: value}``) for batch or sequential optimization runs.  Use these
    to seed :class:`~des_scq.optimization.Optimization` with a diverse set of
    starting points for global search.

Loss function conventions
-------------------------
Every loss function in this module has the signature::

    loss_fn(Spectrum, flux_profile) → (loss: Tensor, metrics: dict)

where

* ``Spectrum``     — list of ``(eigenenergies, states)`` tuples as returned
  by :meth:`~des_scq.circuit.Circuit.spectrumManifold`.
* ``flux_profile`` — the same control manifold that was passed to
  ``spectrumManifold`` (available for flux-dependent weighting).
* ``loss``         — scalar differentiable loss.
* ``metrics``      — ``{str: float}`` dict of auxiliary values logged each
  iteration (anharmonicity, transition energies, degeneracy measure, etc.).

Factory pattern
---------------
Loss functions that depend on target values (e.g., ``lossTransition``) are
implemented as *factories*: a function that closes over the target tensors
and returns the actual loss callable.  This keeps the target data inside the
closure and out of the optimization loop's parameter list.
"""

import torch
from torch import tensor, abs, stack, var, log
from torch.optim import RMSprop
from numpy import meshgrid, linspace, array, log10, random, logspace
from numpy.random import choice
from random import seed
from scipy.stats import truncnorm
from des_scq.components import Parameters


# ---------------------------------------------------------------------------
# Spectral helper
# ---------------------------------------------------------------------------

def anHarmonicity(spectrum):
    """Compute the anharmonicity from the three lowest energy levels.

    Anharmonicity *α = (E₂₀ - E₁₀) - E₁₀ = E₂₁ - E₁₀*

    Parameters
    ----------
    spectrum : Tensor, shape (≥3,)
        Ordered eigenenergies *[E₀, E₁, E₂, ...]*.

    Returns
    -------
    Tensor (scalar)
        Anharmonicity in GHz (negative for transmon-like circuits).
    """
    ground, Ist, IInd = spectrum[:3]
    return (IInd - Ist) - (Ist - ground)


# ---------------------------------------------------------------------------
# Loss function: MSE reference
# ---------------------------------------------------------------------------

MSE = torch.nn.MSELoss()


# ---------------------------------------------------------------------------
# Loss functions (factories)
# ---------------------------------------------------------------------------

def lossTransitionFlatness(Spectrum, flux_profile):
    """Loss that minimizes variance of transition energies across the flux manifold.

    Penalizes any variation in *E₁₀* and *E₂₁* over the control manifold,
    driving the circuit toward a flux-insensitive operating point.

    Parameters
    ----------
    Spectrum : list of (Tensor, None)
        Full spectrum manifold.
    flux_profile : list
        Control manifold (not used directly, kept for API consistency).

    Returns
    -------
    loss : Tensor (scalar)
        Sum of variances of *E₁₀* and *E₂₁* over the manifold.
    metrics : dict
        Empty dict.

    Example
    -------
    >>> optim = Optimization(circuit, flux_profile, lossTransitionFlatness)
    """
    spectrum = stack([spectrum[:3] for spectrum, state in Spectrum])
    loss  = var(spectrum[:, 1] - spectrum[:, 0])
    loss += var(spectrum[:, 2] - spectrum[:, 1])
    return loss, dict()


def lossDegeneracyWeighted(delta0, D0, N=2):
    """Factory: loss that targets a specific degeneracy and anharmonicity ratio.

    Optimizes simultaneously for:

    * **Degeneracy** — the difference in *E₁₀* between two adjacent flux
      points, measured as ``log(|ΔE₁₀| / E₁₀)``.
    * **Anharmonicity ratio** — ``D = log(E₂₀ / E₁₀)``, which characterizes
      the qubit's protection.

    The loss is a weighted linear combination::

        loss = delta * delta0 - D * D0

    Parameters
    ----------
    delta0 : float
        Weight for the degeneracy term (positive penalizes large degeneracy).
    D0 : float
        Weight for the anharmonicity-ratio term (positive rewards large *D*).
    N : int, optional
        Unused (reserved).  Default ``2``.

    Returns
    -------
    callable
        Loss function with signature
        ``(Spectrum, flux_profile) → (loss, metrics)``.

    Notes
    -----
    ``metrics`` contains ``'delta'``, ``'D'``, ``'E10'``, ``'E20'``.
    """
    def Loss(Spectrum, flux_profile):
        E10 = [E[0][1] - E[0][0] for E in Spectrum]
        E20 = [E[0][2] - E[0][0] for E in Spectrum]
        Ist  = abs(E10[0] - E10[1])
        D    = log(E20[0] / E10[0]) / log(tensor(10.))
        delta = log(Ist / E10[0]) / log(tensor(10.))
        loss  = delta * delta0 - D * D0
        return loss, {
            'delta': delta.detach().item(),
            'D':     D.detach().item(),
            'E10':   E10[0].detach().item(),
            'E20':   E20[0].detach().item(),
        }
    return Loss


def lossDegeneracyTarget(delta0, D0):
    """Factory: squared-error loss toward target degeneracy and anharmonicity.

    Minimizes::

        loss = (delta0 + delta)² + (D0 - D)²

    where ``delta = log(|n₁₀ - e₁₀| / e₂₀)`` and ``D = log(e₂₀ / e₁₀)``.

    Parameters
    ----------
    delta0 : float
        Target log-degeneracy value (usually negative, e.g., ``-3.0``).
    D0 : float
        Target log anharmonicity-ratio value.

    Returns
    -------
    callable
        ``(Spectrum, flux_profile) → (loss, metrics)``
    """
    def Loss(Spectrum, flux_profile):
        half      = 0
        neighbour = -1
        e20   = Spectrum[half][0][2]     - Spectrum[half][0][0]
        e10   = Spectrum[half][0][1]     - Spectrum[half][0][0]
        D     = log(e20 / e10)
        n10   = Spectrum[neighbour][0][1] - Spectrum[neighbour][0][0]
        delta = log((n10 - e10).abs() / e20)
        loss  = (delta0 + delta) ** 2 + (D0 - D) ** 2
        return loss, {
            'delta': delta.detach().item(),
            'D':     D.detach().item(),
            'E10':   e10.detach().item(),
            'E20':   e20.detach().item(),
        }
    return Loss


def lossAnharmonicity(alpha):
    """Factory: MSE loss toward a target anharmonicity value.

    Minimizes the mean-squared error between the circuit's anharmonicity
    *α = E₂₁ - E₁₀* (averaged over the control manifold) and the target
    value *alpha*.

    Parameters
    ----------
    alpha : float
        Target anharmonicity in GHz.

    Returns
    -------
    callable
        ``(Spectrum, flux_profile) → (loss, metrics)``

    Notes
    -----
    ``metrics`` contains ``'anharmonicity'`` (the current average value).

    Example
    -------
    >>> loss_fn = lossAnharmonicity(-0.2)   # target -200 MHz anharmonicity
    >>> optim   = Optimization(circuit, flux_profile, loss_fn)
    """
    def lossFunction(Spectrum, flux_profile):
        anharmonicity = tensor(0.0)
        for spectrum, state in Spectrum:
            anharmonicity += anHarmonicity(spectrum)
        anharmonicity = anharmonicity / len(Spectrum)
        loss = MSE(anharmonicity, tensor(alpha))
        return loss, {'anharmonicity': anharmonicity.detach().item()}
    return lossFunction


def lossTransition(E10, E21):
    """Factory: MSE loss toward target transition energies over a flux manifold.

    Minimizes::

        loss = MSE(e₁₀, E10) + MSE(e₂₁, E21)

    where the MSE is computed simultaneously over all flux points in the
    control manifold.

    Parameters
    ----------
    E10 : Tensor, shape (num_flux_points,)
        Target *E₁₀* transition energies in GHz.
    E21 : Tensor, shape (num_flux_points,)
        Target *E₂₁* transition energies in GHz.

    Returns
    -------
    callable
        ``(Spectrum, flux_profile) → (loss, metrics)``

    Notes
    -----
    ``metrics`` contains ``'mid10'`` and ``'mid21'``: the *E₁₀* and *E₂₁*
    values at the middle flux point.

    The number of target values must equal the number of flux points in
    ``control_profile``.  If your flux profile has *N* points, pass
    ``E10`` and ``E21`` as tensors of length *N*.

    Example
    -------
    >>> from torch import tensor, float64 as double
    >>> E10 = tensor([5.0, 4.9, 4.8, 4.9, 5.0], dtype=double)
    >>> E21 = tensor([4.8, 4.7, 4.6, 4.7, 4.8], dtype=double)
    >>> loss_fn = lossTransition(E10, E21)
    """
    def lossFunction(Spectrum, flux_profile):
        spectrum = stack([spectrum[:3] for spectrum, state in Spectrum])
        e10  = spectrum[:, 1] - spectrum[:, 0]
        e21  = spectrum[:, 2] - spectrum[:, 1]
        loss = MSE(e10, E10) + MSE(e21, E21)
        mid  = int(len(flux_profile) / 2)
        log  = {'mid10': e10[mid].detach().item(),
                'mid21': e21[mid].detach().item()}
        return loss, log
    return lossFunction


# ---------------------------------------------------------------------------
# Discovery: initialization helpers
# ---------------------------------------------------------------------------

def initializationSequential(parameters, optimizer, iterations=100, lr=.005,
                              algo=RMSprop):
    """Run sequential single-start optimizations from a list of initial points.

    For each parameter set in *parameters*, re-initializes the circuit,
    rebuilds the optimizer, and runs a fresh optimization.

    Parameters
    ----------
    parameters : list of dict {str: float}
        Initial parameter states, e.g., as returned by
        :func:`uniformParameters`.
    optimizer : Optimization
        An :class:`~des_scq.optimization.Optimization` instance (will be
        mutated in place).
    iterations : int, optional
        Number of gradient steps per start.  Default ``100``.
    lr : float, optional
        Learning rate.  Default ``0.005``.
    algo : torch.optim class, optional
        Optimizer algorithm.  Default ``RMSprop``.

    Returns
    -------
    Search : list of (dLogs, dParams, dCircuit)
        One result tuple per initial parameter set.

    Example
    -------
    >>> parameters = uniformParameters(circuit, ('J', 'C'), n=5, N=20)
    >>> Search     = initializationSequential(parameters, optim, iterations=200)
    >>> # pick the run with the lowest final loss:
    >>> best = min(Search, key=lambda r: r[0]['loss'].iloc[-1])
    """
    Search = []
    for index, parameter in enumerate(parameters):
        print(index, '--------------------')
        optimizer.circuit.initialization(parameter)
        # circuit.initialization rewrites a new tensor
        optimizer.parameters, _ = optimizer.circuitParameters()
        optimizer.initAlgo(lr=lr, algo=algo)
        Search.append(optimizer.optimization(iterations=iterations))
    return Search


# ---------------------------------------------------------------------------
# Parameter-space samplers
# ---------------------------------------------------------------------------

def truncNormalParameters(circuit, subspace, N, var=5):
    """Sample *N* circuit parameter sets from truncated normal distributions.

    Each parameter in *subspace* is sampled from a truncated normal
    distribution centered on its current value, clipped to its physical
    bounds.

    Parameters
    ----------
    circuit : Circuit
        Source of current values and bounds.
    subspace : iterable of str
        Element IDs to sample.
    N : int
        Number of parameter sets to generate.
    var : float, optional
        Standard deviation of the truncated normal (in energy units, GHz).
        Default ``5``.

    Returns
    -------
    list of dict {str: float}
        Length *N*, each a complete circuit state (non-sampled parameters
        hold their current values).

    Notes
    -----
    The random seed is fixed per element index (``101 + index``) for
    reproducibility.
    """
    iDs, domain = [], []
    for index, component in enumerate(circuit.network):
        if component.ID in subspace:
            iDs.append(component.ID)
            loc = component.energy().item()
            a, b = component.bounds()
            if a.is_cuda:
                a, b = a.cpu(), b.cpu()
            a = (a - loc) / var
            b = (b - loc) / var
            domain.append(truncnorm.rvs(a, b, loc, var, size=N,
                                        random_state=random.seed(101 + index)))
    grid = array(domain).T
    return parameterSpace(circuit, grid, iDs)


def uniformParameters(circuit, subspace, n, N, externals=[], random_state=10,
                      logscale=False):
    """Sample *N* parameter sets with uniform random sampling per element.

    For each element in *subspace*, draws *N* values uniformly (or log-
    uniformly) from a grid of *n* equally spaced points spanning the element's
    physical bounds.

    Parameters
    ----------
    circuit : Circuit
        Source of element bounds.
    subspace : iterable of str
        Element IDs to sample.
    n : int
        Number of grid points per dimension.
    N : int
        Total number of parameter sets to generate.
    externals : list of Parameters, optional
        Additional non-circuit parameters to include in the sampling.
        Default ``[]``.
    random_state : int, optional
        Seed for the random number generator.  Default ``10``.
    logscale : bool, optional
        If ``True``, space grid points logarithmically.  Useful for
        parameters that span several orders of magnitude.  Default ``False``.

    Returns
    -------
    list of dict {str: float}
        Length *N*.

    Example
    -------
    >>> parameters = uniformParameters(circuit, ('J', 'C'), n=10, N=50)
    >>> Search     = initializationSequential(parameters, optim)
    """
    iDs, grid = [], []
    seed(random_state)
    for component in circuit.network + externals:
        if component.ID in subspace:
            distribution = uniformUnidimensional(component, n, N, logscale)
            grid.append(distribution)
            iDs.append(component.ID)
    grid = array(grid).T
    return parameterSpace(circuit, grid, iDs)


def uniformUnidimensional(parameter: Parameters, n: int, N: int,
                           logscale=False):
    """Draw *N* uniform random samples for a single parameter.

    Parameters
    ----------
    parameter : Parameters
        Circuit element providing bounds via ``parameter.bounds()``.
    n : int
        Number of grid points to discretize the domain.
    N : int
        Number of samples to draw (with replacement from the *n* grid points).
    logscale : bool, optional
        If ``True``, space grid points on a log scale.  Default ``False``.

    Returns
    -------
    ndarray, shape (N,)
        Sampled energy values in GHz.
    """
    spacing = linspace
    if logscale:
        spacing = logspace
    a, b = parameter.bounds()
    a, b = a.item(), b.item()
    if logscale:
        a = log10(a)
        b = log10(b)
    domain = spacing(a, b, n + 1, endpoint=False)[1:]
    return choice(domain, N)


def domainParameters(domain, circuit, subspace):
    """Enumerate all combinations of given 1-D domains (full grid).

    Builds the Cartesian product of the provided domain arrays and returns
    one parameter dict per grid point.

    Parameters
    ----------
    domain : list of array-like
        Per-element value arrays, e.g. ``[linspace(1, 100, 10),
        linspace(1, 500, 10)]``.  Length must equal ``len(subspace)``.
    circuit : Circuit
        Source of the static (non-scanned) circuit state.
    subspace : list of str
        Element IDs corresponding to each domain array.

    Returns
    -------
    list of dict {str: float}
        One entry per grid point (total length = product of domain sizes).

    Warnings
    --------
    The grid grows exponentially with the number of dimensions.  Use
    :func:`uniformParameters` for higher-dimensional searches.
    """
    grid = array(meshgrid(*domain))
    grid = grid.reshape(len(subspace), -1).T
    return parameterSpace(circuit, grid, subspace)


def parameterSpace(circuit, grid, iDs):
    """Convert a raw parameter grid into a list of circuit state dicts.

    For each row in *grid*, starts from the circuit's current static state
    and overrides the values for the listed IDs.

    Parameters
    ----------
    circuit : Circuit
        Provides the baseline state via ``circuit.circuitState()``.
    grid : array-like, shape (N, len(iDs))
        Parameter values; each row is one sample.
    iDs : list of str
        Element IDs corresponding to grid columns.

    Returns
    -------
    list of dict {str: float}
        Length *N*.
    """
    space = []
    for point in grid:
        state = circuit.circuitState()       # static non-sampled parameters
        state.update(dict(zip(iDs, point)))  # overwrite sampled parameters
        space.append(state)
    return space
