"""
des_scq.dense
=============
Low-level operator algebra for multi-mode superconducting circuits.

This module provides:

* **Basis operators** — charge (*Q*, *D±*) and flux (*F*) operators in the
  charge basis; position (*F*) and momentum (*Q*) operators in the oscillator
  (Fock) basis; flux-bias operators in the flux basis.
* **Tensor product infrastructure** — routines for lifting a single-mode
  operator to the full multi-mode Hilbert space and for computing
  mode-coupling terms.
* **Displacement operators** — exponentiated flux operators used to build
  the Josephson Hamiltonian.
* **Fourier transformation** — ``transformationMatrix`` converts between the
  charge and flux representations (periodic boundary conditions).
* **Oscillator-basis diagonalization helpers** — ``chargeFlux``,
  ``fluxFlux``, etc., rotate oscillator operators into the charge/flux
  eigenbasis for the Kerman representation.

Hilbert-space conventions
-------------------------
* **Charge basis**: states are indexed by Cooper-pair number
  *n ∈ {-n, …, 0, …, n}* for a truncation parameter *n*.
  Hilbert-space dimension = *2n + 1*.
* **Flux basis**: states are uniformly spaced flux points on *(-Φ₀/2, Φ₀/2]*.
  Dimension = *N* (arbitrary grid size).
* **Oscillator basis**: Fock states *|0⟩, …, |N-1⟩*.
  Dimension = *N*.

All operators are returned as ``torch.cdouble`` (complex128) tensors by
default to support Hermitian matrix operations.
"""

from torch import kron
from des_scq.components import *


# ---------------------------------------------------------------------------
# Multi-mode tensor product helpers
# ---------------------------------------------------------------------------

def modeTensorProduct(pre, M, post):
    """Embed a single-mode operator *M* into the full multi-mode basis.

    Constructs  I_{pre[0]} ⊗ … ⊗ I_{pre[-1]} ⊗ M ⊗ I_{post[0]} ⊗ …

    Parameters
    ----------
    pre : list of int
        Hilbert-space dimensions of modes *before* the target mode.
    M : Tensor, shape (d, d)
        The operator to embed.
    post : list of int
        Hilbert-space dimensions of modes *after* the target mode.

    Returns
    -------
    Tensor
        Full-system operator of dimension prod(pre) * d * prod(post).
    """
    H = identity(1)
    for dim in pre:
        H = kron(H, identity(dim))
    H = kron(H, M)
    for dim in post:
        H = kron(H, identity(dim))
    return H


def crossBasisProduct(A, B, a, b):
    """Build a two-mode operator using different operator sets for each mode.

    For mode *i*, uses ``A[i]`` if *i == a*, ``B[i]`` if *i == b*, and the
    identity otherwise.  Used to construct off-diagonal Josephson terms for
    junctions between two non-grounded nodes.

    Parameters
    ----------
    A : list of Tensor
        Operator set for the first selected mode.
    B : list of Tensor
        Operator set for the second selected mode.
    a : int
        Mode index using operators from *A*.
    b : int
        Mode index using operators from *B*.

    Returns
    -------
    Tensor
        Multi-mode operator as a tensor product.
    """
    assert len(A) == len(B)
    n = len(A)
    product = identity(1)
    for i in range(n):
        if i == a:
            product = kron(product, A[i])
        elif i == b:
            product = kron(product, B[i])
        else:
            product = kron(product, identity(len(A[i])))
    return product


def basisProduct(O, indices=None):
    """Tensor product of operators, one per mode.

    For modes in *indices*, uses the corresponding operator in *O*; for all
    other modes, uses the identity of appropriate dimension.

    Parameters
    ----------
    O : list of Tensor
        List of per-mode operators (one per mode in the system).
    indices : list of int, optional
        Modes for which the non-trivial operator is applied.  If ``None``
        (default), applies to all modes.

    Returns
    -------
    Tensor
        Full system operator.
    """
    n = len(O)
    B = identity(1)
    if indices is None:
        indices = arange(n)
    for i in range(n):
        if i in indices:
            B = kron(B, O[i])
        else:
            B = kron(B, identity(len(O[i])))
    return B


def modeProduct(A, i, B, j):
    """Tensor product of mode-*i* operator from *A* and mode-*j* from *B*.

    Parameters
    ----------
    A : list of Tensor
        Operator set for the first mode.
    i : int
        Mode index in *A*.
    B : list of Tensor
        Operator set for the second mode.
    j : int
        Mode index in *B*.

    Returns
    -------
    Tensor
        Full-system operator.
    """
    return mul(basisProduct(A, [i]), basisProduct(B, [j]))


def modeMatrixProduct(A, M, B, mode=(0, 0)):
    """Build a bilinear Hamiltonian term *Aᵀ M B* in the multi-mode basis.

    Computes::

        H = Σᵢⱼ  M[i,j] · (mode-i operator from A) ⊗ (mode-j operator from B)

    The optional *mode* offset shifts the row/column mode indices, allowing
    off-diagonal coupling blocks to be added without building a full
    concatenated operator list.

    Parameters
    ----------
    A : list of Tensor
        Left operator set (one tensor per mode).
    M : Tensor, shape (nA, nB)
        Coupling matrix in mode space (e.g., inverse capacitance matrix).
    B : list of Tensor
        Right operator set.
    mode : tuple of int, optional
        ``(a, b)`` offsets into *A* and *B* respectively.  The matrix row *i*
        couples to mode ``i + a`` from *A* and column *j* couples to mode
        ``j + b`` from *B*.  Default ``(0, 0)``.

    Returns
    -------
    Tensor, shape (prod(dims), prod(dims))
        Full-system Hamiltonian contribution.

    Notes
    -----
    Zero matrix elements are skipped to avoid unnecessary tensor products.
    """
    shape = prod([len(a) for a in A])
    H = null(shape)
    a, b = mode
    nA, nB = M.shape
    for i in range(nA):
        for j in range(nB):
            if not M[i, j] == 0:
                H += M[i, j] * modeProduct(A, i + a, B, j + b)
    return H


def unitaryTransformation(M, U):
    """Apply unitary transformation *M → U†MU*.

    Parameters
    ----------
    M : Tensor
        Operator to transform.
    U : Tensor
        Unitary transformation matrix.

    Returns
    -------
    Tensor
        Transformed operator *U†MU*.
    """
    return U.conj().T @ M @ U


def mul(A, B):
    """Matrix product *A @ B*."""
    return A @ B


# ---------------------------------------------------------------------------
# Elementary operators
# ---------------------------------------------------------------------------

def identity(n, dtype=float, device=None):
    """Return the *n × n* identity matrix.

    Parameters
    ----------
    n : int
        Matrix dimension.
    dtype : torch.dtype, optional
        Default ``torch.float64``.
    device : str or torch.device, optional

    Returns
    -------
    Tensor, shape (n, n)
    """
    return eye(n, dtype=dtype, device=device)


def null(N=1, dtype=complex, device=None):
    """Return the *N × N* zero matrix (complex dtype by default).

    Parameters
    ----------
    N : int, optional
        Matrix dimension.  Default ``1``.
    dtype : torch.dtype, optional
        Default ``torch.cdouble``.
    device : str or torch.device, optional

    Returns
    -------
    Tensor, shape (N, N)
    """
    return zeros(N, N, dtype=complex, device=device)


# ---------------------------------------------------------------------------
# Hilbert-space grids
# ---------------------------------------------------------------------------

def chargeStates(n, dtype=int, device=None):
    """Return a 1-D tensor of charge eigenvalues for truncation *n*.

    The charge grid spans *{-n, -n+1, …, n-1, n}* in descending order
    (matching the convention used for the diagonal charge operator *Q*).

    Parameters
    ----------
    n : int
        Charge-basis truncation.  Hilbert-space dimension = *2n + 1*.

    Returns
    -------
    Tensor, shape (2n+1,)
        Integer Cooper-pair numbers from *+n* down to *-n*.
    """
    return linspace(n, -n, 2 * n + 1, dtype=dtype, device=None)


def fluxStates(N, n=1, dtype=float, device=None):
    """Return a 1-D tensor of flux-basis grid points on *(-n, n]*.

    Parameters
    ----------
    N : int
        Number of grid points.
    n : float, optional
        Half-extent of the flux domain.  Default ``1`` (i.e., one flux quantum).

    Returns
    -------
    Tensor, shape (N,)
        Uniformly spaced flux values, *endpoint excluded* from the negative
        side (open left, closed right — matching periodicity of the Fourier
        transform).
    """
    return linspace(n, -n, N + 1, dtype=dtype, device=device)[1:]


def transformationMatrix(n_charge, n_flux=1, device=None):
    """Discrete Fourier transform between charge and flux bases.

    Constructs the unitary matrix *T* such that::

        |φ_k⟩ = Σ_n  T[k,n] |n⟩      (charge → flux)

    where *T[k,n] = exp(2πi φ_k n) / N* with *φ_k* uniformly spaced on
    *(-1/2, 1/2]* and *n* running over Cooper-pair numbers.

    Parameters
    ----------
    n_charge : int
        Charge-basis truncation.  Sets *N = 2n_charge + 1*.
    n_flux : float, optional
        Flux domain half-extent.  Default ``1``.
    device : str or torch.device, optional

    Returns
    -------
    Tensor, shape (N, N), dtype cdouble
        Unitary Fourier transformation matrix.
    """
    charge_states = chargeStates(n_charge, complex, device)
    N_flux        = 2 * n_charge + 1
    flux_states   = fluxStates(N_flux, n_flux, complex, device) / 2 / n_flux
    # flux points are normalized to (-0.5, 0.5] for the DFT
    T  = outer(flux_states, charge_states)
    T *= 2 * pi * im
    T  = exp(T) / N_flux
    return T


# ---------------------------------------------------------------------------
# Oscillator basis operators
# ---------------------------------------------------------------------------

def basisQo(n, impedance, device=None):
    """Charge (momentum) operator in the oscillator (Fock) basis.

    *Q = -i √(1/2πZ) (a - a†)*  where *a* is the ladder operator.

    Parameters
    ----------
    n : int
        Hilbert-space truncation (number of Fock states).
    impedance : Tensor or float
        Mode impedance *Z = √(C⁻¹/L)* in natural units.

    Returns
    -------
    Tensor, shape (n, n), dtype cdouble
        Anti-Hermitian charge operator (imaginary off-diagonal matrix).
    """
    Qo = arange(1, n, device=device)
    Qo = sqrt(Qo)
    Qo = -diag(Qo, diagonal=1) + diag(Qo, diagonal=-1)
    return Qo * im * sqrt(1 / 2 / pi / impedance)


def basisFo(n, impedance, device=None):
    """Flux (position) operator in the oscillator (Fock) basis.

    *F = √(Z/2π) (a + a†)*

    Parameters
    ----------
    n : int
        Hilbert-space truncation (number of Fock states).
    impedance : Tensor or float
        Mode impedance *Z* in natural units.

    Returns
    -------
    Tensor, shape (n, n), dtype cdouble
        Hermitian flux operator (real symmetric tridiagonal matrix).
    """
    Fo = arange(1, n, device=device)
    Fo = sqrt(Fo)
    Fo = diag(Fo, diagonal=1) + diag(Fo, diagonal=-1)
    return Fo.to(dtype=complex) * sqrt(impedance / 2 / pi)


# ---------------------------------------------------------------------------
# Charge (canonical) basis operators
# ---------------------------------------------------------------------------

def basisQq(n, device=None):
    """Charge operator *Q* in the charge (Cooper-pair) basis.

    *Q = diag(2n, 2(n-1), …, -2n)* — diagonal in the number basis.

    Parameters
    ----------
    n : int
        Charge truncation.  Dimension = *2n + 1*.

    Returns
    -------
    Tensor, shape (2n+1, 2n+1), dtype cdouble
        Diagonal charge operator (units: Cooper pairs, factor of 2 for charge
        in units of the electron charge).
    """
    charge = chargeStates(n, complex, device)
    Q = diag(charge.clone().detach())
    return Q * 2


def basisFq(n, device=None):
    """Flux (phase) operator in the charge basis, via Fourier transform.

    Constructs the flux operator by conjugating the charge representation of
    the diagonal flux operator with the DFT matrix *T*::

        F_charge = T · F_flux · T†

    Parameters
    ----------
    n : int
        Charge truncation.

    Returns
    -------
    Tensor, shape (2n+1, 2n+1), dtype cdouble
        Flux operator in the charge basis.
    """
    Q = basisQq(n, device)
    U = transformationMatrix(n, device=device)
    return U @ Q @ U.conj().T / 2


def basisFf(N, n=1, device=None):
    """Flux operator in the flux basis — diagonal in flux eigenvalues.

    Parameters
    ----------
    N : int
        Grid size (number of flux states).
    n : float, optional
        Half-extent of the flux domain.  Default ``1``.

    Returns
    -------
    Tensor, shape (N, N), dtype cdouble
        Diagonal flux operator.
    """
    flux = fluxStates(N, n, dtype=complex, device=device) / 2 / n # periodicity bound
    return diag(flux)


def basisQf(n, N=1, device=None):
    """Charge operator in the flux basis, via Fourier transform.

    Parameters
    ----------
    n : int
        Charge truncation (sets dimension *2n+1*).
    N : float, optional
        Flux domain half-extent.  Default ``1``.

    Returns
    -------
    Tensor, shape (2n+1, 2n+1), dtype cdouble
        Charge operator in the flux basis.
    """
    F = basisFf(2 * n + 1, N)
    U = transformationMatrix(n, N, device=device)
    return U @ F @ U.conj().T


# ---------------------------------------------------------------------------
# Oscillator-basis diagonalization (for Kerman mode decomposition)
# ---------------------------------------------------------------------------

def fluxFlux(n, impedance):
    """Flux operator diagonalized in the flux eigenbasis.

    Diagonalizes the oscillator flux operator *Fo* and returns *Fo* in its
    own eigenbasis (diagonal matrix of eigenvalues).

    Parameters
    ----------
    n : int
        Truncation (Hilbert-space dimension = *2n+1*).
    impedance : float or Tensor
        Mode impedance.

    Returns
    -------
    Tensor
        Diagonal flux operator in the flux eigenbasis.
    """
    N  = 2 * n + 1
    Po = basisFo(N, impedance)
    D  = diagonalisation(Po)
    return unitaryTransformation(Po, D)


def chargeFlux(n, impedance):
    """Charge operator in the flux eigenbasis.

    Rotates the oscillator charge operator *Qo* into the basis that
    diagonalizes the oscillator flux operator *Fo*.

    Parameters
    ----------
    n : int
        Truncation.
    impedance : float or Tensor
        Mode impedance.

    Returns
    -------
    Tensor
        Charge operator in the flux eigenbasis.
    """
    N  = 2 * n + 1
    Po = basisFo(N, impedance)
    Qo = basisQo(N, impedance)
    D  = diagonalisation(Po)
    return unitaryTransformation(Qo, D)


def chargeCharge(n, impedance):
    """Charge operator diagonalized in the charge eigenbasis.

    Parameters
    ----------
    n : int
        Truncation.
    impedance : float or Tensor
        Mode impedance.

    Returns
    -------
    Tensor
        Diagonal charge operator in the charge eigenbasis.
    """
    N  = 2 * n + 1
    Qo = basisQo(N, impedance)
    D  = diagonalisation(Qo)
    return unitaryTransformation(Qo, D)


def fluxCharge(n, impedance):
    """Flux operator in the charge eigenbasis.

    Parameters
    ----------
    n : int
        Truncation.
    impedance : float or Tensor
        Mode impedance.

    Returns
    -------
    Tensor
        Flux operator in the charge eigenbasis.
    """
    N  = 2 * n + 1
    Po = basisFo(N, impedance)
    Qo = basisQo(N, impedance)
    D  = diagonalisation(Qo)
    return unitaryTransformation(Po, D)


# ---------------------------------------------------------------------------
# Josephson displacement operators (charge basis)
# ---------------------------------------------------------------------------

def chargeDisplacePlus(n, device=None):
    """Cooper-pair lowering operator *e^{-iφ}* in the charge basis.

    Shifts the charge state by *-1* Cooper pair::

        D⁺|n⟩ = |n-1⟩

    (sub-diagonal shift matrix).

    Parameters
    ----------
    n : int
        Charge truncation.  Dimension = *2n + 1*.

    Returns
    -------
    Tensor, shape (2n+1, 2n+1), dtype cdouble
        Sub-diagonal displacement operator.
    """
    diagonal = ones((2 * n + 1) - 1, dtype=complex, device=device)
    return diag(diagonal, diagonal=-1)


def chargeDisplaceMinus(n, device=None):
    """Cooper-pair raising operator *e^{+iφ}* in the charge basis.

    Shifts the charge state by *+1* Cooper pair::

        D⁻|n⟩ = |n+1⟩

    (super-diagonal shift matrix).

    Parameters
    ----------
    n : int
        Charge truncation.  Dimension = *2n + 1*.

    Returns
    -------
    Tensor, shape (2n+1, 2n+1), dtype cdouble
        Super-diagonal displacement operator.
    """
    diagonal = ones((2 * n + 1) - 1, dtype=complex, device=device)
    return diag(diagonal, diagonal=1)


def displacementCharge(n, a, device=None):
    """Generalized flux displacement *e^{i·2π·a·F}* in the charge basis.

    Used for Josephson junctions in circuits with non-trivial loop structure
    (Kerala representation).

    Parameters
    ----------
    n : int
        Charge truncation.
    a : float or Tensor
        Displacement amplitude (in units of flux quantum fraction).

    Returns
    -------
    Tensor, shape (2n+1, 2n+1), dtype cdouble
        Unitary displacement operator.
    """
    D = basisFq(n, device)
    return expm(im * 2 * pi * a * D)


def displacementOscillator(n, z, a, device=None):
    """Generalized flux displacement *e^{i·2π·a·F}* in the oscillator basis.

    Parameters
    ----------
    n : int
        Oscillator truncation (number of Fock states).
    z : float or Tensor
        Mode impedance *Z* in natural units.
    a : float or Tensor
        Displacement amplitude.

    Returns
    -------
    Tensor, shape (n, n), dtype cdouble
        Unitary displacement operator.
    """
    D = basisFo(n, z, device)
    return expm(im * 2 * pi * a * D)


if __name__ == '__main__':
    Qo = basisQo(30, tensor(4.0))
    Fq = basisFq(30)
    print('Fq shape:', Fq.shape)
    Qf = basisQf(30, 4)
    print('Qf shape:', Qf.shape)
