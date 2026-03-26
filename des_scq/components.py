"""
des_scq.components
==================
Lumped-element circuit components and unit-conversion utilities.

Unit system
-----------
The library uses **energy units (GHz)** throughout — all energies are expressed
as frequencies, i.e., divided by Planck's constant *h*.  The chain of
conversions is::

    SI  ──► natural  ──► energy (GHz)

Helper functions for each direction are provided below.  Physical constants
are defined at module scope in SI units.

Reparametrization
-----------------
Every element parameter is stored internally as an *unconstrained* real tensor
``base`` that maps to the physical (bounded) energy range via a sigmoid
transform::

    energy = sigmoid(base) * scale + offset

This keeps gradient-based optimization numerically stable and automatically
enforces parameter bounds without projected-gradient methods.  The inverse
transform ``sigmoidInverse`` is used when initializing an element from a
given energy value.

Classes
-------
Parameters   Abstract base (ID, device, dtype, requires_grad).
Control      Scalar control parameter with configurable bounds.
Elements     Abstract base for two-terminal lumped elements.
J            Josephson junction  —  characterized by Ej (GHz).
C            Capacitor           —  characterized by Ec (GHz).
L            Inductor            —  characterized by El (GHz).
"""

import uuid
from numpy import log, sqrt as sqroot, pi, prod, clip
from torch import tensor, norm, abs, ones, zeros, zeros_like, argsort
from torch import linspace, arange, diagonal, diag, sqrt, eye
from torch.linalg import det, inv, eig as eigsolve, norm
from torch import matrix_exp as expm, exp, outer
from torch import sin, cos, sigmoid, clamp
from torch import cdouble as complex, float64 as float  # default precision

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------

im = 1.0j
root2 = sqroot(2)

e           = 1.60217662e-19    # electron charge [C]
h           = 6.62607004e-34    # Planck's constant [J·s]
hbar        = h / 2 / pi        # reduced Planck's constant [J·s]
flux_quanta = h / 2 / e         # magnetic flux quantum Φ₀ [Wb]
Z0          = flux_quanta / 2 / e  # impedance quantum [Ω]

zero = 1e-12   # numerical zero — lower bound guard for log/sigmoid
inf  = 1e12    # numerical infinity — upper bound guard

# ---------------------------------------------------------------------------
# Default bounds for element energy parameters (GHz)
#
#   energy ∈ (X_, X0 + X_)  (bounds are non-inclusive)
#
#   J0, C0, L0 : scale of sigmoid range
#   J_, C_, L_ : non-zero lower bound (necessary for logarithmic
#                parameterization and physical meaningfulness)
# ---------------------------------------------------------------------------

J0, C0, L0   = 1200, 2500, 1200
J_,  C_,  L_ = 1e-6, 1e-6, 1e-6

A0 = 1e10   # scale for Control parameters
A_ = 1e3    # lower bound for Control parameters


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

def capSINat(cap):
    """Convert capacitance from SI [F] to natural units [e²/h·10⁹]."""
    return cap / (e**2 / h / 1e9)


def capNatSI(cap):
    """Convert capacitance from natural units [e²/h·10⁹] to SI [F]."""
    return (e**2 / 1e9 / h) * cap


def indSINat(ind):
    """Convert inductance from SI [H] to natural units [Φ₀²/h·10⁹]."""
    return ind / (flux_quanta**2 / h / 1e9)


def indNatSI(ind):
    """Convert inductance from natural units [Φ₀²/h·10⁹] to SI [H]."""
    return ind * (flux_quanta**2 / h / 1e9)

# conversion Natural and Energy units

def capEnergy(cap):
    """Convert capacitance from natural units to charging energy [GHz].

    Uses *Ec = 1 / 2C* in natural units.

    Parameters
    ----------
    cap : float or Tensor
        Capacitance in natural units (h·10⁹/e²).

    Returns
    -------
    float or Tensor
        Charging energy *Ec* in GHz.
    """
    return 1.0 / 2 / cap


def indEnergy(ind):
    """Convert inductance from natural units to inductive energy [GHz].

    Uses *El = 1 / 4π²L* in natural units.

    Parameters
    ----------
    ind : float or Tensor
        Inductance in natural units (h·10⁹/Φ₀²).

    Returns
    -------
    float or Tensor
        Inductive energy *El* in GHz.
    """
    return 1.0 / 4 / pi**2 / ind

# conversion SI to Energy units 

def capE(cap):
    """Convert capacitance from SI [F] to charging energy [GHz].

    Parameters
    ----------
    cap : float
        Capacitance in Farads.

    Returns
    -------
    float
        Charging energy *Ec = e²/2Ch* in GHz.

    Example
    -------
    >>> from des_scq.components import capE
    >>> Ec = capE(45e-15)   # 45 fF → ~0.43 GHz
    """
    return 1 / 2 / cap * e * e / h / 1e9


def indE(ind):
    """Convert inductance from SI [H] to inductive energy [GHz].

    Parameters
    ----------
    ind : float
        Inductance in Henrys.

    Returns
    -------
    float
        Inductive energy *El = Φ₀²/4π²Lh* in GHz.

    Example
    -------
    >>> from des_scq.components import indE
    >>> El = indE(10e-9)    # 10 nH → GHz
    """
    return 1 / 4 / pi**2 / ind * flux_quanta**2 / h / 1e9


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def sigmoidInverse(x):
    """Inverse sigmoid (logit), clipped to avoid log(0).

    Used to initialize element ``base`` tensors from a given energy value::

        base = sigmoidInverse((energy - offset) / scale)

    Parameters
    ----------
    x : float
        Value in (0, 1) representing the normalized energy position.

    Returns
    -------
    float
        Unconstrained logit value.
    """
    x = 1 / x - 1
    x = clip(x, zero, inf)
    return -log(x)


def normalize(state, square=True):
    """Normalize a quantum state vector.

    Parameters
    ----------
    state : Tensor
        Raw state vector (may be complex).
    square : bool, optional
        If ``True`` (default), return |ψ|² (probability distribution).
        If ``False``, return the normalized amplitude |ψ|/‖|ψ|‖.

    Returns
    -------
    Tensor
        Normalized state.
    """
    state = abs(state)
    norm_state = norm(state)
    state = state / norm_state
    if square:
        state = abs(state) ** 2
    return state


def diagonalisation(M, reverse=False):
    """Diagonalize a matrix and return its eigenvector matrix sorted by
    eigenvalue.

    Parameters
    ----------
    M : Tensor, shape (N, N)
        Square matrix to diagonalize.
    reverse : bool, optional
        If ``True``, sort eigenvalues in descending order (negates before
        sorting, then the eigenvectors correspond to the reversed spectrum).
        Default is ``False`` (ascending).

    Returns
    -------
    D : Tensor, shape (N, N)
        Matrix whose columns are the eigenvectors of *M*, sorted by the real
        part of the corresponding eigenvalue.
    """
    eig, vec = eigsolve(M)
    if reverse:
        eig = -eig
    indices = argsort(eig.real)
    D = vec[:, indices].clone().detach()
    return D


# ---------------------------------------------------------------------------
# Component base classes
# ---------------------------------------------------------------------------

class Parameters:
    """Abstract base class for all circuit parameters.

    Stores the ID, device placement, dtype, and ``requires_grad`` flag shared
    by all elements.  The ``base`` attribute holds the unconstrained sigmoid
    pre-image of the physical parameter value.

    Parameters
    ----------
    ID : str, optional
        Human-readable identifier (e.g., ``'J1'``, ``'Cx'``).  If ``None``,
        a random UUID hex string is generated automatically.
    device : str or torch.device, optional
        Target device for tensors (e.g., ``'cpu'``, ``'cuda'``).
    dtype : torch.dtype, optional
        Floating-point precision.  Defaults to ``torch.float64``.
    requires_grad : bool, optional
        Whether to track gradients for this parameter.  Default ``True``.
    """

    def __init__(self, ID=None, device=None, dtype=float, requires_grad=True):
        if ID is None:
            ID = uuid.uuid4().hex
        self.ID            = ID
        self.device        = device
        self.dtype         = dtype
        self.requires_grad = requires_grad
        self.base          = None

    def variable(self):
        """Return the raw unconstrained ``base`` tensor."""
        return self.base


class Control(Parameters):
    """Scalar external control parameter with a bounded energy range.

    The physical value is::

        A = sigmoid(base) * A0 + A_

    so *A* lives in the open interval *(A_, A0 + A_)*.

    Parameters
    ----------
    A : float
        Initial value of the control parameter.
    ID : str, optional
        Identifier string.
    A0 : float, optional
        Sigmoid scale.  Default ``1e10``.
    A_ : float, optional
        Lower bound (offset).  Default ``1e3``.
    device, dtype, requires_grad
        Passed to :class:`Parameters`.
    """

    def __init__(self, A, ID=None, A0=A0, A_=A_, device=None, dtype=float,
                 requires_grad=True):
        super().__init__(ID, device, dtype, requires_grad)
        self.A0 = A0
        self.A_ = A_
        self.initControl(A)

    def initControl(self, A):
        """Re-initialize the control from a new value *A*."""
        self.base = tensor(sigmoidInverse((A - self.A_) / self.A0),
                           device=self.device, dtype=self.dtype,
                           requires_grad=self.requires_grad)

    def energy(self):
        """Return the current control value (in native units)."""
        return sigmoid(self.base) * self.A0 + self.A_

    def bounds(self):
        """Return (lower, upper) bound tensors."""
        return tensor(self.A_), tensor(self.A0 + self.A_)


class Elements(Parameters):
    """Abstract base for two-terminal circuit elements.

    Parameters
    ----------
    plus : int
        Node index of the positive terminal.
    minus : int
        Node index of the negative terminal.
    ID, device, dtype, requires_grad
        Passed to :class:`Parameters`.
    """

    def __init__(self, plus, minus, ID=None, device=None, dtype=float,
                 requires_grad=True):
        super().__init__(ID, device, dtype, requires_grad)
        self.plus  = plus
        self.minus = minus


# ---------------------------------------------------------------------------
# Concrete element types
# ---------------------------------------------------------------------------

class J(Elements):
    """Josephson junction.

    The Josephson energy *Ej* lives in *(J_, J0 + J_)* GHz::

        Ej = sigmoid(base) * J0 + J_

    Parameters
    ----------
    plus : int
        Positive node index.
    minus : int
        Negative node index.
    Ej : float
        Josephson energy in GHz.
    ID : str, optional
        Element identifier (e.g., ``'Jx'``).
    J0 : float, optional
        Sigmoid scale.  Default ``1200`` GHz.
    J_ : float, optional
        Lower bound.  Default ``1e-6`` GHz.
    device, dtype, requires_grad
        Passed to :class:`Elements`.

    Examples
    --------
    >>> from des_scq.components import J
    >>> junc = J(plus=0, minus=1, Ej=10.0, ID='J1')
    >>> junc.energy()       # returns a Tensor ≈ 10.0 GHz
    """

    def __init__(self, plus, minus, Ej, ID=None, J0=J0, J_=J_,
                 device=None, dtype=float, requires_grad=True):
        super().__init__(plus, minus, ID, device, dtype, requires_grad)
        self.J0 = J0
        self.J_ = J_
        self.initJunc(Ej)

    def initJunc(self, Ej):
        """Re-initialize from a new Josephson energy *Ej* (GHz)."""
        self.base = tensor(sigmoidInverse((Ej - self.J_) / self.J0),
                           device=self.device, dtype=self.dtype,
                           requires_grad=self.requires_grad)

    def energy(self):
        """Return the current Josephson energy *Ej* in GHz."""
        return sigmoid(self.base) * self.J0 + self.J_

    def bounds(self):
        """Return (lower, upper) bound tensors in GHz."""
        return tensor(self.J_), tensor(self.J0 + self.J_)


class C(Elements):
    """Capacitor described by its charging energy *Ec*.

    *Ec* lives in *(C_, C0 + C_)* GHz::

        Ec = sigmoid(base) * C0 + C_

    Parameters
    ----------
    plus : int
        Positive node index.
    minus : int
        Negative node index.
    Ec : float
        Charging energy *e²/2C·h* in GHz.
    ID : str, optional
        Element identifier (e.g., ``'Cx'``).
    C0 : float, optional
        Sigmoid scale.  Default ``2500`` GHz.
    C_ : float, optional
        Lower bound.  Default ``1e-6`` GHz.
    device, dtype, requires_grad
        Passed to :class:`Elements`.

    Notes
    -----
    To convert from SI Farads: ``Ec = capE(C_farads)``.

    Examples
    --------
    >>> from des_scq.components import C, capE
    >>> cap = C(plus=0, minus=1, Ec=capE(45e-15), ID='C1')
    """

    def __init__(self, plus, minus, Ec, ID=None, C0=C0, C_=C_,
                 device=None, dtype=float, requires_grad=True):
        super().__init__(plus, minus, ID, device, dtype, requires_grad)
        self.C0 = C0
        self.C_ = C_
        self.initCap(Ec)

    def initCap(self, Ec):
        """Re-initialize from a new charging energy *Ec* (GHz)."""
        self.base = tensor(sigmoidInverse((Ec - self.C_) / self.C0),
                           device=self.device, dtype=self.dtype,
                           requires_grad=self.requires_grad)

    def energy(self):
        """Return the current charging energy *Ec* in GHz."""
        return sigmoid(self.base) * self.C0 + self.C_

    def capacitance(self):
        """Return the capacitance in natural units (h·10⁹/e²).

        Computed as *C = 1/2Ec* from the stored *Ec*.
        """
        return capEnergy(self.energy())

    def bounds(self):
        """Return (lower, upper) bound tensors in GHz."""
        return tensor(self.C_), tensor(self.C0 + self.C_)


class L(Elements):
    """Inductor described by its inductive energy *El*.

    *El* lives in *(L_, L0 + L_)* GHz::

        El = sigmoid(base) * L0 + L_

    Parameters
    ----------
    plus : int
        Positive node index.
    minus : int
        Negative node index.
    El : float
        Inductive energy *Φ₀²/4π²L·h* in GHz.
    ID : str, optional
        Element identifier (e.g., ``'Lx'``).
    external : bool, optional
        If ``True``, this inductor is used as an external flux-bias line
        rather than a circuit element.  External inductors contribute loop
        flux but are excluded from gradient updates by convention.
        Default ``False``.
    L0 : float, optional
        Sigmoid scale.  Default ``1200`` GHz.
    L_ : float, optional
        Lower bound.  Default ``1e-6`` GHz.
    device, dtype, requires_grad
        Passed to :class:`Elements`.

    Notes
    -----
    To convert from SI Henrys: ``El = indE(L_henrys)``.
    """

    def __init__(self, plus, minus, El, ID=None, external=False,
                 L0=L0, L_=L_, device=None, dtype=float, requires_grad=True):
        super().__init__(plus, minus, ID, device, dtype, requires_grad)
        self.L0       = L0
        self.L_       = L_
        self.external = external
        self.initInd(El)

    def initInd(self, El):
        """Re-initialize from a new inductive energy *El* (GHz)."""
        self.base = tensor(sigmoidInverse((El - self.L_) / self.L0),
                           device=self.device, dtype=self.dtype,
                           requires_grad=self.requires_grad)

    def energy(self):
        """Return the current inductive energy *El* in GHz."""
        return sigmoid(self.base) * self.L0 + self.L_

    def inductance(self):
        """Return the inductance in natural units (h·10⁹/Φ₀²).

        Computed as *L = 1/4π²El* from the stored *El*.
        """
        return indEnergy(self.energy())

    def bounds(self):
        """Return (lower, upper) bound tensors in GHz."""
        return tensor(self.L_), tensor(self.L0 + self.L_)


if __name__ == '__main__':
    # Sanity check: capEnergy is its own inverse
    print(capEnergy(capEnergy(10)))   # should print ≈ 10
