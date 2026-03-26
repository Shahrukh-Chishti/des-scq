"""
des_scq.models
==============
Pre-built constructors for standard superconducting qubit circuits.

Each function instantiates a circuit representation class (``Charge`` or
``Kerman``) with the appropriate network topology and default component
energies.  All energy parameters are in **GHz**.

Usage pattern
-------------
Every constructor follows the same calling convention::

    circuit = model_name(Rep, basis, <energy_params>, ...)

where

* ``Rep``   — the basis class: ``Charge`` or ``Kerman``
  (imported from ``des_scq.circuit``).
* ``basis`` — Hilbert-space truncation.  A flat list for ``Charge``,
  a dict ``{'O': [...], 'I': [...], 'J': [...]}`` for ``Kerman``.
* Energy parameters — circuit element energies in GHz (see individual
  docstrings for defaults and units).

Symmetry enforcement
--------------------
All constructors accept an optional ``symmetry`` flag.  When ``True``, the
``pairs`` dict is populated so that geometrically equivalent components share
the same trainable tensor.  This halves (or thirds) the optimizable parameter
count and keeps the circuit at its design symmetry point during optimization.

Available models
----------------
transmon          Single-junction transmon qubit.
splitTransmon     Two-junction split transmon with flux tunability.
fluxonium         Fluxonium qubit (junction shunted by superinductance).
fluxoniumArray    Fluxonium with a Josephson-junction array superinductance.
zeroPi            Zero-pi qubit.
prismon           Prismon qubit (three-junction ring).
shuntedQubit      C-shunted three-junction qubit with external flux bias.
fluxShunted       Flux-shunted three-junction qubit.
oscillatorLC      Simple LC oscillator.
box4Branches      Four-branch box circuit.
phaseSlip         Multi-branch phase-slip circuit.
"""

from des_scq.components import J, C, L, sigmoidInverse
from des_scq.components import C0, J0, L0, C_, J_, L_
from torch import tensor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def tensorize(values, variable=True):
    """Convert a list of floats to a list of PyTorch tensors.

    Parameters
    ----------
    values : list of float
    variable : bool, optional
        Whether to set ``requires_grad=True``.  Default ``True``.

    Returns
    -------
    list of Tensor
    """
    return [tensor(val, requires_grad=variable) for val in values]


def sigInv(sig, limit):
    """Apply sigmoid inverse to a list of normalized values.

    Parameters
    ----------
    sig : list of float
        Values in *(0, 1)* representing normalized parameter positions.
    limit : float
        Upper bound of the parameter range.

    Returns
    -------
    list of float
    """
    return [sigmoidInverse(s / limit) for s in sig]


# ---------------------------------------------------------------------------
# Circuit constructors
# ---------------------------------------------------------------------------

def transmon(Rep, basis, Ej=10., Ec=0.3):
    """Single-junction transmon qubit.

    Topology: one Josephson junction and one shunt capacitor between nodes
    0 and 1.  No external flux control.

    Parameters
    ----------
    Rep : class
        Basis representation class (``Charge`` or ``Kerman``).
    basis : list
        Hilbert-space truncation, e.g., ``[256]`` for Charge basis.
    Ej : float, optional
        Josephson energy in GHz.  Default ``10.0``.
    Ec : float, optional
        Charging energy *e²/2C·h* in GHz.  Default ``0.3``.

    Returns
    -------
    Circuit instance

    Example
    -------
    >>> from des_scq.circuit import Charge
    >>> circuit = transmon(Charge, [256], Ej=30., Ec=0.3)
    >>> circuit.circuitComponents()
    """
    transmon_net = [J(0, 1, Ej, 'J'),
                    C(0, 1, Ec, 'C')]
    control_iD   = ()
    return Rep(transmon_net, control_iD, basis)


def splitTransmon(Rep, basis):
    """Two-junction split transmon with a flux-tunable loop.

    Topology: two junctions in a loop bridged by a flux-bias inductor.
    The external flux tunes the effective Josephson energy::

        0 --J-- 1 --L(ext)-- 2 --J-- 0

    with capacitors across each junction.

    Parameters
    ----------
    Rep : class
        Basis representation class.
    basis : list or dict
        Hilbert-space truncation.

    Returns
    -------
    Circuit instance

    Notes
    -----
    Control channel: ``'I'`` (external flux, string ID).
    """
    transmon_net = [J(0, 1, 10.),  C(0, 1, 100.),
                    L(1, 2, .0003, 'I', external=True),
                    J(2, 0, 10.),  C(2, 0, 100.)]
    control_iD   = ('I',)
    return Rep(transmon_net, control_iD, basis)


def fluxonium(Rep, basis, El=.0003, Ec=.1, Ej=20.):
    """Fluxonium qubit.

    Topology: a Josephson junction shunted by a large linear superinductance
    and a capacitor::

        0 --J(Ej)-- 1
        0 --L(El)-- 1    (external flux bias)
        0 --C(Ec)-- 1

    Parameters
    ----------
    Rep : class
        Basis representation class.
    basis : list or dict
        Hilbert-space truncation.  For oscillator-dominant regime, use
        Kerman with ``basis = {'O': [64], 'I': [], 'J': []}``.
    El : float, optional
        Inductive energy in GHz.  Default ``0.0003``.
    Ec : float, optional
        Charging energy in GHz.  Default ``0.1``.
    Ej : float, optional
        Josephson energy in GHz.  Default ``20.0``.

    Returns
    -------
    Circuit instance

    Notes
    -----
    Control channel: ``'I'`` (external flux, string ID).

    Example
    -------
    >>> from des_scq.circuit import Kerman
    >>> basis   = {'O': [64], 'I': [], 'J': []}
    >>> circuit = fluxonium(Kerman, basis, El=0.1, Ec=1., Ej=50.)
    """
    circuit    = [C(0, 1, Ec, 'Cap'),
                  J(0, 1, Ej, 'JJ'),
                  L(0, 1, El, 'I',  external=True)]
    control_iD = ('I',)
    return Rep(circuit, control_iD, basis)


def fluxoniumArray(Rep, basis, shunt=None, gamma=1.5, N=0, Ec=100., Ej=150.):
    """Fluxonium with a Josephson-junction array as the superinductance.

    The array consists of *N+1* large junctions (energy *γ · Ej*) connecting
    islands in series, replacing the linear inductor of a standard fluxonium.

    Parameters
    ----------
    Rep : class
        Basis representation class.
    basis : list or dict
        Hilbert-space truncation.
    shunt : float, optional
        Shunt capacitor energy across each array junction (GHz).  If ``None``,
        defaults to ``Ec / gamma``.
    gamma : float, optional
        Junction energy ratio for array junctions relative to *Ej*.
        Default ``1.5``.
    N : int, optional
        Number of additional islands in the array.  ``N=0`` gives a standard
        SQUID-like two-junction circuit.  Default ``0``.
    Ec : float, optional
        Charging energy of the main capacitor in GHz.  Default ``100``.
    Ej : float, optional
        Josephson energy of the main junction in GHz.  Default ``150``.

    Returns
    -------
    Circuit instance
    """
    circuit = [C(0, 1, Ec, 'Cap'),
               J(0, 1, Ej, 'Junc')]
    if shunt is None:
        shunt = Ec / gamma
    for i in range(N):
        circuit += [J(1 + i, 2 + i, gamma * Ej, 'junc' + str(i)),
                    C(1 + i, 2 + i, shunt,       'cap'  + str(i), C_=0.)]
    circuit += [J(1 + N, 0, gamma * Ej, 'junc' + str(N)),
                C(1 + N, 0, shunt,      'cap'  + str(N), C_=0.)]
    control_iD = ()
    return Rep(circuit, control_iD, basis)


def zeroPi(Rep, basis, Ej=10., Ec=50., El=10., EcJ=100., symmetry=False,
           _L_=(L_, L0), _C_=(C_, C0), _J_=(J_, J0), _CJ_=(4 * C_, 4 * C0),
           device=None):
    """Zero-pi qubit.

    A four-node circuit with two junctions, two inductors, and four
    capacitors arranged so that the ground-state manifold is doubly
    degenerate and protected from both charge and flux noise::

        0 --L(Lx)-- 1 --J(Jx)-- 3
        |            |            |
        C(Cy)       C(Cx)       C(CJx)
        |            |            |
        3 --L(Ly)-- 2 --J(Jy)-- 0

    Parameters
    ----------
    Rep : class
        Basis representation class.
    basis : list or dict
        Hilbert-space truncation.
    Ej : float, optional
        Josephson energy in GHz.  Default ``10.0``.
    Ec : float, optional
        Charging energy of shunt capacitors in GHz.  Default ``50.0``.
    El : float, optional
        Inductive energy in GHz.  Default ``10.0``.
    EcJ : float, optional
        Charging energy of junction self-capacitances in GHz.
        Default ``100.0``.
    symmetry : bool, optional
        If ``True``, enforce *Lx = Ly*, *Cx = Cy*, *Jx = Jy*, *CJx = CJy*.
        Default ``False``.
    _L_, _C_, _J_, _CJ_ : tuple (lower_bound, scale), optional
        Override default sigmoid bounds for each element type.
    device : str or torch.device, optional

    Returns
    -------
    Circuit instance

    Notes
    -----
    Control channels: ``'Lx'``, ``'Ly'`` (external flux on both inductors).

    Example
    -------
    >>> from des_scq.circuit import Kerman
    >>> basis   = {'O': [32], 'I': [], 'J': [8, 8]}
    >>> circuit = zeroPi(Kerman, basis, Ej=10., Ec=50., El=0.5, symmetry=True)
    """
    circuit  = [L(0, 1, El, 'Lx',  external=True, L0=_L_[1], L_=_L_[0]),
                L(2, 3, El, 'Ly',  external=True, L0=_L_[1], L_=_L_[0])]
    circuit += [C(1, 2, Ec,  'Cx',  C0=_C_[1], C_=_C_[0]),
                C(3, 0, Ec,  'Cy',  C0=_C_[1], C_=_C_[0])]
    circuit += [J(1, 3, Ej,  'Jx',  J0=_J_[1], J_=_J_[0]),
                J(2, 0, Ej,  'Jy',  J0=_J_[1], J_=_J_[0])]
    circuit += [C(1, 3, EcJ, 'CJx', C0=_CJ_[1], C_=_CJ_[0]),
                C(2, 0, EcJ, 'CJy', C0=_CJ_[1], C_=_CJ_[0])]
    pairs = {}
    if symmetry:
        pairs = {'Ly': 'Lx', 'Cy': 'Cx', 'Jy': 'Jx', 'CJy': 'CJx'}
    control_iD = ('Lx', 'Ly')
    return Rep(circuit, control_iD, basis, pairs, device)


def prismon(Rep, basis, Ej=10., Ec=50., El=10., EcJ=100., symmetry=False,
            _L_=(L_, L0), _C_=(C_, C0), _J_=(J_, J0), _CJ_=(4 * C_, 4 * C0)):
    """Prismon qubit — three-junction ring with inductive shunts.

    A six-node circuit with three-fold topology: three inductors, three shunt
    capacitors, three junctions, and three junction self-capacitances.

    Parameters
    ----------
    Rep : class
        Basis representation class.
    basis : list or dict
        Hilbert-space truncation.
    Ej : float, optional
        Josephson energy in GHz.  Default ``10.0``.
    Ec : float, optional
        Shunt capacitor charging energy in GHz.  Default ``50.0``.
    El : float, optional
        Inductive energy in GHz.  Default ``10.0``.
    EcJ : float, optional
        Junction self-capacitance charging energy in GHz.  Default ``100.0``.
    symmetry : bool, optional
        If ``True``, enforce three-fold symmetry (*Ja = Jb = Jc*, etc.).
        Default ``False``.

    Returns
    -------
    Circuit instance

    Notes
    -----
    Control channels: ``'La'``, ``'Lb'``, ``'Lc'`` (three external flux lines).
    """
    circuit  = [L(0, 1, El, 'La', external=True, L0=_L_[1], L_=_L_[0]),
                C(0, 2, Ec, 'Ca', C0=_C_[1], C_=_C_[0]),
                J(1, 2, Ej, 'Ja', J0=_J_[1], J_=_J_[0]),
                C(1, 2, EcJ,'CJa',C0=_CJ_[1],C_=_CJ_[0])]
    circuit += [L(2, 3, El, 'Lb', external=True, L0=_L_[1], L_=_L_[0]),
                C(1, 5, Ec, 'Cb', C0=_C_[1], C_=_C_[0]),
                J(0, 4, Ej, 'Jb', J0=_J_[1], J_=_J_[0]),
                C(0, 4, EcJ,'CJb',C0=_CJ_[1],C_=_CJ_[0])]
    circuit += [L(5, 4, El, 'Lc', external=True, L0=_L_[1], L_=_L_[0]),
                C(4, 3, Ec, 'Cc', C0=_C_[1], C_=_C_[0]),
                J(3, 5, Ej, 'Jc', J0=_J_[1], J_=_J_[0]),
                C(3, 5, EcJ,'CJc',C0=_CJ_[1],C_=_CJ_[0])]
    pairs = {}
    if symmetry:
        pairs = {'Jb': 'Ja', 'Jc': 'Ja',
                 'Cb': 'Ca', 'Cc': 'Ca',
                 'Lb': 'La', 'Lc': 'La',
                 'CJb': 'CJa', 'CJc': 'CJa'}
    control_iD = ('La', 'Lb', 'Lc')
    return Rep(circuit, control_iD, basis, pairs)


def shuntedQubit(Rep, basis, josephson=[120., 50., 120.], cap=[10., 50., 10.],
                 ind=100., symmetry=False,
                 _C_=(C_, C0), _J_=(J_, J0)):
    """C-shunted three-junction qubit with external flux bias.

    Topology: three junctions in a loop with shunt capacitors and a single
    external-flux-bias inductor::

        1 --J1(C1)-- 2 --J2(C2)-- 3 --J3(C3)-- 0
        0 --L(I,ext)-- 1

    Parameters
    ----------
    Rep : class
        Basis representation class.
    basis : list or dict
        Hilbert-space truncation.
    josephson : list of float, optional
        ``[Ej1, Ej2, Ej3]`` Josephson energies in GHz.
        Default ``[120., 50., 120.]``.
    cap : list of float, optional
        ``[Ec1, Ec2, Ec3]`` charging energies in GHz.
        Default ``[10., 50., 10.]``.
    ind : float, optional
        Inductive energy *El* of the flux-bias inductor in GHz.
        Default ``100.0``.
    symmetry : bool, optional
        If ``True``, enforce *J3 = J1* and *C3 = C1*.  Default ``False``.
    _C_, _J_ : tuple, optional
        Override sigmoid bounds.

    Returns
    -------
    Circuit instance

    Notes
    -----
    Control channel: ``'I'`` (external flux, string ID).

    Example
    -------
    >>> from des_scq.circuit import Kerman
    >>> from des_scq.components import capE
    >>> C1 = capE(45e-15)
    >>> circuit = shuntedQubit(Kerman, {'O':[4],'J':[8,8],'I':[]},
    ...                        cap=[C1, capE(1e-15), C1])
    """
    Ej1, Ej2, Ej3 = josephson
    C1,  C2,  C3  = cap
    circuit  = [J(1, 2, Ej1, 'JJ1'), C(1, 2, C1, 'C1', _C_[1], _C_[0])]
    circuit += [J(2, 3, Ej2, 'JJ2'), C(2, 3, C2, 'C2', _C_[1], _C_[0])]
    circuit += [J(3, 0, Ej3, 'JJ3'), C(3, 0, C3, 'C3', _C_[1], _C_[0])]
    circuit += [L(0, 1, ind, 'I', external=True)]
    pairs = {}
    if symmetry:
        pairs = {'JJ3': 'JJ1', 'C3': 'C1'}
    return Rep(circuit, ['I'], basis, pairs)


def fluxShunted(Rep, basis, josephson=[120., 50., 120.], cap=[10., 50., 10.],
                symmetry=False, _C_=(C_, C0), _J_=(J_, J0)):
    """Flux-shunted three-junction qubit.

    Similar to ``shuntedQubit`` but without a linear shunt inductor — the
    three junctions form a closed loop with no external flux-bias line.

    Parameters
    ----------
    Rep : class
        Basis representation class.
    basis : list or dict
        Hilbert-space truncation.
    josephson : list of float, optional
        ``[Ej1, Ej2, Ej3]`` in GHz.  Default ``[120., 50., 120.]``.
    cap : list of float, optional
        ``[Ec1, Ec2, Ec3]`` in GHz.  Default ``[10., 50., 10.]``.
    symmetry : bool, optional
        If ``True``, enforce *J2 = J1* and *C2 = C1*.  Default ``False``.

    Returns
    -------
    Circuit instance

    Notes
    -----
    No control channels (no external flux line).
    """
    Ej1, Ej2, Ej3 = josephson
    C1,  C2,  C3  = cap
    circuit  = [J(0, 1, Ej1, 'JJ1'), C(0, 1, C1, 'C1', _C_[1], _C_[0])]
    circuit += [J(0, 2, Ej2, 'JJ2'), C(0, 2, C2, 'C2', _C_[1], _C_[0])]
    circuit += [J(1, 2, Ej3, 'JJ3'), C(1, 2, C3, 'C3', _C_[1], _C_[0])]
    pairs = {}
    if symmetry:
        pairs = {'JJ2': 'JJ1', 'C2': 'C1'}
    return Rep(circuit, (), basis, pairs)


def oscillatorLC(Rep, basis, El=.00031, Ec=51.6256):
    """Simple LC oscillator.

    A single inductor and capacitor — useful for benchmarking basis
    truncations and verifying the harmonic-oscillator spectrum.

    Parameters
    ----------
    Rep : class
        Basis representation class.
    basis : list or dict
        Hilbert-space truncation.
    El : float, optional
        Inductive energy in GHz.  Default ``0.00031``.
    Ec : float, optional
        Charging energy in GHz.  Default ``51.6256``.

    Returns
    -------
    Circuit instance

    Notes
    -----
    The harmonic oscillator frequency is
    *ω/2π = √(8 El Ec)* GHz.  Level spacing should be uniform to within
    truncation error.
    """
    circuit    = [L(0, 1, El, 'L'),
                  C(0, 1, Ec, 'C')]
    control_iD = ()
    return Rep(circuit, control_iD, basis)


def box4Branches(Rep, basis, Ej, Ec, El):
    """Four-node box circuit with one branch per side.

    Each side of the square carries one junction, one capacitor, and one
    external-flux-bias inductor.

    Parameters
    ----------
    Rep : class
        Basis representation class.
    basis : list or dict
        Hilbert-space truncation.
    Ej : list of float, length 4
        Josephson energies [GHz] for each branch.
    Ec : list of float, length 4
        Charging energies [GHz] for each branch.
    El : list of float, length 4
        Inductive energies [GHz] for each branch.

    Returns
    -------
    Circuit instance

    Notes
    -----
    Control channel: ``'L0'`` (external flux on branch 0).
    """
    circuit  = [L(0, 1, El[0], 'L0', external=True, L0=L0, L_=L_),
                C(0, 1, Ec[0], 'C0', C0, C_),
                J(0, 1, Ej[0], 'J0', J0, J_)]
    circuit += [L(1, 2, El[1], 'L1', external=True, L0=L0, L_=L_),
                C(1, 2, Ec[1], 'C1', C0, C_),
                J(1, 2, Ej[1], 'J1', J0, J_)]
    circuit += [L(2, 3, El[2], 'L2', external=True, L0=L0, L_=L_),
                C(2, 3, Ec[2], 'C2', C0, C_),
                J(2, 3, Ej[2], 'J2', J0, J_)]
    circuit += [L(3, 0, El[3], 'L3', external=True, L0=L0, L_=L_),
                C(3, 0, Ec[3], 'C3', C0, C_),
                J(3, 0, Ej[3], 'J3', J0, J_)]
    control_iD = ['L0']
    return Rep(circuit, control_iD, basis)


def phaseSlip(Rep, basis,
              inductance=[.001, .0005, .00002, .00035, .0005],
              capacitance=[100, 30, 30, 30, 30, 40, 10]):
    """Multi-branch phase-slip circuit.

    A complex eight-node circuit modelling a quantum phase-slip device with
    two inductive arms and multiple Josephson junctions.

    Parameters
    ----------
    Rep : class
        Basis representation class.
    basis : list or dict
        Hilbert-space truncation.
    inductance : list of float, length 5
        ``[La, Lb, Lc, Ld, Le]`` inductive energies in GHz.
    capacitance : list of float, length 7
        ``[Ca, Cb, Cc, Cd, Ce, Cf, Cg]`` charging energies in GHz.

    Returns
    -------
    Circuit instance

    Notes
    -----
    Control channels: ``'Ltl'``, ``'Lbl'``, ``'Ltr'``, ``'Lbr'``, ``'Ll'``
    (five external flux lines).
    """
    La, Lb, Lc, Ld, Le = inductance
    Ca, Cb, Cc, Cd, Ce, Cf, Cg = capacitance
    circuit  = [C(0, 1, Ca)]
    circuit += [L(1, 3, La, 'Ltl', external=True),
                L(1, 4, Lb, 'Lbl', external=True)]
    circuit += [J(3, 2, 10.), J(4, 2, 10.)]
    circuit += [C(3, 2, Cb),  C(4, 2, Cc)]
    circuit += [C(2, 5, Cd),  C(2, 6, Ce)]
    circuit += [J(2, 5, 10.), J(2, 6, 100.)]
    circuit += [C(2, 0, Cf)]
    circuit += [L(5, 7, Lc, 'Ltr', external=True),
                L(6, 7, Ld, 'Lbr', external=True)]
    circuit += [L(1, 7, Le, 'Ll',  external=True)]
    circuit += [C(7, 0, Cg)]
    control_iD = ('Ltl', 'Lbl', 'Ltr', 'Lbr', 'Ll')
    return Rep(circuit, control_iD, basis)


if __name__ == '__main__':
    from des_scq.circuit import Kerman
    from torch import tensor

    basis   = {'O': [10], 'I': [], 'J': [4, 4]}
    circuit = shuntedQubit(Kerman, basis)
    print(circuit.kermanDistribution())
    H_LC = circuit.hamiltonianLC()
    H_J  = circuit.hamiltonianJosephson({'I': tensor(.25)})
