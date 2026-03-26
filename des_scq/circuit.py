"""
des_scq.circuit
===============
Circuit graph, Hamiltonian assembly, and exact diagonalization.

This module contains three classes that implement superconducting circuit
physics at progressively higher levels of abstraction:

``Circuit``
    Base class.  Parses the element network into a ``networkx.MultiGraph``,
    builds the spanning tree, constructs node-capacitance and
    branch-inductance matrices, and provides the common Hamiltonian API
    (``hamiltonianLC``, ``hamiltonianJosephson``, ``hamiltonianChargeOffset``,
    ``spectrumManifold``).

``Charge``
    Subclass for circuits whose every mode is described in the charge
    (Cooper-pair number) basis.  The full Hilbert space is a tensor product
    of per-node charge spaces.  Suitable for single- and few-mode qubits
    (transmon, simple fluxonium, etc.) where all modes have a Josephson or
    capacitive origin.

``Kerman``
    Subclass implementing the mode decomposition of Kerman (2020).  The
    *Nn* circuit modes are partitioned into three types:

    * **Oscillator modes** (*No*) — modes that appear in the inductive part
      of the Hamiltonian.  Represented in the Fock basis.
    * **Island modes** (*Ni*) — topologically isolated nodes (no inductive
      path to ground).  Represented in the charge basis.
    * **Josephson modes** (*Nj = Nn - No - Ni*) — remaining modes with
      Josephson-junction coupling.  Represented in the charge basis.

    The Kerman representation is beneficial for circuits with a mix of
    harmonic and anharmonic degrees of freedom (zero-pi, fluxonium with
    large linear inductance, prismon, etc.).

Reference
---------
A. J. Kerman, *Efficient numerical simulation of complex Josephson quantum
circuits*, arXiv:2010.14929 (2020).

Conventions
-----------
* Node **0** is always ground.
* External flux values are reduced flux *Φ/Φ₀ ∈ [0, 1]*.
* All energies are in **GHz** (*h* = 1 in these units).
* Eigenvalues are returned in ascending order.
"""

import networkx
import copy
import torch
from contextlib import nullcontext
from torch import exp, det, norm, tensor, arange, zeros, sqrt, diagonal, eye
from torch.linalg import eigvalsh, inv
from numpy.linalg import matrix_rank, eigvalsh as eigenvalues
from numpy import prod, sort
from des_scq.dense import *
from des_scq.components import diagonalisation, J, L, C, im, pi, complex, float


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def inverse(A, zero=1e-15):
    """Compute the matrix inverse, with a guard for singular matrices.

    Parameters
    ----------
    A : Tensor
        Square matrix to invert.
    zero : float, optional
        Threshold below which the determinant is considered zero.

    Returns
    -------
    Tensor
        Inverse of *A*.
    """
    if det(A) == 0:
        D = A.diag()
        A[D == 0, D == 0] = tensor(1 / zero)
        import pdb; pdb.set_trace()
    try:
        A = inv(A)
    except Exception:
        print('check inverse !! possibly det = 0')
    return A


def phase(phi):
    """Return the complex phase factor *e^{i·2π·φ}*.

    Parameters
    ----------
    phi : Tensor or float
        Reduced flux *Φ/Φ₀*.

    Returns
    -------
    Tensor (complex)
        Phase factor used in the Josephson Hamiltonian.
    """
    return exp(im * 2 * pi * phi)


def hamiltonianEnergy(H):
    """Diagonalize a Hermitian Hamiltonian and return its eigenenergies.

    Parameters
    ----------
    H : Tensor, shape (N, N)
        Hermitian Hamiltonian matrix.

    Returns
    -------
    Tensor, shape (N,)
        Eigenvalues in ascending order (GHz).

    Example
    -------
    >>> from des_scq.circuit import hamiltonianEnergy
    >>> energies = hamiltonianEnergy(H_LC + H_J)
    >>> E10 = energies[1] - energies[0]
    """
    return eigvalsh(H)


# ---------------------------------------------------------------------------
# Base circuit class
# ---------------------------------------------------------------------------

class Circuit:
    """Base class for superconducting circuit Hamiltonian assembly.

    Represents the circuit as a multigraph, builds the capacitance and
    inductance matrices using nodal analysis, and provides the core
    Hamiltonian building blocks that subclasses specialize.

    Parameters
    ----------
    network : list of Elements
        All lumped elements (``J``, ``C``, ``L``) in the circuit.  Node
        indices inside each element define the graph topology.
    control_iD : tuple or list
        IDs of control elements.  String IDs correspond to inductive
        branches (external flux control); integer IDs correspond to nodes
        (charge-offset control).  The order must match the control vectors
        supplied to ``spectrumManifold``.
    basis : list or dict
        Hilbert-space truncation.  For ``Charge``: a list of per-node
        truncations, e.g., ``[256]``.  For ``Kerman``: a dict with keys
        ``'O'``, ``'I'``, ``'J'``, e.g., ``{'O': [32], 'I': [], 'J': [8, 8]}``.
    pairs : dict, optional
        Symmetry constraints of the form ``{'slave_ID': 'master_ID'}``.
        The slave component shares the master's ``base`` tensor so that
        gradient updates respect the enforced symmetry.  Default ``{}``.
    device : str or torch.device, optional
        Tensor device for all computations.  Default ``None`` (CPU).

    Attributes
    ----------
    G : networkx.MultiGraph
        Circuit graph.
    spanning_tree : networkx.Graph
        Minimum spanning tree of the graph (used to compute loop fluxes).
    nodes : dict {int: node_label}
        Mapping from internal index to circuit node label (ground excluded).
    edges : dict {int: (u, v, key)}
        All circuit edges indexed in the order: inductors first,
        then capacitors and junctions.
    edges_inductive : dict {int: (u, v, key)}
        Inductive edges only.
    Nn : int
        Number of active (non-ground) nodes.
    spectrum_limit : int
        Number of eigenstates computed by ``eigenSpectrum``.  Default 4.

    Notes
    -----
    Topology rules enforced by the code:

    * No two external flux inductors in parallel (would be redundant).
    * No *LC* pair in parallel (degenerate inversion of capacitance matrix).
    * Josephson junctions may be placed in parallel.
    """

    def __init__(self, network, control_iD, basis, pairs=dict(), device=None):
        super().__init__()
        self.network     = network
        self.control_iD  = control_iD
        self.G           = self.parseCircuit()
        self.spanning_tree = self.spanningTree()
        self.nodes, self.nodes_ = self.nodeIndex()
        self.edges, self.edges_inductive = self.edgesIndex()
        self.Nn = len(self.nodes)
        self.Ne = len(self.edges)
        self.Nb = len(self.edges_inductive)
        self.pairs = pairs
        self.symmetrize(self.pairs)

        self.Cn_, self.Ln_ = self.componentMatrix()

        self.basis  = basis
        self.device = device
        self.null_flux = tensor(0., device=self.device)
        self.null      = null(self.basisSize(), device=self.device)

        self.spectrum_limit = 4
        self.ii_limit       = 3
        self.grad_calc      = True

    # -----------------------------------------------------------------------
    # Initialization and parameter management
    # -----------------------------------------------------------------------

    def initialization(self, parameters):
        """Re-initialize all circuit elements from a parameter dictionary.

        Parameters
        ----------
        parameters : dict {str: float}
            Mapping from element ID to energy value in GHz.  Only elements
            whose IDs appear in *parameters* are updated.

        Notes
        -----
        After calling this method the capacitance/inductance matrices
        (``Cn_``, ``Ln_``) are *not* automatically recomputed.  They are
        rebuilt lazily the next time a Hamiltonian is assembled.
        """
        for component in self.network:
            if component.__class__ == C:
                component.initCap(parameters[component.ID])
            elif component.__class__ == L:
                component.initInd(parameters[component.ID])
            elif component.__class__ == J:
                component.initJunc(parameters[component.ID])
        self.symmetrize(self.pairs)

    def named_parameters(self, subspace=(), recurse=False):
        """Iterate over ``(ID, base_tensor)`` pairs for optimizable parameters.

        Slave parameters (those constrained by ``pairs``) are excluded, as
        their gradients accumulate through the master's tensor.

        Parameters
        ----------
        subspace : tuple, optional
            If non-empty, restrict iteration to element IDs in this tuple.
            Default ``()`` (all elements).

        Yields
        ------
        (str, Tensor)
            Element ID and its unconstrained ``base`` tensor.
        """
        parameters = []
        IDs        = []
        slaves     = self.pairs.keys()
        for component in self.network:
            if component.ID in subspace or len(subspace) == 0:
                parameter = component.base
                if component.ID not in slaves:
                    IDs.append(component.ID)
                    parameters.append(parameter)
        return zip(IDs, parameters)

    def symmetrize(self, pairs):
        """Enforce symmetry constraints by pointing slave ``base`` tensors
        to their master's tensor.

        Parameters
        ----------
        pairs : dict {str: str}
            ``{'slave_ID': 'master_ID'}`` mapping.
        """
        components = self.circuitComposition()
        for slave, master in pairs.items():
            master = components[master]
            slave  = components[slave]
            slave.base = master.base

    # -----------------------------------------------------------------------
    # Graph construction
    # -----------------------------------------------------------------------

    def parseCircuit(self):
        """Build the circuit multigraph from the element network.

        Josephson junctions are given a weight equal to their current *Ej*
        (used by the minimum spanning tree algorithm).  All other elements
        receive a small nominal weight.

        Returns
        -------
        networkx.MultiGraph
        """
        G = networkx.MultiGraph()
        for component in self.network:
            weight = 1e-3
            if component.__class__ == J:
                weight = component.energy().item()
            G.add_edge(component.plus, component.minus,
                       key=component.ID, weight=weight, component=component)
        return G

    def nodeIndex(self):
        """Map node labels to contiguous integer indices (ground excluded).

        Returns
        -------
        nodes : dict {int: label}
            Index → node label.
        nodes_ : dict {label: int}
            Node label → index (ground maps to -1).
        """
        nodes = list(self.G.nodes())
        nodes.remove(0)
        nodes  = dict([*enumerate(nodes)])
        nodes_ = {val: key for key, val in nodes.items()}
        nodes_[0] = -1
        return nodes, nodes_

    def edgesIndex(self):
        """Assign integer indices to all edges, inductors first.

        Inductive edges are assigned indices from 0 upward; capacitive and
        Josephson edges fill in from the top downward.  This ordering is
        required by the branch-inductance and connection-polarity matrices.

        Returns
        -------
        edges : dict {int: (u, v, key)}
            All edges indexed.
        edges_inductive : dict {int: (u, v, key)}
            Inductive edges only (subset of *edges*).
        """
        edges_G = self.G.edges(keys=True)
        index_plus  = 0
        index_minus = len(edges_G) - 1
        edges, edges_inductive = dict(), dict()
        for u, v, key in edges_G:
            component = self.G[u][v][key]['component']
            if component.__class__ == L:
                edges_inductive[index_plus] = (u, v, key)
                edges[index_plus]           = (u, v, key)
                index_plus  += 1
            else:
                edges[index_minus] = (u, v, key)
                index_minus -= 1
        return edges, edges_inductive

    def spanningTree(self):
        """Compute the minimum spanning tree of the inductive sub-graph.

        The spanning tree is used to determine unique paths between nodes for
        loop-flux threading.

        Returns
        -------
        networkx.Graph
            Minimum spanning tree (weights = 1/*Ej* for junctions, small
            constant for inductors).
        """
        GL = self.graphGL()
        return networkx.minimum_spanning_tree(GL)

    def graphGL(self, elements=[C]):
        """Return a copy of the circuit graph with *elements* removed.

        Parameters
        ----------
        elements : list of class, optional
            Element classes to exclude.  Default ``[C]`` (removes capacitors,
            keeping inductors and junctions for spanning-tree computation).

        Returns
        -------
        networkx.MultiGraph
        """
        GL    = copy.deepcopy(self.G)
        edges = []
        for u, v, component in GL.edges(data=True):
            component = component['component']
            if component.__class__ in elements:
                edges.append((u, v, component.ID))
        GL.remove_edges_from(edges)
        return GL

    # -----------------------------------------------------------------------
    # State and composition queries
    # -----------------------------------------------------------------------

    def circuitState(self):
        """Return a dict of current element energies (GHz).

        Returns
        -------
        dict {str: float}
            ``{element_ID: energy_value}`` for all elements.
        """
        return {component.ID: component.energy().item()
                for component in self.network}

    def circuitComposition(self):
        """Return a dict mapping element IDs to element objects.

        Returns
        -------
        dict {str: Elements}
        """
        return {component.ID: component for component in self.network}

    def circuitComponents(self):
        """Return a dict of physical component values in natural units.

        Returns
        -------
        dict {str: float}
            Capacitors: capacitance (natural units).
            Inductors: inductance (natural units).
            Junctions: Josephson energy (GHz).
        """
        circuit_components = {}
        for component in self.network:
            if component.__class__ == C:
                circuit_components[component.ID] = component.capacitance().item()
            elif component.__class__ == L:
                circuit_components[component.ID] = component.inductance().item()
            elif component.__class__ == J:
                circuit_components[component.ID] = component.energy().item()
        return circuit_components

    # -----------------------------------------------------------------------
    # Matrix construction (nodal analysis)
    # -----------------------------------------------------------------------

    def componentMatrix(self):
        """Build the inverse node-capacitance and node-inductance matrices.

        Computes::

            Cn_ = Cn⁻¹                 (inverse node capacitance)
            Ln_ = Rbn^T · Lb⁻¹ · Rbn  (projected inverse inductance)

        where *Rbn* is the branch–node connection polarity matrix and *Lb*
        is the diagonal branch-inductance matrix.

        Returns
        -------
        Cn_ : Tensor, shape (Nn, Nn)
            Inverse node capacitance matrix.
        Ln_ : Tensor, shape (Nn, Nn)
            Projected inverse inductance matrix.
        """
        Cn   = self.nodeCapacitance()
        assert not det(Cn) == 0
        Cn_  = inverse(Cn)
        Rbn  = self.connectionPolarity()
        Lb   = self.branchInductance()
        M    = self.mutualInductance()
        L_inv = inverse(Lb + M)
        Ln_  = Rbn.conj().T @ L_inv @ Rbn
        return Cn_, Ln_

    def nodeCapacitance(self):
        """Build the node capacitance matrix *Cn* (Nn × Nn).

        Diagonal entries: sum of capacitances connected to each node.
        Off-diagonal entries: negative capacitance of branches between nodes.

        Returns
        -------
        Tensor, shape (Nn, Nn)
        """
        Cn = zeros((self.Nn, self.Nn), dtype=float)
        for i, node in self.nodes.items():
            for u, v, component in self.G.edges(node, data=True):
                component = component['component']
                if component.__class__ == C:
                    capacitance = component.capacitance()
                    Cn[i, i]  += capacitance
                    if not (u == 0 or v == 0):
                        Cn[self.nodes_[u], self.nodes_[v]] = -capacitance
                        Cn[self.nodes_[v], self.nodes_[u]] = -capacitance
        return Cn

    def branchInductance(self):
        """Build the diagonal branch-inductance matrix *Lb* (Nb × Nb).

        Returns
        -------
        Tensor, shape (Nb, Nb)
        """
        Lb = zeros((self.Nb, self.Nb), dtype=float)
        for index, (u, v, key) in self.edges_inductive.items():
            component = self.G[u][v][key]['component']
            if component.__class__ == L:
                Lb[index, index] = component.inductance()
        return Lb

    def mutualInductance(self):
        """Build the mutual-inductance matrix *M* (Nb × Nb).

        Currently returns zero (mutual inductance not implemented).

        Returns
        -------
        Tensor, shape (Nb, Nb)
            Zero matrix.
        """
        return zeros((self.Nb, self.Nb), dtype=float)

    def connectionPolarity(self):
        """Build the branch–node connection polarity matrix *Rbn* (Nb × Nn).

        *Rbn[b, i] = +1* if branch *b* leaves node *i*,
        *Rbn[b, i] = -1* if branch *b* arrives at node *i*.

        Returns
        -------
        Tensor, shape (Nb, Nn)
        """
        Rbn = zeros((self.Nb, self.Nn), dtype=float)
        for index, (u, v, key) in self.edges_inductive.items():
            if u != 0:
                Rbn[index][self.nodes_[u]] =  1
            if v != 0:
                Rbn[index][self.nodes_[v]] = -1
        return Rbn

    # -----------------------------------------------------------------------
    # Loop-flux threading
    # -----------------------------------------------------------------------

    def loopFlux(self, u, v, key, external_fluxes):
        """Compute the total external flux threading the loop containing
        edge *(u, v, key)*.

        Traverses the unique path between *u* and *v* in the spanning tree
        and accumulates the external flux from any inductors marked
        ``external=True`` along that path.

        Parameters
        ----------
        u, v : int
            Node indices of the Josephson junction whose loop flux is needed.
        key : str
            Edge identifier.
        external_fluxes : dict {str: Tensor}
            Mapping from inductor ID to reduced flux value *Φ/Φ₀*.

        Returns
        -------
        Tensor (scalar)
            Total reduced loop flux.
        """
        flux     = self.null_flux.clone()
        external = set(external_fluxes.keys())
        S        = self.spanning_tree
        path     = networkx.shortest_path(S, u, v)
        for i in range(len(path) - 1):
            multi = S.get_edge_data(path[i], path[i + 1])
            match = external.intersection(set(multi.keys()))
            assert len(match) <= 2
            if len(match) == 1:
                component = multi[match.pop()]['component']
                assert component.__class__ == L
                assert component.external == True
                flux += external_fluxes[component.ID]
        return flux

    # -----------------------------------------------------------------------
    # Component accessors
    # -----------------------------------------------------------------------

    def josephsonComponents(self):
        """Return the edges and Josephson energies of all junctions.

        Returns
        -------
        edges : list of (u, v, key)
        Ej : list of Tensor
        """
        edges, Ej = [], []
        for index, (u, v, key) in self.edges.items():
            component = self.G[u][v][key]['component']
            if component.__class__ == J:
                edges.append((u, v, key))
                Ej.append(component.energy())
        return edges, Ej

    def fluxBiasComponents(self):
        """Return edges and inductance functions of all external flux biases.

        Returns
        -------
        edges : list of (u, v, key)
        L_ext : list of callables
            Each callable returns the inductance value in natural units.
        """
        edges, L_ext = [], []
        for index, (u, v, key) in self.edges.items():
            component = self.G[u][v][key]['component']
            if component.__class__ == L and component.external:
                edges.append((u, v, key))
                L_ext.append(component.inductance)
        return edges, L_ext

    def modeImpedance(self):
        """Return per-mode impedance *Z_i = √(Cn_[i,i] / Ln_[i,i])*.

        Returns
        -------
        list of Tensor
        """
        Cn_, Ln_, basis = self.Cn_, self.Ln_, self.basis
        return [sqrt(Cn_[i, i] / Ln_[i, i]) for i in range(len(basis))]

    def islandModes(self):
        """Count the number of island modes.

        An island is a connected component of the graph that has no inductive
        (or Josephson) path to ground.  Its flux coordinate does not appear
        in the Hamiltonian.

        Returns
        -------
        int
            Number of island modes *Ni*.
        """
        islands = self.graphGL(elements=[C])
        islands = list(networkx.connected_components(islands))
        return sum(1 for sub in islands if 0 not in sub)

    def basisSize(self, modes=False):
        """Return the total Hilbert-space dimension.

        For the base ``Circuit`` class the basis is a flat list; dimension =
        product of all elements.

        Parameters
        ----------
        modes : bool, optional
            If ``True``, return the list of per-mode dimensions instead of
            their product.

        Returns
        -------
        int or list of int
        """
        N = [size for size in self.basis]
        if modes:
            return N
        return prod(N)

    # -----------------------------------------------------------------------
    # Hamiltonian assembly — to be overridden by subclasses
    # -----------------------------------------------------------------------

    def hamiltonianLC(self):
        """Assemble the LC (kinetic + potential) part of the Hamiltonian.

        Must be implemented by subclasses.

        Returns
        -------
        Tensor, shape (N, N), dtype cdouble
            LC Hamiltonian in GHz.
        """
        raise NotImplementedError

    def hamiltonianJosephson(self, external_fluxes=dict()):
        """Assemble the Josephson part of the Hamiltonian.

        Must be implemented by subclasses.

        Parameters
        ----------
        external_fluxes : dict {str: Tensor}, optional
            Maps inductor IDs to reduced flux values *Φ/Φ₀*.

        Returns
        -------
        Tensor, shape (N, N), dtype cdouble
            Josephson Hamiltonian in GHz.
        """
        raise NotImplementedError

    def hamiltonianChargeOffset(self, charge_offset=dict()):
        """Assemble the charge-offset correction to the Hamiltonian.

        Must be implemented by subclasses.

        Parameters
        ----------
        charge_offset : dict {int: Tensor}, optional
            Maps node indices to charge-offset values in units of Cooper pairs.

        Returns
        -------
        Tensor, shape (N, N), dtype cdouble
        """
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Control dispatch
    # -----------------------------------------------------------------------

    def controlData(self, control):
        """Split a flat control vector into flux and charge dictionaries.

        Parameters
        ----------
        control : list or tuple
            Ordered control values matching the order of ``control_iD``.
            String IDs map to flux channels; integer IDs map to charge channels.

        Returns
        -------
        control_flux : dict {str: Tensor}
        control_charge : dict {int: Tensor}
        """
        if len(self.control_iD) == 0:
            return dict(), dict()
        assert len(control) == len(self.control_iD)
        control_flux, control_charge = dict(), dict()
        for iD, ctrl in zip(self.control_iD, control):
            assert type(iD) is str or type(iD) is int
            if type(iD) is str:
                control_flux[iD]   = ctrl
            elif type(iD) is int:
                control_charge[iD] = ctrl
        return control_flux, control_charge

    def circuitHamiltonian(self, control):
        """Assemble the full Hamiltonian for a given control point.

        Parameters
        ----------
        control : tuple (control_flux, control_charge)
            Pre-parsed control dictionaries as returned by ``controlData``.

        Returns
        -------
        Tensor, shape (N, N), dtype cdouble
            Full Hamiltonian in GHz.
        """
        H = self.hamiltonianLC()
        control_flux, control_charge = control
        H += self.hamiltonianJosephson(control_flux)
        if len(control_charge) > 0:
            H += self.hamiltonianChargeOffset(control_charge)
        return H

    # -----------------------------------------------------------------------
    # Spectrum computation
    # -----------------------------------------------------------------------

    def eigenSpectrum(self, control):
        """Compute eigenenergies for a single control point.

        Parameters
        ----------
        control : list or tuple
            Raw control vector (will be parsed by ``controlData``).

        Returns
        -------
        spectrum : Tensor, shape (spectrum_limit,)
            Lowest *spectrum_limit* eigenenergies in GHz.
        states : None
            Eigenstates are not returned in the current implementation.
        """
        control = self.controlData(control)
        with torch.inference_mode() if self.grad_calc is False else nullcontext() as null:
            H        = self.circuitHamiltonian(control)
            spectrum = eigvalsh(H)[:self.spectrum_limit]
            states   = None
        return spectrum, states

    def spectrumManifold(self, manifold):
        """Compute eigenenergies over a list of control points.

        Parameters
        ----------
        manifold : list of control vectors
            Each element is a list/tuple of control values whose order matches
            ``control_iD``.  For a circuit with no controls, pass ``[[]]`` or
            ``[dict()]``.

        Returns
        -------
        list of (Tensor, None)
            One ``(spectrum, states)`` tuple per control point.  The spectrum
            tensor has shape *(spectrum_limit,)* and contains eigenenergies
            in GHz, in ascending order.

        Example
        -------
        >>> flux_range   = tensor(linspace(0, 1, 51))
        >>> flux_profile = [[phi] for phi in flux_range]
        >>> Spectrum     = circuit.spectrumManifold(flux_profile)
        >>> E10 = [(s[1] - s[0]).item() for s, _ in Spectrum]
        """
        Spectrum = []
        for control in manifold:
            spectrum, states = self.eigenSpectrum(control)
            Spectrum.append((spectrum, states))
        return Spectrum


# ---------------------------------------------------------------------------
# Kerman mixed-mode representation
# ---------------------------------------------------------------------------

class Kerman(Circuit):
    """Circuit in the Kerman mixed-mode (O/I/J) basis.

    Implements the mode decomposition of Kerman (2020), which partitions the
    *Nn* circuit degrees of freedom into:

    * **Oscillator modes** (*No*) — harmonic modes arising from the inductive
      network.  Represented in the Fock (number-state) basis.
    * **Island modes** (*Ni*) — isolated charge islands with no inductive
      connection to ground.  Represented in the charge basis.
    * **Josephson modes** (*Nj*) — anharmonic modes with Josephson-junction
      coupling.  Represented in the charge basis.

    The partition is determined automatically from the circuit topology via
    ``kermanDistribution``.

    Parameters
    ----------
    network, control_iD, basis, pairs, device
        As in :class:`Circuit`.  ``basis`` must be a dict with keys
        ``'O'``, ``'I'``, ``'J'``, each a list of per-mode truncation values,
        e.g.::

            basis = {'O': [32], 'I': [], 'J': [8, 8]}

    Attributes
    ----------
    No, Ni, Nj : int
        Number of oscillator, island, and Josephson modes.
    R : Tensor
        Kerman transformation matrix (real part of eigenvectors of *Ln_*).

    Notes
    -----
    The Kerman basis is most beneficial when *No > 0* (i.e., the circuit has
    at least one harmonic mode), as it allows a much smaller Fock-space
    truncation for that mode compared to the charge basis.

    Reference
    ---------
    A. J. Kerman, arXiv:2010.14929.
    """

    def __init__(self, network, control_iD, basis, pairs=dict(), device=None):
        super().__init__(network, control_iD, basis, pairs, device)
        self.No, self.Ni, self.Nj = self.kermanDistribution()
        self.N  = self.basisSize()
        self.R  = self.kermanTransform().real
        self.L_, self.C_ = self.modeTransformation()

    def basisSize(self, modes=False):
        """Return total Hilbert-space dimension or per-mode sizes.

        Parameters
        ----------
        modes : bool, optional
            If ``True``, return a dict ``{'O': [...], 'I': [...], 'J': [...]}``.

        Returns
        -------
        int or dict
        """
        N      = dict()
        basis  = self.basis
        N['O'] = [size          for size in basis['O']]
        N['I'] = [2 * size + 1  for size in basis['I']]
        N['J'] = [2 * size + 1  for size in basis['J']]
        if modes:
            return N
        return int(prod(N['O']) * prod(N['I']) * prod(N['J']))

    def kermanDistribution(self):
        """Determine the O/I/J mode partition for this circuit.

        * *No* = rank(*Ln_*)  (number of non-zero inductive eigenvalues)
        * *Ni* = number of connected components not containing ground
        * *Nj = Nn - No - Ni*

        Returns
        -------
        No, Ni, Nj : int, int, int
        """
        Ln_ = self.Ln_
        Ni = self.islandModes()
        if Ln_.is_cuda:
            No = matrix_rank(Ln_.cpu().detach().numpy())
        else:
            No = matrix_rank(Ln_.detach().numpy())
        Nj = self.Nn - Ni - No
        return No,Ni,Nj

    def kermanTransform(self):
        """Compute the mode transformation matrix *R*.

        *R* is the matrix whose columns are the eigenvectors of *Ln_* sorted
        by descending eigenvalue (oscillator modes first, zero modes last).

        Returns
        -------
        Tensor, shape (Nn, Nn)
            Real transformation matrix.
        """
        Ln_ = self.Ln_
        R   = diagonalisation(Ln_.detach(), reverse=True).to(float)
        return R

    def kermanComponents(self):
        """Extract block-diagonal O/I/J components from the transformed
        capacitance and inductance matrices.

        Returns
        -------
        Lo_ : Tensor, shape (No, No)
            Oscillator-mode inductance block.
        C_  : tuple of Tensors
            ``(Co_, Coi_, Coj_, Ci_, Cij_, Cj_)`` — all capacitance coupling
            blocks in the transformed basis.
        """
        L_, C_  = self.L_, self.C_
        No, Ni, Nj = self.No, self.Ni, self.Nj
        Lo_  = L_[:No, :No]
        Co_  = C_[:No,        :No]
        Coi_ = C_[:No,        No:No+Ni]
        Coj_ = C_[:No,        No+Ni:]
        Ci_  = C_[No:No+Ni,   No:No+Ni]
        Cij_ = C_[No:No+Ni,   No+Ni:]
        Cj_  = C_[No+Ni:,     No+Ni:]
        return Lo_, (Co_, Coi_, Coj_, Ci_, Cij_, Cj_)

    def modeTransformation(self):
        """Rotate *Cn_* and *Ln_* into the Kerman mode basis.

        Applies the transformation::

            L_ = (R^T)⁻¹ · Ln_ · R⁻¹
            C_ = R · Cn_ · R^T

        Returns
        -------
        L_ : Tensor, shape (Nn, Nn)
        C_ : Tensor, shape (Nn, Nn)
        """
        Cn_, Ln_ = self.Cn_, self.Ln_
        R   = self.R
        L_  = inv(R.T) @ Ln_ @ inv(R)
        C_  = R @ Cn_ @ R.T
        return L_, C_

    def oscillatorImpedance(self):
        """Return per-oscillator-mode impedance *Z_i = √(Cn_[i,i] / Ln_[i,i])*.

        Returns
        -------
        list of Tensor, length No
        """
        Cn_, Ln_, basis = self.Cn_, self.Ln_, self.basis
        return [sqrt(Cn_[i, i] / Ln_[i, i]) for i in range(len(basis['O']))]

    def linearCombination(self, index):
        """Return row *index* of *R⁻¹* (one row of the mode decomposition).

        Used to construct the linear combination of mode displacements for
        the Josephson Hamiltonian.

        Parameters
        ----------
        index : int
            Node index in the rotated basis.

        Returns
        -------
        Tensor, shape (Nn,)
        """
        invR        = inv(self.R)
        combination = invR[index]
        assert len(combination) == self.Nn
        return combination

    def displacementCombination(self, combination):
        """Build displacement operator lists for a given mode combination.

        For each element of *combination*, constructs the forward (+) and
        backward (−) displacement operator in the appropriate basis (oscillator
        or charge).

        Parameters
        ----------
        combination : Tensor, shape (Nn,)
            Linear combination coefficients for the mode displacements.

        Returns
        -------
        Dplus, Dminus : list of Tensor
            Displacement operators for the forward and backward directions.
        """
        basis = self.basis
        No,Ni,Nj = self.No,self.Ni,self.Nj
        O = combination[:No]
        I = combination[No:No+Ni]
        J = combination[No+Ni:]
        
        Z = self.oscillatorImpedance() * 2 # cooper pair factor
        # re-calculation impedance factor with circuit variation
        DO_plus = [displacementOscillator(basis_max,z,o) for o,z,basis_max in zip(O,Z,basis['O'])]
        DO_minus = [displacementOscillator(basis_max,z,-o) for o,z,basis_max in zip(O,Z,basis['O'])]
        
        DI_plus = [displacementCharge(basis_max,i) for i,basis_max in zip(I,basis['I'])]
        DI_minus = [displacementCharge(basis_max,-i) for i,basis_max in zip(I,basis['I'])]
        DJ_plus = [displacementCharge(basis_max,j) for j,basis_max in zip(J,basis['J'])]
        DJ_minus = [displacementCharge(basis_max,-j) for j,basis_max in zip(J,basis['J'])]
        
        Dplus = DO_plus+DI_plus+DJ_plus
        Dminus = DO_minus+DI_minus+DJ_minus
        assert len(combination)==len(Dplus)
        assert len(combination)==len(Dminus)
        return Dplus,Dminus

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=True)
    def hamiltonianLC(self):
        """Assemble the Kerman-basis LC Hamiltonian.

        Constructs all kinetic and potential coupling blocks in the
        O/I/J mode basis, including cross-mode capacitive coupling terms.

        Returns
        -------
        Tensor, shape (N, N), dtype cdouble
            LC Hamiltonian (GHz).
        """
        basis = self.basis
        self.Cn_,self.Ln_ = self.componentMatrix()
        self.L_,self.C_ = self.modeTransformation()
        Lo_,C_ = self.kermanComponents()
        Co_,Coi_,Coj_,Ci_,Cij_,Cj_ = C_

        No,Ni,Nj = self.kermanDistribution()

        Z = sqrt(diagonal(Co_)/diagonal(Lo_))
        Qo = [basisQo(basis_max,Zi) for Zi,basis_max in zip(Z,basis['O'])]
        Qi = [basisQq(basis_max) for basis_max in basis['I']]
        Qj = [basisQq(basis_max) for basis_max in basis['J']]
        Q = Qo + Qi + Qj

        H = modeMatrixProduct(Q,Co_,Q,(0,0))/2

        Fo = [basisFo(basis_max,Zi) for Zi,basis_max in zip(Z,basis['O'])]
        Fi = [basisFq(basis_max) for basis_max in basis['I']]
        Fj = [basisFq(basis_max) for basis_max in basis['J']]
        F = Fo + Fi + Fj

        H += modeMatrixProduct(F,Lo_,F,(0,0))/2
        
        H += modeMatrixProduct(Q,Coi_,Q,(0,No))
        H += modeMatrixProduct(Q,Coj_,Q,(0,No+Ni))
        H += modeMatrixProduct(Q,Cij_,Q,(No,No+Ni))

        H += modeMatrixProduct(Q,Ci_,Q,(No,No))/2
        H += modeMatrixProduct(Q,Cj_,Q,(No+Ni,No+Ni))/2        
        return H

    def hamiltonianJosephson(self, external_fluxes=dict()):
        """Assemble the Kerman-basis Josephson Hamiltonian.

        Uses the generalized displacement operators
        (``displacementCombination``) to handle junctions in circuits with
        mixed oscillator/charge modes and non-trivial loop structure.

        Parameters
        ----------
        external_fluxes : dict {str: Tensor}, optional
            Maps external-inductor IDs to reduced flux *Φ/Φ₀*.

        Returns
        -------
        Tensor, shape (N, N), dtype cdouble
            Josephson Hamiltonian (GHz).
        """
        edges, Ej = self.josephsonComponents()
        N = self.basisSize()
        H = null(N)
        for (u, v, key), E in zip(edges, Ej):
            i, j = self.nodes_[u], self.nodes_[v]
            flux  = self.loopFlux(u, v, key, external_fluxes)
            if i < 0 or j < 0:
                i           = max(i, j)
                combination = self.linearCombination(i)
                Dplus, Dminus = self.displacementCombination(combination)
                Jplus  = basisProduct(Dplus)
                Jminus = basisProduct(Dminus)
            else:
                combination = (self.linearCombination(i)
                               - self.linearCombination(j))
                Dplus, Dminus = self.displacementCombination(combination)
                Jplus  = basisProduct(Dplus)
                Jminus = basisProduct(Dminus)
            H -= E * (Jplus * phase(flux) + Jminus * phase(-flux))
        return H / 2
    
    def hamiltonianChargeOffset(self,charge_offset=dict()):
        charge = zeros(self.Nn,dtype=float)
        for node,dQ in charge_offset.items():
            charge[self.nodes_[node]] = dQ
        charge = self.R@charge

        No,Ni,Nj = self.kermanDistribution() #No,self.Ni,self.Nj

        self.Cn_,self.Ln_ = self.componentMatrix()
        self.L_,self.C_ = self.modeTransformation()
        Lo_,C_ = self.kermanComponents()
        Co_,Coi_,Coj_,Ci_,Cij_,Cj_ = C_

        Z = sqrt(diagonal(Co_)/diagonal(Lo_))
        Qo = [basisQo(basis_max,Zi) for Zi,basis_max in zip(Z,basis['O'])]
        Qi = [basisQq(basis_max) for basis_max in basis['I']]
        Qj = [basisQq(basis_max) for basis_max in basis['J']]
        Q = Qo + Qi + Qj
        Io = [identity(2*basis_max+1)*0.0 for basis_max in basis['O']]
        Ii = [identity(2*basis_max+1)*charge[index+No]*2 for index,basis_max in enumerate(basis['I'])]
        Ij = [identity(2*basis_max+1)*charge[index+No+Ni]*2 for index,basis_max in enumerate(basis['J'])]
        I = Io + Ii + Ij
        
        H = modeMatrixProduct(Q,Coi_,I,(0,No))
        H += modeMatrixProduct(Q,Coj_,I,(0,No+Ni))
        
        H += modeMatrixProduct(I,Coj_,Q,(No+Ni,No))
        H += modeMatrixProduct(Q,Coj_,I,(No+Ni,No))
        H -= modeMatrixProduct(I,Coj_,I,(No+Ni,No))
        
        H += modeMatrixProduct(I,Ci_,Q,(No,No))/2
        H += modeMatrixProduct(Q,Ci_,I,(No,No))/2
        H += modeMatrixProduct(I,Ci_,I,(No,No))/2
        
        H += modeMatrixProduct(I,Cj_,Q,(No+Ni,No+Ni))/2
        H += modeMatrixProduct(Q,Cj_,I,(No+Ni,No+Ni))/2
        H += modeMatrixProduct(I,Cj_,I,(No+Ni,No+Ni))/2

        return -H

# ---------------------------------------------------------------------------
# Charge basis representation
# ---------------------------------------------------------------------------

class Charge(Circuit):
    """Circuit in the pure charge (Cooper-pair number) basis.

    Every mode is described by its Cooper-pair occupation number.  The basis
    for a circuit with *Nn* modes and truncation list ``basis = [n0, n1, …]``
    has Hilbert-space dimension *∏ (2nᵢ + 1)*.

    Parameters
    ----------
    network, control_iD, basis, pairs, device
        As in :class:`Circuit`.  ``basis`` must be a flat list of per-node
        truncation integers, e.g., ``[256]`` for a single-mode circuit or
        ``[8, 8, 8]`` for a three-node circuit.

    Notes
    -----
    The charge basis is suitable for circuits where all modes are strongly
    anharmonic (e.g., transmon, simple fluxonium, multi-node Josephson
    circuits without large linear inductance).  For circuits with oscillator
    modes use :class:`Kerman`.

    Convergence: Check that eigenenergies are stable as you increase each
    ``basis[i]``.  Typical values: 128–512 for single-mode transmon/fluxonium;
    8–16 per mode for multi-mode circuits.
    """

    def __init__(self, network, control_iD, basis, pairs=dict(), device=None):
        super().__init__(network, control_iD, basis, pairs, device)
        self.N = self.basisSize()

    def basisSize(self, modes=False):
        """Return total dimension or per-mode dimensions.

        Dimension of mode *i* = *2 * basis[i] + 1*.

        Parameters
        ----------
        modes : bool, optional
            If ``True``, return list of per-mode dimensions.

        Returns
        -------
        int or list of int
        """
        N = [2 * size + 1 for size in self.basis]
        if modes:
            return N
        return prod(N)

    def hamiltonianLC(self):
        """Assemble the charge-basis LC Hamiltonian.

        Builds::

            H_LC = ½ Qᵀ Cn⁻¹ Q  +  ½ Fᵀ Ln⁻¹ F

        where *Q* and *F* are the charge and flux operators in the charge
        basis, and the tensor product over modes is handled by
        ``modeMatrixProduct``.

        Returns
        -------
        Tensor, shape (N, N), dtype cdouble
            LC Hamiltonian (GHz).
        """
        Cn_, Ln_ = self.Cn_, self.Ln_
        basis = self.basis
        Q = [ basisQq(basis_max) for basis_max in basis]
        F = [ basisFq(basis_max) for basis_max in basis]
        H = modeMatrixProduct(Q, Cn_, Q)
        H += modeMatrixProduct(F, Ln_, F)

        return H / 2
    
    def hamiltonianJosephson(self, external_fluxes=dict()):
        """Assemble the charge-basis Josephson Hamiltonian.

        For each junction, builds the displacement-operator pair
        *e^{±iφ} · e^{±i·2π·Φ_ext/Φ₀}* and accumulates::

            H_J = -½ Σ_j  Ej [D⁺ e^{i·2πΦ_j} + D⁻ e^{-i·2πΦ_j}]

        Grounded junctions act on a single node; floating junctions use
        ``crossBasisProduct`` to couple two nodes.

        Parameters
        ----------
        external_fluxes : dict {str: Tensor}, optional
            Maps external-inductor IDs to reduced flux *Φ/Φ₀*.

        Returns
        -------
        Tensor, shape (N, N), dtype cdouble
            Josephson Hamiltonian (GHz).
        """
        basis  = self.basis
        Dplus  = [chargeDisplacePlus(basis_max)  for basis_max in basis]
        Dminus = [chargeDisplaceMinus(basis_max) for basis_max in basis]
        edges, Ej = self.josephsonComponents()
        N = self.basisSize()
        H = null(N)
        for (u, v, key), E in zip(edges, Ej):
            i, j = self.nodes_[u], self.nodes_[v]
            flux  = self.loopFlux(u, v, key, external_fluxes)
            if i < 0 or j < 0:
                # Grounded junction — single-mode displacement
                i      = max(i, j)
                Jplus  = basisProduct(Dplus,  [i])
                Jminus = basisProduct(Dminus, [i])
            else:
                # Floating junction — two-mode cross-displacement
                Jplus  = crossBasisProduct(Dplus,  Dminus, i, j)
                Jminus = crossBasisProduct(Dplus,  Dminus, j, i)
            H -= E * (Jplus * phase(flux) + Jminus * phase(-flux))
        return H / 2

    def hamiltonianChargeOffset(self, charge_offset=dict()):
        """Charge-offset correction to the Hamiltonian.

        Models a quasi-static offset charge on specified nodes, as occurs
        due to charge noise or an intentional gate voltage.  The correction
        term is::

            H_offset = ½ [Qᵀ Cn⁻¹ Q_offset + Q_offset Cn⁻¹ Q + Q_offset Cn⁻¹ Q_offset]

        Parameters
        ----------
        charge_offset : dict {int: Tensor}
            Maps node index to offset charge value (in units of Cooper pairs,
            so *ng = 0.5* means half a Cooper pair of offset charge).

        Returns
        -------
        Tensor, shape (N, N), dtype cdouble
            Charge-offset Hamiltonian contribution (GHz).

        Example
        -------
        >>> offset = {1: tensor(0.5)}   # 0.5 Cooper pairs on node 1
        >>> H_off  = circuit.hamiltonianChargeOffset(offset)
        """
        charge = zeros(self.Nn, dtype=float)
        basis  = self.basis
        Cn_    = self.Cn_
        for node, dQ in charge_offset.items():
            charge[self.nodes_[node]] = dQ
        Q = [basisQq(basis_max) for basis_max in basis]
        I = [identity(2 * basis_max + 1, complex) * charge[index] * 2
             for index, basis_max in enumerate(basis)]
        H  = modeMatrixProduct(Q, Cn_, I)
        H += modeMatrixProduct(I, Cn_, Q)
        H += modeMatrixProduct(I, Cn_, I)
        return H / 2.

    def potentialCharge(self, external_fluxes=dict()):
        """Potential energy (inductive + Josephson) in the charge basis.

        Parameters
        ----------
        external_fluxes : dict {str: Tensor}, optional

        Returns
        -------
        Tensor, shape (N, N), dtype cdouble
        """
        Ln_  = self.Ln_
        basis = self.basis
        F = [basisFq(basis_max) for basis_max in basis]
        H  = modeMatrixProduct(F, Ln_, F) / 2
        H += self.hamiltonianJosephson(external_fluxes)
        return H

if __name__ == '__main__':
    import des_scq.models as models
    torch.set_num_threads(12)

    # Transmon charge manifold
    circuit = models.transmon(Charge, [256])
    circuit.control_iD  = [1]
    charge_profile = [[tensor(0.)], [tensor(.25)], [tensor(.5)]]
    Spectrum = circuit.spectrumManifold(charge_profile)
    print(Spectrum[0][0][:4])

    # Shunted qubit in Kerman basis
    basis   = {'O': [10], 'I': [], 'J': [4, 4]}
    circuit = models.shuntedQubit(Kerman, basis)
    H_LC    = circuit.hamiltonianLC()
    H_J     = circuit.hamiltonianJosephson
    eigen   = eigvalsh(H_LC + H_J({'I': tensor(.225)}))
    print(eigen[:4])
