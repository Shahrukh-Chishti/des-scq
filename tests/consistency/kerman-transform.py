# Kerman mode transformation: Josephson/island columns of M = R^-1 are the indicator
# vectors of the inductive pieces (lattice of identified shifts); oscillator columns are free.
#
# python tests/consistency/kerman-transform.py
import itertools
import torch
from torch import tensor
from torch.linalg import inv
from des_scq.circuit import Kerman
from des_scq.components import J, C, L

torch.set_default_dtype(torch.float64)
EJ_OFF = .01     # junctions 'off': the Josephson modes are free rotors
# stiff inductors keep the oscillators above the compared rotor levels


def transitions(circuit, levels=5):
    circuit.spectrum_limit = levels
    E, _ = circuit.eigenSpectrum([])
    E = E.detach()
    return E[1:] - E[0]


def rotorReference(network, pieces, levels=5, nmax=3):
    """Low transitions of free rotors H = 4 N^T E N, N integer (Cooper pairs on each piece),
    E = c_red^-1, c_red[a,b] = 1_a^T c_n 1_b, c_n the node capacitance matrix in units of
    1/Ec (so a single capacitor Ec gives E = Ec). Oscillators only shift every level equally."""
    nodes = sorted({n for element in network for n in (element.plus, element.minus)} - {0})
    index = {n: i for i, n in enumerate(nodes)}
    c = torch.zeros(len(nodes), len(nodes))
    for element in network:
        if isinstance(element, C):
            w = 1 / element.energy().detach()
            for u, v in ((element.plus, element.minus), (element.minus, element.plus)):
                if u != 0:
                    c[index[u], index[u]] += w
                    if v != 0:
                        c[index[u], index[v]] -= w
    one = torch.zeros(len(nodes), len(pieces))
    for k, piece in enumerate(pieces):
        for n in piece:
            one[index[n], k] = 1.
    E = inv(one.T @ c @ one)
    energies = sorted(float(4 * tensor(N, dtype=torch.float64) @ E @ tensor(N, dtype=torch.float64))
                      for N in itertools.product(range(-nmax, nmax + 1), repeat=len(pieces)))
    return tensor(energies[1:levels]) - energies[0]


def checkLattice(circuit):
    """J/I columns of M are integer, lie in null(Ln_); L_ is block-diagonal; junction phases
    contain the J/I modes with integer coefficients."""
    No = circuit.No
    M = inv(circuit.R)
    assert torch.allclose(M[:, No:], M[:, No:].round(), atol=1e-12), M
    assert float(circuit.L_[No:, :].abs().max()) < 1e-9 * float(circuit.Ln_.abs().max())
    for (u, v, key) in circuit.josephsonComponents()[0]:
        i, j = circuit.nodes_[u], circuit.nodes_[v]
        c = circuit.linearCombination(max(i, j)) if min(i, j) < 0 else \
            circuit.linearCombination(i) - circuit.linearCombination(j)
        assert torch.allclose(c[No:], c[No:].round(), atol=1e-12), (key, c)


def oscillatorFreedom(circuit, seed=0):
    """Replace the oscillator columns by M_O G + M_IJ B: the spectrum must not change."""
    generator = torch.Generator().manual_seed(seed)
    No, Nn = circuit.No, circuit.Nn
    M = inv(circuit.R)
    G = torch.eye(No) + .4 * torch.rand(No, No, generator=generator)
    B = torch.rand(Nn - No, No, generator=generator) - .5
    M = torch.column_stack([M[:, :No] @ G + M[:, No:] @ B, M[:, No:]])
    circuit.R = inv(M)
    circuit.L_, circuit.C_ = circuit.modeTransformation()


def check(name, network, basis, pieces, tolerance):
    circuit = Kerman(network, (), basis)
    checkLattice(circuit)
    E = transitions(circuit)
    reference = rotorReference(network, pieces, levels=len(E) + 1)
    deviation = float((E - reference).abs().max() / reference.abs().max())
    print(f'{name}: E0i {[round(float(e), 4) for e in E]}  rotors {[round(float(e), 4) for e in reference]}'
          f'  relative deviation {deviation:.1e}')
    assert deviation < tolerance, deviation
    oscillatorFreedom(circuit)
    shift = float((transitions(circuit) - E).abs().max())
    print(f'{name}: oscillator columns replaced by M_O G + M_IJ B, max |dE0i| = {shift:.1e} GHz')
    assert shift < 1e-4 * float(E.abs().max()), shift      # truncation error of the small Fock bases


def testOnePiece():
    """Nodes 1, 2 joined by an inductor (one piece, |c| = 2), node 1 to ground by a junction:
    one oscillator, one Josephson rotor with 4 Ec, Ec of the piece's total capacitance."""
    network = [L(1, 2, 60., 'L'), J(1, 0, EJ_OFF, 'J'), C(1, 0, 1.2, 'C1'), C(2, 0, 2., 'C2')]
    check('one piece', network, {'O': [20], 'I': [], 'J': [6]}, [{1, 2}], 2e-3)


def testTwoPieces():
    """Pieces {1,2} and {3,4}, each joined internally by an inductor; junctions 2-3 and 4-ground:
    two oscillators, two Josephson rotors coupled through the capacitance matrix."""
    network = [L(1, 2, 60., 'La'), L(3, 4, 50., 'Lb'), J(2, 3, EJ_OFF, 'Jab'), J(4, 0, EJ_OFF, 'Jb'),
               C(1, 0, 1.2, 'C1'), C(2, 0, 2., 'C2'), C(3, 0, 1.5, 'C3'), C(4, 0, 1.8, 'C4'),
               C(2, 3, 3., 'C23')]
    check('two pieces', network, {'O': [6, 6], 'I': [], 'J': [3, 3]}, [{1, 2}, {3, 4}], 2e-3)


if __name__ == '__main__':
    testOnePiece()
    testTwoPieces()
    print('kerman-transform: passed')
