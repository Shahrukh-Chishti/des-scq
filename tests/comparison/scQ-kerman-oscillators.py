# Josephson junctions across coupled oscillator modes (Kerman basis) against scqubits.
# The Josephson displacement must be built with the same mode impedance as the LC
# Hamiltonian (Kerman.oscillatorImpedance, mode basis). Before, it used the node-basis
# impedance, which differs whenever the mode rotation mixes oscillator nodes, so
# multi-oscillator circuits (box, zero-pi) had wrong spectra; single-mode circuits
# (fluxonium) were unaffected.
# Charge-mode displacements (dense.displacementCharge) must raise Q like the oscillator
# ones, [F, Q] = +i/pi. With the opposite sign, junctions spanning an oscillator and a
# charge mode (C-shunted qubit: JJ1 between the inductive node and a junction node) gave
# wrong spectra; junctions touching one mode type were unaffected (cos is even).
#
# python tests/comparison/scQ-kerman-oscillators.py
import torch
from torch import tensor
import scqubits as scq
from des_scq import models
from des_scq.circuit import Kerman
from des_scq.components import J, C

torch.set_default_dtype(torch.float64)

# rf-SQUID box: every branch J || C || L (GHz)
Ej = [10., 12., 9., 11.]
Ec = [.3, .4, .35, .45]
El = [10., 8., 12., 9.]
levels = 5


def scqSpectrum(cutoff=14):
    branches = []
    for (u, v), ej, ec, el in zip([(0, 1), (1, 2), (2, 3), (3, 0)], Ej, Ec, El):
        branches += [f'    - [L, {u}, {v}, {el}]',
                     f'    - [C, {u}, {v}, {ec}]',
                     f'    - [JJ, {u}, {v}, {ej}, 1e5]']   # negligible junction capacitance
    circuit = scq.Circuit('branches:\n' + '\n'.join(branches) + '\n',
                          from_file=False, ext_basis='harmonic')
    for name in circuit.cutoff_names:
        setattr(circuit, name, cutoff)
    for flux in circuit.external_fluxes:
        setattr(circuit, flux.name, 0.)
    energy = circuit.eigenvals(evals_count=levels)
    return energy[1:] - energy[0]


def kermanSpectrum(n=12):
    circuit = models.box4Branches(Kerman, {'O': [n] * 3, 'I': [], 'J': []}, Ej, Ec, El)
    circuit.spectrum_limit = levels
    energy, _ = circuit.eigenSpectrum([tensor(0.)])
    energy = energy.detach().numpy()
    return energy[1:] - energy[0]


def testImpedanceIsModeBasis():
    circuit = models.box4Branches(Kerman, {'O': [4] * 3, 'I': [], 'J': []}, Ej, Ec, El)
    circuit.hamiltonianLC()
    Z = circuit.oscillatorImpedance()
    assert Z.shape == (circuit.No,)
    Lo_, C_ = circuit.kermanComponents()
    assert torch.allclose(Z, (torch.diagonal(C_[0]) / torch.diagonal(Lo_)).sqrt())
    print('oscillatorImpedance: mode basis, one value per oscillator mode')


def testSpectrum():
    SCQ, DeS = scqSpectrum(), kermanSpectrum()
    deviation = abs(SCQ - DeS).max()
    print('scqubits:', SCQ.round(4))
    print('des-scq :', DeS.round(4))
    print(f'max |E0i(des-scq) - E0i(scqubits)| = {deviation:.2e} GHz')
    assert deviation < 1e-3


def testJunctionAcrossModes():
    """C-shunted qubit: one oscillator mode (inductive node 1), two junction modes;
    JJ1 (1-2) spans the oscillator and a junction mode."""
    josephson, cap, ind = [20., 9., 20.], [.8, 1.6, .8], 100.
    reference = scq.Circuit(f"""branches:
    - [JJ, 1, 2, {josephson[0]}, {cap[0]}]
    - [JJ, 2, 3, {josephson[1]}, {cap[1]}]
    - [JJ, 3, 0, {josephson[2]}, {cap[2]}]
    - [L, 0, 1, {ind}]
    """, from_file=False, ext_basis='harmonic')
    for name in reference.cutoff_names:
        setattr(reference, name, 10)
    circuit = models.shuntedQubit(Kerman, {'O': [10], 'I': [], 'J': [6, 6]}, josephson, cap, ind)
    circuit.spectrum_limit = 4
    deviation = 0.
    for flux in (0., .5):
        for symbol in reference.external_fluxes:
            setattr(reference, symbol.name, flux)
        SCQ = reference.eigenvals(evals_count=4)
        DeS, _ = circuit.eigenSpectrum([tensor(flux)])
        DeS = DeS.detach().numpy()
        SCQ, DeS = SCQ[1:] - SCQ[0], DeS[1:] - DeS[0]
        print(f'C-shunted qubit, flux {flux}: scqubits {SCQ.round(4)}  des-scq {DeS.round(4)}')
        deviation = max(deviation, abs(SCQ - DeS).max())
    print(f'max |E0i(des-scq) - E0i(scqubits)| = {deviation:.2e} GHz')
    assert deviation < 1e-2


def testDoubleLoopFluxOffsets():
    """Double-loop flux qubit (two junction modes) under two mesh fluxes and two node
    offsets. scqubits variables: theta1 = phi2, theta2 = phi1 - phi2, so
    ng1 = q1 + q2, ng2 = q1; its fluxes sit on JJ(1,2) (mesh A + B) and JJ(2,0) (mesh B)."""
    network = [J(0, 1, 4., 'J1'), C(0, 1, .8, 'C1'), J(1, 2, 4., 'J2'), C(1, 2, .8, 'C2'),
               J(2, 0, 1.5, 'J3'), J(2, 0, 2., 'J4'), C(2, 0, 1.2, 'C3')]
    circuit = Kerman(network, ('A', 'B', 1, 2), {'O': [], 'I': [], 'J': [10, 10]})
    assert circuit.closures == {'J2': (1, ['A', 'B']), 'J4': (1, ['B'])}
    circuit.spectrum_limit = 4
    reference = scq.Circuit("""branches:
    - [JJ, 0, 1, 4., 0.8]
    - [JJ, 1, 2, 4., 0.8]
    - [JJ, 2, 0, 1.5, 1.2]
    - [JJ, 2, 0, 2., 1e5]
    """, from_file=False, ext_basis='harmonic')
    assert (reference.transformation_matrix == [[1, 1], [1, 0]]).all()
    reference.cutoff_n_1 = reference.cutoff_n_2 = 10
    deviation = 0.
    for A, B, q1, q2 in [(.3, .4, 0., 0.), (0., 0., .25, .1), (.5, .2, .4, .3),
                         (.3, .4, .25, .1), (.3, .4, .25, -.1), (.1, .35, -.2, .3)]:
        reference.Φ1, reference.Φ2 = A + B, B
        reference.ng1, reference.ng2 = q1 + q2, q1
        SCQ = reference.eigenvals(evals_count=4)
        DeS, _ = circuit.eigenSpectrum([tensor(A), tensor(B), tensor(q1), tensor(q2)])
        deviation = max(deviation, abs(SCQ - DeS.detach().numpy()).max())
    print(f'double loop, 2 fluxes + 2 offsets: max |E(des-scq) - E(scqubits)| = {deviation:.1e} GHz')
    assert deviation < 1e-4


def testBoxMeshFluxes():
    """rf-SQUID box (three mixed oscillator modes) with flux in meshes B, C, D at once,
    against scqubits (closures J2: -B, J1: -C, J3: +D), and periodicity in each mesh.
    Mesh E is left out: it also lies inside the loop closed by inductor L2, whose
    external flux is not applied (inductor-loop flux unsupported)."""
    circuit = models.box4Branches(Kerman, {'O': [12] * 3, 'I': [], 'J': []}, Ej, Ec, El)
    assert {key: closure for key, closure in circuit.closures.items() if key in ('J1', 'J2', 'J3')} \
        == {'J2': (-1, ['B', 'E', 'A']), 'J1': (-1, ['C']), 'J3': (1, ['D'])}
    circuit.control_iD = ('B', 'C', 'D')
    circuit.spectrum_limit = levels
    branches = []
    for (u, v), ej, ec, el in zip([(0, 1), (1, 2), (2, 3), (3, 0)], Ej, Ec, El):
        branches += [f'    - [L, {u}, {v}, {el}]', f'    - [C, {u}, {v}, {ec}]',
                     f'    - [JJ, {u}, {v}, {ej}, 1e5]']
    reference = scq.Circuit('branches:\n' + '\n'.join(branches) + '\n',
                            from_file=False, ext_basis='harmonic')
    for name in reference.cutoff_names:
        setattr(reference, name, 14)
    deviation = 0.
    for B, C_, D in [(.2, .35, .1), (.5, 0., .25)]:
        reference.Φ1, reference.Φ2, reference.Φ3, reference.Φ4, reference.Φ5 = 0., -C_, 0., -B, D
        SCQ = reference.eigenvals(evals_count=levels)
        DeS, _ = circuit.eigenSpectrum([tensor(B), tensor(C_), tensor(D)])
        DeS = DeS.detach().numpy()
        deviation = max(deviation, abs(SCQ - DeS).max())
        for mesh in range(3):
            shifted = [B, C_, D]
            shifted[mesh] += 1.
            periodic, _ = circuit.eigenSpectrum([tensor(x) for x in shifted])
            assert abs(periodic.detach().numpy() - DeS).max() < 1e-9
    print(f'box, fluxes in meshes B, C, D: max |E(des-scq) - E(scqubits)| = {deviation:.1e} GHz, '
          f'periodic in each mesh')
    assert deviation < 1e-3


if __name__ == '__main__':
    print('Coupled oscillator modes with junctions (Kerman) v/s scqubits')
    testImpedanceIsModeBasis()
    testSpectrum()
    testJunctionAcrossModes()
    testDoubleLoopFluxOffsets()
    testBoxMeshFluxes()
    print('PASS')
