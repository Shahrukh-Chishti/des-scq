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


if __name__ == '__main__':
    print('Coupled oscillator modes with junctions (Kerman) v/s scqubits')
    testImpedanceIsModeBasis()
    testSpectrum()
    testJunctionAcrossModes()
    print('PASS')
