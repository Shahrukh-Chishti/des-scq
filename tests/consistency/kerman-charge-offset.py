# Offset charges in the Kerman basis, driven through charge control (integer control_iD),
# against the Charge basis.
#
# python tests/consistency/kerman-charge-offset.py
import torch
from torch import tensor
from des_scq.circuit import Charge, Kerman
from des_scq.components import J, C, L

torch.set_default_dtype(torch.float64)

offsets = [0., .25, .5]


def spectra(network, control_iD, kerman_basis, charge_basis, points):
    kerman = Kerman(network, control_iD, kerman_basis)
    charge = Charge(network, control_iD, charge_basis)
    kerman.spectrum_limit = charge.spectrum_limit = 4
    for point in points:
        point = [tensor(value) for value in point]
        Ek, _ = kerman.eigenSpectrum(point)
        Eq, _ = charge.eigenSpectrum(point)
        yield Ek.detach(), Eq.detach()


def testTransmon():
    """Charge-sensitive transmon (one junction mode), offset on node 1."""
    network = [J(0, 1, 4., 'J'), C(0, 1, .4, 'C')]
    transitions = []
    for Ek, Eq in spectra(network, (1,), {'O': [], 'I': [], 'J': [12]}, [12],
                          [[ng] for ng in offsets]):
        deviation = float((Ek - Eq).abs().max())     # absolute energies, not only transitions
        assert deviation < 1e-9, deviation
        transitions.append(float(Ek[1] - Ek[0]))
    print('transmon E01 over n_g', offsets, ':', [round(e, 4) for e in transitions])
    assert transitions[0] - transitions[-1] > 1e-2   # the offset reaches the spectrum


def testIslandJunction():
    """Transmon coupled to a floating island (one island, one junction mode):
    offsets on both nodes exercise the island-junction cross terms (Cij)."""
    network = [J(0, 1, 4., 'J'), C(0, 1, .4, 'C'),
               C(1, 2, 1., 'Cc'), C(2, 0, .8, 'Ci')]
    points = [[ng, .15] for ng in offsets]
    deviation = max(float((Ek - Eq).abs().max())
                    for Ek, Eq in spectra(network, (1, 2), {'O': [], 'I': [8], 'J': [12]}, [12, 8], points))
    print(f'island + junction: max |E_kerman - E_charge| = {deviation:.1e} GHz')
    assert deviation < 1e-9, deviation


def testOscillatorJunction():
    """Resonator capacitively coupled to a transmon (one oscillator, one junction
    mode): oscillator modes carry no offset but set the Hilbert-space shape."""
    network = [L(0, 1, 20., 'L'), C(0, 1, .2, 'Cr'),
               J(0, 2, 4., 'J'), C(0, 2, .4, 'C'), C(1, 2, 2., 'Cc')]
    deviation = max(float((Ek - Eq).abs().max())
                    for Ek, Eq in spectra(network, (2,), {'O': [30], 'I': [], 'J': [12]}, [14, 12],
                                          [[ng] for ng in offsets]))
    print(f'oscillator + junction: max |E_kerman - E_charge| = {deviation:.1e} GHz')
    assert deviation < 1e-9, deviation


if __name__ == '__main__':
    print('Charge offset in the Kerman basis')
    testTransmon()
    testIslandJunction()
    testOscillatorJunction()
    print('PASS')
