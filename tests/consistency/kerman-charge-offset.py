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


def meshesAndOffsets(network, kerman_basis, charge_basis, closures, control_iD, points):
    """Kerman against Charge under mesh fluxes and node offsets together. Points with
    one offset reversed at fixed flux differ, so a relative sign error between offset
    and flux (or between modes) shows."""
    assert Kerman(network, (), kerman_basis).closures == closures
    spectra_ = list(spectra(network, control_iD, kerman_basis, charge_basis, points))
    deviation = max(float((Ek - Eq).abs().max()) for Ek, Eq in spectra_)
    reversal = float((spectra_[-1][1] - spectra_[-2][1]).abs().max())
    return deviation, reversal


def testDoubleLoop():
    """Double-loop flux qubit (two junction modes): meshes A (J1-J2-J4) and B (SQUID
    J3-J4), offsets on both nodes."""
    network = [J(0, 1, 4., 'J1'), C(0, 1, .8, 'C1'), J(1, 2, 4., 'J2'), C(1, 2, .8, 'C2'),
               J(2, 0, 1.5, 'J3'), J(2, 0, 2., 'J4'), C(2, 0, 1.2, 'C3')]
    points = [[.3, 0., 0., 0.], [0., .4, 0., 0.], [0., 0., .25, .1], [.5, .2, .4, .3],
              [.3, .4, .25, .1], [.3, .4, .25, -.1]]
    deviation, reversal = meshesAndOffsets(network, {'O': [], 'I': [], 'J': [8, 8]}, [8, 8],
                                           {'J2': (1, ['A', 'B']), 'J4': (1, ['B'])},
                                           ('A', 'B', 1, 2), points)
    print(f'double loop, 2 fluxes + 2 offsets: max |E_kerman - E_charge| = {deviation:.1e} GHz '
          f'(offset reversal moves levels {reversal:.2f} GHz)')
    assert deviation < 1e-9, deviation
    assert reversal > 5e-2


def testShuntedSquid():
    """C-shunted qubit with a SQUID middle junction (one oscillator, two junction
    modes): meshes A (JJ1-JJ2b-JJ3-I) and B (SQUID), offsets on both junction nodes."""
    network = [J(1, 2, 4., 'JJ1'), C(1, 2, .8, 'C1'), J(2, 3, 1.5, 'JJ2a'), J(2, 3, 2., 'JJ2b'),
               C(2, 3, 1.6, 'C2'), J(3, 0, 4., 'JJ3'), C(3, 0, .8, 'C3'), L(0, 1, 100., 'I')]
    points = [[.5, .1, .45, .2], [.3, .2, .3, .15], [.3, .2, -.3, .15]]
    deviation, reversal = meshesAndOffsets(network, {'O': [8], 'I': [], 'J': [5, 5]}, [5, 5, 5],
                                           {'JJ3': (1, ['A', 'B']), 'JJ2b': (-1, ['B'])},
                                           ('A', 'B', 2, 3), points)
    print(f'shunted SQUID, 2 fluxes + 2 offsets: max |E_kerman - E_charge| = {deviation:.1e} GHz '
          f'(offset reversal moves levels {reversal:.2f} GHz)')
    assert deviation < 1e-3, deviation   # Charge-basis truncation
    assert reversal > 5e-2


if __name__ == '__main__':
    print('Charge offset in the Kerman basis')
    testTransmon()
    testIslandJunction()
    testOscillatorJunction()
    testDoubleLoop()
    testShuntedSquid()
    print('PASS')
