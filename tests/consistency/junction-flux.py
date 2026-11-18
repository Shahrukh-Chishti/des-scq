# External flux assigned to a junction (loops closed by junctions only).
# Circuit.loopFlux adds external_fluxes[key] for the junction `key` itself and skips
# junction IDs on the spanning-tree path, so a junction-only loop carries the flux once.
#
# python tests/consistency/junction-flux.py
import torch
from torch import tensor
from des_scq import models
from des_scq.circuit import Charge, Kerman

torch.set_default_dtype(torch.float64)

N = 6
josephson = [115., 50., 115.]      # SCILLA flux qubit: JJ1 (0-1), JJ2 (0-2), JJ3 (1-2)
cap = [4.0, 4.0, 0.4]              # charging energies (GHz)
fluxes = [0., 0.1, 0.25, 0.4, 0.5]


def fluxQubit(junction):
    circuit = models.fluxShunted(Charge, [N, N], josephson, cap)
    circuit.control_iD = (junction,)
    circuit.spectrum_limit = 3
    return circuit


def transitions(circuit, flux):
    energy, _ = circuit.eigenSpectrum([tensor(float(flux))])
    return (energy[1:] - energy[0]).detach()


def testFluxEntersSpectrum():
    circuit = fluxQubit('JJ3')
    E0, E5 = transitions(circuit, 0.), transitions(circuit, .5)
    print('E01 at flux 0, 1/2:', float(E0[0]), float(E5[0]))
    assert abs(float(E0[0] - E5[0])) > 1e-3


def testGaugeInvariance():
    """Flux on any of the three junctions: same spectrum (loop flux counted once)."""
    for flux in fluxes:
        spectra = [transitions(fluxQubit(junction), flux) for junction in ('JJ1', 'JJ2', 'JJ3')]
        deviation = max(float((spectrum - spectra[0]).abs().max()) for spectrum in spectra)
        print(f'flux {flux}: max deviation between junction gauges {deviation:.2e} GHz')
        assert deviation < 1e-9


def testPeriodicityAndSymmetry():
    circuit = fluxQubit('JJ3')
    for flux in fluxes:
        E = transitions(circuit, flux)
        assert float((transitions(circuit, flux + 1.) - E).abs().max()) < 1e-9
        assert float((transitions(circuit, -flux) - E).abs().max()) < 1e-9
    print('periodic in 1 and symmetric under flux -> -flux')


def testLoopFluxAdditivity():
    """Fluxonium: inductor flux unchanged; a junction flux adds to it."""
    torch.set_default_dtype(torch.float64)
    circuit = models.fluxonium(Kerman, {'O': [16], 'I': [], 'J': []}, El=.5, Ec=1., Ej=5.)
    a, b = tensor(.13), tensor(.31)
    assert float(circuit.loopFlux(0, 1, 'JJ', {'I': b})) == float(b)
    assert float(circuit.loopFlux(0, 1, 'JJ', {'JJ': a})) == float(a)
    assert abs(float(circuit.loopFlux(0, 1, 'JJ', {'JJ': a, 'I': b})) - float(a + b)) < 1e-15
    print('loopFlux: inductor flux, junction flux and their sum')


def testFluxGradient():
    """d E01 / d flux through the junction flux: autograd against central FD."""
    circuit = fluxQubit('JJ3')
    circuit.grad_calc = True
    flux = tensor(.3, requires_grad=True)
    energy, _ = circuit.eigenSpectrum([flux])
    gradient, = torch.autograd.grad(energy[1] - energy[0], flux)
    h = 1e-6
    fd = (transitions(circuit, .3 + h)[0] - transitions(circuit, .3 - h)[0]) / (2 * h)
    print(f'dE01/dflux: autograd {float(gradient):.10f}, central FD {float(fd):.10f}')
    assert abs(float(gradient - fd)) <= 1e-6 * max(1., abs(float(fd)))


if __name__ == '__main__':
    print('Junction flux (loop closed by junctions only)')
    testFluxEntersSpectrum()
    testGaugeInvariance()
    testPeriodicityAndSymmetry()
    testLoopFluxAdditivity()
    testFluxGradient()
    print('PASS')
