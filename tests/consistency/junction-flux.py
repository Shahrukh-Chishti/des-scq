# External flux through meshes closed by junctions (loops with junctions only).
# Flux is keyed by mesh label (circuit.meshes); Circuit.loopFlux sums, for each
# spanning-tree chord, the fluxes of the meshes its loop encloses, so a mesh flux
# enters the Hamiltonian once whichever junction closes the mesh.
#
# python tests/consistency/junction-flux.py
import networkx
import torch
from torch import tensor
from des_scq import models
from des_scq.circuit import Charge, Kerman, parseMeshes
from des_scq.components import L

torch.set_default_dtype(torch.float64)

N = 6
josephson = [115., 50., 115.]      # SCILLA flux qubit: JJ1 (0-1), JJ2 (0-2), JJ3 (1-2)
cap = [4.0, 4.0, 0.4]              # charging energies (GHz)
fluxes = [0., 0.1, 0.25, 0.4, 0.5]


def fluxQubit():
    circuit = models.fluxShunted(Charge, [N, N], josephson, cap)
    circuit.control_iD = ('A',)    # the single mesh JJ1-JJ3-JJ2
    circuit.spectrum_limit = 3
    return circuit


def transitions(circuit, flux):
    energy, _ = circuit.eigenSpectrum([tensor(float(flux))])
    return (energy[1:] - energy[0]).detach()


def retree(circuit, chord):
    """Rebuild closures with the spanning tree that leaves out junction `chord`."""
    GL = circuit.graphGL()
    tree = networkx.MultiGraph([(u, v, k) for u, v, k in GL.edges(keys=True) if k != chord])
    rank = {key: index for index, (u, v, key) in circuit.edges.items()}
    circuit.closures, circuit.meshes = parseMeshes(GL, tree, rank)
    assert list(circuit.closures) == [chord] and list(circuit.meshes) == ['A']
    return circuit


def testFluxEntersSpectrum():
    circuit = fluxQubit()
    E0, E5 = transitions(circuit, 0.), transitions(circuit, .5)
    print('E01 at flux 0, 1/2:', float(E0[0]), float(E5[0]))
    assert abs(float(E0[0] - E5[0])) > 1e-3


def testGaugeInvariance():
    """Mesh A closed by each of the three junctions: same spectrum."""
    circuits = [retree(fluxQubit(), chord) for chord in ('JJ1', 'JJ2', 'JJ3')]
    for flux in fluxes:
        spectra = [transitions(circuit, flux) for circuit in circuits]
        deviation = max(float((spectrum - spectra[0]).abs().max()) for spectrum in spectra)
        print(f'flux {flux}: max deviation between spanning trees {deviation:.2e} GHz')
        assert deviation < 1e-9


def testPeriodicityAndSymmetry():
    circuit = fluxQubit()
    for flux in fluxes:
        E = transitions(circuit, flux)
        assert float((transitions(circuit, flux + 1.) - E).abs().max()) < 1e-9
        assert float((transitions(circuit, -flux) - E).abs().max()) < 1e-9
    print('periodic in 1 and symmetric under flux -> -flux')


def testLoopFlux():
    """loopFlux: chord loops sum their meshes with the chord sign; tree branches carry none."""
    circuit = models.fluxonium(Kerman, {'O': [16], 'I': [], 'J': []}, El=.5, Ec=1., Ej=5.)
    b = tensor(.31)
    assert circuit.closures == {'JJ': (1, ['A'])}
    assert float(circuit.loopFlux(0, 1, 'JJ', {'A': b})) == float(b)
    assert float(circuit.loopFlux(0, 1, 'I', {'A': b})) == 0.
    assert float(circuit.loopFlux(0, 1, 'JJ', {})) == 0.

    circuit = models.box4Branches(Kerman, {'O': [2] * 3, 'J': [], 'I': []},
                                  [10.] * 4, [10.] * 4, [10.] * 4)
    components = circuit.circuitComposition()
    chord = {meshes[0]: key for key, (_, meshes) in circuit.closures.items()}
    free = [mesh for mesh in circuit.meshes if components[chord[mesh]].__class__ != L]
    phi = {mesh: tensor(10. ** i) for i, mesh in enumerate(free)}   # distinct digit per mesh
    edges = {key: (u, v) for u, v, key in circuit.edges.values()}
    for key, (sign, meshes) in circuit.closures.items():
        expected = sign * sum(float(phi[mesh]) for mesh in meshes if mesh in phi)
        assert float(circuit.loopFlux(*edges[key], key, phi)) == expected, key
    for key in set(edges) - set(circuit.closures):
        assert float(circuit.loopFlux(*edges[key], key, phi)) == 0., key
    blocked = [mesh for mesh in circuit.meshes if mesh not in free]
    for mesh in blocked:
        try:
            circuit.loopFlux(*edges['J0'], 'J0', {mesh: tensor(.1)})
            raise AssertionError(f'mesh {mesh} closed by an inductor accepted flux')
        except NotImplementedError:
            pass
    print(f'loopFlux: box4Branches meshes {free} summed per chord, {blocked} rejected')


def testFluxGradient():
    """d E01 / d mesh flux: autograd against central FD."""
    circuit = fluxQubit()
    circuit.grad_calc = True
    flux = tensor(.3, requires_grad=True)
    energy, _ = circuit.eigenSpectrum([flux])
    gradient, = torch.autograd.grad(energy[1] - energy[0], flux)
    h = 1e-6
    fd = (transitions(circuit, .3 + h)[0] - transitions(circuit, .3 - h)[0]) / (2 * h)
    print(f'dE01/dflux: autograd {float(gradient):.10f}, central FD {float(fd):.10f}')
    assert abs(float(gradient - fd)) <= 1e-6 * max(1., abs(float(fd)))


if __name__ == '__main__':
    print('Mesh flux (loop closed by junctions only)')
    testFluxEntersSpectrum()
    testGaugeInvariance()
    testPeriodicityAndSymmetry()
    testLoopFlux()
    testFluxGradient()
    print('PASS')
