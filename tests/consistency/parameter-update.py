# Parameter updates reach the Charge-basis Hamiltonian.
# Charge.hamiltonianLC rebuilds Cn_, Ln_ from the current elements (as Kerman does);
# before, they were frozen at construction, so capacitor updates were ignored in the
# forward pass. Also: inverse() no longer stops at a breakpoint on a singular matrix.
#
# python tests/consistency/parameter-update.py
import torch
from torch import tensor
from des_scq import models
from des_scq.circuit import Charge, inverse

torch.set_default_dtype(torch.float64)

N = 15
Ej, Ec = 20., .3


def transmonFrequency(Ej, Ec):
    return (8 * Ej * Ec) ** .5 - Ec


def f01(circuit):
    energy, _ = circuit.eigenSpectrum([])
    return energy[1] - energy[0]


def testCapacitorUpdate():
    circuit = models.transmon(Charge, [N], Ej=Ej, Ec=Ec)
    before = float(f01(circuit))
    state = circuit.circuitState()
    state['C'] = .2
    circuit.initialization(state)
    after = float(f01(circuit))
    print(f'f01(Ec=0.3) = {before:.6f} (~{transmonFrequency(Ej, .3):.6f}), '
          f'f01(Ec=0.2) = {after:.6f} (~{transmonFrequency(Ej, .2):.6f})')
    assert abs(before / transmonFrequency(Ej, .3) - 1) < .02
    assert abs(after / transmonFrequency(Ej, .2) - 1) < .02


def testCapacitorGradient():
    """d f01 / d (C base): autograd against central FD on the base (in-place updates,
    as an optimizer does)."""
    circuit = models.transmon(Charge, [N], Ej=Ej, Ec=Ec)
    base = circuit.circuitComposition()['C'].base
    gradient, = torch.autograd.grad(f01(circuit), base)
    h = 1e-5
    with torch.no_grad():
        base += h
        plus = float(f01(circuit))
        base -= 2 * h
        minus = float(f01(circuit))
        base += h
    fd = (plus - minus) / (2 * h)
    print(f'df01/dbase_C: autograd {float(gradient):.10f}, central FD {fd:.10f}')
    assert abs(fd) > 1e-3
    assert abs(float(gradient) - fd) <= 1e-6 * abs(fd)


def testSingularInverse():
    A = tensor([[1., 0.], [0., 0.]])
    B = inverse(A.clone())
    print('inverse of a singular matrix returns:', B.tolist())
    assert torch.isfinite(B).all()


if __name__ == '__main__':
    print('Parameter updates in the Charge basis')
    testCapacitorUpdate()
    testCapacitorGradient()
    testSingularInverse()
    print('PASS')
