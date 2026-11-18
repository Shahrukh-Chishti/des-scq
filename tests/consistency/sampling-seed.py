# uniformParameters is reproducible: random_state seeds numpy, which draws the samples
# (numpy.random.choice); before, it seeded Python's `random` and had no effect.
#
# python tests/consistency/sampling-seed.py
import torch
from des_scq import models
from des_scq.circuit import Charge
from des_scq.discovery import uniformParameters

torch.set_default_dtype(torch.float64)


def sample(random_state):
    circuit = models.transmon(Charge, [5], Ej=20., Ec=.3)
    return uniformParameters(circuit, ('J', 'C'), n=50, N=8, random_state=random_state)


def testSameSeedSameSamples():
    assert sample(3) == sample(3)
    print('same random_state -> same parameter sets')


def testDifferentSeedDifferentSamples():
    assert sample(3) != sample(4)
    print('different random_state -> different parameter sets')


if __name__ == '__main__':
    print('Seeded uniform sampling')
    testSameSeedSameSamples()
    testDifferentSeedDifferentSamples()
    print('PASS')
