# Flux-shunted qubit with an external flux through its junction-only loop:
# des-scq (flux through mesh A) against scqubits (flux on its closure branch).
#
# python tests/comparison/scQ-flux-shunted-flux.py
import torch
from torch import tensor
from numpy import array, linspace
import scqubits as scq
from des_scq import models
from des_scq.circuit import Charge
from des_scq.utils import plotCompare

torch.set_default_dtype(torch.float64)

Ej = 30.02 ; Ec = 10.2
N = 8
josephson = [2*Ej, 12.5*Ej, 2*Ej]
cap = [25*Ec, Ec, 5*Ec]
flux_list = linspace(0, 1, 41)


def scqCircuit(N, josephson, cap):
    parameters = array(list(zip(josephson, cap))).flatten()
    yaml_inp = """branches:
    - [JJ, 0, 1, {}, {}]
    - [JJ, 0, 2, {}, {}]
    - [JJ, 1, 2, {}, {}]
    """.format(*parameters)
    circuit = scq.Circuit(yaml_inp, from_file=False, ext_basis="harmonic")
    circuit.cutoff_n_1 = N
    circuit.cutoff_n_2 = N
    return circuit


def comparison(N, josephson, cap, levels=4):
    circuit = models.fluxShunted(Charge, [N, N], josephson, cap)
    circuit.control_iD = ('A',)   # the single mesh JJ1-JJ3-JJ2
    circuit.spectrum_limit = levels
    reference = scqCircuit(N, josephson, cap)
    symbol = reference.external_fluxes[0]
    SCQ, DeS = [], []
    for flux in flux_list:
        setattr(reference, symbol.name, flux)
        spectrum = reference.eigenvals(evals_count=levels)
        SCQ.append(spectrum - spectrum[0])
        energy, _ = circuit.eigenSpectrum([tensor(float(flux))])
        energy = energy.detach().numpy()
        DeS.append(energy - energy[0])
    return array(SCQ).T, array(DeS).T


if __name__ == '__main__':
    print('Flux Shunted qubit with loop flux: des-scq v/s scqubits')
    SCQ, DeS = comparison(N, josephson, cap)
    print('max |E0i(des-scq) - E0i(scqubits)| over flux =', abs(SCQ - DeS).max(), 'GHz')
    plotCompare(flux_list,
                {'SCQ-1': SCQ[1], 'SCQ-2': SCQ[2], 'SCQ-3': SCQ[3],
                 'des_scq-1': DeS[1], 'des_scq-2': DeS[2], 'des_scq-3': DeS[3]},
                'Flux Shunted qubit Spectrum vs loop flux - Library comparison',
                'flux (Phi0)', 'E - E0 (GHz)', width=5, html=True)
