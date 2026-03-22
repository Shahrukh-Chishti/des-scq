from des_scq import models
from torch import tensor,set_num_threads,stack
from numpy import array,linspace
import scqubits as scq
from des_scq.circuit import Kerman
from des_scq.utils import plotCompare
set_num_threads(32)

Ej = 30.02 ; Ec = 10.2
El = .5
N = 256
flux_range = linspace(0,1,21)
limit = 4

def comparison(N):
    basis = {'O':[N],'I':[],'J':[]}
    SCQ = []
    for flux in flux_range:
        fluxonium = scq.Fluxonium(EJ = Ej,
                                EC = Ec,
                                EL = El,flux=flux,
                                cutoff = N)
        spectrum = fluxonium.eigenvals(evals_count=limit)
        SCQ.append(spectrum)
    #SCQ = fluxonium.get_spectrum_vs_paramvals('flux',flux_range,limit)
    SCQ = array(SCQ).T
    circuit = models.fluxonium(Kerman,basis,El,Ec,Ej)
    print('Fluxonium Parameters:',circuit.circuitComponents())
    flux_manifold = [[tensor(flux)] for flux in flux_range]
    DeS = circuit.spectrumManifold(flux_manifold)
    DeS = stack([val for val,vec in DeS]).detach().numpy().T
    return SCQ,DeS

if __name__ == '__main__':
    print('SCQubit Fluxonium Spectrum')
    SCQ,DeS = comparison(N)
    plotCompare(flux_range,
                {'SCQ-0':SCQ[0],'SCQ-1':SCQ[1],'SCQ-2':SCQ[2],
                 'DeS-0':DeS[0],'DeS-1':DeS[1],'DeS-2':DeS[2]},
                 'Fluxonium Spectrum - Library comparison','flux_ext','spectrum(GHz)',width=5)