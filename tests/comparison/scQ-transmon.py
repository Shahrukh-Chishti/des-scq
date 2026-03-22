from des_scq import models
from torch import tensor
from torch.linalg import eigvalsh
from torch import set_num_threads
from numpy import array,linspace
import scqubits as scq
from des_scq.circuit import Charge
from des_scq.utils import plotCompare
set_num_threads(32)

Ej = 30.02 ; Ec = 10.2
Ej = Ec*1.
N = 512
ng_list = linspace(-2, 2, 220)

def comparison(N):
    basis = [N]
    circuit = models.transmon(Charge,basis,Ej,Ec)
    H_LC = circuit.hamiltonianLC()
    H_J = circuit.hamiltonianJosephson()

    print('Transmon Parameters:',circuit.circuitComponents())

    SCQ,DeS = [],[]

    for q in ng_list:
        # calculating eigspectrum at each charge offset
        tmon = scq.Transmon(EJ=Ej,EC=Ec,ng=q,ncut=512)
        spectrum = tmon.eigenvals(evals_count=4)
        SCQ.append(spectrum)
        offset = dict([(1,tensor(q))])
        H_off = circuit.hamiltonianChargeOffset(offset)
        H = H_LC+H_J+H_off
        spectrum = eigvalsh(H)[:4]
        DeS.append(spectrum.detach().numpy())

    return array(SCQ).T,array(DeS).T

if __name__ == '__main__':
    print('SCQubit Transmon Spectrum')
    tmon = scq.Transmon(
        EJ=Ej,
        EC=Ec,
        ng=0.0,
        ncut=N)

    #fig, axes = tmon.plot_evals_vs_paramvals('ng', ng_list, evals_count=3, subtract_ground=False)
    SCQ,DeS = comparison(N)
    plotCompare(ng_list,
                {'SCQ-0':SCQ[0],'SCQ-1':SCQ[1],'SCQ-2':SCQ[2],
                 'des_scq-0':DeS[0],'des_scq-1':DeS[1],'des_scq-2':DeS[2]},
                 'Transmon Spectrum - Library comparison','n_g','spectrum(GHz)',width=5)
