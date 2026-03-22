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
N = 8
josephson = [2*Ej,12.5*Ej,2*Ej]
cap = [25*Ec,Ec,5*Ec]
ng_list = linspace(-1, 1, 101)

def fluxShunted(Rep,basis,josephson=[120.,50,120.],cap=[10.,50.,10.],symmetry=False):
    from des_scq.models import J,C
    Ej1,Ej2,Ej3 = josephson
    C1,C2,C3 = cap

    circuit = [J(0,1,Ej1,'JJ1'),C(0,1,C1,'C1')]
    circuit += [J(0,2,Ej2,'JJ2'),C(0,2,C2,'C2')]
    circuit += [J(1,2,Ej3,'JJ3'),C(1,2,C3,'C3')]
    
    # inbuilt symmetry
    pairs = dict()
    if symmetry:
        pairs['JJ2'] = 'JJ1'
        pairs['C2'] = 'C1'
    
    circuit = Rep(circuit,(),basis,pairs)
    return circuit

def scqCircuit(N,josephson,cap):
    parameters = array(list(zip(josephson,cap))).flatten()
    yaml_inp = """branches:
    - [JJ, 0, 1, {}, {}]
    - [JJ, 0, 2, {}, {}]
    - [JJ, 1, 2, {}, {}]
    """.format(*parameters)
    circuit = scq.Circuit(yaml_inp, from_file=False, ext_basis="harmonic")
    circuit.cutoff_n_1 = N
    circuit.cutoff_n_2 = N

    return circuit

def comparison(N,josephson,cap):
    basis = [N,N]
    circuit = fluxShunted(Charge,basis,josephson,cap)
    H_LC = circuit.hamiltonianLC()
    H_J = circuit.hamiltonianJosephson()

    print('Flux Shunted qubit - Parameters:',circuit.circuitComponents())

    SCQ,DeS = [],[]

    # scq node indexing is offset by 1, from des_scq
    for q in ng_list:
        # calculating eigspectrum at each charge offset
        flunt = scqCircuit(N,josephson,cap)
        flunt.ng2 = q
        spectrum = flunt.eigenvals(evals_count=4)
        SCQ.append(spectrum)
        offset = dict([(1,tensor(q))])
        H_off = circuit.hamiltonianChargeOffset(offset)
        H = H_LC+H_J+H_off
        spectrum = eigvalsh(H)[:4]
        DeS.append(spectrum.detach().numpy())

    return array(SCQ).T,array(DeS).T

if __name__ == '__main__':
    print('SCQubit Flux Shunted qubit Spectrum')
    flunt = scqCircuit(N,josephson,cap)

    #fig, axes = flunt.plot_evals_vs_paramvals('ng', ng_list, evals_count=3, subtract_ground=False)
    SCQ,DeS = comparison(N,josephson,cap)
    plotCompare(ng_list,
                {'SCQ-0':SCQ[0],'SCQ-1':SCQ[1],'SCQ-2':SCQ[2],
                 'des_scq-0':DeS[0],'des_scq-1':DeS[1],'des_scq-2':DeS[2]},
                 'Flux Shunted qubit Spectrum - Library comparison','n_g','spectrum(GHz)',width=5,html=True)
