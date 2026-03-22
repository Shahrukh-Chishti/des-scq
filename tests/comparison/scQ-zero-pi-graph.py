from des_scq import models
from torch import tensor
from torch.linalg import eigvalsh
from torch import set_num_threads
from numpy import array,linspace,pi
import scqubits as scq
from des_scq.circuit import Charge
from des_scq.utils import plotCompare
set_num_threads(32)

Ej = 30.02 ; Ec = 10.2
EcJ = 50. ; El = 5.
N = 3
ng_list = linspace(-1, 1, 101)
flux_list = linspace(0,1,21)

def scqCircuitGraph(N,Ej,Ec,El,EcJ):
    # graphical implementation of ZeroPi circuit
    parameters = array([El,El,Ec,Ec,Ej,EcJ,Ej,EcJ])
    yaml_inp = """branches:
    - [L, 0, 1, {}]
    - [L, 2, 3, {}]
    - [C, 1, 2, {}]
    - [C, 3, 0, {}]
    - [JJ, 1, 3, {}, {}]
    - [JJ, 2, 0, {}, {}]
    """.format(*parameters)
    circuit = scq.Circuit(yaml_inp, from_file=False, ext_basis="harmonic")
    print(circuit.cutoff_names)
    print(circuit.sym_external_fluxes())
    circuit.cutoff_n_1 = N
    circuit.cutoff_ext_2 = N
    circuit.cutoff_ext_3 = N
    return circuit

def scqZeroPi(N,Ej,Ec,El,EcJ):
    # inbuilt reduced-symmetric ZeroPi model
    phi_grid = scq.Grid1d(-6*pi, 6*pi, 200)
    circuit = scq.ZeroPi(grid = phi_grid,
                           EJ   = Ej,
                           EL   = El,
                           ECJ  = EcJ,
                           EC   = Ec,
                           ng   = 0.,
                           flux = 0.,
                           ncut = N)
    return circuit

def comparison(N):
    zeroPi = scqZeroPi(N,Ej,Ec,El,EcJ)
    graphical = scqCircuitGraph(N,Ej,Ec,El,EcJ)
    basis = [N,N,N]
    circuit = models.zeroPi(Charge,basis,Ej,Ec,El,EcJ,symmetry=True)
    print(circuit.control_iD)

    print('Zero-Pi qubit - Parameters:',circuit.circuitComponents())

    ZP = zeroPi.get_spectrum_vs_paramvals(param_name='flux', param_vals=flux_list)
    ZP = ZP.energy_table.T[:3]
    SCQ = []
    for flux in flux_list:
        graphical.Φ1 = flux
        spectrum  = graphical.eigenvals(evals_count=4)
        SCQ.append(spectrum)

    flux_manifold = [[tensor(flux),tensor(0.)] for flux in flux_list]
    DeS = [spec.detach().numpy() for spec,vec in circuit.spectrumManifold(flux_manifold)]
    return ZP,array(SCQ).T,array(DeS).T

if __name__ == '__main__':
    print('Comparing SCQubit Zero-Pi(graphical) Spectrum')

    ZP,SCQ,DeS = comparison(N)
    plotCompare(flux_list,
                {'ZP-0':ZP[0]-ZP[0],'ZP-1':ZP[1]-ZP[0],'ZP-2':ZP[2]-ZP[0],
                 'SCQ-0':SCQ[0]-SCQ[0],'SCQ-1':SCQ[1]-SCQ[0],'SCQ-2':SCQ[2]-SCQ[0],
                 'des_scq-0':DeS[0]-DeS[0],'des_scq-1':DeS[1]-DeS[0],'des_scq-2':DeS[2]-DeS[0]},
                 'Zero-Pi qubit Spectrum - Library comparison','flux','spectrum(GHz)',width=5,html=True)
