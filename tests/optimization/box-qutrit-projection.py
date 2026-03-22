from des_scq.discovery import uniformParameters,initializationSequential
from des_scq.optimization import Optimization
from des_scq.components import capE,indE,e,h,complex
from torch import tensor,float32 as float,diag,sqrt,abs,log,diagonal
from torch.linalg import norm
from numpy import arange,linspace,array,around
from random import sample,seed
from des_scq.utils import plotCompare,plotHeatmap,plotOptimization
from numpy.linalg import norm
from torch import set_num_threads
from torch.optim import SGD,RMSprop,Adam
from des_scq.circuit import Kerman, hamiltonianEnergy, phase
from des_scq.components import J,C,L,pi,h
from des_scq.components import C0,J0,L0,capE,indE,C_,J_,L_
set_num_threads(36)
seed(10)

El,Ec,Ej = 10,10,10
flux_range = linspace(0,1,1,endpoint=True)

circuit = [L(0,1,El,'L0',True,L0,L_),C(0,1,Ec,'C0',C0,C_),J(0,1,Ej,'J0',J0,J_)]
circuit += [L(1,2,El,'L1',True,L0,L_),C(1,2,Ec,'C1',C0,C_),J(1,2,Ej,'J1',J0,J_)]
circuit += [L(2,3,El,'L2',True,L0,L_),C(2,3,Ec,'C2',C0,C_),J(2,3,Ej,'J2',J0,J_)]
circuit += [L(3,0,El,'L3',True,L0,L_),C(3,0,Ec,'C3',C0,C_),J(3,0,Ej,'J3',J0,J_)]

n_base = 2
basis = {'O':[n_base]*3,'J':[],'I':[]}; rep = 'K'
control_iD = ['L0']

circuit = Kerman(circuit,control_iD,basis)
N = circuit.basisSize()

def nodeChargeExcitation(a,node,b):
    Lo_,C_ = circuit.kermanComponents()
    Co_,Coi_,Coj_,Ci_,Cij_,Cj_ = C_
    Z = sqrt(diagonal(Co_)/diagonal(Lo_))
    #print('Impedance',Z)

    O = circuit.backend.null(N,complex)
    basis = circuit.basis['O']
    R = circuit.R.T

    Qo = [circuit.backend.basisQo(basis_max,Zi) for Zi,basis_max in zip(Z,basis)]
    for i,r in enumerate(R[node]):
        O += circuit.backend.basisProduct(Qo,[i])*r
    transition = a.conj().T@O@b
    return abs(transition)

def ladderQutrit(detuning,cutoff=16,node=None,levels=True):
    def costFunction(Spectrum,flux_profile):
        metrics = dict()
        loss = tensor(0.)
        if levels:
            e32 = Spectrum[0][0][3]-Spectrum[0][0][2]
            e21 = Spectrum[0][0][2]-Spectrum[0][0][1]
            e30 = Spectrum[0][0][3]-Spectrum[0][0][0]
            e20 = Spectrum[0][0][2]-Spectrum[0][0][0]
            e10 = Spectrum[0][0][1]-Spectrum[0][0][0]

            d1 = abs(e21-e10); d2 = abs(e32-e21)-detuning
            d3 = abs(e32-e10)-detuning ; d4 = abs(e30-cutoff)
            loss += (d1**2)+(d2**2)+(d3**2)+(d4**2)
            loss /= 16.
            distances = [d.detach().item() for d in (d1,d2,d3,d4)]
            metrics.update(dict(zip(('d1','d2','d3','d4'),distances)))
        if node is not None:
            loss *= 16.
            ground = Spectrum[0][1][0]
            I = Spectrum[0][1][1]
            II = Spectrum[0][1][2]
            ground.retain_grad()
            if ground.grad is not None:
                print(norm(ground.grad))
            d5 = abs(nodeChargeExcitation(ground,node,I)-1)
            d6 = abs(nodeChargeExcitation(I,node,II)-1)
            d7 = abs(nodeChargeExcitation(ground,node,II))
            loss += d5**2 + d6**2 + d7**2
            loss /= 49.
            distances = [d.detach().item() for d in (d5,d6,d7)]
            metrics.update(dict(zip(('d5','d6','d7'),distances)))
        return loss,metrics
    return costFunction

if __name__ == '__main__':
    flux_profile = [[flux] for flux in tensor(flux_range)]

    iterations = 100
    lr = .01 # without control overlap
    lossObjective = ladderQutrit(5,node=None,levels=True)
    optim = Optimization(circuit,flux_profile,loss_function=lossObjective)

    circuit.vectors_calc = True
    optim.log_grad = True

    subspace = [component.ID for component in circuit.network]
    n_grid = 16
    n_init = 2
    parameters = uniformParameters(circuit,subspace,n_grid,n_init,logscale=True)

    FullSpace = initializationSequential(parameters,optim,iterations=iterations,lr=lr,algo=Adam)
    index = 0
    print(FullSpace[index][0])
    print(FullSpace[index][1])
    dLogs,dParams,dCircuit = FullSpace[0]
    plotCompare(dLogs.index,dLogs,'Optimizing Box-Qutrit','iteration')