from des_scq.models import transmon
from des_scq.components import float
from des_scq.discovery import lossTransition
from des_scq.optimization import Optimization
from des_scq.circuit import Charge
from torch import tensor
from des_scq.utils import plotCompare
from torch import set_num_threads
set_num_threads(30)

if __name__ == "__main__":
    basis = [256]
    circuit = transmon(Charge,basis)
    print(circuit.circuitComponents())
    flux_profile = [[]]

    lossObjective = lossTransition(tensor([.5],dtype=float),tensor([.25],dtype=float))
    optim = Optimization(circuit,flux_profile,lossObjective)
    optim.initAlgo(lr=1.)
    print(optim.optimizer)
    dLogs,dParams,dCircuit = optim.optimization(iterations=1000)

    plotCompare(dLogs.index,dLogs,'Optimizing Transmon','iteration')
    plotCompare(dParams.index,dParams,None,"iteration","parameters")
    plotCompare(dCircuit.index,dCircuit,None,"iteration","energy")