from multiprocess import Pool,set_start_method
from des_scq.discovery import uniformParameters,lossTransition
from des_scq import models
from des_scq.optimization import Optimization
from des_scq.circuit import Kerman
from torch import tensor, float32 as float, float64 as double
from numpy import linspace
import pickle,dill
from torch import set_num_threads
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
set_num_threads(6)

def loadTarget():
    with open('./tests/optimization/target_fluxqubit.p', 'rb') as content:
        target_info = pickle.load(content)
    target_spectrum = target_info['spectrum']
    E10 = target_spectrum[:,1] - target_spectrum[:,0]
    E21 = target_spectrum[:,2] - target_spectrum[:,1]
    target = {'E10':E10[[0,20,-1]],'E21':E21[[0,20,-1]]}
    return target

def buildOptimizer(target):
    basis = {'O':[2],'J':[3,3],'I':[]}
    circuit = models.shuntedQubit(Kerman,basis)

    flux_range = linspace(0,1,3,endpoint=True)
    flux_range = tensor(flux_range)
    flux_profile = [[tensor(flux)] for flux in flux_range]

    lossObjective = lossTransition(tensor(target['E10'],dtype=double),tensor(target['E21'],dtype=double))
    optim = Optimization(circuit,flux_profile,loss_function=lossObjective)
    subspace = [component.ID for component in circuit.network]
    return optim,circuit,subspace

def optimization(args):
    parameters,lr,iterations = args
    target = loadTarget()
    optim,circuit,subspace = buildOptimizer(target)
    optim.circuit.initialization(parameters)
    optim.parameters,_ = optim.circuitParameters()
    optim.initAlgo(lr=lr)
    return optim.optimization(iterations=iterations)

if __name__ == '__main__':
    set_start_method('spawn',force=True)
    num_process = 8
    num_inits = 16
    target = loadTarget()
    optim,circuit,subspace = buildOptimizer(target)
    parameters = uniformParameters(circuit,subspace,10,num_inits)
    lr = 1e-3; iterations = 50
    
    arguments = [(para,lr,iterations) for para in parameters]
    optimization(arguments[0])
    print('single agent optimization:pass')
    with Pool(num_process) as multi:
        Search = multi.map(optimization,arguments)
    print(len(Search))