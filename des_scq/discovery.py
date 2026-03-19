import torch
from torch import tensor,abs,stack,var,log
from torch.optim import RMSprop
from numpy import meshgrid,linspace,array,log10,random,logspace
from numpy.random import choice
from random import seed
from scipy.stats import truncnorm
from des_scq.components import Parameter

# Characteristics

def anHarmonicity(spectrum):
    ground,Ist,IInd = spectrum[:3]
    return (IInd-Ist)-(Ist-ground)

# Loss Functions

MSE = torch.nn.MSELoss()

# Full batch of flux_profile
def lossTransitionFlatness(Spectrum,flux_profile):
    spectrum = stack([spectrum[:3] for spectrum,state in Spectrum])
    loss = var(spectrum[:,1]-spectrum[:,0])
    loss += var(spectrum[:,2]-spectrum[:,1])
    return loss,dict()

def lossDegeneracyWeighted(delta0,D0,N=2):
    def Loss(Spectrum,flux_profile):
        spot = 0
        
        E10 = [E[0][1]-E[0][0] for E in Spectrum]
        E20 = [E[0][2]-E[0][0] for E in Spectrum]
        
        Ist = abs(E10[0]-E10[1])
        #E10 = tensor(E10,requires_grad=True)

        # n10 = Spectrum[neighbour][0][1]-Spectrum[neighbour][0][0]
        #        degen_point = flux_profile[0]['Lx']
        #delta = Spectrum[neighbour][0][1]-Spectrum[neighbour][0][0]-e10
        #delta = delta/e10
        #delta = delta.abs()
        #sensitivity = grad(e10,degen_point,create_graph=True)[0] # local fluctuations
        #sensitivity = log(sensitivity.abs())
        
        D = log((E20[0])/(E10[0]))/log(tensor(10.))
        delta = log(Ist/E10[0])/log(tensor(10.))
        loss = delta*delta0 - D*D0
        return loss,{'delta':delta.detach().item(),'D':D.detach().item(),'E10':E10[0].detach().item(),'E20':E20[0].detach().item()}
    return Loss

def lossDegeneracyTarget(delta0,D0):
    def Loss(Spectrum,flux_profile):
        half = 0#int(len(flux_profile)/2)
        neighbour = -1
        e20 = Spectrum[half][0][2]-Spectrum[half][0][0]
        e10 = Spectrum[half][0][1]-Spectrum[half][0][0]
        D = log(e20/e10)
        degen_point = flux_profile[0]['Lx']
        n10 = Spectrum[neighbour][0][1]-Spectrum[neighbour][0][0]
        delta = log((n10-e10).abs()/e20)
        loss = (delta0+delta)**2 + (D0-D)**2
        return loss,{'delta':delta.detach().item(),'D':D.detach().item(),'E10':e10.detach().item(),'E20':e20.detach().item()}
    return Loss

def lossAnharmonicity(alpha):
    def lossFunction(Spectrum,flux_profile):
        anharmonicity = tensor(0.0)
        for spectrum,state in Spectrum:
            anharmonicity += anHarmonicity(spectrum)
        anharmonicity = anharmonicity/len(Spectrum)
        loss = MSE(anharmonicity,tensor(alpha))
        return loss,{'anharmonicity':anharmonicity.detach().item()}
    return lossFunction

def lossTransition(E10,E21):
    def lossFunction(Spectrum,flux_profile):
        spectrum = stack([spectrum[:3] for spectrum,state in Spectrum])
        e10 = spectrum[:,1]-spectrum[:,0]
        e21 = spectrum[:,2]-spectrum[:,1]
        loss = MSE(e10,E10) + MSE(e21,E21)
        mid = int(len(flux_profile)/2)
        log = {'mid10':e10[mid].detach().item(),'mid21':e21[mid].detach().item()}
        return loss,log
    return lossFunction

# Discovery Initialization

def initializationSequential(parameters,optimizer,iterations=100,lr=.005,algo=RMSprop):
    Search = []
    for index,parameter in enumerate(parameters):
        print(index,'--------------------')
        optimizer.circuit.initialization(parameter)
        # circuit.initialization rewrites a new tensor
        optimizer.parameters,_ = optimizer.circuitParameters()
        optimizer.initAlgo(lr=lr,algo=algo)
        Search.append(optimizer.optimization(iterations=iterations))
    return Search

def initializationParallel(optimizer,iterations=100,lr=.005,algo=RMSprop):
    def optimization(parameters):
        optimizer.circuit.initialization(parameters)
        optimizer.parameters,_ = optimizer.circuitParameters()
        optimizer.initAlgo(lr=lr,algo=algo)
        return optimizer.optimization(iterations=iterations)
    return optimization

# Search Space Population

def truncNormalParameters(circuit,subspace,N,var=5):
    # var : std of normal distribution
    iDs,domain = [],[]
    for index,component in enumerate(circuit.network):
        if component.ID in subspace:
            iDs.append(component.ID)
            loc = component.energy().item()
            a,b = component.bounds()
            if a.is_cuda:
                a,b = a.cpu(),b.cpu()
            a = (a - loc)/var ; b = (b - loc)/var
            domain.append(truncnorm.rvs(a,b,loc,var,size=N,random_state=random.seed(101+index)))
    grid = array(domain).T
    return parameterSpace(circuit,grid,iDs)

def uniformParameters(circuit,subspace,n,N,externals=[],random_state=10,logscale=False):
    iDs,grid = [],[]
    seed(random_state)
    
    for component in circuit.network+externals:
        if component.ID in subspace:
            distribution = uniformUnidimensional(component,n,N,logscale)
            grid.append(distribution)
            iDs.append(component.ID)
            
    grid = array(grid).T
    return parameterSpace(circuit,grid,iDs)

def uniformUnidimensional(parameter:Parameter,n:int,N:int,logscale=False):
    spacing = linspace
    if logscale:
        spacing = logspace
    a,b = parameter.bounds()
    a,b = a.item(),b.item()
    if logscale:
        a = log10(a); b = log10(b)
    domain = spacing(a,b,n+1,endpoint=False)[1:]
    return choice(domain,N)

def domainParameters(domain,circuit,subspace):
    grid = array(meshgrid(*domain))
    grid = grid.reshape(len(subspace),-1).T
    return parameterSpace(circuit,grid,subspace)

def parameterSpace(circuit,grid,iDs):
    space = []
    for point in grid:
        state = circuit.circuitState() # static subspace
        state.update(dict(zip(iDs,point)))
        space.append(state)
    return space
