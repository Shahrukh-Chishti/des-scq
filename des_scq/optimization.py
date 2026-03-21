from numpy import array,isnan
from time import perf_counter
import pandas
from torch.optim import Adam,LBFGS
from torch import tensor
from des_scq.utils import empty
from des_scq.components import Control
from des_scq.circuit import Circuit
from inspect import signature

class Optimization:
    def __init__(self,circuit:Circuit,control_profile=[],loss_function=None,external=[]):
        # circuit & data parallel - control profile
        self.circuit = circuit
        # optimization control
        self.control_profile = control_profile
        self.loss_function = loss_function
        # 2 : w/o external parameters, 3 : [ext_params]
        self.loss_arg_count = len(signature(self.loss_function).parameters)
        self.initialization(external=external)
        # depending upon the choice of optimization & loss
        self.vectors_calc = False
        self.grad_calc = True
        self.log_grad = False
        self.log_spectrum = False
        self.log_hessian = False
        self.iteration = 0
        self.initAlgo()

    def initialization(self,parameters=dict(),external=[]):
         # parameters,IDs : circuit space + external
        if len(parameters) > 0:
            self.circuit.initialization(parameters) # update circuit
        self.parameters,self.IDs = self.circuitParameters() # rebuild
        self.Bounds = self.parameterBounds()
        self.external = []
        if len(external) > 0:
            assert self.loss_arg_count > 2
            for parameter in external:
                self.externalParameter(parameter)

    def logInit(self):
        self.logs = []
        self.dParams = [self.parameterState()]
        self.dCircuit = [self.circuitState()]

    def logCompile(self):
        dLog = pandas.DataFrame(self.logs)
        if len(dLog)>0:
            dLog['time'] = dLog['time'].diff()
        else:
            #import pdb;pdb.set_trace()
            print('Failure initial')
        dParams = pandas.DataFrame(self.dParams)
        dCircuit = pandas.DataFrame(self.dCircuit)
        return dLog,dParams,dCircuit

    def circuitID(self):
        IDs = []
        for component in self.circuit.network:
            IDs.append(component.ID)
        return IDs

    def externalParameter(self,parameter:Control):
        # add free parameter, beyond circuit space    
        self.external.append(parameter)
        self.parameters.append(parameter.base)
        self.IDs.append(parameter.ID)
        self.Bounds.append(parameter.bounds())

    def circuitParameters(self,subspace=()):
        # base parameters
        parameters = []; IDs = []
        for ID,parameter in self.circuit.named_parameters(subspace):
            parameters.append(parameter)
            IDs.append(ID)
        return parameters,IDs

    def circuitState(self):
        components = self.circuit.circuitComposition()
        parameters = {}
        for iD in components:
            parameters[iD] = components[iD].energy().item()
        for parameter in self.external:
            parameters[parameter.ID] = parameter.energy().item()
        return parameters

    def parameterState(self):
        components = self.circuit.circuitComposition()
        parameters = {}
        for iD in components:
            parameters[iD] = components[iD].base.item()
        for parameter in self.external:
            parameters[parameter.ID] = parameter.base.item()
        return parameters

    def parameterBounds(self):
        components = self.circuit.circuitComposition()
        Bounds = []
        for iD in components:
            bound = components[iD].bounds()
            Bounds.append(bound) # positive boundary exclusive
        return Bounds

    def modelParameters(self):
        # complete list of model parameters
        # circuit components, external parameters
        parameters = []
        components = self.circuit.circuitComposition()
        for iD in components:
            parameters.append(components[iD].energy())
        for parameter in self.external:
            parameters.append(parameter.energy())
        return parameters

    def loss(self):
        Spectrum = self.circuit.spectrumManifold(self.control_profile)
        loss,metrics = self.loss_function(Spectrum,self.control_profile)
        return loss,metrics,Spectrum

    def lossScape(self,scape,static=dict()):
        # parascape : {A:[...],B:[....]} | A,B in circuit.parameters
        A,B = scape.keys()
        Loss = empty((len(scape[A]),len(scape[B])))
        for id_A,a in enumerate(scape[A]):
            for id_B,b in enumerate(scape[B]):
                point = static.copy()
                point.update({A:a,B:b})
                self.circuit.initialization(point)
                loss,metrics,Spectrum = self.loss()
                Loss[id_A,id_B] = loss.detach().item()
        return Loss.transpose() # A -> X-axis , B -> Y-axis

    def breakPoint(self,logs):
        # break optimization loop : stagnation / spiking
        loss = pandas.DataFrame(logs)['loss'].to_numpy()
        if isnan(loss[-1]):
            print('Loss::NaN')
            print(self.circuitState())
            return True
        if loss[-1] > 1e12 and len(loss) > 10:
            print('Loss:',loss[-1])
            print(self.circuitState())
            return True    
        return False

    def gradients(self):
        gradients = [parameter.grad.detach().item() for parameter in self.parameters]
        gradients = dict(zip(['grad-'+ID for ID in self.IDs],gradients))
        return gradients      

    def initAlgo(self,algo=Adam,lr=1e-3):
        self.optimizer = algo(self.parameters,lr)

    def logger(self,metrics,Spectrum):
        """
            * log every Spectrum decomposition
            * indexing via iteration count
            * additional gradient,Hessian logging
            * assuming apriori backward call
        """
        self.dParams.append(self.parameterState())
        self.dCircuit.append(self.circuitState())
        if self.log_spectrum:
            spectrum = dict()
            for level in range(3):
                spectrum['0-level-'+str(level)] = Spectrum[0][0][level].detach().item()
                spectrum['pi-level-'+str(level)] = Spectrum[int(len(self.control_profile)/2)][0][level].detach().item()
            metrics.update(spectrum)
        if self.log_grad:
            gradients = self.gradients()
            metrics.update(gradients)
        
        self.logs.append(metrics)

    def closure(self):
        self.optimizer.zero_grad()
        loss,metrics,Spectrum = self.loss()
        metrics['loss'] = loss.detach().item()
        metrics['iter'] = self.iteration
        loss.backward(retain_graph=True)
        self.logger(metrics,Spectrum)
        return loss

    def optimization(self,iterations=100):
        start = perf_counter()
        self.logInit()
        for self.iteration in range(iterations):
            self.optimizer.step(self.closure)
            self.logs[-1]['time'] = perf_counter()-start
            if self.breakPoint(self.logs[-15:]):
                print('Optimization Break Point :',self.iteration)
                break

        return self.logCompile()

if __name__=='__main__':
    import torch,sys
    from torch import tensor
    from discovery import lossTransition
    from models import transmon,fluxonium,zeroPi
    from datetime import timedelta
    from models import transmon
    from discovery import lossTransition
    from circuit import Charge
    from torch import tensor
    basis = [128]
    circuit = transmon(Charge,basis)
    print(circuit.circuitComponents())
    flux_profile = [[]]

    lossObjective = lossTransition(tensor([.5],dtype=float),tensor([.25],dtype=float))
    optim = Optimization(circuit,flux_profile,lossObjective)
    optim.initAlgo(lr=1.)
    print(optim.optimizer)
    dLogs,dParams,dCircuit = optim.optimization(iterations=100)
    print(dLogs.iloc[:5])
    print(dLogs.iloc[-5:])