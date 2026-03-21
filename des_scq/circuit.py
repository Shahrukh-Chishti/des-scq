import networkx,copy,torch
from contextlib import nullcontext
from torch import exp,det,norm,tensor,arange,zeros,sqrt,diagonal,lobpcg,rand,eye
from torch.linalg import eigvalsh,inv
from numpy.linalg import matrix_rank,eigvalsh as eigenvalues
from numpy import prod,sort
from des_scq.dense import *
from des_scq.components import diagonalisation,J,L,C,im,pi,complex,float

### Computational Sub-routines

def inverse(A,zero=1e-15):
    if det(A) == 0:
        #return zeros_like(A)
        D = A.diag()
        A[D==0,D==0] = tensor(1/zero)
        import pdb;pdb.set_trace()
    try:
        A = inv(A)
    except:
        import pdb;pdb.set_trace()
    #import pdb;pdb.set_trace()    
    #A[A<=zero] = tensor(0.)
    return A

def phase(phi):
    # phi = flux/flux_quanta
    return exp(im*2*pi*phi)

def hamiltonianEnergy(H):
    eigenenergies = eigvalsh(H)
    return eigenenergies

### Circuit Modue
class Circuit():
    """
        * no external fluxes must be in parallel : redundant
        * no LC component must be in parallel : redundant
        * only Josephson elements allowed in parallel
        * u,v : indices of raw graph : GL, minimum_spanning_tree
        * i,j : indices of indexed graph : mode correspondence
    """

    def __init__(self,network,control_iD,basis,pairs=dict(),device=None):
        super().__init__()
        # circuit network
        self.network = network
        self.control_iD = control_iD
        self.G = self.parseCircuit()
        self.spanning_tree = self.spanningTree()
        self.nodes,self.nodes_ = self.nodeIndex()
        self.edges,self.edges_inductive = self.edgesIndex()
        self.Nn = len(self.nodes)
        self.Ne = len(self.edges)
        self.Nb = len(self.edges_inductive)
        self.pairs = pairs
        self.symmetrize(self.pairs)
        # circuit components
        self.Cn_,self.Ln_ = self.componentMatrix()

        self.basis = basis
        # basis : list of basis_size of ith mode
        # basis : dict of O,I,J : list of basis_size

        self.device = device
        self.null_flux = tensor(0.,device=self.device)
        # basis Size undefined in Root class
        self.null = null(self.basisSize(),device=self.device)

        self.spectrum_limit = 4
        self.ii_limit = 3
        self.grad_calc = True

    def initialization(self,parameters):
        # parameters : GHz unit
        for component in self.network:
            if component.__class__ == C :
                component.initCap(parameters[component.ID])
            elif component.__class__ == L :
                component.initInd(parameters[component.ID])
            elif component.__class__ == J :
                component.initJunc(parameters[component.ID])
        self.symmetrize(self.pairs)

    def named_parameters(self,subspace=(),recurse=False):
        parameters = []; IDs = []
        slaves = self.pairs.keys()
        for component in self.network:
            if component.ID in subspace or len(subspace)==0:
                parameter = component.base
                if not component.ID in slaves:
                    IDs.append(component.ID)
                    parameters.append(parameter)
        return zip(IDs,parameters)

    def symmetrize(self,pairs):
        components = self.circuitComposition()
        for slave,master in pairs.items():
            master = components[master]
            slave = components[slave]
            slave.base = master.base

    def parseCircuit(self):
        G = networkx.MultiGraph()
        for component in self.network:
            weight = 1e-3
            if component.__class__ == J:
                weight = component.energy().item() # scalar value
            G.add_edge(component.plus,component.minus,key=component.ID,weight=weight,component=component)
        return G

    def nodeIndex(self):
        # nodes identify variable placeholders
        nodes = list(self.G.nodes())
        nodes.remove(0) # removing ground from active nodes
        nodes = dict([*enumerate(nodes)])

        nodes_ = {val:key for key,val in nodes.items()}
        nodes_[0] = -1
        return nodes,nodes_

    def edgesIndex(self):
        """
            {(u,v):[,,,]} : MultiGraph edges
            inductive edges ... josephson/capactive edges
        """
        edges_G = self.G.edges(keys=True)
        index_plus,index_minus = 0, len(edges_G)-1
        edges,edges_inductive = dict(), dict()
        for u,v,key in edges_G:
            component = self.G[u][v][key]['component']
            if component.__class__ == L:
                edges_inductive[index_plus] = (u,v,key)
                edges[index_plus] = (u,v,key)
                index_plus += 1
            else:
                edges[index_minus] = (u,v,key)
                index_minus -= 1
        return edges,edges_inductive

    def spanningTree(self):
        GL = self.graphGL()
        S = networkx.minimum_spanning_tree(GL)
        return S

    def graphGL(self,elements=[C]):
        GL = copy.deepcopy(self.G)
        edges = []
        for u,v,component in GL.edges(data=True):
            component = component['component']
            if component.__class__ in elements:
                edges.append((u,v,component.ID))
        GL.remove_edges_from(edges)

        return GL

    def circuitState(self):
        parameters = {}
        for component in self.network:
            parameters[component.ID] = component.energy().item()
        return parameters

    def circuitComposition(self):
        components = dict()
        for component in self.network:
            components[component.ID] = component
        return components

    def circuitComponents(self):
        circuit_components = dict()
        for component in self.network:
            if component.__class__ == C :
                circuit_components[component.ID] = component.capacitance().item()
            elif component.__class__ == L :
                circuit_components[component.ID] = component.inductance().item()
            elif component.__class__ == J :
                circuit_components[component.ID] = component.energy().item()
        return circuit_components

    def componentMatrix(self):
        Cn = self.nodeCapacitance()
        assert not det(Cn)==0
        Cn_ = inverse(Cn)
        Rbn = self.connectionPolarity()
        Lb = self.branchInductance()
        M = self.mutualInductance()
        L_inv = inverse(Lb+M)
        Ln_ = Rbn.conj().T @ L_inv @ Rbn

        return Cn_,Ln_

    def loopFlux(self,u,v,key,external_fluxes):
        """
            external_fluxes : {identifier:flux_value}
        """
        flux = self.null_flux.clone()
        external = set(external_fluxes.keys())
        S = self.spanning_tree
        path = networkx.shortest_path(S,u,v)
        for i in range(len(path)-1):
            multi = S.get_edge_data(path[i],path[i+1])
            match = external.intersection(set(multi.keys()))
            # multiple external flux forbidden on same branch
            assert len(match) <= 2
            if len(match)==1 :
                component = multi[match.pop()]['component']
                assert component.__class__ == L
                assert component.external == True
                flux += external_fluxes[component.ID]

        return flux

    def josephsonComponents(self):
        edges,Ej = [],[]
        for index,(u,v,key) in self.edges.items():
            component = self.G[u][v][key]['component']
            if component.__class__ == J:
                edges.append((u,v,key))
                Ej.append(component.energy())
        return edges,Ej

    def fluxBiasComponents(self):
        """
            Inducer : Inductor introduces external flux bias
        """
        edges,L_ext = [],[]
        for index,(u,v,key) in self.edges.items():
            component = self.G[u][v][key]['component']
            if component.__class__ == L:
                if component.external:
                    edges.append((u,v,key))
                    L_ext.append(component.inductance)
        return edges,L_ext

    def nodeCapacitance(self):
        Cn = zeros((self.Nn,self.Nn),dtype=float)
        for i,node in self.nodes.items():
            for u,v,component in self.G.edges(node,data=True):
                component = component['component']
                if component.__class__ == C:
                    capacitance = component.capacitance()
                    Cn[i,i] += capacitance
                    if not (u==0 or v==0):
                        Cn[self.nodes_[u],self.nodes_[v]] = -capacitance
                        Cn[self.nodes_[v],self.nodes_[u]] = -capacitance
        return Cn

    def branchInductance(self):
        Lb = zeros((self.Nb,self.Nb),dtype=float)
        #fill_diagonal(Lb,L_limit)
        for index,(u,v,key) in self.edges_inductive.items():
            component = self.G[u][v][key]['component']
            if component.__class__ == L:
                #if not component.external:
                Lb[index,index] = component.inductance()
        return Lb

    def mutualInductance(self):
        M = zeros((self.Nb,self.Nb),dtype=float) 
        return M

    def connectionPolarity(self):
        Rbn = zeros((self.Nb,self.Nn),dtype=float)
        for index,(u,v,key) in self.edges_inductive.items():
            if not u==0:
                Rbn[index][self.nodes_[u]] = 1
            if not v==0:
                Rbn[index][self.nodes_[v]] = -1

        return Rbn

    def modeImpedance(self):
        Cn_,Ln_,basis = self.Cn_,self.Ln_,self.basis
        impedance = [sqrt(Cn_[i,i]/Ln_[i,i]) for i in range(len(basis))]
        return impedance
    
    def islandModes(self):
        islands = self.graphGL(elements=[C])
        islands = networkx.connected_components(islands)
        islands = list(islands)
        Ni = 0
        for sub in islands:
            if 0 not in sub:
                Ni += 1
        return Ni

    def fluxInducerEnergy(self):
        basis = self.basis
        # basis for Flux modes
        fluxModes = [basisFf(basis_max) for basis_max in basis]
        edges,L_ext = self.fluxBiasComponents()
        H = null() # tensor([[0.0]])
        for (u,v,key),L in zip(edges,L_ext):
            i,j = self.nodes_[u],self.nodes_[v]
            if i<0 or j<0 :
                # grounded inducer
                i = max(i,j)
                P = basisProduct(fluxModes,[i])
            else:
                P = basisProduct(fluxModes,[i]) - basisProduct(fluxModes,[j])
            H = H + P@P / 2 / L
        return H

    def basisSize(self,modes=False):
        N = [size for size in self.basis]
        if modes:
            return N
        return prod(N)

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=False)
    def circuitHamiltonian(self,control):
        # torch reconstruct computation graph repeatedly
        H = self.hamiltonianLC()
        # Offset & Josephson sector
        control_flux,control_charge= control
        H += self.hamiltonianJosephson(control_flux)
        if len(control_charge) > 0:
            H += self.hamiltonianChargeOffset(control_charge)
        return H

    def controlData(self,control):
        if len(self.control_iD) == 0:
            return dict(),dict()
        assert len(control) == len(self.control_iD)
        # control channel segregation
        # str : branch, flux
        # int : node, charge
        control_flux,control_charge = dict(),dict()
        for iD,ctrl in zip(self.control_iD,control):
            assert type(iD) is str or type(iD) is int
            if type(iD) is str:
                control_flux[iD] = ctrl
            elif type(iD) is int:
                control_charge[iD] = ctrl
        control = control_flux,control_charge
        return control

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=False)
    def eigenSpectrum(self,control):
        # control data : array/list of Tensor/s
        control = self.controlData(control)
        with torch.inference_mode() if self.grad_calc is False else nullcontext() as null:
            H = self.circuitHamiltonian(control)
            spectrum = eigvalsh(H)[:self.spectrum_limit]
            states = None
        return spectrum,states

    def spectrumManifold(self,manifold):
        # sequential execution within batch of control points
        Spectrum = [] # check autograd over creation
        # States = []
        for control in manifold:
            spectrum,states = self.eigenSpectrum(control)
            Spectrum.append((spectrum,states))
            # Spectrum.append(spectrum);States.append(states)
        return Spectrum

    def fluxScape(self,flux_points,flux_manifold):
        H_LC = self.hamiltonianLC()
        H_J = self.hamiltonianJosephson
        E0,EI = self.spectrumManifold(flux_points,flux_manifold,H_LC,H_J,excitation=1)
        E0,EII = self.spectrumManifold(flux_points,flux_manifold,H_LC,H_J,excitation=2)
        EI = tensor(EI).detach().numpy()
        EII = tensor(EII).detach().numpy()
        return EI,EII

### Basis Representation
class Kerman(Circuit):
    def __init__(self,network,control_iD,basis,pairs=dict(),device=None):
        super().__init__(network,control_iD,basis,pairs,device)
        self.No,self.Ni,self.Nj = self.kermanDistribution()
        self.N = self.basisSize()
        self.R = self.kermanTransform().real
        self.L_,self.C_ = self.modeTransformation()

    def basisSize(self,modes=False):
        N = dict()
        basis = self.basis
        N['O'] = [size for size in basis['O']]
        N['I'] = [2*size+1 for size in basis['I']]
        N['J'] = [2*size+1 for size in basis['J']]
        if modes:
            return N
        N = prod(N['O'])*prod(N['I'])*prod(N['J'])
        return int(N)

    def kermanDistribution(self):
        Ln_ = self.Ln_
        Ni = self.islandModes()
        if Ln_.is_cuda:
            No = matrix_rank(Ln_.cpu().detach().numpy())
        else:
            No = matrix_rank(Ln_.detach().numpy())
        Nj = self.Nn - Ni - No
        return No,Ni,Nj

    def kermanTransform(self):
        Ln_ = self.Ln_
        R = diagonalisation(Ln_.detach(),True).to(float)
        return R

    def kermanComponents(self):
        L_,C_ = self.L_,self.C_
        No,Ni,Nj = self.No,self.Ni,self.Nj #self.kermanDistribution()
        N = self.Nn

        Lo_ = L_[:No,:No]
        
        Co_ = C_[:No,:No]
        Coi_ = C_[:No,No:No+Ni]
        Coj_ = C_[:No,No+Ni:]
        Ci_ = C_[No:No+Ni,No:No+Ni]
        Cij_ = C_[No:No+Ni,No+Ni:]
        Cj_ = C_[No+Ni:,No+Ni:]

        C_ = Co_,Coi_,Coj_,Ci_,Cij_,Cj_

        return Lo_,C_

    def modeTransformation(self):
        Cn_,Ln_ = self.Cn_,self.Ln_ #componentMatrix()
        R = self.R
        L_ = inv(R.T) @ Ln_ @ inv(R)
        C_ = R @ Cn_ @ R.T
        return L_,C_

    def oscillatorImpedance(self):
        Cn_,Ln_,basis = self.Cn_,self.Ln_,self.basis
        impedance = [sqrt(Cn_[i,i]/Ln_[i,i]) for i in range(len(basis['O']))]
        return impedance

    def linearCombination(self,index):
        invR = inv(self.R)
        combination = invR[index]
        assert len(combination) == self.Nn
        return combination

    def displacementCombination(self,combination):
        basis = self.basis
        No,Ni,Nj = self.No,self.Ni,self.Nj # self.kermanDistribution()
        O = combination[:No]
        I = combination[No:No+Ni]
        J = combination[No+Ni:]
        
        Z = self.oscillatorImpedance() * 2 # cooper pair factor
        # re-calculation impedance factor with circuit variation
        DO_plus = [displacementOscillator(basis_max,z,o) for o,z,basis_max in zip(O,Z,basis['O'])]
        DO_minus = [displacementOscillator(basis_max,z,-o) for o,z,basis_max in zip(O,Z,basis['O'])]
        
        DI_plus = [displacementCharge(basis_max,i) for i,basis_max in zip(I,basis['I'])]
        DI_minus = [displacementCharge(basis_max,-i) for i,basis_max in zip(I,basis['I'])]
        DJ_plus = [displacementCharge(basis_max,j) for j,basis_max in zip(J,basis['J'])]
        DJ_minus = [displacementCharge(basis_max,-j) for j,basis_max in zip(J,basis['J'])]
        
        Dplus = DO_plus+DI_plus+DJ_plus
        Dminus = DO_minus+DI_minus+DJ_minus
        assert len(combination)==len(Dplus)
        assert len(combination)==len(Dminus)
        return Dplus,Dminus

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=True)
    def hamiltonianLC(self):
        """
            basis : {O:(,,,),I:(,,,),J:(,,,)}
        """
        basis = self.basis
        self.Cn_,self.Ln_ = self.componentMatrix()
        self.L_,self.C_ = self.modeTransformation()
        Lo_,C_ = self.kermanComponents()

        #Lo_ = self.Lo_
        Co_,Coi_,Coj_,Ci_,Cij_,Cj_ = C_
        No,Ni,Nj = self.kermanDistribution() #No,self.Ni,self.Nj

        Z = sqrt(diagonal(Co_)/diagonal(Lo_))
        Qo = [basisQo(basis_max,Zi) for Zi,basis_max in zip(Z,basis['O'])]
        Qi = [basisQq(basis_max) for basis_max in basis['I']]
        Qj = [basisQq(basis_max) for basis_max in basis['J']]
        Q = Qo + Qi + Qj

        H = modeMatrixProduct(Q,Co_,Q,(0,0))/2

        Fo = [basisFo(basis_max,Zi) for Zi,basis_max in zip(Z,basis['O'])]
        Fi = [basisFq(basis_max) for basis_max in basis['I']]
        Fj = [basisFq(basis_max) for basis_max in basis['J']]
        F = Fo + Fi + Fj

        H += modeMatrixProduct(F,Lo_,F,(0,0))/2
        
        H += modeMatrixProduct(Q,Coi_,Q,(0,No))
        H += modeMatrixProduct(Q,Coj_,Q,(0,No+Ni))
        H += modeMatrixProduct(Q,Cij_,Q,(No,No+Ni))

        H += modeMatrixProduct(Q,Ci_,Q,(No,No))/2
        H += modeMatrixProduct(Q,Cj_,Q,(No+Ni,No+Ni))/2

        return H

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=True)
    def hamiltonianJosephson(self,external_fluxes=dict()):
        edges,Ej = self.josephsonComponents()
        N = self.basisSize()
        H = null(N)
        for (u,v,key),E in zip(edges,Ej):
            i,j = self.nodes_[u],self.nodes_[v]
            flux = self.loopFlux(u,v,key,external_fluxes)
            if i<0 or j<0 :
                # grounded josephson
                i = max(i,j)
                combination = self.linearCombination(i)
                Dplus,Dminus = self.displacementCombination(combination)

                Jplus = basisProduct(Dplus)
                Jminus = basisProduct(Dminus)
            else:
                combination = self.linearCombination(i) - self.linearCombination(j)
                Dplus,Dminus = self.displacementCombination(combination)

                Jplus = basisProduct(Dplus)
                Jminus = basisProduct(Dminus)
            H -= E*(Jplus*phase(flux) + Jminus*phase(-flux))

        return H/2

    def hamiltonianChargeOffset(self,charge_offset=dict()):
        charge = zeros(self.Nn,dtype=float)
        for node,dQ in charge_offset.items():
            charge[self.nodes_[node]] = dQ
        charge = self.R@charge
        
        No,Ni,Nj = self.kermanDistribution() #No,self.Ni,self.Nj
        
        Qo = [basisQo(basis_max) for basis_max in basis['O']]
        Qi = [basisQq(basis_max) for basis_max in basis['I']]
        Qj = [basisQq(basis_max) for basis_max in basis['J']]
        Q = Qo + Qi + Qj
        Io = [identity(2*basis_max+1)*0.0 for basis_max in basis['O']]
        Ii = [identity(2*basis_max+1)*charge[index+No]*2 for index,basis_max in enumerate(basis['I'])]
        Ij = [identity(2*basis_max+1)*charge[index+No+Ni]*2 for index,basis_max in enumerate(basis['J'])]
        I = Io + Ii + Ij
        
        self.Cn_,self.Ln_ = self.componentMatrix()
        self.L_,self.C_ = self.modeTransformation()
        Lo_,C_ = self.kermanComponents()
        Co_,Coi_,Coj_,Ci_,Cij_,Cj_ = C_
        
        H = modeMatrixProduct(Q,Coi_,I,(0,No))
        H += modeMatrixProduct(Q,Coj_,I,(0,No+Ni))
        
        H += modeMatrixProduct(I,Coj_,Q,(No+Ni,No))
        H += modeMatrixProduct(Q,Coj_,I,(No+Ni,No))
        H -= modeMatrixProduct(I,Coj_,I,(No+Ni,No))
        
        H += modeMatrixProduct(I,Ci_,Q,(No,No))/2
        H += modeMatrixProduct(Q,Ci_,I,(No,No))/2
        H += modeMatrixProduct(I,Ci_,I,(No,No))/2
        
        H += modeMatrixProduct(I,Cj_,Q,(No+Ni,No+Ni))/2
        H += modeMatrixProduct(Q,Cj_,I,(No+Ni,No+Ni))/2
        H += modeMatrixProduct(I,Cj_,I,(No+Ni,No+Ni))/2

        return -H

class Charge(Circuit):
    def __init__(self,network,control_iD,basis,pairs=dict(),device=None):
        super().__init__(network,control_iD,basis,pairs,device)
        self.N = self.basisSize()

    def basisSize(self,modes=False):
        N = [2*size+1 for size in self.basis]
        if modes:
            return N
        return prod(N)

    def hamiltonianLC(self):
        """
            basis : [basis_size] charge
        """
        Cn_,Ln_ = self.Cn_,self.Ln_
        basis = self.basis
        Q = [basisQq(basis_max) for basis_max in basis]
        F = [basisFq(basis_max) for basis_max in basis]
        H = modeMatrixProduct(Q,Cn_,Q)
        H += modeMatrixProduct(F,Ln_,F)

        return H/2

    def hamiltonianJosephson(self,external_fluxes=dict()):
        basis = self.basis
        Dplus = [chargeDisplacePlus(basis_max) for basis_max in basis]
        Dminus = [chargeDisplaceMinus(basis_max) for basis_max in basis]
        edges,Ej = self.josephsonComponents()
        N = self.basisSize()
        H = null(N)
        for (u,v,key),E in zip(edges,Ej):
            i,j = self.nodes_[u],self.nodes_[v]
            flux = self.loopFlux(u,v,key,external_fluxes)
            if i<0 or j<0 :
                # grounded josephson
                i = max(i,j)
                Jplus = basisProduct(Dplus,[i])
                Jminus = basisProduct(Dminus,[i])
            else:
                Jplus = crossBasisProduct(Dplus,Dminus,i,j)
                Jminus = crossBasisProduct(Dplus,Dminus,j,i)
                #assert (Jminus == Jplus.conj().T).all()
                
            H -= E*(Jplus*phase(flux) + Jminus*phase(-flux))

        return H/2

    def hamiltonianChargeOffset(self,charge_offset=dict()):
        charge = zeros(self.Nn,dtype=float)
        basis = self.basis
        Cn_ = self.Cn_
        for node,dQ in charge_offset.items():
            charge[self.nodes_[node]] = dQ
            
        Q = [basisQq(basis_max) for basis_max in basis]
        I = [identity(2*basis_max+1,complex)*charge[index]*2 for index,basis_max in enumerate(basis)]
        H = modeMatrixProduct(Q,Cn_,I)
        H += modeMatrixProduct(I,Cn_,Q)
        H += modeMatrixProduct(I,Cn_,I)
        
        return H/2.

    def potentialCharge(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        F = [basisFq(basis_max) for basis_max in basis]
        H = modeMatrixProduct(F,Ln_,F)/2
        H += self.josephsonCharge(external_fluxes)
        return H

if __name__=='__main__':
    import models
    torch.set_num_threads(12)
    # Transmon Charge Manifold
    circuit = models.transmon(Charge,[256])
    circuit.control_iD = [1]
    charge_profile = [[tensor(0.)],[tensor(.25)],[tensor(.5)]]
    Spectrum = circuit.spectrumManifold(charge_profile)
    print(Spectrum[0][0][:4])
    # Shunted Qubit example
    basis = [6]*3 ; Rep = Charge
    basis = {'O':[10],'I':[],'J':[4,4]} ; Rep = Kerman
    circuit = models.shuntedQubit(Rep,basis)
    flux_manifold = zip(arange(0,1,.01))
    H_LC = circuit.hamiltonianLC()
    H_J = circuit.hamiltonianJosephson
    eigen = eigvalsh(H_LC+H_J({'I':tensor(.225)}))
    print(eigen[:4])