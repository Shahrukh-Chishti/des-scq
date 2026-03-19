import uuid
from numpy import log,sqrt as sqroot,pi,prod,clip
from torch import tensor,norm,abs,ones,zeros,zeros_like,argsort
from torch import linspace,arange,diagonal,diag,sqrt,eye
from torch.linalg import det,inv,eig as eigsolve,norm
from torch import matrix_exp as expm,exp,outer
from torch import sin,cos,sigmoid,clamp
from torch import cdouble as complex, float64 as float # default precision

im = 1.0j
root2 = sqroot(2)
e = 1.60217662 * 10**(-19)
h = 6.62607004 * 10**(-34)
hbar = h/2/pi
flux_quanta = h/2/e
Z0 = flux_quanta / 2 / e
zero = 1e-12
inf = 1e12

# Bounds(non-inclusive) enclose domain of parameter space
# Energy units in GHz
J0,C0,L0 = 1200,2500,1200 ; J_,C_,L_ = 1e-6,1e-6,1e-6
A0 = 1e10 ; A_ = 1e3
# non-zero lower bound is necessary for logarithmic transform & practical sense

# def null(H):
#     def empty(*args):
#         return H*0
#     return empty

# conversion SI and Natural units

def capSINat(cap):
    return cap/(e**2/h/1e9)

def capNatSI(cap):
    return (e**2/1e9/h)*cap

def indSINat(ind):
    return ind/(flux_quanta**2/h/1e9)

def indNatSI(ind):
    return ind*(flux_quanta**2/h/1e9)

# conversion Natural and Energy units

def capEnergy(cap):
    # natural <-> energy
    return 1. / 2 / cap # GHz

def indEnergy(ind):
    # natural <-> energy
    return 1. / 4 / pi**2 / ind # GHz

# natural -- involution -- energy

# conversion SI to Energy units 

def capE(cap):
    return 1 / 2 / cap * e * e / h / 1e9

def indE(ind):
    return 1 / 4 / pi**2 / ind * flux_quanta**2 / h / 1e9

def sigmoidInverse(x):
    x = 1/x -1
    x = clip(x,zero,inf)
    return -log(x)

def normalize(state,square=True):
    state = abs(state)
    norm_state = norm(state)
    state = state/norm_state
    if square:
        state = abs(state)**2
    return state

def diagonalisation(M,reverse=False):
    eig,vec = eigsolve(M)
    if reverse:
        eig = -eig
    indices = argsort(eig.real)
    D = vec[:,indices].clone().detach()
    return D

# components are standardized with natural & energy units

class Parameters:
    def __init__(self,ID=None,device=None,dtype=float,requires_grad=True):
        if ID is None:
            ID = uuid.uuid4().hex
        self.ID = ID
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.base = None

    def variable(self):
        return self.base

class Control(Parameters):
    def __init__(self,A,ID=None,A0=A0,A_=A_,device=None,dtype=float,requires_grad=True):
        super().__init__(ID, device, dtype, requires_grad)
        self.A0 = A0; self.A_ = A_
        self.initControl(A)

    def initControl(self,A):
        self.base = tensor(sigmoidInverse((A-self.A_)/self.A0),device=self.device,dtype=self.dtype,requires_grad=self.requires_grad)

    def energy(self):
        return sigmoid(self.base) * self.A0 + self.A_
    
    def bounds(self):
        upper = tensor(self.A0+self.A_)
        lower = tensor(self.A_)
        return lower,upper

class Elements(Parameters):
    def __init__(self,plus,minus,ID=None,device=None,dtype=float,requires_grad=True):
        super().__init__(ID,device,dtype,requires_grad)
        self.plus = plus
        self.minus = minus

class J(Elements):
    def __init__(self,plus,minus,Ej,ID=None,J0=J0,J_=J_,device=None,dtype=float,requires_grad=True):
        super().__init__(plus,minus,ID,device,dtype,requires_grad)
        self.J0 = J0; self.J_ = J_
        self.initJunc(Ej) # Ej[GHz]
        
    def initJunc(self,Ej):
        self.base = tensor(sigmoidInverse((Ej-self.J_)/self.J0),device=self.device,dtype=self.dtype,requires_grad=self.requires_grad) ##
 
    def energy(self):
        return sigmoid(self.base) * self.J0 + self.J_ # GHz
    
    def bounds(self):
        upper = tensor(self.J0 + self.J_)
        lower = tensor(self.J_)
        return lower,upper

class C(Elements):
    def __init__(self,plus,minus,Ec,ID=None,C0=C0,C_=C_,device=None,dtype=float,requires_grad=True):
        super().__init__(plus,minus,ID,device,dtype,requires_grad)
        self.C0 = C0; self.C_ = C_
        self.initCap(Ec) # Ec[GHz]
        
    def initCap(self,Ec):
        self.base = tensor(sigmoidInverse((Ec-self.C_)/self.C0),device=self.device,dtype=self.dtype,requires_grad=self.requires_grad) ##

    def energy(self):
        return sigmoid(self.base)*self.C0 + self.C_ # GHz

    def capacitance(self):
        return capEnergy(self.energy()) # he9/e/e : natural unit
    
    def bounds(self):
        upper = tensor(self.C0 + self.C_)
        lower = tensor(self.C_)
        return lower,upper

class L(Elements):
    def __init__(self,plus,minus,El,ID=None,external=False,L0=L0,L_=L_,device=None,dtype=float,requires_grad=True):
        super().__init__(plus,minus,ID,device,dtype,requires_grad)
        self.L0 = L0; self.L_ = L_
        self.external = external # duplication : doesn't requires_grad
        self.initInd(El) # El[GHz]
        
    def initInd(self,El):
        self.base = tensor(sigmoidInverse((El-self.L_)/self.L0),device=self.device,dtype=self.dtype,requires_grad=self.requires_grad) ##
 
    def energy(self):
        return sigmoid(self.base)*self.L0 + self.L_ # GHz

    def inductance(self):
        return indEnergy(self.energy()) # 4e9 e^2/h : natural unit
    
    def bounds(self):
        upper = tensor(self.L0 + self.L_)
        lower = tensor(self.L_)
        return lower,upper

if __name__=='__main__':
    print(capEnergy(capEnergy(10)))
