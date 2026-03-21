from torch import kron
from des_scq.components import *

""" Operator Computation """

def modeTensorProduct(pre,M,post):
    """
        extend mode to full system basis
        sequentially process duplication
    """
    H = identity(1) ##
    for dim in pre:
        H = kron(H,identity(dim))
    H = kron(H,M)
    for dim in post:
        H = kron(H,identity(dim))
    return H

def crossBasisProduct(A,B,a,b):
    assert len(A)==len(B)
    n = len(A)
    product = identity(1) ##
    for i in range(n):
        if i==a:
            product = kron(product,A[i])
        elif i==b:
            product = kron(product,B[i])
        else:
            product = kron(product,identity(len(A[i])))
    return product

def basisProduct(O,indices=None):
    n = len(O)
    B = identity(1) ##
    if indices is None:
        indices = arange(n)
    for i in range(n):
        if i in indices:
            B = kron(B,O[i])
        else:
            B = kron(B,identity(len(O[i])))
    return B

def modeProduct(A,i,B,j):
    return mul(basisProduct(A,[i]),basisProduct(B,[j]))

def modeMatrixProduct(A,M,B,mode=(0,0)):
    """
        M : mode operator, implementing mode interactions
        B : list : basis operators
        A : list : basis operators(transpose)
        cross_mode : indicates if A!=B, assumed ordering : AxB
        returns : prod(nA) x prod(nB) mode state Hamiltonian matrix
    """
    shape = prod([len(a) for a in A])
    H = null(shape)
    a,b = mode
    nA,nB = M.shape
    for i in range(nA):
        for j in range(nB):
            if not M[i,j]==0:
                H += M[i,j]*modeProduct(A,i+a,B,j+b)

    return H

def unitaryTransformation(M,U):
    M = U.conj().T@ M@ U
    return M

def mul(A,B):
    return A@B

""" Operator Tensors """

def identity(n,dtype=float,device=None):
    return eye(n,dtype=dtype,device=device) ##

def null(N=1,dtype=complex,device=None):
    return zeros(N,N,dtype=complex,device=device) ##

# States Grid

def chargeStates(n,dtype=int,device=None):
    """
        Charge grid

        Parameter
        ---------
        n : int
            extent of charge domain
    """
    charge = linspace(n,-n,2*n+1,dtype=dtype,device=None)
    return charge

def fluxStates(N,n=1,dtype=float,device=None):
    """
        Flux grid 

        Parameter
        ---------
        N : int
            grid points on flux domain
        n : float
            extent of flux line
    """
    flux = linspace(n,-n,N+1,dtype=dtype,device=device)[1:]
    return flux

def transformationMatrix(n_charge,n_flux=1,device=None):
    """
        Fourier transformation to interchange quantization

        Parameter
        ---------
        n_charge : int
            extent of charge domain
        n_flux : float
            extent of flux line
    """
    charge_states = chargeStates(n_charge,complex,device)
    N_flux = 2*n_charge+1 # dimensionality of Hilbert space
    flux_states = fluxStates(N_flux,n_flux,complex,device)/2/n_flux
    # domain of flux bound to (-.5,.5] : Fourier transform
    T = outer(flux_states,charge_states)
    T *= 2*pi*im # Fourier Phase
    T = exp(T)/N_flux # Normalization
    return T # unitary transformation

# Oscillator Basis

def basisQo(n,impedance,device=None):
    Qo = arange(1,n,device=device)
    Qo = sqrt(Qo)
    Qo = -diag(Qo,diagonal=1) + diag(Qo,diagonal=-1)
    return Qo*im*sqrt(1/2/pi/impedance)

def basisFo(n,impedance,device=None):
    Fo = arange(1,n,device=device) ##
    Fo = sqrt(Fo)
    Fo = diag(Fo,diagonal=1) + diag(Fo,diagonal=-1) ##
    return Fo.to(dtype=complex)*sqrt(impedance/2/pi)

# Canonical Basis

def basisQq(n,device=None):
    # charge basis
    charge = chargeStates(n,complex,device)
    Q = diag(charge.clone().detach()) ##
    return Q * 2

def basisFqKerman(n):
    # charge basis
    N = 2*n+1
    P = zeros((N,N),dtype=complex)
    charge = chargeStates(n)
    for q in charge:
        for p in charge:
            if not p==q:
                P[q,p] = (-(n+1)*sin(2*pi*(q-p)*n/N) + n*sin(2*pi*(q-p)*(n+1)/N))
                P[q,p] /= -im*N*(1-cos(2*pi*(q-p)/N))*N
    return P

def basisFq(n,device=None):
    Q = basisQq(n,device)
    U = transformationMatrix(n,device=device)
    return U@Q@U.conj().T/2

def basisFf(N,n=1,device=None):
    flux = fluxStates(N,n,dtype=complex,device=device)/2/n # periodicity bound
    F = diag(flux)
    return F

def basisQf(n,N=1,device=None):
    F = basisFf(2*n+1,N)
    U = transformationMatrix(n,N,device=device)
    return U@F@U.conj().T

# Oscillator basis Diagonalization

def fluxFlux(n,impedance):
    N = 2*n+1
    Po = basisFo(N,impedance)
    D = diagonalisation(Po)
    Pp = unitaryTransformation(Po,D)
    return Pp

def chargeFlux(n,impedance):
    N = 2*n+1
    Po = basisFo(N,impedance)
    Qo = basisQo(N,impedance)
    D = diagonalisation(Po)
    Qp = unitaryTransformation(Qo,D)
    return Qp

def chargeCharge(n,impedance):
    N = 2*n+1
    Qo = basisQo(N,impedance)
    D = diagonalisation(Qo)
    Qq = unitaryTransformation(Qo,D)
    return Qq

def fluxCharge(n,impedance):
    N = 2*n+1
    Po = basisFo(N,impedance)
    Qo = basisQo(N,impedance)
    D = diagonalisation(Qo)
    Pq = unitaryTransformation(Po,D)
    return Pq

# Junction Displacement Operators    

def chargeDisplacePlus(n,device=None):
    """n : charge basis truncation"""
    diagonal = ones((2*n+1)-1,dtype=complex,device=device)
    D = diag(diagonal,diagonal=-1)
    return D

def chargeDisplaceMinus(n,device=None):
    """n : charge basis truncation"""
    diagonal = ones((2*n+1)-1,dtype=complex,device=device)
    D = diag(diagonal,diagonal=1)
    return D

def displacementCharge(n,a,device=None):
    D = basisFq(n,device)
    D = expm(im*2*pi*a*D)
    return D

def displacementOscillator(n,z,a,device=None):
    D = basisFo(n,z,device)
    D = expm(im*2*pi*a*D)
    return D

if __name__=='__main__':
    Qo = basisQo(30,tensor(4))
    Fq = basisFq(30)
    print(Fq.shape)
    Qf = basisQf(30,4)
    print(Qf.shape)
