from des_scq.components import J,C,L,sigmoidInverse
from des_scq.components import C0,J0,L0,C_,J_,L_
from torch import tensor

def tensorize(values,variable=True):
    tensors = []
    for val in values:
        tensors.append(tensor(val,requires_grad=variable))
    return tensors

def sigInv(sig,limit):
    return [sigmoidInverse(s/limit) for s in sig]

# input order enumerates into node index

def box4Branches(Rep,basis,Ej,Ec,El):
    circuit = [L(0,1,El[0],'L0',True,L0,L_),C(0,1,Ec[0],'C0',C0,C_),J(0,1,Ej[0],'J0',J0,J_)]
    circuit += [L(1,2,El[1],'L1',True,L0,L_),C(1,2,Ec[1],'C1',C0,C_),J(1,2,Ej[1],'J1',J0,J_)]
    circuit += [L(2,3,El[2],'L2',True,L0,L_),C(2,3,Ec[2],'C2',C0,C_),J(2,3,Ej[2],'J2',J0,J_)]
    circuit += [L(3,0,El[3],'L3',True,L0,L_),C(3,0,Ec[3],'C3',C0,C_),J(3,0,Ej[3],'J3',J0,J_)]
    control_iD = ['L0']

    circuit = Rep(circuit,control_iD,basis)
    return circuit

def zeroPi(Rep,basis,Ej=10.,Ec=50.,El=10.,EcJ=100.,symmetry=False,_L_=(L_,L0),_C_=(C_,C0),_J_=(J_,J0),_CJ_=(4*C_,4*C0),device=None):
    circuit = [L(0,1,El,'Lx',True,_L_[1],_L_[0]),L(2,3,El,'Ly',True,_L_[1],_L_[0])]
    circuit += [C(1,2,Ec,'Cx',_C_[1],_C_[0]),C(3,0,Ec,'Cy',_C_[1],_C_[0])]
    circuit += [J(1,3,Ej,'Jx',_J_[1],_J_[0]),J(2,0,Ej,'Jy',_J_[1],_J_[0])]
    circuit += [C(1,3,EcJ,'CJx',_CJ_[1],_CJ_[0]),C(2,0,EcJ,'CJy',_CJ_[1],_CJ_[0])]
    pairs = dict()
    if symmetry:
        pairs['Ly'] = 'Lx'
        pairs['Cy'] = 'Cx'
        pairs['Jy'] = 'Jx'
        pairs['CJy'] = 'CJx'
    
    control_iD = ('Lx','Ly')
    circuit = Rep(circuit,control_iD,basis,pairs,device)
    return circuit

def prismon(Rep,basis,Ej=10.,Ec=50.,El=10.,EcJ=100.,symmetry=False,_L_=(L_,L0),_C_=(C_,C0),_J_=(J_,J0),_CJ_=(4*C_,4*C0)):
    circuit =  [L(0,1,El,'La',True,_L_[1],_L_[0]),C(0,2,Ec,'Ca',_C_[1],_C_[0]),J(1,2,Ej,'Ja',_J_[1],_J_[0]),C(1,2,EcJ,'CJa',_CJ_[1],_CJ_[0])]
    circuit += [L(2,3,El,'Lb',True,_L_[1],_L_[0]),C(1,5,Ec,'Cb',_C_[1],_C_[0]),J(0,4,Ej,'Jb',_J_[1],_J_[0]),C(0,4,EcJ,'CJb',_CJ_[1],_CJ_[0])]
    circuit += [L(5,4,El,'Lc',True,_L_[1],_L_[0]),C(4,3,Ec,'Cc',_C_[1],_C_[0]),J(3,5,Ej,'Jc',_J_[1],_J_[0]),C(3,5,EcJ,'CJc',_CJ_[1],_CJ_[0])]
    
    # inbuilt symmetry
    pairs = dict()
    if symmetry:
        pairs['Jb'] = 'Ja' ; pairs['Jc'] = 'Ja'
        pairs['Cb'] = 'Ca' ; pairs['Cc'] = 'Ca'
        pairs['Lb'] = 'La' ; pairs['Lc'] = 'La'
        pairs['CJb'] = 'CJa' ; pairs['CJc'] = 'CJa'
        
    control_iD = ('La','Lb','Lc')
    circuit = Rep(circuit,control_iD,basis,pairs)
    return circuit

def transmon(Rep,basis,Ej=10.,Ec=0.3):
    transmon = [J(0,1,Ej,'J')]
    transmon += [C(0,1,Ec,'C')]
    control_iD = ()

    transmon = Rep(transmon,control_iD,basis)
    return transmon

def splitTransmon(Rep,basis):
    transmon = [J(0,1,10),C(0,1,100)]
    transmon += [L(1,2,.0003,'I',True)]
    transmon += [J(2,0,10),C(2,0,100)]
    control_iD = ('I')
    transmon = Rep(transmon,control_iD,basis)
    return transmon

def oscillatorLC(Rep,basis,El=.00031,Ec=51.6256):
    circuit = [L(0,1,El,'L'),C(0,1,Ec,'C')]
    control_iD = ()
    return Rep(circuit,control_iD,basis)

def fluxoniumArray(Rep,basis,shunt=None,gamma=1.5,N=0,Ec=100,Ej=150):
    # N : number of islands
    circuit = [C(0,1,Ec,'Cap')]
    circuit += [J(0,1,Ej,'Junc')]
    if shunt is None:
        shunt = Ec/gamma
    for i in range(N):
        circuit += [J(1+i,2+i,gamma*Ej,'junc'+str(i))]
        circuit += [C(1+i,2+i,shunt,'cap'+str(i),C_=0.)]
    circuit += [J(1+N,0,gamma*Ej,'junc'+str(N))]
    circuit += [C(1+N,0,shunt,'cap'+str(N),C_=0.)]
    
    control_iD = ()
    circuit = Rep(circuit,control_iD,basis)
    return circuit

def fluxonium(Rep,basis,El=.0003,Ec=.1,Ej=20):
    circuit = [C(0,1,Ec,'Cap')]
    circuit += [J(0,1,Ej,'JJ')]
    circuit += [L(0,1,El,'I',True)]
    control_iD = ('I')

    circuit = Rep(circuit,control_iD,basis)
    return circuit

def shuntedQubit(Rep,basis,josephson=[120.,50,120.],cap=[10.,50.,10.],ind=100.,symmetry=False,_C_=(C_,C0),_J_=(J_,J0)):
    Ej1,Ej2,Ej3 = josephson
    C1,C2,C3 = cap

    circuit = [J(1,2,Ej1,'JJ1'),C(1,2,C1,'C1',_C_[1],_C_[0])]
    circuit += [J(2,3,Ej2,'JJ2'),C(2,3,C2,'C2',_C_[1],_C_[0])]
    circuit += [J(3,0,Ej3,'JJ3'),C(3,0,C3,'C3',_C_[1],_C_[0])]
    circuit += [L(0,1,ind,'I',True)]
    
    # inbuilt symmetry
    pairs = dict()
    if symmetry:
        pairs['JJ3'] = 'JJ1'
        pairs['C3'] = 'C1'
    
    circuit = Rep(circuit,['I'],basis,pairs)
    return circuit

def fluxShunted(Rep,basis,josephson=[120.,50,120.],cap=[10.,50.,10.],symmetry=False,_C_=(C_,C0),_J_=(J_,J0)):
    Ej1,Ej2,Ej3 = josephson
    C1,C2,C3 = cap

    circuit = [J(0,1,Ej1,'JJ1'),C(0,1,C1,'C1',_C_[1],_C_[0])]
    circuit += [J(0,2,Ej2,'JJ2'),C(0,2,C2,'C2',_C_[1],_C_[0])]
    circuit += [J(1,2,Ej3,'JJ3'),C(1,2,C3,'C3',_C_[1],_C_[0])]
    
    # inbuilt symmetry
    pairs = dict()
    if symmetry:
        pairs['JJ2'] = 'JJ1'
        pairs['C2'] = 'C1'
    
    circuit = Rep(circuit,(),basis,pairs)
    return circuit

def phaseSlip(Rep,basis,inductance=[.001,.0005,.00002,.00035,.0005],capacitance=[100,30,30,30,30,40,10]):
    La,Lb,Lc,Ld,Le = inductance
    Ca,Cb,Cc,Cd,Ce,Cf,Cg = capacitance
    circuit = [C(0,1,Ca)]
    circuit += [L(1,3,La,'Ltl',True),L(1,4,Lb,'Lbl',True)]
    circuit += [J(3,2,10),J(4,2,10)]
    circuit += [C(3,2,Cb),C(4,2,Cc)]
    circuit += [C(2,5,Cd),C(2,6,Ce)]
    circuit += [J(2,5,10),J(2,6,100)]
    circuit += [C(2,0,Cf)]
    circuit += [L(5,7,Lc,'Ltr',True),L(6,7,Ld,'Lbr',True)]
    circuit += [L(1,7,Le,'Ll',True)]
    circuit += [C(7,0,Cg)]
    control_iD = ('Ltl','Lbl','Ltr','Lbr','Ll')
    circuit = Rep(circuit,control_iD,basis)
    return circuit

if __name__=='__main__':
    from circuit import Kerman
    basis = {'O':[10],'I':[0],'J':[10,10]}
    circuit = shuntedQubit(Kerman,basis)
    print(circuit.kermanDistribution())
    H_LC = circuit.hamiltonianLC()
    H_J = circuit.hamiltonianJosephson({'I':tensor(.25)})