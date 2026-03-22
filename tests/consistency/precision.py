from torch import cfloat,cdouble,tensor
from des_scq.models import transmon
from des_scq.circuit import Charge
from des_scq.utils import plotCompare
from numpy import linspace,array

def transmonOffsetCharge(N,dtype,ng_list,limit=3):
    Ej = 30.02 ; Ec = 10.2
    circuit = transmon(Charge,[N],Ej=Ej,Ec=Ec)
    H_LC = circuit.hamiltonianLC()
    H_J = circuit.hamiltonianJosephson()
    dTorch = {'valsh':[],'eigh':[]}
    dNumpy = {'valsh':[],'eigh':[]}
    for offset in ng_list:
        offset = dict([(1,tensor(offset))])
        H_off = circuit.hamiltonianChargeOffset(offset)
        H = H_LC + H_J + H_off
        H = H.to(dtype)
        from torch.linalg import eigvalsh,eigh
        dTorch['valsh'].append(eigvalsh(H)[:limit].detach().numpy())
        dTorch['eigh'].append(eigh(H)[0][:limit].detach().numpy())
        from numpy.linalg import eigvalsh,eigh
        H = H.detach().numpy()
        dNumpy['valsh'].append(eigvalsh(H)[:limit])
        dNumpy['eigh'].append(eigh(H)[0][:limit])
    return dTorch,dNumpy

if __name__ == "__main__":
    print('Comparison of numerical encoding Precision')
    print('Transmon Spectrum v/s offset charge n_g')
    N = 512
    ng_list = linspace(-2, 2, 220)

    Float32 = transmonOffsetCharge(N,cfloat,ng_list)
    Double64 = transmonOffsetCharge(N,cdouble,ng_list)

    FloatTorch = array(Float32[0]['valsh']).T
    FloatNumpy = array(Float32[1]['valsh']).T

    DoubleTorch = array(Double64[0]['valsh']).T
    DoubleNumpy = array(Double64[1]['valsh']).T

    plotCompare(ng_list,
                {'numpy-0':FloatNumpy[0],'numpy-1':FloatNumpy[1],'numpy-2':FloatNumpy[2],
                 'torch-0':FloatTorch[0],'torch-1':FloatTorch[1],'torch-2':FloatTorch[2]},
                 'Transmon Spectrum - Float Precision','n_g','spectrum(GHz)',width=5)
    
    plotCompare(ng_list,
                {'numpy-0':DoubleNumpy[0],'numpy-1':DoubleNumpy[1],'numpy-2':DoubleNumpy[2],
                 'torch-0':DoubleTorch[0],'torch-1':DoubleTorch[1],'torch-2':DoubleTorch[2]},
                 'Transmon Spectrum - Double Precision','n_g','spectrum(GHz)',width=5)

    
    