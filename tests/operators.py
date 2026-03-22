from des_scq.dense import *
from torch import tensor
from torch.linalg import eigvalsh

def main(N,Z):
    Qo = basisQo(N,Z)
    Fo = basisFo(N,Z)
    print('Qo:',eigvalsh(Qo))
    print('Fo:',eigvalsh(Fo))

    Qq = basisQq(N)
    Fq = basisFq(N)
    print('Qq:',eigvalsh(Qq))
    print('Fq:',eigvalsh(Fq))

    Ff = basisFf(N)
    Qf = basisQf(N)
    print('Qf:',eigvalsh(Qf))
    print('Ff:',eigvalsh(Ff))

if __name__ == '__main__':
    print('Eigenspectrum of Operators')
    main(N=11,Z=tensor(1.))