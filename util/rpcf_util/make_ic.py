import os
import sys
from configparser import ConfigParser

import numpy as np

def make_ic(casedir):
    # read grid data from params file
    config = ConfigParser()
    config.read(os.path.join(casedir, 'params'))
    Ny = int(config.get('params', 'Ny')[:-1])
    Nz = int(config.get('params', 'Nz')[:-1])

    # initialise random field (or sinuisoidal)
    d1 = 1e-3*np.random.randn(Ny, Nz)
    # d1k = np.fft.rfft(d1, axis=1)
    # d1k[:, [0, 1]] = 0.0
    # d1k[:, 3:] = 0.0
    # d1 = np.fft.irfft(d1k, axis=1)
    d1 = np.c_[d1, d1[:, 0]]
    d1.tofile(os.path.join(casedir, '0.000000/psi'))
    np.zeros((3, Ny, Nz + 1), dtype=np.float64).tofile(os.path.join(casedir, '0.000000/U'))

if __name__ == '__main__':
    # parse input arguments
    if len(sys.argv) == 1:
        print('Invalid arguments given! Missing simulation directory path!')
        sys.exit(1)
    casedir = sys.argv[1]

    # make the initial condition
    make_ic(casedir)
