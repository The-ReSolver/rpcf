# This file provides the functionality to quickly update the params file of a
# given simulation directory.

import os
from configparser import ConfigParser

def update_params(casedir, **kwargs):
    # unpack keyword arguments
    Ny = kwargs.get('Ny', '64')
    Nz = kwargs.get('Nz', '32')
    Re = kwargs.get('Re', '10')
    Ro = kwargs.get('Ro', '0.5')
    L = kwargs.get('L', '8')
    dt = kwargs.get('dt', '0.01')
    T = kwargs.get('T', '100')
    n_it_out = kwargs.get('n_it_out', '100')
    t_restart = kwargs.get('t_restart', '0.000')
    stretch_factor = kwargs.get('stretch_factor', '1e-12')
    n_threads = kwargs.get('n_threads', '1')

    # initialise config parser object
    config = ConfigParser()
    config.optionxform = str

    # read the params file
    config.read(os.path.join(casedir, 'params'))

    # update all the fields
    config['params']['Ny'] = Ny + ';'
    config['params']['Nz'] = Nz + ';'
    config['params']['Re'] = Re + ';'
    config['params']['Ro'] = Ro + ';'
    config['params']['L'] = L + ';'
    config['params']['dt'] = dt + ';'
    config['params']['T'] = T + ';'
    config['params']['n_it_out'] = n_it_out + ';'
    config['params']['t_restart'] = t_restart + ';'
    config['params']['stretch_factor'] = stretch_factor + ';'
    config['params']['n_threads'] = n_threads + ';'

    # write back to the params file
    with open(os.path.join(casedir, 'params'), 'w') as paramsfile:
        config.write(paramsfile)
