# This script will create the file structure required to initialise the initial
# conditions of a simulation and thereafter run it.

import os
import sys

def generate_sim_directory(casedir):
    """
    Generate a simulation directory for a given path location.
    """
    os.mkdir(casedir)
    os.mkdir(os.path.join(casedir, '0.000000/'))
    with open(os.path.join(casedir, 'params'), 'w') as f:
        f.write('[params]\n')
        f.write('Ny             = 64;\n')
        f.write('Nz             = 32;\n')
        f.write('Re             = 10;\n')
        f.write('Ro             = 0.5;\n')
        f.write('L              = 8;\n')
        f.write('dt             = 0.01;\n')
        f.write('T              = 100;\n')
        f.write('n_it_out       = 100;\n')
        f.write('t_restart      = 0.000;\n')
        f.write('t_offset       = 0.0;\n')
        f.write('stretch_factor = 1e-12;\n')
        f.write('n_threads      = 1;\n')
    open(os.path.join(casedir, '0.000000/U'), 'w').close()
    open(os.path.join(casedir, '0.000000/psi'), 'w').close()


if __name__ == '__main__':
    # parse command line arguments
    if len(sys.argv) == 1:
        casedir = os.path.join(os.getcwd(), 'tmp/')
    else:
        casedir = sys.argv[1]

    # generate simulation directory
    generate_sim_directory(casedir)
