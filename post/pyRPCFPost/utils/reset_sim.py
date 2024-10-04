# This script will reset a directory that contains simulation data back to a valid
# initialisation state.

import os
import sys
import shutil

def reset_sim(casedir):
    # remove all except 0.000000 directory and params file
    for file in os.listdir(casedir):
        if file != '0.000000' and file != 'params':
            shutil.rmtree(os.path.join(casedir, file))

    # remove metadata and omega files from 0.000000
    os.remove(os.path.join(casedir, '0.000000/metadata'))
    os.remove(os.path.join(casedir, '0.000000/omega'))

if __name__ == '__main__':
    # parse input arguments
    if len(sys.argv) == 1:
        print('Invalid arguments given! Missing simulation directory path!')
        sys.exit(1)
    casedir = sys.argv[1]

    # reset the simulation directory
    reset_sim(casedir)
