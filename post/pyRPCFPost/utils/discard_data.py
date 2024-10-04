# This script will discard data given by a time range. Will keep the initial
# condition by default.

import os
import sys
import shutil

def discard_data(casedir, time_range_list):
    # remove all range of data (excluding initial and params)
    for file in os.listdir(casedir):
        if file != '0.000000' and file != 'params':
            if float(file) < float(time_range_list[0]) or float(file) > float(time_range_list[1]):
                shutil.rmtree(os.path.join(casedir, file))

if __name__ == '__main__':
    # parse input arguments
    if len(sys.argv) == 1:
        print('Invalid arguments given! Missing simulation directory path!')
        sys.exit(1)
    casedir = sys.argv[1]
    try:
        time_range = sys.argv[2]
    except IndexError:
        print("Time range of data to be retained missing!")
        sys.exit(1)

    # generate range of time data to keep
    time_range_list = time_range.split('-')

    # discard the data
    discard_data(casedir, time_range_list)
