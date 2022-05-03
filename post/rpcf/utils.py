__author__ = """Davide Lasagna, Aerodynamics and
			    Flight Mechanics group, Soton Uni."""

#from rpcf.simulation import Sim
import numpy as np
from scipy.interpolate import interp1d

def write_params_to_file(params, filename):

	""" Write parameters dictionary to paams file. """
	with open(filename, "w") as f:
		f.write("[params]\n")
		f.writelines("\n".join(["%s = %s;" % (k, v) for k, v in params.iteritems()]))
		f.write("\n")

def nice(x, y):
    f = interp1d(x, y, kind='cubic')
    xx = np.linspace(min(x), max(x), 100)
    return xx, f(xx)
