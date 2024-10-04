__author__ = """Davide Lasagna, Aerodynamics and
			    Flight Mechanics group, Soton Uni."""

import numpy as np 

def linearStabilityBoundary(Omega, j=1, k=1):
	""" Marginal stability for the linear stability analysis. """
	return np.sqrt(j**2 + k**2)**3/np.sqrt(1-Omega)/np.sqrt(Omega)/k

def eigenvalue_energy_method(n, m, Re):
	""" Compute value of eigenvalue corresponding to eigenmode
	    (n, m) at Reynolds number Re.
	"""
	return -(2*(m**2+n**2) + m*Re/np.sqrt(m**2+n**2))/(2*Re)

def eigenvalue_linear_method(n, m, Re, Omega):
	""" Compute value of eigenvalue corresponding to eigenmode
	    (n, m) at Reynolds number Re.
	"""
	return -(m**2+n**2)/float(Re) + m/np.sqrt(m**2+n**2)*np.sqrt((1-Omega)*Omega)

def firstNModesByEigenvalue(N, Re):
	""" Return the first `N` eigenmodes sorted by the corresponding 
		eigenvalue computed at Reynolds number equal to `Re`. A list 
		of three elementes tuples (n, m, eigenvalue(n, abs(m))) is 
		returned. Modes are sorted according to Sergei's Mathematica 
		notebook, and the sorting is made regardless of the sign of 
		the integer m.
	"""
	modes = [(n, m, max(eigenvalue(n, m, Re), eigenvalue(n, -m, Re))) 
				for n in range(1, N+5) for m in range(-N-4, N+5)]
	sortedModes = sorted(modes, key=lambda x: (x[2], x[1]), reverse=True)
	return sortedModes[:N]

def firstNModesByEigenvalueMask(N, Re, shape):
	""" Creates an array  of shape `shape` of boolean values which are True
		if one given mode is active.  Active modes are selected
		as by the result of the function firstNModesByEigenvalue.
	"""
	mask = np.zeros(shape, dtype=np.bool)
	modes = list(set([(m[0], abs(m[1])) for m in firstNModesByEigenvalue(N, Re)]))
	mask[zip(*modes)] = True
	return mask


