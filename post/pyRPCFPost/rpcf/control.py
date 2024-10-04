__author__ = """Davide Lasagna, Aerodynamics and
			    Flight Mechanics group, Soton Uni."""

import numpy as np
from pyRPCFPost.rpcf.field import integral, Field

def f(y, gammak):
	return -1./8.*np.exp(-gammak*(1+y))*(1+y)*(1-y)**3

def fp(y, gammak):
	return (y - 1)**2*(0.125*-gammak*(y - 1)*(y + 1) + 0.5*y + 0.25)*np.exp(-gammak*(y + 1))

def cfunc(Ny, Nz, L, k, gamma, ftype):
	""" Build control functions

		Parameters
		----------
		Ny, Nz : ints, size of the mesh
		L : float, axial length of the domain
		k : integer, wavenumber associated to the control function
		gamma : float > 0, decay rate of the control function
		ftype : str, in ['s', 'c'], type of the control function, sice or cosine


		Returns 
		-------
		cfunc : Field instance, 
	"""
	
	# parameters
	alpha = 2*np.pi/L

	# make mesh
	yy = np.linspace(-1, 1, Ny)
	zz = np.linspace(0, L, Nz+1)[:-1]
	z, y = np.meshgrid(zz, yy)

	# allocate
	U = np.zeros((3, Ny, Nz), dtype=np.float64)
	
	# depending on type of cfunction
	if ftype == 'c':
		U[1] =  0
	elif ftype == 's':
		U[1] = f(y, gamma*k)*np.cos(k*alpha*z)*k*alpha

	if ftype == 'c':
		U[2] = 0
	elif ftype == 's':
		U[2] = -fp(y, gamma*k)*np.sin(k*alpha*z)

	# create Field instance
	f = Field(U, {'Ny':Ny,
	              'Nz':Nz, 
	              'stretch_factor':1e-10, 
	              'L':L, 
	              'alpha':alpha})
	# normalize
	f /= integral(f**2)**0.5
	return f

