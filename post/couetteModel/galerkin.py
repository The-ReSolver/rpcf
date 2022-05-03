__author__ = """Davide Lasagna, Aerodynamics and
			    Flight Mechanics group, Soton Uni."""

import numpy as np
from numpy import cos, sin, sqrt, sign

from couetteModel.operators import dot, integral, Field, laplacian, grad
from couetteModel.utils import makeGrid
import copy


class LinearStabilityModeGenerator():
	def __init__(self, params):
		""" A callable object that return eigenfunctions of the
			linear stability problem, as instance of the Field class.

			This object should be called with a (j, k) tuple, where j > 0.
		"""
		self.X = makeGrid(params)
		self.params = params

	def __call__(self, jk):
		j, k = jk
		U = np.zeros_like(self.X)
		if j == 0:
			U[0] =          sin(k*self.X[2])*sqrt(1-self.params['Omega'])/sqrt(self.params['Omega'])
			U[1] = -sign(k)*sin(k*self.X[2])
			U[2] =  0.0
		elif k == 0:
			U[0] =  sin(j*self.X[1])
			U[1] =  0.0
			U[2] =  0.0
		else:
			U[0] =    sin(j*self.X[1])*cos(k*self.X[2])*sqrt(j**2 + k**2)*sqrt((1-self.params['Omega'])/self.params['Omega'])
			U[1] =  k*sin(j*self.X[1])*cos(k*self.X[2])
			U[2] = -j*cos(j*self.X[1])*sin(k*self.X[2])
			
		f = Field(data=U, metadata=self.params)
		f.metadata['jk'] = jk
		return f/sqrt(integral(dot(f, f)))

class GalerkinSystem:
	def __init__(self, modes, Re, Omega):
		self.modes = modes

		# make parts
		self.Lambda, self. W, self.R = self._buildLinearPart(Omega)
		self.Q = self._buildNonLinearPart()
		self.L = self.Lambda/Re + self.W + self.R

		# filter out small bits
		self.Q[abs(self.Q) < 1e-13] = 0
		self.Lambda[abs(self.Lambda) < 1e-13] = 0
		self.L[abs(self.L) < 1e-13] = 0
		self.W[abs(self.W) < 1e-13] = 0
		
	def __call__(self, a, t):
		""" Compute derivative of modal amplitudes. """
		return np.dot(self.L, a) + np.einsum("ijk, j, k", self.Q, a, a ) 

	def _buildLinearPart(self, Omega):
		""" Build linear term of Galerkin system. """
		n = len(self.modes)

		# matrix of the linear part
		Lambda = np.zeros((n, n))
		W = np.zeros((n, n))
		R = np.zeros((n, n))
		for i, ui in enumerate(self.modes): 
			for j, uj in enumerate(self.modes): 
				Lambda[i, j]  = integral(dot(ui, laplacian(uj)))
				W[i, j] = -integral(Field(data=ui.data[0]*uj.data[1], metadata=ui.metadata))
				R[i, j] = Omega*integral(Field(data=ui.data[0]*uj.data[1] - ui.data[1]*uj.data[0], metadata=ui.metadata))
		return Lambda, W, R

	def _buildNonLinearPart(self):
		""" Build non linear part """
		n = len(self.modes)

		# matrix of the nonlinear part
		Q = np.zeros((n, n, n))
		for i, ui in enumerate(self.modes): 
			for j, uj in enumerate(self.modes): 
				for k, uk in enumerate(self.modes): 
					Q[i, j, k] = -integral(dot(ui, dot(uj, grad(uk))))
		return Q

def orthonormalize(modes):
	""" Make modes an orthonormal set of basis functions."""
	this_copy = copy.deepcopy(modes)
	new_modes = []
	for i, mode in enumerate(this_copy):
		for m in new_modes:
			mode -= m*integral(dot(m , mode))
		mode /= sqrt(integral(dot(mode, mode)))
		new_modes.append(mode)
	return new_modes

	# this is a check
	# for i, ui in enumerate(self.modes):
	#    for j, uj in enumerate(self.modes):
	#		print i, j, integral(dot(ui, uj))
