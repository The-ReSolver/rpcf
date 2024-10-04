import subprocess, os

import numpy as np

from scipy.sparse import dok_matrix
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import factorized
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.io as sio

from pyRPCFPost.rpcf.field import Field, integral, grad

def trunc(x, d=6):
	""" Truncate x to d decimals digits"""
	return float(int(10**d*x))/10**d

def buildB(Re, Ro, Ny, L, n):
	""" Build matrix B of eigenproblem """
	# wavenumbers
	alpha = 2*np.pi/L
	k = n*alpha
	k2 = k*k

	# step size
	h = 2.0/(Ny-1)
	h2 = h*h

	# number of internal grid points
	N = Ny - 2

	# size of the problem
	s = 2*N-2

	B = dok_matrix((s, s), dtype=np.float64)

	# # part due to f
	for i in xrange(N):
		B[i, i] += -Re # B1

	# part due to g
	for i in xrange(N-2):
		B[i+N,   i+N] += Re*k2  #B3
		B[i+N,   i+N] += 2*Re/h2 #B2

		# off diagonal B2
		if i == 0:
			B[i+N, i+N] += -Re/h2/4
		else:
			B[i+N, i+N-1] += -Re/h2 

		if i == N-3:
			B[i+N, i+N] += -Re/h2/4
		else:
			B[i+N, i+N+1] += -Re/h2 
		
	return B.tocsc()

def buildA(Re, Ro, Ny, L, n, ftype='c'):
	""" Build matrix A """
	if ftype == 'c':
		sign = -1
	elif ftype == 's':
		sign = 1
	else:
		raise ValueError

	# wavenumbers
	alpha = 2*np.pi/L
	k = n*alpha
	k2 = k*k
	k4 = k2*k2

	# step size
	h = 2.0/(Ny-1)
	h2 = h*h
	h4 = h2*h2

	# number of internal grid points
	N = Ny - 2

	# size of the problem
	s = 2*N-2
	
	A = dok_matrix((s, s), dtype=np.float64)

	# part due to f
	for i in xrange(N):
		# A1
		A[i, i] += 2/h2
		if i > 0:
			A[i, i-1] += -1/h2
		if i < N-1:
			A[i, i+1] += -1/h2

		# A2
		A[i, i] += k*k

		# A3 
		if i==0:
			A[i, i+N] += sign*Re*(Ro-1)*k/4.0
		elif i==N-1:
			A[i, i+N-2] += sign*Re*(Ro-1)*k/4.0
		else:
			A[i, i+N-1] += sign*Re*(Ro-1)*k

	# # part due to g
	for i in xrange(N-2):
		# A6
		A[i+N, i+1] += sign*Ro*k*Re

		# A5
		A[i+N, i+N] += -4*k2/h2

		if i==0:
			A[i+N, i+N] += 2*k2/h2/4.0
		else:
			A[i+N, i+N-1] += 2*k2/h2
		
		if i==N-3:
			A[i+N, i+N]  += 2*k2/h2/4.0
		else:
			A[i+N, i+N+1]  += 2*k2/h2

		# A7
		A[i+N, i+N] += -k4

		# A4
		if i==0:
			A[i+N, i+N  ] += -5.0/h4
			A[i+N, i+N+1] +=  4.0/h4
			A[i+N, i+N+2] += -1.0/h4
		elif i==1:
			A[i+N, i+N-1] += 15.0/4.0/h4
			A[i+N, i+N  ] +=     -6.0/h4
			A[i+N, i+N+1] +=      4.0/h4
			A[i+N, i+N+2] +=     -1.0/h4
		elif i==N-3:
			A[i+N, i+N  ] += -5.0/h4
			A[i+N, i+N-1] +=  4.0/h4
			A[i+N, i+N-2] += -1.0/h4
		elif i==N-4:
			A[i+N, i+N+1] += 15.0/4.0/h4
			A[i+N, i+N  ] +=     -6.0/h4
			A[i+N, i+N-1] +=      4.0/h4
			A[i+N, i+N-2] +=     -1.0/h4
		else:
			A[i+N, i+N+2] += -1.0/h4
			A[i+N, i+N+1] +=  4.0/h4
			A[i+N, i+N  ] += -6.0/h4
			A[i+N, i+N-1] +=  4.0/h4
			A[i+N, i+N-2] += -1.0/h4

	return A.tocsc()

def unpack(v, N):
	""" Unpack an eigenvector into its f and g components. Also apply the b.c."""
	f, g = v[:N], v[N:]
	f = np.r_[0, f, 0]
	g = np.r_[0, 0.25*g[0], g, 0.25*g[-1], 0]
	return f.real, g.real

def myeigs(A, B, k):
	""" Code taken from answer to 
		http://scicomp.stackexchange.com/questions/10940/solving-a-generalised-eigenvalue-problem
	"""
	B_inv = factorized(B);
	def mv(v):
		temp = A.dot(v)
		return B_inv(temp)
	BA = LinearOperator(A.shape, matvec=mv) 
	vals, vecs = eigs(BA, k, which='LR', maxiter=100000)
	# vals, vecs = _eigs(A, B, k)
	N = (vecs.shape[0] + 2)/2.0
	return vals.real, [unpack(v, N) for v in vecs.T]
	
def solver(Re, Ro, Ny, L, m, ftype, n):
	""" Return m eigenfunctions with largest eigenvalue. 

		Parameters
		----------
		Re : float - Reynolds number
		Ro : float - Rotation number 
		Ny : int   - number of grid points along y
		L  : float - length of the domain along z
		ftype : str - type of eigenfunction, 'c' for cosine, or 's' for sine 
		n : int or list - wave numbers at which computations will be performed

		Returns
		-------
		vals : list of floats - the sorted eigenvalues, first is least stable
		ns   : list of ints   - the wavenumbers corresponding to vals
		fs, gs  : lists of two arrays with the solution of the eigenvalue problem.
	"""
	# maximum number of eigenvalues to solve for at each run
	MAXK = m

	# initialise 
	best = {-1000 - i: (None, None, None) for i in range(m)}
	for nn in np.atleast_1d(n):
		# solve eigenproblem analytically for n=0
		if nn == 0:
			if ftype == 's':
				vals = [-np.pi*np.pi*k*k/Re for k in range(1, MAXK+1)]
				vecs = [(np.zeros(Ny, dtype=np.float64), np.cos(k*np.pi*(np.linspace(0, 2, Ny)))-1) for k in range(1, MAXK+1)]
			elif ftype == 'c':
				vals = [-np.pi*np.pi*k*k/Re/4.0 for k in range(1, MAXK+1)]
				vecs = [(np.sin(k*np.pi*0.5*(np.linspace(0, 2, Ny))), np.zeros(Ny, dtype=np.float64)) for k in range(1, MAXK+1)]
		else:
			A = buildA(Re, Ro, Ny, L, nn, ftype)
			B = buildB(Re, Ro, Ny, L, nn)
			vals, vecs = myeigs(A, B, MAXK)

		# update data structure so that it contains
		# the least stable eigenfunctions.
		for i in range(len(vals)):
			m = min(best.keys())
			if vals[i] > m:
				del best[m]
				best[vals[i]] = (nn, (vecs[i][0]).reshape(-1, 1), (vecs[i][1]).reshape(-1, 1))

	# extract data
	vals = best.keys()
	ns, fs, gs = zip(*best.values())
	return vals, ns, fs, gs

def ddy(g, y):
	""" Compute derivative of g along y using spline interpolation. """
	der = InterpolatedUnivariateSpline(y, g, k=3).derivative()
	return der(y).reshape(-1, 1)

def mode_generator(Re, Ro, Ny, Nz, L, m, ftype, n):
	""" Generate the linear stability eigenfunctions for the RPCF problem. 

		Parameters
		----------
		Re : float - Reynolds number
		Ro : float - Rotation number 
		Ny : int   - number of grid points along y
		Nz : int   - number of grid points along z
		L  : float - length of the domain along z
		m  : int   - number of eigenfunctions that we want
		ftype : str - type of eigenfunction, 'c' for cosine, or 's' for sine, 'b' for both.
		ns : int or list - wave numbers at which computations will be performed

	"""

	# how long is the domain?
	alpha = 2*np.pi/L

	# create grid
	yy = np.linspace(-1, 1, Ny)
	zz = np.linspace(0, L, Nz+1)
	z, y = np.meshgrid(zz, yy)

	# initialise to empty list
	uis = []

	# add cosine functions
	if ftype in ['c', 'b']:
		vals, ns, fs, gs = solver(Re, Ro, Ny, L, m, 'c', n)
		for i in range(m):
			U = np.zeros((3, Ny, Nz+1), dtype=np.float64)
			U[0] =      fs[i] * np.cos(ns[i]*alpha*z)
			U[1] =      gs[i] * np.cos(ns[i]*alpha*z)*ns[i]*alpha
			U[2] = -ddy(gs[i], yy)* np.sin(ns[i]*alpha*z)

			# make eigenfunction always look the same
			sign = 1
			if not U[0, 1, 1] == 0:
				sign = np.sign(U[0, 1, 1])
				U *= sign

			# note: we truncate the eigenvalue 
			# to a limite precision, so that sine and cosine
			# eigenfunctions will have the same value, and eventually
			# by sorting them cosine will always be first, 
			# since sorting is stable.
			params = {'Ny':Ny,
					  'Nz':Nz, 
					  'n':ns[i], 
					  'ftype':"cos",
					  'stretch_factor':1e-13, 
					  'L':L, 
					  'alpha':alpha,
					  'lambda':trunc(vals[i])}

			# create a Field instance for psi
			psi = Field(sign*gs[i]*np.sin(ns[i]*alpha*z), params)

			# create basis function
			f = Field(U, dict(params.items() + {'psi':psi}.items()))

			# normalize
			div = integral(f**2)**0.5
			f /= div
			f.metadata["psi"].data /= div
			uis.append(f)

	# add sine functions
	if ftype in ['s', 'b']:
		vals, ns, fs, gs = solver(Re, Ro, Ny, L, m, 's', n)
		for i in range(m):
			U = np.zeros((3, Ny, Nz+1), dtype=np.float64)
			U[0] =      fs[i] * np.sin(ns[i]*alpha*z)
			U[1] =     -gs[i] * np.sin(ns[i]*alpha*z)*ns[i]*alpha
			U[2] = -ddy(gs[i], yy)* np.cos(ns[i]*alpha*z)

			sign = 1
			if not U[0, 1, 1] == 0:
				sign = np.sign(U[0, 1, 1])
				U *= sign

			params = {'Ny':Ny,
					  'Nz':Nz, 
					  'n':ns[i], 
					  'ftype':"sin",
					  'stretch_factor':1e-13, 
					  'L':L, 
					  'alpha':alpha,
					  'lambda':trunc(vals[i])}

			# create a Field instance for psi
			psi = Field(sign*gs[i]*np.cos(ns[i]*alpha*z), params)

			# create basis function
			f = Field(U, dict(params.items() + {'psi':psi}.items()))

			# normalize
			div = integral(f**2)**0.5
			f /= div
			f.metadata["psi"].data /= div
			uis.append(f)


	return sorted(uis, key=lambda ui: ui.metadata['lambda'], reverse=True)

def mode_selector(G, levels=1, tol=1e-8):
	""" Select modes based on energy flow analysis. 

		Parameters
		----------
		G : list - of stability eigenfunctions
		levels : int - number of levels for triadic interactions

		Notes
		-----
		We select the basis functions in such a way that the thrid order
		tensor associated to the non linear term is the most possible
		dense, meaning that the energy transfer is adequately captured.
		In addition the modes used will be sorted by eigenvalue
		and level of interaction. 

	"""
	# initialise to unstable eigenfunctions
	M = [ui for ui in G if ui.metadata['lambda'] > 0]

	# remove them from G
	for ui in M:
		G.pop(G.index(ui))

	for l in range(levels):
		new_M = []
		for uj in M:
			for uk in M:
				rhs = uj*grad(uk)
				for ui in G:
					val = integral(ui*rhs)
					if abs(val) > tol:
						if not ui in new_M:
							new_M.append(ui)

		for ui in new_M:
			G.pop(G.index(ui)) 
		M.extend(new_M)

	return sorted(M, key=lambda ui: ui.metadata['lambda'], reverse=True)

	



