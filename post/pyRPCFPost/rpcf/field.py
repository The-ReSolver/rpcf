load__author__ = """Davide Lasagna, Aerodynamics and
				Flight Mechanics group, Soton Uni."""

import copy

import numpy as np 
from scipy.integrate import simpson

from pyRPCFPost.rpcf import lru

def ddt(f, dt, order=2):
	""" Compute time derivative of time series f, 
		using a second order finite difference scheme."""
	out = np.zeros_like(f)
	if order == 1:
		out[:-1] = f[1:]-f[:-1]
		out[-1] = f[-1]-f[-2]
	elif order == 2:
		out[1:-1] = 0.5*(f[2:] - f[:-2])
		out[0] = (-3*f[0] + 4*f[1] - f[2])/2
		out[-1] = (3*f[-1] - 4*f[-2] + f[-3])/2
	elif order == 4:
		out[2:-2] = (f[:-4] - 8*f[1:-3] + 8*f[3:-1] - f[4:])/12
	else:
		raise ValueError("order must 1 or 2")

	return out/dt

def _makeGrid(metadata):
	"""	Create a 2d grid for visualization. 

		The grid will contain one more point along z, 
		to plot the entire domain. This means that the 
		shape of the two components will be (Ny, Nz+1).

	"""
	Ny = metadata['Ny']
	Nz = metadata['Nz']
	d = float(metadata['stretch_factor'])
	L = float(metadata['L'])
	out = np.empty((2, Ny, Nz+1), dtype=np.float64)
	x = np.mgrid[0:Ny, 0:Nz+1]
	out[0] = np.tanh(d*(x[0]/(Ny-1.0)-0.5))/np.tanh(d/2.0)
	out[1] = x[1]*L/Nz
	return out

def _der2(data, y):
	""" Second derivative on non-uniform grid """
	out = np.empty_like(data)
	for i in range(1, len(y)-1):
		hp = y[i+1] - y[i]
		hm = y[i] - y[i-1]
		ai =  2/hm/(hp+hm)
		bi = -2/(hp*hm)
		ci =  2/hp/(hp+hm)
		out[..., i, :] = (ai*data[..., i-1, :] + 
						  bi*data[..., i,   :] + 
						  ci*data[..., i+1, :] )
	h0 = y[1] - y[0]
	h1 = y[2] - y[1]
	out[..., 0, :] = (out[..., 1, :]*(1+h0/h1) -
					  out[..., 2, :]*h0/h1     )

	hn = y[-1] - y[-2]
	hm = y[-2] - y[-3]
	out[..., -1, :] = ( out[..., -3, :]*(1-(hn+hm)/hm) + 
						out[..., -2, :]*((hn+hm)/hm)  ) 
	return out

def _der1(data, y):
	""" Differentiate data along columns using a non uniform
		spacing y. This one work for scalar and vector fields. """
	out = np.empty(data.shape, dtype=np.float64)
	for i in range(1, len(y)-1):
		hp = y[i+1] - y[i]
		hm  = y[i] - y[i-1]
		ai = -hp/hm/(hp+hm)
		bi = (hp - hm)/(hp*hm)
		ci =  hm/hp/(hp+hm)
		out[..., i, :] = (ai*data[..., i-1, :] + 
						  bi*data[..., i,   :] + 
						  ci*data[..., i+1, :] )

	h1 = y[1] - y[0]
	h2  = y[2] - y[1]
	a0 = -(2*h1+h2)/(h1*(h1+h2))
	b0 =  (h1+h2)/(h1*h2)
	c0 = -h1/(h2*(h1+h2))
	out[..., 0, :] = (a0*data[..., 0, :] + 
					  b0*data[..., 1, :] + 
					  c0*data[..., 2, :] )

	h1 = y[-1] - y[-2]
	h2  = y[-2] - y[-3]
	an = h1/(h2*(h1+h2))
	bn = -(h1+h2)/(h1*h2)
	cn = (2*h1+h2)/(h1*(h1+h2))
	out[..., -1, :] = (an*data[..., -3, :] +
					   bn*data[..., -2, :] + 
					   cn*data[..., -1, :] )

	return out

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#@lru.lru_cache(maxsize=50)
def integral(f, direction=None):
	""" Compute integrals of a scalar field. """
	if not f.rank == 0:
		raise ValueError("integral only defined for scalar fields")
	if direction == None:
		out = simpson(f.data, x=f.y[:,0], axis=0)
		return simpson(out, dx=float(f.metadata['L'])/f.metadata['Nz'])
	elif direction == 'y':
		return simpson(f.data, x=f.y[:,0], axis=0)
	elif direction == 'z':
		return simpson(f.data, dx=float(f.metadata['L'])/f.metadata['Nz'], axis=1)
	else:
		raise ValueError("direction not understood")

#@lru.lru_cache(maxsize=50)
def laplacian(f):
	""" Laplacian of a field f. """
	return der(f, 'yy') + der(f, 'zz')

#@lru.lru_cache(maxsize=50)
def der(f, mode):
	""" Derivative of field f. """
	if mode == 'y':
		data = _der1(f.data, f.y[:,0])
	elif mode == 'yy':
		data = _der2(f.data, f.y[:,0])
	elif mode == 'z':
		ks = np.arange(f.metadata['Nz']/2.0+1.0)
		ks[-1] *= -1.0
		outK = np.fft.rfft(f.data[..., :-1])
		out = np.fft.irfft(1j*ks*f.metadata['alpha']*outK)
		shape = list(out.shape); shape[-1] += 1
		data = np.empty(shape, dtype=np.float64)
		data[..., :-1] = out
		data[..., -1]  = out[..., 0]
	elif mode == 'zz':
		kk = np.arange(f.metadata['Nz']/2.0+1)
		outK = np.fft.rfft(f.data[..., :-1], axis=-1)
		out = np.fft.irfft(-kk**2*f.metadata['alpha']**2*outK, axis=-1)
		shape = list(out.shape); shape[-1] += 1
		data = np.empty(shape, dtype=np.float64)
		data[..., :-1] = out
		data[..., -1]  = out[..., 0]
	else:
		raise ValueError("mode %s not understood" % mode)
	return Field(data, f.metadata)

#@lru.lru_cache(maxsize=50)
def curl(f):
	""" Compute curl of vector field """
	out = Field(np.empty((3, f.metadata['Ny'], f.metadata['Nz']+1), dtype=np.float64), f.metadata)
	out[0] =  der(f[2], 'y') - der(f[1], 'z') 
	out[1] =  der(f[0], 'z')
	out[2] = -der(f[0], 'y')
	return out

#@lru.lru_cache(maxsize=50)
def grad(f):
	""" Gradient of a vector field. 

		We compute derivatives of the first three components
		of f, which represent the velocity components. If the 
		field f has more than three components, then it is 
		discarded. 
	"""
	if f.rank != 1:
		raise TypeError("gradient only implemented for vector fields ")
	out = np.empty((3, 3, f.metadata['Ny'], f.metadata['Nz']+1), dtype=np.float64)
	dy = der(f, 'y')
	dz = der(f, 'z')
	out[:, 0, ...] = 0.0 # d/dx = 0
	out[:, 1, ...] = dy.data
	out[:, 2, ...] = dz.data
	return Field(out, f.metadata)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Field():
	# We override multiplication of an array with a field instance.
	__array_priority__ = 100
	_grid = None

	def __init__(self, f, metadata):
		""" This represents any tensor field, including
			scalar fields, as the divergence, velocity fields
			and tensor fields, such as the stress tensor field.

			Parameters
			----------
			f : np.ndarray of shape depending on rank 

				rank = 0 : (Ny, Nz+1)
				rank = 1 : (3, Ny, Nz+1)
				rank = 2 : (3, 3, Ny, Nz+1)

			metadata : dict with extra information.

			Notes
			----- 
			The shape of the data has the +1 because
			data files are written including the right
			boundary.

		"""
		self.rank = len(f.shape) - 2
		self.data = f # do not create a copy
		self.metadata = copy.copy(metadata)

	def __neg__(self):
		""" Negation makes a copy """
		return Field(-self.data, self.metadata)
	
	def __getitem__(self, slice):	
		if isinstance(slice, int):
			return Field(self.data[slice], self.metadata)
		else:
			return self.data[slice]

	def __setitem__(self, slice, val):
		""" ahahahaha """
		if isinstance(val, Field):
			self.data[slice] = val.data
		else:
			self.data[slice] = val

	def __add__(self, f2):
		""" Add something to field. """
		if isinstance(f2, Field):
			if not self.rank == f2.rank:
				raise ValueError("Fields must have same rank to be added")
			else:
				return Field(self.data + f2.data, self.metadata)
		else: 
			try:
				return Field(self.data + f2, self.metadata)
			except TypeError:
				raise ValueError("cannot add instance of type %s " 
													"to Field instance" % type(f2))

	def __sub__(self, f2):
		""" Subtract somethign from field. """
		if isinstance(f2, Field):
			if not self.rank == f2.rank:
				raise ValueError("Fields must have same rank to be subtracted")
			else:
				return Field(self.data - f2.data, self.metadata)
		else: 
			try:
				return Field(self.data - f2, self.metadata)
			except TypeError:
				raise ValueError("cannot subtract instance of type %s " 
													"from Field instance" % type(f2))

	def __rsub__(self, f1):
		""" Subtract somethign from field. """
		if isinstance(f1, Field):
			if not self.rank == f1.rank:
				raise ValueError("Fields must have same rank to be subtracted")
			else:
				return Field(f1.data - self.data, self.metadata)
		else: 
			try:
				return Field(f1 - self.data, self.metadata)
			except TypeError:
				raise ValueError("cannot subtract instance of type %s " 
													"from Field instance" % type(f1))

	def __div__(self, f2):
		""" Divide field by something. """
		if isinstance(f2, Field):
			if not self.rank == f2.rank:
				raise ValueError("Fields must have same rank to be divided")
			else:
				return Field(self.data/f2.data, self.metadata)
		else: 
			try:
				return Field(self.data/f2, self.metadata)
			except TypeError:
				raise ValueError("cannot divide Field instance wit instance of type %s" % type(f2))

	def __mul__(self, f2):
		""" Multiply field by something. """
		if isinstance(f2, Field):
			if self.rank == 0 and f2.rank == 0:
				# scalar with scalar: elementwise
				expr = "kl, kl -> kl" 
			elif self.rank == 1 and f2.rank == 1: 
				# vector dot vector: reduction
				expr = "ikl, ikl -> kl"
			elif self.rank == 2 and f2.rank == 2: 
				# tensor tensor: elementwise
				expr = "ijkl, ijkl -> ijkl"
			elif self.rank == 1 and f2.rank == 2: 
				# vector dot tensor: reduction
				expr = "jkl, ijkl -> ikl"
			else:
				raise TypeError("operation not supported")
			return Field(np.einsum(expr, self.data, f2.data), self.metadata)

		else: 
			if isinstance(f2, (float, int)):
				return Field(self.data*f2, self.metadata)
			elif isinstance(f2, np.ndarray):
				if len(f2.shape) == 2: 
					raise TypeError("cannot right-multiply Field instance with matrix")
				elif len(f2.shape) == 1: #multiply by a vector
					return Field(np.einsum("j, jkl -> kl", f2, self.data), self.metadata)
			else:
				raise ValueError("cannot multiply Field instance with %s " 
													"instance" % type(f2))

	def __rmul__(self, f1):
		""" Mirror operator. """
		#multiply by a vector or matrix 
		if isinstance(f1, np.ndarray):
			if len(f1.shape) == 2: 
				return Field(np.einsum("ij, jkl -> ikl", f1, self.data), self.metadata)
			elif len(f1.shape) == 1:
				if self.rank == 0:
					return Field(np.einsum("j, kl -> jkl", f1, self.data), self.metadata)
				elif self.rank == 1:
					return Field(np.einsum("j, jkl -> kl", f1, self.data), self.metadata)
		else:
			try:	
				return Field(self.data*f1, self.metadata)
			except TypeError:
				raise ValueError("cannot multiply Field instance with %s " 
													"instance" % type(f1))

	def __iadd__(self, f2):
		""" Add something to field. """
		if isinstance(f2, Field):
			if not self.rank == f2.rank:
				raise ValueError("Fields must have same rank to be added")
			else:
				self.data += f2.data
		else: 
			try:
				self.data += f2
			except TypeError:
				raise ValueError("cannot add instance of type %s " 
													"to Field instance" % type(f2))	
		return self

	def __pow__(self, p):
		""" Square operation """
		if p == 2:
			if self.rank == 0: # elementwise
				expr = "kl, kl-> kl"
			elif self.rank == 1: # reduction
				expr = "ikl, ikl -> kl"
			elif self.rank == 2: # elementwise
				expr = "ijkl, ijkl -> ijkl"
			else:
				raise TypeError("squaring not defined for rank %f" % p)
			return Field(np.einsum(expr, self.data, self.data), self.metadata)
		else:
			raise ValueError("Are you sure using p=%f?" % p)

	@property 
	def z(self):
		if Field._grid is None:
			Field._grid = _makeGrid(self.metadata)
		elif Field._grid.shape != (2, self.metadata['Ny'], self.metadata['Nz']+1):
			Field._grid = _makeGrid(self.metadata)
		return Field._grid[1]

	@property 
	def y(self):
		if Field._grid is None:
			Field._grid = _makeGrid(self.metadata)
		elif Field._grid.shape != (2, self.metadata['Ny'], self.metadata['Nz']+1):
			Field._grid = _makeGrid(self.metadata)
		return Field._grid[0]