__author__ = """Davide Lasagna, Aerodynamics and
			    Flight Mechanics group, Soton Uni."""

import copy
import h5py
import numpy as np 

from pyRPCFPost.couetteModel.fspace import fproduction, flaplacian, fintegralEnergy, \
								fdissipation, fgradient3D, fcurl, fgradient,\
								wave_number_array3D

from pyRPCFPost.couetteModel import lru

class Field():
	def __init__(self, metadata, data=None, dataK=None):
		""" Data container in physical space.
			This contains azimuthal velocity 
			and streamfunction. """
		self._data = data 
		self._dataK = dataK
		self.metadata = copy.copy(metadata)

	def to_velocity_components(self):
		# get wavenumbers
		K = wave_number_array3D((self.metadata['N'], self.metadata['N']))
		dataK = np.zeros((3, self.metadata['N'], self.metadata['N']/2+1), dtype=np.complex128)
		np.seterr(divide='ignore')
		dataK[0] = self.dataK[0]
		dataK[1] = self.dataK[1]/1j/K[2]
		dataK[2] = self.dataK[1]/1j/K[1]
		dataK[1, :, 0] = 0.0
		dataK[2, 0, :] = 0.0
		return Field(dataK=dataK, metadata=self.metadata)

	@property
	def rank(self):
		try:
			return len(self._dataK.shape) - 2
		except AttributeError:
			return len(self._data.shape) - 2		

	@property
	def data(self):
		""" Lazy evaluation of Fourier transform. """
		if self._data is None:
			self._data = np.fft.irfft2(self._dataK)
		return self._data

	@property
	def dataK(self):
		""" Lazy evaluation of Fourier transform. """
		if self._dataK is None:
			self._dataK = np.fft.rfft2(self._data)
		return self._dataK
		
	def tofile(self, filename):
		outh = open(filename, "wb")
		for i in range(self._data.shape[0]):
			im = self.dataK[i].ravel().imag
			re = self.dataK[i].ravel().real
			np.c_[re, im].ravel().tofile(outh)
		outh.close()

	def __getitem__(self, sl):
		""" Implement slicing. """
		return self._data[sl]

	def __neg__(self):
		""" Change sign of field data """
		return Field(data=-self.data, metadata=self.metadata)

	def __abs__(self):
		""" Compute absolute value of Field data """
		return Field(np.abs(self.data), metadata=self.metadata)

	def __add__(self, f2):
		if isinstance(f2, Field):
			if self.rank == f2.rank:
				return Field(data=self.data + f2.data, metadata=self.metadata)
			else:
				raise ValueError("Fields must have same rank to be summed")
		else:
			try:
				return Field(data=self.data + f2, metadata=self.metadata)
			except TypeError:
				raise ValueError("cannot add Field instance to %s instance" 
															% type(f2))

	def __sub__(self, f2):
		if isinstance(f2, Field):
			if self.rank == f2.rank:
				return Field(data=self.data - f2.data, metadata=self.metadata)
			else:
				raise ValueError("Fields must have same rank to be subtracted")
		else:
			try:
				return Field(data=self.data - f2, metadata=self.metadata)
			except TypeError:
				raise ValueError("cannot subtract %s instance to Field" 
													"instance" % type(f2))

	def __mul__(self, f2):
		if isinstance(f2, Field):
			if self.rank == f2.rank:
				return Field(data=self.data * f2.data, metadata=self.metadata)
			else:
				raise ValueError("Fields must have same rank to be multiplied")
		else:
			try:
				return Field(data=self.data * f2, metadata=self.metadata)
			except TypeError:
				raise ValueError("cannot multiply Field instance with %s " 
													"instance" % type(f2))

	def __div__(self, f2):
		if isinstance(f2, Field):
			if self.rank == f2.rank:
				return Field(data=self.data / f2.data, metadata=self.metadata)
			else:
				raise ValueError("Fields must have same rank to be divided")
		else:
			try:
				return Field(data=self.data / f2, metadata=self.metadata)
			except TypeError:
				raise ValueError("cannot divide Field instance with %s " 
													"instance" % type(f2))

	def __pow__(self, exp):
		try:
			return Field(data=self.data ** exp, metadata=self.metadata)
		except TypeError:
			raise ValueError("cannot computer power Field instance with %s " 
												"instance" % type(exp))

	def __radd__(self, f2):
		try:
			return Field(data=self.data + f2, metadata=self.metadata)
		except TypeError:
			raise ValueError("cannot add Field instance to %s instance" 
															% type(f2))

	def __rsub__(self, f2):
		try:
			return Field(data=self.data - f2, metadata=self.metadata)
		except TypeError:
			raise ValueError("cannot subtract Field instance from %s instance"
															% type(f2))

	def __rmul__(self, f2):
		try:
			return Field(data=self.data * f2, metadata=self.metadata)
		except TypeError:
			raise ValueError("cannot multiply Field instance with %s instance" 
															% type(f2))

	def __iadd__(self, f2):
		if isinstance(f2, Field):
			if self.rank == f2.rank:
				self.data += f2.data
				return self
			else:
				raise ValueError("Fields must have same rank to be summed")
		else:
			try:
				self.data += f2
				return self			
			except TypeError:
				raise ValueError("cannot add Field instance to %s instance" 
															% type(f2))

	def __isub__(self, f2):
		if isinstance(f2, Field):
			if self.rank == f2.rank:
				self.data -= f2.data
				return self
			else:
				raise ValueError("Fields must have same rank to be subtracted")
		else:
			try:
				self.data -= f2
				return self			
			except TypeError:
				raise ValueError("cannot subtract %s instance to Field" 
												"instance" % type(f2))


#@lru.lru_cache(maxsize=50)
def dot(f, g):
	""" Dot product between two fields """
	# vector vector product u dot u
	if f.rank == 1 and g.rank == 1:
		out = np.einsum("ijk, ijk -> jk", f.data, g.data)
	# vector tensor product u dot grad u
	elif f.rank == 1 and g.rank == 2:
		out = np.einsum("jmn, ijmn -> imn", f.data, g.data)
	# tensor vector product 
	elif f.rank == 2 and g.rank == 1:
		out = np.einsum("jmn, ijmn -> imn", g.data, f.data)
	else:
		raise ValueError("dot product is not defined for\
			 fields with rank %d and %d" % (f.rank and g.rank))
	return Field(data=out, metadata=f.metadata)

#@lru.lru_cache(maxsize=50)
def integral(f):
	""" Compute integral of scalar field over 2*PI square domain """
	if not f.rank == 0:
		raise ValueError("integral only defined for scalar fields")
	return _trapz2d(_fill(f.data), dx=2*np.pi/f.data.shape[1], dy=2*np.pi/f.data.shape[0])

def integralComponent(f, idx):
	""" Compute integral of scalar field over 2*PI square domain """
	return _trapz2d(_fill(f.data[idx]), dx=2*np.pi/f.data.shape[2], dy=2*np.pi/f.data.shape[1])

#@lru.lru_cache(maxsize=50)
def grad(f):
	""" Compute velocity gradient tensor. """
	if f.rank == 1:
		return Field(dataK=fgradient3D(f.dataK), metadata=f.metadata)
	if f.rank == 0:
		return Field(dataK=fgradient(f.dataK), metadata=f.metadata)

#@lru.lru_cache(maxsize=50)
def laplacian(f):
	""" Laplacian of a field. """
	return Field(dataK=flaplacian(f.dataK), metadata=f.metadata)

#@lru.lru_cache(maxsize=50)
def curl(f):
	"""Compute curl of vector field"""
	return Field(dataK=fcurl(f.dataK), metadata=f.metadata)

#@lru.lru_cache(maxsize=50)
def Soper(f, g):
	if not f.rank == 1 and not g.rank == 1:
		raise ValueError("Soper is defined for vector fields only")
	return dot(f, grad(g)) + dot(g, grad(f)) 

def integralEnergy(f):
	""" Return integral of kinetic energy over the domain. """
	try:
		return f.metadata['K']
	except KeyError:
		return fintegralEnergy(f.dataK)

def dissipationK(f):
	""" Return dissipation computed in Fourier space """
	return fdissipation(f.dataK)/f.metadata['Re']

def productionK(f):
	""" Return production computed in Fourier space """
	return fproduction(f.dataK)


# --- helper functions ----
def _trapz2d(z, x=None, y=None, dx=1., dy=1.):
    ''' Integrates a regularly spaced 2D grid using the composite trapezium rule. 
    IN:
       z : 2D array
       x : (optional) grid values for x (1D array)
       y : (optional) grid values for y (1D array)
       dx: if x is not supplied, set it to the x grid interval
       dy: if y is not supplied, set it to the x grid interval
    '''
    
    if x != None:
        dx = (x[-1]-x[0])/(np.shape(x)[0]-1)
    if y != None:
        dy = (y[-1]-y[0])/(np.shape(y)[0]-1)    
    
    s1 = z[0,0] + z[-1,0] + z[0,-1] + z[-1,-1]
    s2 = np.sum(z[1:-1,0]) + np.sum(z[1:-1,-1]) + np.sum(z[0,1:-1]) + np.sum(z[-1,1:-1])
    s3 = np.sum(z[1:-1,1:-1])
    
    return 0.25*dx*dy*(s1 + 2*s2 + 4*s3)

def _fill(data):
	""" Create copy of data filled to represent periodic boundary conditions. """
	new = np.zeros((data.shape[0]+1, data.shape[1]+1), dtype=data.dtype)
	new[:-1, :-1] = data
	new[:-1, -1] = data[:,0]
	new[-1, :-1] = data[0, :]
	new[-1, -1] = data[0,0]
	return new
