__author__ = """Davide Lasagna, Aerodynamics and
			    Flight Mechanics group, Soton Uni."""

from operator import itemgetter
from ConfigParser import ConfigParser, NoSectionError
import os, glob

import h5py
import numpy as np 

from couetteModel.operators import Field, integralEnergy
from couetteModel.fspace import fcorrection

# ---------------------
# Some helper functions
# ---------------------
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

def makeGrid(params):
	"""	Create a 2d grid for visualization. """
	N1, N2 = params['N'], params['N']
	X = np.zeros((3, N1, N2), dtype=np.float64)
	x = np.mgrid[0:N1, 0:N2]
	X[1] = x[0]*2*np.pi/N1
	X[2] = x[1]*2*np.pi/N2
	return X

def randomField2(Ny, Nz, K0, initfile):
	""" Create random field and dump data to file."""
	# create random data
	u = np.random.normal(0, K0, (2, Ny, Nz))
	u -= np.mean(np.mean(u, axis=1), axis=1).reshape(2, 1, 1)
	uk = np.fft.rfft2(u)
	outh = open(initfile, "wb")
	for i in range(2):
		im = uk[i].ravel().imag
		re = uk[i].ravel().real
		np.c_[re, im].ravel().tofile(outh)
	outh.close()

def randomField(Ny, Nz, K0, initfile=None):
	""" Create random field and dump data to file."""
	# create random data
	u = np.random.normal(0, 1, (3, Ny, Nz))
	u -= np.mean(np.mean(u, axis=1), axis=1).reshape(3, 1, 1)

	# add correction for incompressibility
	uk = np.fft.rfft2(u) 
	uk += fcorrection(uk)
	uk[:, 0, 0] = 0

	# create field and then apply correction
	f = Field(dataK=uk, metadata={'Ny':Ny, 'Nz':Nz})
	
	# normalize
	f.dataK /= integralEnergy(f)**0.5 / K0**0.5

	# save to disk
	if initfile:
		f.tofile(initfile)
	else:
		return f

# -----------------------------------------------------------
# A class to generate the eigenfunctions of the energy method
# -----------------------------------------------------------
class EigenGen():
	def __init__(self, params):
		""" A callable object that return eigenfunctions of the
			energy problem, as instance of the Field class.
		"""
		self.X = makeGrid(params)
		self.params = params

	def __call__(self, nm):
		n, m = nm
		U = np.zeros_like(self.X)
		U[0] =    np.cos(m*self.X[2])*np.sin(n*self.X[1])/np.sqrt(2.0)/np.pi
		U[1] =  m*np.cos(m*self.X[2])*np.sin(n*self.X[1])/np.sqrt(2.0)/np.pi/np.sqrt(m*m+n*n)
		U[2] = -n*np.sin(m*self.X[2])*np.cos(n*self.X[1])/np.sqrt(2.0)/np.pi/np.sqrt(m*m+n*n)
		return Field(data=U, metadata=self.params)


# ----------------------------------------
# A class to load results from simulations
# ----------------------------------------
class SimulationResults():
	def __init__(self, casedir, verbose=False):
		""" An iterator to iterate over time on the simulation results. 
			
			Parameters
			----------
			casedir : str
				path of the directory with simulation results

			verbose: Boolean
				wheter to print debug stuff


			Notes
			-------
			The iterator returns Field istances
		"""
		# where are we
		self.casedir = casedir

		if not os.path.exists(self.casedir):
			raise ValueError("casedir not found")

		self.verbose = verbose

		# load parameters from the same folder as data
		c = ConfigParser()
		c.optionxform=str
		c.read(os.path.join(casedir, "params"))

		# convert to appropriate values
		floats = ['Re', 'Omega', 'dt', 'T']
		convert = lambda k, v : float(v.strip(";")) if k in floats else int(v.strip(";"))
		self.params = {k:convert(k, v) for k, v in c.items("params")}

		# where are we when iterating
		self._current = 0

		# do we have a time array?
		self._t = None

	def getK(self):
		""" Load time history of perturbation kinetic energy. """
		return np.array([self._parse_metadata(t)['K'] for t in self.t])

	@property
	def t(self):
		""" Try to compute a time array. """
		if self._t is None:
			#self._t = np.arange(len(self))*self.params['dt']*self.params['n_it_out']
			self._t = np.array(sorted(map(float, [os.path.basename(d) for d in glob.glob(os.path.join(self.casedir, "*.*"))])))
		return self._t

	def rewind(self):
		self._current = 0

	def getSnapshotAt(self, t):
		""" Load and return snapshot of the flow field at some given time. """
		return self._load(t)
		
	def __len__(self):
		""" Length is the number of snapshots taken. """
		return len(glob.glob(os.path.join(self.casedir, "*.*")))

	def __iter__(self):
		return self

	def next(self):
		""" Get next snapshot to process. """
		if self._current == len(self):
			# rewind simulation
			self._current = 0
			raise StopIteration
		else:
			field = self._load(self.params['n_it_out']*
							   self.params['dt']*
							   self._current)
			self._current += 1
			return field

	def _parse_metadata(self, t):
		""" Parse metadata file """
		if self.verbose: print "%f" % t
		c = ConfigParser()
		c.optionxform = str
		if not os.path.exists(os.path.join(self.casedir, "%f/metadata" %t)):
			raise IOError("metadata file not found at t = %f" % t)
		c.read(os.path.join(self.casedir, "%f/metadata" %t))
		return {k:float(v) for k, v in c.items("metadata")}


	def _load(self, t):
		""" Load data from snapshot at time t """

		data = np.fromfile(os.path.join(self.casedir, "%f/data" %t), dtype=np.complex128)
		U_K = data.reshape((2, self.params['N'], self.params['N']/2+1))

		# update parameters
		metadata = self.params.copy()
		metadata.update(self._parse_metadata(t))

		# that's all folks
		return Field(dataK=U_K, metadata=metadata)
