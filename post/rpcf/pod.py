import numpy as np
from rpcf.field import Field, integral


class POD:
	def __init__(self, sim, ts, demean=False):
		""" Compute POD modes of simulations data. 

			Parameters
			----------
			sim : a SimulationResults instance

			ts : an iterable of time instant at which snapshots
				of the flow will be extracted for computation of 
				the POD.
		"""
		self.sim = sim 
		self.ts = ts

		# build snapshot matrix. this is a n_features x n_samples matrix
		snapshots = np.empty((3*self.sim.params['Ny']*(self.sim.params['Nz'] + 1), len(ts)))

		# fill it
		for i, t in enumerate(ts):
			snapshots[:, i] = self.sim[t].U.data.ravel()

		# demean snapshots
		if demean:
			snapshots -= np.mean(snapshots, axis=1).reshape(-1, 1)

		# compute decomposition
		self._u, self._s, dummy = np.linalg.svd(snapshots, full_matrices=False)

	def getMode(self, i):
		""" Return i-th pod mode. Modes are orthonormal. """
		data = self._u[:,i].reshape(3, self.sim.params['Ny'], self.sim.params['Nz'] + 1)
		f = Field(data, self.sim.params)
		return f/integral(f*f)**0.5

	def spectrum(self):
		""" Return spectrum of correlations matrix. """
		return self._s**2 * (4*np.pi**2/len(self.ts)/self.sim.params['Ny']/(self.sim.params['Nz'] + 1))

	def getModeAmplitude(self, js, ts):
		""" Compute temporal modal coefficients. 

			Parameters
			----------
			js : iterable of int values
				the index of the modes of interest

			ts : iterable of float values
				time instants at which projection is computed

			Returns
			-------
			ais : np.ndarray of shape (n_modes, n_snapshots)
		"""

		# compute modes
		modes = [self.getMode(j) for j in js]

		# allocate memory
		ais = np.empty((len(js), len(ts)))

		# compute projections
		for j, t in enumerate(ts):
			up = self.sim.getSnapshotAt(t) - self.meanFlow
			for i, mode in enumerate(modes):
				ais[i, j] = integral(dot(up, mode))

		return ais
