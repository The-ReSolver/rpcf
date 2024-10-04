import numpy as np
from pyRPCFPost.couetteModel.operators import Field, integral, dot

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
		self._snapshots = np.empty((self.sim.params['N']*self.sim.params['N'], len(ts)))

		# fill it
		for i, t in enumerate(ts):
			self._snapshots[:, i] = self.sim.getSnapshotAt(t).data[0].ravel()

		# compute mean flow
		self._mean = np.mean(self._snapshots, axis=1)
		self.meanFlow = Field(data=self._mean.reshape(self.sim.params['N'], self.sim.params['N']), metadata={})

		# demean snapshots
		if demean:
			self._snapshots -= self._mean.reshape(-1, 1)

		# compute decomposition
		self._u, self._s, dummy = np.linalg.svd(self._snapshots, full_matrices=False)

	def getMode(self, i):
		""" Return i-th pod mode. Modes are orthonormal. """
		data = self._u[:,i].reshape(3, self.sim.params['N'], self.sim.params['N'])
		f = Field(data=data, metadata=self.sim.params)
		return f/integral(dot(f, f))**0.5

	def spectrum(self):
		""" Return spectrum of correlations matrix. """
		return self._s**2 * (4*np.pi**2/len(self.ts)/self.sim.params['N']/self.sim.params['N'])

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



if __name__ == '__main__':
	
	from itertools import product
	import numpy.testing as npt
	from pyRPCFPost.couetteModel.utils import SimulationResults

	# load data
	sim = SimulationResults("test-pod.hdf5")

	# use these snapshots
	ts = np.arange(0, 10, 0.1)

	# compute pod
	pod = POD(sim, ts)

	# use this number of modes
	N = 100

	## 1 # verify modes are orthonormal. (This proves orthogonality and unit norm.)
	modes = [pod.getMode(i) for i in range(N)]
	delta = np.array([integral(dot(modeI, modeJ)) for modeI, modeJ in product(modes, modes)])
	npt.assert_array_almost_equal(delta.reshape(N, N), np.eye(N), 10)
					
	## verify reconstruction is working. (This also proves the amplitude is fine.)
	ais = pod.getModeAmplitude(range(len(ts)), [0])
	modes = [pod.getMode(i) for i in range(len(ts))]
	u0 = sum(mode*ai for mode, ai in zip(modes, ais))
	npt.assert_array_almost_equal(u0.data, (sim.getSnapshotAt(0) - pod.meanFlow).data, 10)

	## 2 # verify amplitudes have zero mean value
	ais = pod.getModeAmplitude(range(N), ts)
	npt.assert_array_almost_equal( [np.mean(ai) for ai in ais], np.zeros(N), 10 )

	## 3 # verify amplitudes have variance equal to eigenvalue
	npt.assert_array_almost_equal( [np.var(ai) for ai in ais], pod.spectrum()[:N], 10 )
	




