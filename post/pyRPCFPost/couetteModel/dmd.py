import numpy as np
from pyRPCFPost.couetteModel.operators import Field, integral, dot

class DMD:

    def __init__(self, sim, ts, mean=None):
        """ 

            Parameters
            ----------
            sim : a SimulationResults instance

            ts : an iterable of time instant at which snapshots
                of the flow will be extracted for computation of 
                the DMD.
        """
        self.sim = sim 
        self.ts = ts

        # build snapshot matrix. this is a n_features x n_samples matrix
        self._snapshots = np.asmatrix(np.empty((3*self.sim.params['Ny']*self.sim.params['Ny'], len(ts))))

        # fill it
        for i, t in enumerate(ts):
            if mean:
                self._snapshots[:, i] = (self.sim.getSnapshotAt(t) - mean).data.reshape(-1, 1)
            else:
                self._snapshots[:, i] = self.sim.getSnapshotAt(t).data.reshape(-1, 1)

        # compute svd of data
        self._U, Sigma, W = np.linalg.svd(self._snapshots[:, :-1], full_matrices=False)

        # compute inverse os diagonal matrix
        SigmaI = np.asmatrix(np.diag(1/Sigma))

        # get our S
        S = self._U.H * self._snapshots[:, 1:] * W.H * SigmaI

        # compute eigenvalues
        mu, self._Y = np.linalg.eig(S)

        # apply transform to go to continuos space
        self.lambdas = np.log(mu) / (ts[1]-ts[0])

    def mode(self, i):
        phi = (self._U*self._Y[:,i])
        return np.asarray(phi).reshape(3, self.sim.params['Ny'], self.sim.params['Nz'])
    
    def residual(self):
        a, res, rank, s = np.linalg.lstsq(self._snapshots[:, :-1], self._snapshots[:, -1])
        return res


if __name__ == '__main__':
    

    from itertools import product
    import numpy.testing as npt
    from couetteModel.utils import SimulationResults
    from pylab import show, plot

    # dimensionality of the snapshots
    M = 5

    # create random system matrix 
    A = np.diag(np.linspace(1, 10, M*M))

    # generate exact snapshots
    V  = np.random.normal(0, 1, (M*M, 2*M) )
    for i in range(2*M-1):
        V[:, i+1] = np.dot(A, V[:, i])

    # compute dmd
    dmd = DMD(V)

    # compare eigenvalues
    print sorted(dmd.lambdas.real)
    print sorted(np.linalg.eigvals(A).real)
    # npt.assert_array_almost_equal(np.linalg.eig(A), dmd.lambdas)




