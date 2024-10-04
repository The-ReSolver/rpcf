__author__ = """Davide Lasagna, Aerodynamics and
                Flight Mechanics group, Soton Uni."""

import itertools
from operator import itemgetter
from configparser import ConfigParser, NoSectionError
import os, glob, copy

import numpy as np 
from pyRPCFPost.rpcf.field import Field

# ----------------------------------------
# A class to load results from simulations
# of the rotating plane couette flow.
# ----------------------------------------
class SnapshotTimeError(Exception):
    pass

class SimulationResults():
    def __init__(self, casedir):
        """ An iterator to iterate over simulation results. 
            
            Parameters
            ----------
            casedir : str
                path of the directory with simulation results


            Notes
            -------
            The iterator returns Field istances
        """
        # where are we
        self.casedir = casedir

        if not os.path.exists(self.casedir):
            raise ValueError("casedir not found")

        # load parameters from the same folder as data
        c = ConfigParser()
        c.optionxform=str
        c.read(os.path.join(casedir, "params"))

        # convert to appropriate values
        floats = ['Re', 'Ro', 'dt', 'T', 'L', 't_restart', 't_offset', 'stretch_factor', 'A', 'eta', 'steady_halt']
        convert = lambda k, v : float(v.strip(";")) if k in floats else int(v.strip(";"))
        self.params = {k:convert(k, v) for k, v in c.items("params")}

        # add alpha
        self.params['alpha'] = 2*np.pi/self.params['L']

        # where are we when iterating
        self._current = 0

        # time array
        self.t = np.array(sorted(map(float, [os.path.basename(d) for d in glob.glob(os.path.join(self.casedir, "*.*"))])))

        # number of snapshots available
        self._len = len(self.t)

    def getK(self, ts=None):
        """ Load time history of perturbation kinetic energy. """
        if ts is None:
            ts = self.t
        try:
            return np.array([self._parse_metadata(t)['K'] for t in ts])
        except IOError: # ts is a single time and we return a single value
            raise SnapshotTimeError("snapshot not found. """)
        
        return self._parse_metadata(ts)['K']

    def __getitem__(self, slice_obj):
        """ Load and return snapshot of the flow field at some given time. """
        # when we want only one snapshot
        if isinstance(slice_obj, float) or isinstance(slice_obj, int):
            try:
                return self._load(slice_obj)
            except IOError:
                raise SnapshotTimeError("snapshots not available at time %f." % slice_obj)

        # when we want many of them, i.e. a selection of snapshots
        if isinstance(slice_obj, slice):

            # default is snapshot spacing
            step = slice_obj.step if slice_obj.step is not None else self.t[1]
            if abs(divmod(step, self.t[1])[1] > 1e-10):
                raise IndexError("step size is not a mutiple of available data time step") 

            # if stop is not provided we set the maximum            
            stop = slice_obj.stop if slice_obj.stop <= self.t[-1]  else self.t[-1] + step

            return (self[t] for t in np.arange(slice_obj.start, stop, step))

    def __len__(self):
        """ Length is the number of snapshots taken. """
        return self._len

    def rewind(self):
        self._current = 0

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
        c = ConfigParser()
        c.optionxform = str
        if not os.path.exists(os.path.join(self.casedir, "%f/metadata" %t)):
            raise IOError("metadata file not found at t = %f" % t)
        c.read(os.path.join(self.casedir, "%f/metadata" %t))
        return {k:float(v) for k, v in c.items("metadata")}

    def _load(self, t):
        """ Lazyily loading data into the workspace """
        U = np.memmap(filename=os.path.join(self.casedir, "%f/U" % t),
                      dtype=np.float64,
                      shape=(3, self.params['Ny'], self.params['Nz']+1),
                      mode='r', 
                      order='C')

        psi = np.memmap(filename=os.path.join(self.casedir, "%f/psi" % t),
                       dtype=np.float64,
                       shape=(self.params['Ny'], self.params['Nz']+1),
                       mode='r', 
                       order='C')

        omega = np.memmap(filename=os.path.join(self.casedir, "%f/omega" % t),
                        dtype=np.float64,
                        shape=(self.params['Ny'], self.params['Nz']+1),
                        mode='r', 
                        order='C')

        # update parameters
        metadata = self.params.copy()
        metadata.update(self._parse_metadata(t))

        # that's all folks
        return Snapshot(U, psi, omega, metadata)

class Snapshot():
    def __init__(self, U, psi, omega, metadata):
        """ Light class for loading data in the workspace. """

        # all metadata of the snapshot
        self.metadata = metadata
        
        # metadata that will be transmitted to Fields instances
        meta = ["Ny", "Nz", "t", "L", "alpha", "stretch_factor"]
        self._partial_metadata = {meta_i: metadata[meta_i] for meta_i in meta}

        # lazily store data
        self.U  = Field(U, self._partial_metadata)
        self.psi  = Field(psi, self._partial_metadata)
        self.omega  = Field(omega, self._partial_metadata)