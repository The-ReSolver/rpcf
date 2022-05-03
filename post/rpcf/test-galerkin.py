import unittest

import numpy as np
import numpy.testing as npt
from numpy import sin, cos

from pylab import *

from rpcf.field import Field
from rpcf.galerkin import orthonormalize

finiteDifferenceDecimalsPrecision = 3
integralDecimalsPrecision = 5
decimalsPrecision = 10



class TestGalerkin(unittest.TestCase):
	def setUp(self):
		self.metadata = { "Ny": 200,
						  "Nz": 100, 
						  "L" : 2,
						  "stretch_factor" : 3,
						  "alpha" : 2*np.pi/2}

	def test1(self):

		N = 5

		# make ten random modes
		datas = [np.random.normal(0, 1, (2, self.metadata['Ny'], self.metadata['Ny'])) for i in range(N)]
		modes = [Field(data, self.metadata) for data in datas]

		# make them orthonormal
		modes = orthonormalize(modes)

		for m in modes:
			matshow(m.data[0])

		show()





if __name__ == '__main__':
	unittest.main()