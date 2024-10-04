import unittest

import numpy as np
import numpy.testing as npt
from numpy import sin, cos

from pyRPCFPost.couetteModel.operators import Field, dot, integral, \
 			                       laplacian, curl, grad, dissipationK, \
				                    productionK, integralEnergy
from pyRPCFPost.couetteModel.utils import makeGrid

decimalsPrecision = 10


class TestField(unittest.TestCase):
	def setUp(self):
		self.metadata = { "Ny": 64,
						  "Nz": 64,
						  "Re": 1,}

		# prepare test flow field
		self.X = makeGrid(self.metadata)
		x, y, z = self.X
		self.U = np.zeros_like(self.X)
		self.U[0] =  cos(y)*cos(z)
		self.U[1] =  sin(y)*cos(z)
		self.U[2] = -cos(y)*sin(z)

		self.field = Field(data=self.U, metadata=self.metadata)


	def testGrad(self):
		""" Test computation of gradient. """
		trueGrad = np.zeros((3, 3, self.metadata['Ny'], self.metadata['Nz']), 
														dtype=np.float64)
		x, y, z = self.X
		trueGrad[0,0] =  0.0
		trueGrad[0,1] = -sin(y)*cos(z)
		trueGrad[0,2] = -cos(y)*sin(z)
		trueGrad[1,0] =  0.0
		trueGrad[1,1] =  cos(y)*cos(z)
		trueGrad[1,2] = -sin(y)*sin(z)
		trueGrad[2,0] =  0.0
		trueGrad[2,1] =  sin(y)*sin(z)
		trueGrad[2,2] = -cos(y)*cos(z)

		npt.assert_array_almost_equal(trueGrad, 
									  grad(self.field).data, 
									  decimalsPrecision)

	def testDot(self):
		""" Test dot product. """

		# test vector vector
		npt.assert_array_almost_equal(np.sum(self.U**2, axis=0), 
									  dot(self.field, self.field).data, 
									  decimalsPrecision)

		# build true gradient field
		trueGrad = np.zeros((3, 3, self.metadata['Ny'], self.metadata['Nz']), 
														dtype=np.float64)
		x, y, z = self.X
		trueGrad[0,0] =  0.0
		trueGrad[0,1] = -sin(y)*cos(z)
		trueGrad[0,2] = -cos(y)*sin(z)
		trueGrad[1,0] =  0.0
		trueGrad[1,1] =  cos(y)*cos(z)
		trueGrad[1,2] = -sin(y)*sin(z)
		trueGrad[2,0] =  0.0
		trueGrad[2,1] =  sin(y)*sin(z)
		trueGrad[2,2] = -cos(y)*cos(z)

		gradField = Field(data=trueGrad, metadata=self.metadata)


		# build true result
		trueOut = np.zeros((3, self.metadata['Ny'], self.metadata['Nz']), 
													dtype=np.float64)
		trueOut[0] = -sin(y)**2*cos(z)**2 + cos(y)**2*sin(z)**2
		trueOut[1] = sin(y)*cos(y)
		trueOut[2] = sin(z)*cos(z)

		# test vector/tensor
		npt.assert_array_almost_equal(trueOut, 
									  dot(self.field, gradField).data, 
									  decimalsPrecision)
		npt.assert_array_almost_equal(dot(gradField, self.field).data, 
								      trueOut, 
								      decimalsPrecision)

	def testIntegral(self):
		""" Test integral by computing energy of the test field. """
		npt.assert_almost_equal(integral(dot(self.field, self.field)), 
							    3*np.pi*np.pi,
							    decimalsPrecision)

	def testIntegralEnergy(self):
		""" Test the value of the L2 norm of the field. """
		npt.assert_almost_equal(integralEnergy(self.field), 
							    3*np.pi*np.pi/2,
							    decimalsPrecision)

	def testLaplacian(self):
		""" Test laplacian operator """
		trueLaplacian = -2*self.U
		npt.assert_array_almost_equal(trueLaplacian, 
									  laplacian(self.field).data, 
									  decimalsPrecision)

	def testCurl(self):
		""" Test curl of vector field """
		trueCurl = np.zeros((3, self.metadata['Ny'], self.metadata['Nz']), dtype=np.float64) 
		x, y, z = self.X
		trueCurl[0] =  2*sin(y)*sin(z)
		trueCurl[1] = -cos(y)*sin(z)
		trueCurl[2] = sin(y)*cos(z)

		npt.assert_array_almost_equal(trueCurl, 
									  curl(self.field).data, 
									  decimalsPrecision)

	def testDissipation(self):
		""" Test that dissipation rate computed using standard vector notation
			is equal to that computed in Fourier space. """
		npt.assert_almost_equal(integral(dot(self.field, laplacian(self.field))),
								np.sum(dissipationK(self.field)),
								decimalsPrecision)


class TestProduction(unittest.TestCase):
	def setUp(self):
		""" Set up a flow field which has a production different from zero, 
			using an eigenfunction of the energy method. """
		self.metadata = { "Ny": 64,
						  "Nz": 64,
						  "Re": 1,}

		# eigenfunction number
		n, m = 1, 1

		# prepare test flow field
		self.X = makeGrid(self.metadata)
		x, y, z = self.X
		self.U = np.zeros_like(self.X)
		self.U[0] =    cos(m*z)*sin(n*y)/np.sqrt(2)/np.pi
		self.U[1] =  m*cos(m*z)*sin(n*y)/np.sqrt(2)/np.pi/np.sqrt(m**2 + n**2)
		self.U[2] = -n*sin(m*z)*cos(n*y)/np.sqrt(2)/np.pi/np.sqrt(m**2 + n**2)

		self.field = Field(data=self.U, metadata=self.metadata)

	def testProduction(self):
		""" Test that production rate computed using standard vector notation
			is equal to that computed in Fourier space. """
		npt.assert_almost_equal(-integral(Field(data=self.U[0]*self.U[1], 
								metadata=self.metadata)),
								np.sum(productionK(self.field)),
								decimalsPrecision)

if __name__ == '__main__':
	unittest.main()