import unittest

import numpy as np
import numpy.testing as npt
from numpy import sin, cos

from rpcf.field import Field, _makeGrid, grad, _der1, _der2, der, laplacian
from rpcf.field import curl, integral

finiteDifferenceDecimalsPrecision = 3
integralDecimalsPrecision = 5
decimalsPrecision = 10



class Test_Der(unittest.TestCase):
	def setUp(self):
		# vector column
		x = 1 + np.cos(np.linspace(np.pi, 0, 1000)).reshape(-1, 1)
		
		# pack columns into a matrix
		self.x = np.tile(x, (1, 4))

		# function values
		self.y = np.exp(self.x)*x

		# exact first derivative
		self.yp = (1+x)*np.exp(self.x)

		# exact second derivative
		self.ypp = (2+x)*np.exp(self.x)

	def test_der1(self):
		npt.assert_array_almost_equal(self.yp, 
									  _der1(self.y, self.x), 
									  finiteDifferenceDecimalsPrecision)

	def test_der2(self):
		npt.assert_array_almost_equal(self.ypp, 
									  _der2(self.y, self.x), 
									  finiteDifferenceDecimalsPrecision)


class TestField(unittest.TestCase):
	def setUp(self):
		self.metadata = { "Ny": 200,
						  "Nz": 100, 
						  "L" : 2,
						  "stretch_factor" : 3,
						  "alpha" : 2*np.pi/2}

		# prepare test flow field
		y, z = _makeGrid(self.metadata)
		alpha = self.metadata['alpha']
		self.U = np.empty((3, 
						   self.metadata['Ny'], 
						   self.metadata['Nz']+1), dtype=np.float64)

		self.U[0] = y**2*cos(alpha*z)
		self.U[1] = y**3*cos(alpha*z)
		self.U[2] = y**4*sin(alpha*z)

		self.field = Field(self.U, self.metadata)

	def test_der_y(self):
		""" Test derivative along y """
		truedUdy = np.empty((3, 
						      self.metadata['Ny'], 
						      self.metadata['Nz']+1), dtype=np.float64)

		y, z = _makeGrid(self.metadata)
		alpha = self.metadata['alpha']
		truedUdy[0] = 2*y*cos(alpha*z)
		truedUdy[1] = 3*y**2*cos(alpha*z)
		truedUdy[2] = 4*y**3*sin(alpha*z)

	 	truedUdy = Field(truedUdy, self.metadata)
	 	npt.assert_array_almost_equal(truedUdy.data, 
	 								  der(self.field, 'y').data, 
	 								  finiteDifferenceDecimalsPrecision)

	def test_der2_y(self):
		""" Test derivative along y """
		trued2Udy2 = np.empty((3, 
						      self.metadata['Ny'], 
						      self.metadata['Nz']+1), dtype=np.float64)

		y, z = _makeGrid(self.metadata)
		alpha = self.metadata['alpha']
		trued2Udy2[0] =       2*cos(alpha*z)
		trued2Udy2[1] =     6*y*cos(alpha*z)
		trued2Udy2[2] = 12*y**2*sin(alpha*z)

	 	trued2Udy2 = Field(trued2Udy2, self.metadata)
	 	npt.assert_array_almost_equal(trued2Udy2.data[1], 
	 								  der(self.field, 'yy').data[1], 
	 								  finiteDifferenceDecimalsPrecision)

	def test_der_z(self):
		""" Test derivative along z """
		truedUdz = np.empty((3, 
						      self.metadata['Ny'], 
						      self.metadata['Nz']+1), dtype=np.float64)

		y, z = _makeGrid(self.metadata)
		alpha = self.metadata['alpha']
		truedUdz[0] = -y**2*alpha*sin(alpha*z)
		truedUdz[1] = -y**3*alpha*sin(alpha*z)
		truedUdz[2] =  y**4*alpha*cos(alpha*z)

	 	truedUdz = Field(truedUdz, self.metadata)
	 	npt.assert_array_almost_equal(truedUdz.data, 
	 								  der(self.field, 'z').data, 
	 								  decimalsPrecision)

	def test_der2_z(self):
		""" Test derivative along z """
		trued2Udz2 = np.empty((3, 
						      self.metadata['Ny'], 
						      self.metadata['Nz']+1), dtype=np.float64)

		y, z = _makeGrid(self.metadata)
		alpha = self.metadata['alpha']
		trued2Udz2[0] = -y**2*alpha**2*cos(alpha*z)
		trued2Udz2[1] = -y**3*alpha**2*cos(alpha*z)
		trued2Udz2[2] = -y**4*alpha**2*sin(alpha*z)

	 	trued2Udz2 = Field(trued2Udz2, self.metadata)
	 	npt.assert_array_almost_equal(trued2Udz2.data, 
	 								  der(self.field, 'zz').data, 
	 								  decimalsPrecision)

	def testGrad(self):
		""" Test computation of gradient. """
		trueGrad = np.zeros((3, 3, self.metadata['Ny'], self.metadata['Nz']+1), 
														dtype=np.float64)
		y, z = _makeGrid(self.metadata)
		alpha = self.metadata['alpha']
		trueGrad[0,0] =  0.0
		trueGrad[0,1] =    2*y*cos(alpha*z)
		trueGrad[0,2] =  -y**2*sin(alpha*z)*alpha
		trueGrad[1,0] =  0.0
		trueGrad[1,1] =  3*y**2*cos(alpha*z)
		trueGrad[1,2] =   -y**3*sin(alpha*z)*alpha
		trueGrad[2,0] =  0.0
		trueGrad[2,1] =  4*y**3*sin(alpha*z)
		trueGrad[2,2] =    y**4*cos(alpha*z)*alpha
		trueGrad = Field(trueGrad, self.metadata)

		for i in range(3):
			for j in range(3):
				if j == 2:
					npt.assert_array_almost_equal(trueGrad.data[i, j], 
									  	grad(self.field).data[i, j], 
									  	decimalsPrecision)
				else:
					npt.assert_array_almost_equal(trueGrad.data[i, j], 
									  	grad(self.field).data[i, j], 
									  	finiteDifferenceDecimalsPrecision)

	def testLaplacian(self):
		""" Test laplacian operator """
		trueLapl = np.zeros((3, self.metadata['Ny'], self.metadata['Nz']+1), 
														dtype=np.float64)
		y, z = _makeGrid(self.metadata)
		alpha = self.metadata['alpha']
		trueLapl[0] =       2*cos(alpha*z) - alpha**2*cos(alpha*z)*y**2
		trueLapl[1] =     6*y*cos(alpha*z) - alpha**2*cos(alpha*z)*y**3
		trueLapl[2] = 12*y**2*sin(alpha*z) - alpha**2*sin(alpha*z)*y**4
		trueLapl = Field(trueLapl, self.metadata)

		for j in range(3):
			npt.assert_array_almost_equal(trueLapl.data[j], 
									  	laplacian(self.field).data[j], 
									  	finiteDifferenceDecimalsPrecision)

	def testCurl(self):
		""" Test curl of vector field """
		trueCurl = np.zeros((3, self.metadata['Ny'], self.metadata['Nz']+1), dtype=np.float64) 

		y, z = _makeGrid(self.metadata)
		alpha = self.metadata['alpha']
		trueCurl[0] =  4*y**3*sin(alpha*z) + y**3*alpha*sin(alpha*z)
		trueCurl[1] = -alpha*y**2*sin(alpha*z)    
		trueCurl[2] = -2*y*cos(alpha*z)
		trueCurl = Field(trueCurl, self.metadata)

		for j in range(3):
			npt.assert_array_almost_equal(trueCurl.data[j], 
									  	  curl(self.field).data[j], 
									  	  finiteDifferenceDecimalsPrecision)

	def test_integral_constant(self):
		# constant value case
		npt.assert_almost_equal(integral(0*self.field[0] + 1.0), self.metadata['L']*2)

	def test_integral_along_y(self):
		# integral along y
		trueval = 2.0/3.0*cos(self.metadata['alpha']*self.field.z[0])
		npt.assert_almost_equal(integral(self.field[0], 'y'), trueval, integralDecimalsPrecision)

	def test_integral_along_z(self):
		# integral along z of u**2
		trueval = self.field.y[:,0]**4*np.pi/self.metadata['alpha']
		npt.assert_almost_equal(integral(self.field[0]*self.field[0], 'z'), 
								trueval, 
								integralDecimalsPrecision)

	def test_integral_on_domain(self):
		# integral over the domain of u**2
		npt.assert_almost_equal(integral(self.field[0]*self.field[0]), 
								2*np.pi/self.metadata['alpha']/5, 
								integralDecimalsPrecision)

	def test_dot_scalar(self):
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# test product with a scalar value
		npt.assert_array_almost_equal((self.field * 2).data,
									  self.U * 2,
									  decimalsPrecision)

	def test_dot_vector(self):
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# test product with a single vector value
		npt.assert_array_almost_equal((self.field * np.array([3, 4, 1])).data,
									  np.sum(self.U * np.array([3, 4, 1]).reshape(3, 1, 1), axis=0),
									  decimalsPrecision)

	def test_dot_vector_field(self):
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# test product with another vector field
		w = self.field
		v = self.field*5
		npt.assert_array_almost_equal((w*v).data,
									  np.sum(self.U**2*5, axis=0),
									  decimalsPrecision)
		
	def test_dot_matrix(self):
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# test product with a single matrix value
		mat = np.array([[3, 4, 1], [1, 2, 1], [-1, 0, 1]])
		trueval = np.empty_like(self.U, dtype=np.float64)
		for i in range(3):
			trueval[i] = np.sum(mat[i, :].reshape(3, 1, 1)*self.U, axis=0)
		npt.assert_array_almost_equal((mat*self.field).data,
									  trueval,
									  decimalsPrecision)

	def test_dot_itself(self):
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# test product with itself
		npt.assert_array_almost_equal( (self.field*self.field).data,
										np.sum(self.U**2, axis=0),
										decimalsPrecision)

	def test_dot_grad(self):
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# test vector tensor product, for computation of  v * grad(v)
		actualval = self.field*grad(self.field)

		# true value
		y, z = _makeGrid(self.metadata)
		alpha = self.metadata['alpha']
		u, v, w = self.U
		trueval = np.zeros((3, self.metadata['Ny'], self.metadata['Nz']+1), dtype=np.float64) 
		trueval[0] = v*   2*y*cos(alpha*z) - w*y**2*alpha*sin(alpha*z)
		trueval[1] = v*3*y**2*cos(alpha*z) - w*y**3*alpha*sin(alpha*z)
		trueval[2] = v*4*y**3*sin(alpha*z) + w*y**4*alpha*cos(alpha*z)

		npt.assert_array_almost_equal(actualval.data, 
									  	  trueval, 
									  	  finiteDifferenceDecimalsPrecision)



if __name__ == '__main__':
	unittest.main()