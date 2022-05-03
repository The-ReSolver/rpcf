__author__ = """Davide Lasagna, Aerodynamics and
			    Flight Mechanics group, Soton Uni."""

import numpy as np
from numpy import pi as pi


def zero_pad(U_K, M):
	""" Pad Fourier coefficients in the middle """
	# this will be useful
	N1, N2 = number_of_modes(U_K)
	# initialize. Note that since we use real fourier transform
	# the shape of U_K is not exactly that of U, but one the last dimension
	# it has size /2+1.
	# This is really a bad trick
	final_shape =  [len(M)]*(len(U_K.shape) - len(M)) + list(M)
	final_shape[-1] = final_shape[-1]/2+1
	U_K_padded = np.zeros(final_shape, dtype=U_K.dtype)
	# top left quadrant
	U_K_padded[..., :N1/2.0+1, :N2/2.0+1] = U_K[..., :N1/2.0+1, :N2/2.0+1]
	# bottom left quadrant
	U_K_padded[..., -N1/2.0:, :N2/2.0+1] = U_K[..., -N1/2.0:, :N2/2.0+1]
	return U_K_padded

def zero_crop(U_K, N):
	""" """
	# this will be useful
	M = number_of_modes(U_K)
	N1, N2 = N
	# initialize. Note that since we use real fourier transform
	# the shape of U_K is not exactly that of U, but one the last dimension
	# it has size /2+1.
	# This is really a bad trick
	final_shape =  [len(N)]*(len(U_K.shape) - len(N)) + list(N)
	final_shape[-1] = final_shape[-1]/2+1
	U_K_cropped = np.zeros(final_shape, dtype=U_K.dtype)

	# top left quadrant
	U_K_cropped[..., :N1/2.0, :N2/2.0+1] = U_K[..., :N1/2.0, :N2/2.0+1]
	# bottom left quadrant
	U_K_cropped[..., -N1/2.0:, :N2/2.0+1] = U_K[..., -N1/2.0:, :N2/2.0+1]
	return U_K_cropped

def wave_number_array(N):
	""" Build wave number array.

		Parameters
		----------
		N : d-dimensional tuple
			the tuple N contains the number of Fourier modes
			in the physical space along each spatial direction

		Returns
		-------
		K : np.ndarray of shape (d, N1, N2, Nd/2+1)
			this array contains the wavenumbers, see notes for details

		Notes
		-----
		Note that K has a structure which follows the structure of the output 
		of a rfftn. Thus element [0, 0] is the wavenumber correspoding
		to the mean mode. Element [2, 1] is the wavenumber (2, 1), that
		correspoding to exp(i2x + iy).
	"""
	# unpack tuple #FIXME for 3D
	N1, N2 = N

	# this must have shape (d, N1, N2, ..., Nd/2+1)
	K = np.zeros((len(N), N1, N2/2+1), dtype=np.float64) # FIXME 3D

	# fill first direction 
	vec = np.fft.fftshift(np.arange(-N1/2, N1/2)).reshape(-1,1)
	K[0] = np.tile(vec, (1, N2/2+1))

	# fill second direction 
	vec = np.r_[np.arange(N2/2), [-N2/2]]
	K[1] = np.tile(vec, (N1, 1))

	return K

def wave_number_array3D(N):
	""" Build wave number array.

		Parameters
		----------
		N : d-dimensional tuple
			the tuple N contains the number of Fourier modes
			in the physical space along each spatial direction

		Returns
		-------
		K : np.ndarray of shape (d, N1, N2, Nd/2+1)
			this array contains the wavenumbers, see notes for details

		Notes
		-----
		Note that K has a structure which follows the structure of the output 
		of a rfftn. Thus element [0, 0] is the wavenumber correspoding
		to the mean mode. Element [2, 1] is the wavenumber (2, 1), that
		correspoding to exp(i2x + iy).
	"""
	# unpack tuple #FIXME for 3D
	N1, N2 = N

	# this must have shape (d, N1, N2, ..., Nd/2+1)
	K = np.zeros((3, N1, N2/2+1), dtype=np.float64) 

	# fill first direction 
	vec = np.fft.fftshift(np.arange(-N1/2, N1/2)).reshape(-1,1)
	K[1] = np.tile(vec, (1, N2/2+1))

	# fill second direction 
	vec = np.r_[np.arange(N2/2), [-N2/2]]
	K[2] = np.tile(vec, (N1, 1))

	return K

def number_of_modes(U_K):
	""" Get the number of Fourier modes from
		the Fourier decomposition 'U_K' of the d-dimensional
		vector field 'U', where 'U_K' is the real discrete 
		Fourier transform of 'U'. This is just an helper function.

		Parameters
		----------
		U_K : np.ndarray of shape (d, N1, ..., Nd/2+1)
			the real discrete Fourier transform of 'U'

		Returns
		-------
		out : np.ndarray of length 'd'
			out is (N1, ..., Nd)

		Notes
		-----
		It is assumed that the number of modes in each 
		direction is always an ever number. Otherwise bad 
		things can happen.
	"""
	return np.array([U_K.shape[-2], 2*(U_K.shape[-1]-1)])

def fnon_linear_term_rotational(U_K, M):
	""" Compute non linear term of evolution equation, using a 
		pseudo-spectral approximation, with de-aliasing by padding.

		This corresponds to the term 'Omega cross U'.

		Parameters
		----------
		U_K : np.ndarray, shape (d, N1, N2, ..., Nd/2+1)
			The real discrete Fourier transform of the
			data in physical space 'U', which has shape
			(d, N1, N2, ..., Nd)

		Returns
		-------
		N_K : np.ndarray, shape (d, N1, N2, ..., Nd/2+1)

	"""
	# get shape of data in physical space
	N = number_of_modes(U_K)
	# build wavenumber array
	K = wave_number_array(N)
	# compute velocity gradient tensor in fourier space
	# shape is (d, d, N1, N2, Nd/2+1)
	Omega_K = vorticity(U_K)
	# now get to physical space. Note that we transform along the last two
	# axes which are the axes of the physical coordinates
	Omega = np.fft.irfft2(zero_pad(Omega_K, M))*np.prod(M)/np.prod(N)
	# also compute velocity vector field in physical space, so that we can 
	# perform the multiplication there
	# this has shape (d, N1, ..., Nd)
	U = np.fft.irfftn(zero_pad(U_K, M), axes=range(-len(N), 0))*np.prod(M)/np.prod(N)
	# perform multiplication in physical space and transform to fourier space
	prod = np.zeros_like(U)
	prod[0] = -Omega*U[1]
	prod[1] = Omega*U[0]
	#out = np.fft.rfftn(np.einsum('imn, mn -> imn', U, Omega), s=M, axes=range(-len(N), 0))
	out = np.fft.rfftn(prod, s=M, axes=range(-len(N), 0))
	return zero_crop(out, N)*np.prod(N)/np.prod(M)

def fgradient(U_K):
	""" Compute the velocity gradient tensor. """
	# get shape of data in physical space
	N = number_of_modes(U_K)
	# build wavenumber array
	K = wave_number_array(N)
	return 1j*np.einsum('imn, mn -> imn', K, U_K) 

def fgradient3D(U_K):
	""" Compute the velocity gradient tensor. """
	# get shape of data in physical space
	N = number_of_modes(U_K)
	# build wavenumber array
	K = wave_number_array3D(N)
	return 1j*np.einsum('imn, jmn -> jimn', K, U_K) 

def flaplacian(U_K):
	""" Computes laplacian of vector field. """
	K = wave_number_array(number_of_modes(U_K))
	return - U_K * np.sum(K**2, axis=0)

def fcurl(U_K):
	""" Compute vorticity vector in Fourier space.

		Parameters
		----------
		U_K : np.ndarray of shape (d, N1, ..., Nd/2+1)
			the real discrete fourier transform of the
			velocity vector field in physical space

		Returns
		-------
		Omega_K : np.ndarray of shape (d, N1, ..., Nd/2+1)
			the Fourier transform of the vorticity vector
			field
	"""
	K = wave_number_array3D(number_of_modes(U_K))
	Omega_K = np.zeros_like(U_K)
	Omega_K[0] =  1j*K[1]*U_K[2] - 1j*K[2]*U_K[1]
	Omega_K[1] =  1j*K[2]*U_K[0]
	Omega_K[2] = -1j*K[1]*U_K[0]
	return Omega_K

def fdivergence(U_K):
	""" Compute divergence of the flow field in Fourier space.

		Parameters
		----------
		U_K : np.ndarray of shape (d, N1, ..., Nd/2+1)
			the real discrete fourier transform of the
			velocity vector field in physical space

		Returns
		-------
		div_K : np.ndarray of shape (d, N1, ..., Nd/2+1)
			the divergence of the the flow field in Fourier space
	"""
	K = wave_number_array3D(number_of_modes(U_K))
	return 1j*np.einsum('ijk, ijk -> jk', K, U_K)

def fintegralEnergy(U_K):
	""" Compute energy using parseval identity. """
	N1, N2 = number_of_modes(U_K)
	U_K_tosum = U_K.copy(); U_K_tosum[:, :, 1:-1] *= 2**0.5
	return 2*np.pi*np.pi*np.sum(np.abs(U_K_tosum[0])**2)/(N1*N2)**2 + \
		   2*np.pi*np.pi*np.sum(np.abs(U_K_tosum[1])**2)/(N1*N2)**2 + \
		   2*np.pi*np.pi*np.sum(np.abs(U_K_tosum[2])**2)/(N1*N2)**2

def fintegralEnstrophy(U_K):
	""" Compute integral of enstrophy using parseval identity. """
	N1, N2 = number_of_modes(U_K)
	Omega_K = fcurl(U_K);
	Omega_K[:, :, 1:-1] *= 2**0.5
	return 2*np.pi*np.pi*np.sum(np.abs(Omega_K)**2)/(N1*N2)**2

def fdissipation(U_K):
	""" Compute dissipation rate in Fourier space. 
		You have to divide by the Reynolds number.
		Dissipation is always a negative quantity.
	"""
	N1, N2 = number_of_modes(U_K)
	U_K_tosum = U_K.copy(); U_K_tosum[:, :, 1:-1] *= 2**0.5
	return -(4*np.pi**2*np.sum(np.abs(U_K_tosum)**2, axis=0)*
		np.sum(wave_number_array3D((N1, N2))**2, axis=0)/(N1*N2)**2)

def fproduction(U_K):
	""" Compute production in Fourier space. """
	N1, N2 = number_of_modes(U_K)
	U_K_tosum = U_K.copy(); U_K_tosum[:, :, 1:-1] *= 2**0.5
	return 4*np.pi**2*(U_K_tosum[0]*U_K_tosum[1]).real/(N1*N2)**2

def fcorrection(U_K):
	""" Compute correction to 'U_K' such that the
		divergence of 'U' is zero. This correction 
		must then be added to 'U_K';.
		
		Parameters
		----------
		U_K : np.ndarray of shape (d, N1, ..., Nd/2+1)
			the real discrete Fourier transform of 'U'

		Returns 
		-------
		C_K : np.ndarray of shape (d, N1, ..., Nd/2+1)
			an array with the same shape as 'U_K'
	"""
	K = wave_number_array3D(number_of_modes(U_K))
	den = np.sum(K**2, axis=0); den[0, 0] = 1
	return -K*np.einsum('ijk, ijk -> jk', K, U_K)/den