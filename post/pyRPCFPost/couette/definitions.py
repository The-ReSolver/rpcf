import numpy as np


def laminarProfile(r, R_i, R_o, Omega_i, Omega_o):
	""" Return the laminar solution of the azimuthal velocity. """
	# radii ratio
	eta = R_i/R_o

	# coefficients of the laminar profile
	A = (Omega_o - Omega_i*eta**2)/(1 - eta**2)
	B = (R_i**2)/(1 - eta**2)*(Omega_i - Omega_o)

	return A*r + B/r

