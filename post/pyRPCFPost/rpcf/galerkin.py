__author__ = """Davide Lasagna, Aerodynamics and
                Flight Mechanics group, Soton Uni."""

import pickle
import copy
import numpy as np
from numpy.linalg import norm
from scipy.misc import comb
from scipy.integrate import odeint

from pyRPCFPost.rpcf.field import integral, laplacian, grad, Field
from pyRPCFPost.rpcf.lru import lru_cache

# use memoisation
grad = lru_cache(maxsize=50)(grad)
integral = lru_cache(maxsize=50)(integral)
laplacian = lru_cache(maxsize=50)(laplacian)

def orthonormalize(modes):
    """ Make modes an orthonormal set of basis functions."""
    this_copy = copy.deepcopy(modes)
    new_modes = []
    for i, mode in enumerate(this_copy):
        for m in new_modes:
            mode -= m*integral(m*mode)
        mode /= np.sqrt(integral(mode*mode))
        new_modes.append(mode)
    return new_modes

def _f(y, gamma, k):
    """ y function that defines the control functions """
    k = k if k else 1
    return -1./8.*np.exp(-gamma*k*(1+y))*(1+y)*(1-y)**3

def _fp(y, gamma, k):
    """ y-derivative  of f """
    k = k if k else 1
    return (y - 1)**2*(0.125*-gamma*k*(y - 1)*(y + 1) + 0.5*y + 0.25)*np.exp(-gamma*k*(y + 1))

def exponential_cf(Ny, Nz, L, k, gamma, ftype):
    """ Build control function.

        Parameters
        ----------
        Ny, Nz : ints, size of the mesh
        L : float, axial length of the domain
        k : integer, wavenumber associated to the control function
        gamma : float > 0, decay rate of the control function
        ftype : str, in ['s', 'c'], type of the control function, sice or cosine


        Returns 
        -------
        cfunc : Field instance, 
    """
    
    # parameters
    alpha = 2*np.pi/L

    # make mesh
    yy = np.linspace(-1, 1, Ny)
    zz = np.linspace(0, L, Nz+1)
    z, y = np.meshgrid(zz, yy)

    # allocate
    U = np.zeros((3, Ny, Nz+1), dtype=np.float64)
    
    # depending on type of cfunction
    if ftype == 'c':
        U[1] = -_f(y, gamma, k)*np.sin(k*alpha*z)*k*alpha
    elif ftype == 's':
        U[1] = _f(y, gamma, k)*np.cos(k*alpha*z)*k*alpha

    if ftype == 'c':
        U[2] = -_fp(y, gamma, k)*np.cos(k*alpha*z)
    elif ftype == 's':
        U[2] = -_fp(y, gamma, k)*np.sin(k*alpha*z)

    # create Field instance
    f = Field(U, {'Ny':Ny,
                  'Nz':Nz, 
                  'stretch_factor':1e-10, 
                  'L':L, 
                  'alpha':alpha})
    return f

def znmf_cf(Ny, Nz, L, j=1):
    """ Build a control function that satisfies
        the znmf condition, and whose shape depends
        on the parameter j > 0.

        It is obtained as combination of Bernstein polynomials.

    """

    # this is the bernstein polynomial
    def Bernstein(i, n):
        return lambda x : comb(n, i)*x**i*(1.0 - x)**(n-i)

    # helper function to get the y profile
    def phi(j):
        c = - (j+2)/(j+1.0)
        return lambda x: Bernstein(0, j)(x) + c*Bernstein(1, j+1)(x)

    # make mesh
    alpha = 2*np.pi/L
    yy = np.linspace(0, 1, Ny) # we start from zero because of 
                               # B. polynomials are defined on 
                               # that interval
    zz = np.linspace(0, L, Nz+1)
    z, y = np.meshgrid(zz, yy)

    # allocate
    U = np.zeros((3, Ny, Nz+1), dtype=np.float64)
    
    # compute y profile
    U[2] = phi(j)(y)

    # create Field instance
    f = Field(U, {'Ny':Ny,
                  'Nz':Nz, 
                  'stretch_factor':1e-10, 
                  'L':L, 
                  'alpha':alpha})
    return f

def _cf(Ny, Nz, L, ftype):
    """ Build a control function that satifies 
        a zero net mass flux condition. 
    """
    
    alpha = 2*np.pi/L

    # make mesh
    yy = np.linspace(-1, 1, Ny)
    zz = np.linspace(0, L, Nz+1)
    z, y = np.meshgrid(zz, yy)

    # allocate
    U = np.zeros((3, Ny, Nz+1), dtype=np.float64)
    
    if ftype == "znmf":
        U[2] = 0.75*y**2 - 0.5*y - 0.25
    elif ftype == "zpg":
        U[2] = 0.5 - 0.5*y
    else:
        raise ValueError("ftype not understood")

    # create Field instance
    f = Field(U, {'Ny':Ny,
                  'Nz':Nz, 
                  'stretch_factor':1e-10, 
                  'L':L, 
                  'alpha':alpha})
    return f

def load_from_file(filename):
    return pickle.load(open(filename, 'r'))

def filter(A, threshold=1e-4):
    """ Set to zero small coefficients in array A """
    A[abs(A) < threshold] = 0

def dissipation(Lambda, a, Re):
    """ Compute energy dissipation rate """
    return np.einsum("ti, ij, tj -> t", 
                     np.atleast_2d(a), 
                     Lambda, 
                     np.atleast_2d(a))/Re


class GalerkinSystem:
    def __init__(self, modes, params, threshold=1e-4, doNL=True):
        """ Build a Galerkin system from projection of a set of basis functions
            onto the governing equations. 
        """
        self.params = params
        self.n = len(modes)
        # make linear and nonlinear parts of the model
        self.Lambda, self. W = self._buildLinearPart(modes, params)
        if doNL: self.Q = self._buildNonLinearPart(modes)

        # linear part is symmetric. (AGO 2015) NOPE! system is non normla for \Omega !+ 0.5
        #self.Lambda = (self.Lambda + self.Lambda.T)/2.0
        #self.W = (self.W + self.W.T)/2.0

        # nonlinear part has some symmetries
        # thes comes for the defintion of the third order tensor
        # N_{ijk} = <u_i, u_j . \nabla u_k>
        # and because of the identity <u_i, u_j . \nabla u_k> = - <u_k, u_j . \nabla u_i>
        if doNL:
            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.n):
                        a = self.Q[i, j, k]
                        b = self.Q[k, j, i]
                        self.Q[i, j, k] = abs((a-b)/2)*np.sign(a)
                        self.Q[k, j, i] = abs((a-b)/2)*np.sign(b)
        
        # combine linear part
        self.L = self.Lambda/params['Re'] + self.W
        
        filter(self.L, threshold)
        filter(self.Lambda, threshold)
        filter(self.W, threshold)
        if doNL: filter(self.Q, threshold)

    def tofile(self, filename):
        """ Save model to file """
        pickle.dump(self, open(filename, 'w'))

    def copy(self):
        """ Make a copy of the model. """
        c = copy.copy(self)
        c.L = np.copy(self.L)
        c.Q = np.copy(self.Q)
        return c

    def __call__(self, a, t):
        """ Compute derivative of modal amplitudes. """
        return np.dot(self.L, a) + np.einsum("ijk, j, k", self.Q, a, a ) 

    def _buildLinearPart(self, modes, params):
        """ Build linear term of Galerkin system. """
        Lambda = np.zeros((self.n, self.n), dtype=np.float64)
        W = np.zeros((self.n, self.n), dtype=np.float64)
        A = np.array([[0, params['Ro']-1.0, 0], [-params['Ro'], 0, 0], [0, 0, 0]])
        for i, ui in enumerate(modes): 
            for j, uj in enumerate(modes): 
                Lambda[i, j]  = integral(ui*laplacian(uj))
                W[i, j] = integral(ui*(A*uj))
        return Lambda, W

    def _buildNonLinearPart(self, modes):
        """ Build non linear part """
        Q = np.zeros((self.n, self.n, self.n), dtype=np.float64)
        for i, ui in enumerate(modes): 
            for j, uj in enumerate(modes): 
                for k, uk in enumerate(modes): 
                    if (i == j and j == k):
                        continue
                    if (i == k):
                        continue
                    Q[i, j, k] = -integral(ui*(uj*grad(uk)))
        return Q

class BoundaryControl:
    def __init__(self, modes, cfuncs, params):
        """ Builds the matrices required for implementation of 
            control via the boundary.

            Parameters
            ----------
            modes  : list of basis functions
            cfuncs : list of control functions
            params : a dict containing values for the keys
                     "Re" and "Ro", the Reynolds and 
                      rotation numbers.
        """
        self.params = params
        self.n = len(modes)
        self.m = len(cfuncs)

        # build system matrices
        self.M = self._buildM(modes, cfuncs, params)
        self.R = self._buildR(modes, cfuncs, params)
        self.G = self._buildG(modes, cfuncs, params)
        self.E = self._buildE(modes, cfuncs, params)
        self.F = self._buildF(modes, cfuncs, params)

    def __call__(self, a, t, b, bdot):
        return (np.einsum("ij, j",     self.M, bdot) +
                np.einsum("ij, j",     self.R, b   ) +
                np.einsum("ijk, j, k", self.G, b, b) +
                np.einsum("ijk, j, k", self.F, b, a) +
                np.einsum("ijk, j, k", self.E, a, b) )
    @property
    def C(self):
        return np.squeeze(self.E) + np.squeeze(self.F)

    def _buildM(self, modes, cfuncs, params):
        M = np.zeros((self.n, self.m), dtype=np.float64)
        for i, ui in enumerate(modes):
            for j, psij in enumerate(cfuncs):
                M[i, j] = -integral(ui*psij)
        filter(M)
        return M

    def _buildR(self, modes, cfuncs, params):
        R = np.zeros((self.n, self.m), dtype=np.float64)
        A = np.array([[0, params['Ro']-1, 0], [-params['Ro'], 0, 0], [0, 0, 0]])
        for i, ui in enumerate(modes):
            for j, psij in enumerate(cfuncs):
                R[i, j] = integral(ui*(A*psij + laplacian(psij)/params['Re']))
        filter(R)
        return R

    def _buildG(self, modes, cfuncs, params):
        G = np.zeros((self.n, self.m, self.m), dtype=np.float64)
        for i, ui in enumerate(modes):
            for j, psij in enumerate(cfuncs):
                for k, psik in enumerate(cfuncs):
                    G[i, j, k] = -integral(ui*(psij*grad(psik)))
        filter(G)
        return G
    
    def _buildF(self, modes, cfuncs, params):
        F = np.zeros((self.n, self.m, self.n), dtype=np.float64)
        for i, ui in enumerate(modes):
            for j, psij in enumerate(cfuncs):
                for k, uk in enumerate(modes):
                    F[i, j, k] = -integral(ui*(psij*grad(uk)))
        filter(F)
        return F

    def _buildE(self, modes, cfuncs, params):
        E = np.zeros((self.n, self.n, self.m), dtype=np.float64)
        for i, ui in enumerate(modes):
            for j, uj in enumerate(modes):
                for k, psik in enumerate(cfuncs):
                    E[i, j, k] = -integral(ui*(uj*grad(psik)))
        filter(E)
        return E

class ControlledGalerkinSystem(GalerkinSystem):
    def __init__(self, modes, cfuncs, controller, params, threshold=1e-6, doNL=True):
        """ Simulation of a controlled Galerkin system. 

            Parameters
            ----------
            modes  : a list of basis functions
            cfuncs : a list of control functions
            controller : the controller, a callable
            params : parameters dictionary 

            Notes
            -----
            The controller is a function that takes in input
            the state vector and the time, and returns two 
            vectors, b and bdot. Simple.

        """
        self.params = params
        self.f = GalerkinSystem(modes, params, threshold, doNL)
        self.g = BoundaryControl(modes, cfuncs, params)
        self.controller = controller
        self.n = len(modes)
        self.m = len(cfuncs)

    def copy(self):
        """ Returns a new instance """
        return copy.deepcopy(self)

    def __call__(self, a, t):
        """ Time integration routine """
        b, bdot = self.controller(a, t)
        return self.f(a, t) + self.g(a, t, b, bdot)
        

def pretty_print(rom, prec=5):
    """ Display a view of the model """

    strs = []
    for i in range(rom.n):
        s = ""
        # linear part 
        for j in range(rom.n):
            if rom.f.L[i, j] != 0:
                sign = "+" if rom.f.L[i, j] > 0 else "-"
                s += " %s %.16f*a_%d " % (sign, abs(rom.f.L[i, j]), j)
        # nonlinear part
        for j in range(rom.n):
            for k in range(rom.n):
                if rom.f.Q[i, j, k] != 0:
                    sign = "+" if rom.f.Q[i, j, k] > 0 else "-"
                    if j == k:
                        s += " %s %.16f*a_%d^2" % (sign, abs(rom.f.Q[i, j, k]), j)
                    else:
                        s += " %s %.16f*a_%d*a_%d" % (sign, abs(rom.f.Q[i, j, k]), j, k)
        strs.append(s)

    strs_control = []
    for i in range(rom.n):
        s = ""
        # control
        for j in range(rom.f.n):
            if rom.g.C[i, j] != 0:
                sign = "+" if rom.g.C[i, j] > 0 else "-"
                s += " %s %.16f a_%d b" % (sign, abs(rom.g.C[i, j]), j)
        strs_control.append(s)
        
    return strs, strs_control

def pretty_print_latex(rom, prec=5):
    """ """
    strs = []
    for i in range(rom.f.n):
        # linear part 
        s = "%.16f a_%d " % (rom.f.L[i, i], i)
        
        # nonlinear part
        for j in range(rom.f.n):
            for k in range(rom.f.n):
                if rom.f.Q[i, j, k] != 0:
                    sign = "+" if rom.f.Q[i, j, k] > 0 else "-"
                    if j == k:
                        s += " %s %.16f a_%d^2" % (sign, abs(rom.f.Q[i, j, k]), j)
                    else:
                        s += " %s %.16f a_%d a_%d" % (sign, abs(rom.f.Q[i, j, k]), j, k)
                  
        s += " @ "       
        # control
        for j in range(rom.f.n):
            if rom.g.C[i, j] != 0:
                sign = "+" if rom.g.C[i, j] > 0 else "-"
                s += "%s %.16f a_%d b " % (sign, abs(rom.g.C[i, j]), j)

        strs.append(s)
    return strs


# ~~~ Controllers ~~~
def harmonic_motion(A, eta, ton=-1, toff=-1):
    def wrapped(a, t):
        masked = np.squeeze(A*np.atleast_1d((t > ton) & (t < toff)))
        return [masked*np.sin(eta*t)], [masked*eta*np.cos(eta*t)]
    return wrapped

def constant_motion(w0):
    def wrapped(a, t):
        return [w0 + 0.0*t], [0.0]
    return wrapped

#  ~~~ Functions to run systems till they are stationary ~~~
def run_till_stationary(rom, Ts,
                        rtol=1e-7, 
                        maxiter=1000, 
                        a0=None):
    """ Run a rom till it reaches a steady value. """
    # integrate here
    t = np.linspace(0, Ts, 100)

    # term due to control 
    if hasattr(rom, "controller"):
        phi_b = 16*np.trapz((rom.controller(None, t)[0][0])**2, t)/t[-1]/rom.params['Re']
    else:
        phi_b = 0.0

    # initial value
    phi_old = 2e4

    # get lambda matrix
    if hasattr(rom, "Lambda"):
        Lambda = rom.Lambda
    else:
        Lambda = rom.f.Lambda

    # run till min tolerance or, maxiter
    for i in range(maxiter):
        # get initial condition from last simulation if available
        # otherwise get it from argument or pick it at random.
        if i > 0:
            a0 = a[-1]
        else :
            a0 = a0 if a0 is not None else np.random.normal(0, 0.1, rom.n)
        # integrate model
        a = odeint(rom, a0, t)
        
        # calculate dissipation
        phi_a = -np.trapz(dissipation(Lambda, a, rom.params['Re']), t)/t[-1]
        phi = phi_a + phi_b

        # did we reach a steady value
        if abs(phi - phi_old)/phi < rtol:
            break
        phi_old = phi
        
    return phi_a, phi_b