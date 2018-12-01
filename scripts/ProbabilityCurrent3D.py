from abc import ABC, abstractmethod
from math import sqrt

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.constants import e, epsilon_0, hbar, m_e
from scipy.fftpack import dst, idst
from scipy.misc import factorial
from scipy.special import genlaguerre, sph_harm

from CoordinateField3D import CoordinateField3D

plt.rcParams["animation.html"] = "jshtml"

"""
Class QuantumSystem3D
Purpose: A generalized 1D quantum system
Parameters:
    - L: Linear size of the system
    - M: Mass of the particle
    - N: Number of sample points per dimension
    - x: The sample point along the x axis
    - name: Unique name of the system (ex: PIB, SHO)
    - psi0: The initial state of the system
    - cn: The inital eigen decomposition coefficients

Methods:
    set_psi0: Initializes the state
        - psi0_func: A user defined function which computes psi0
        - args: the arugments of psi0_func
    generate_initial_cn: Initializes the eigen decomposition coefficients
        - n: the number of eigenstates you want to consider
    normalize: normalizes a given state
        - psi: the state to be normalized
    psi: returns the state at time t
        - t: The time to compute the state at
    psi_conj: return the complex conjugate of psi(t)
        - t: The time to find the state at
    psi_squared: returns |psi(t)|^2
        - t: The time to coupute psi(t) at
    derivative: Takes the numerical derivative of a given psi(t)
        - psi_t: the state at time t
    probability current: Compute the probability current at a time t
        - t: The time to compute the probability current at
    create_animation: Creates an animation of the system
        - start: the starting time for the animation
        - end: the ending time of the animation
        - frames: the number of frames in the animation

"""


class QuantumSystem3D(ABC):

    def __init__(self, L, M, N=250, name='', realdim=True):
        self.M = M
        self.L = L
        self.N = N
        self.name = name
        self.realdim = realdim
        return

    def normalize(self, psi):
        norm = np.sum(psi)*psi.volume_element
        return psi/sqrt(norm)

    def psi_conj(self):
        psi_conj = self.psi.copy()
        psi_conj.data = np.conj(self.psi.data)
        return psi_conj

    def psi_squared(self):
        return np.abs(self.psi)**2

    def gradient(self, psi):
        def grad_func(x, y, z, psi):
            psi_grad = np.empty(
                (x.shape[0], x.shape[1], x.shape[2], 3), np.complex128)
            volume_element = [x[1, 0, 0] - x[0, 0, 0],
                              y[1, 0, 0] - y[0, 0, 0], z[1, 0, 0] - z[0, 0, 0]]

            i = 0
            dv = volume_element[i]
            print(dv)
            psi_grad[0, :, :, i] = (psi[1, :, :] - psi[0, :, :])/(dv)
            psi_grad[-1, :, :, i] = (psi[-1, :, :] - psi[-2, :, :])/(dv)
            psi_grad[1:-1, :, :, i] = (psi[2:, :, :] - psi[:-2, :, :])/(2*dv)

            i = 1
            dv = volume_element[i]
            psi_grad[:, 0, :, i] = (psi[:, 1, :] - psi[:, 0, :])/(dv)
            psi_grad[:, -1, :, i] = (psi[:, -1, :] - psi[:, -2, :])/(dv)
            psi_grad[:, 1:-1, :, i] = (psi[:, 2:, :] - psi[:, :-2, :])/(2*dv)

            i = 2
            dv = volume_element[i]
            psi_grad[:, :, 0, i] = (psi[:, :, 1] - psi[:, :, 0])/(dv)
            psi_grad[:, :, -1, i] = (psi[:, :, -1] - psi[:, :, -2])/(dv)
            psi_grad[:, :, 1:-1, i] = (psi[:, :, 2:] - psi[:, :, :-2])/(2*dv)

            return psi_grad
        psi_grad = CoordinateField3D(
            self.L, self.L, self.L, self.N, self.N, self.N)
        psi_grad.fillContainer(grad_func, (psi.data,))
        return psi_grad

    def find_probability_current(self):
        psi_conj = self.psi_conj
        psi_grad = self.gradient(self.psi)
        A = hbar/(2*self.M*1j)
        term1 = psi_conj.data*psi_grad.data
        term2 = self.psi.data*self.gradient(psi_conj).data
        J = A*(term1-term2)
        self.J = J.real

    @abstractmethod
    def set_wavefunction(self, n, l, m, realdim=False):
        pass


"""
Class: HydrogenAtom
Parent: QuantumSystem3D
Purpose: A particle in a box quantum system
Parameters:
    - eigenstates: an array holding the first n eigenstates of the system
    - En: an array holding the energy of each eigenstate
Methods:
    generate_eigenstates: Find the first n eigenstates of the system
        - n: The number of eigenstates to generate
"""


class HydrogenAtom(QuantumSystem3D):

    def __init__(self, L, N=250, realdim=True):
        QuantumSystem3D.__init__(
            self, L, m_e, N=N, name="HydrogenAtom", realdim=True)
        return

    def set_wavefunction(self, n, l, m):
        # real dimensions for the reduced Bohr radius
        if self.realdim:
            a_0 = (4.0 * np.pi * epsilon_0 * hbar**2) / (m_e * e**2)
        else:
            a_0 = 1

        # normalization
        C = np.sqrt((2)/(n * a_0)**3 * factorial(n-l-1) /
                    (2 * n * factorial(n+l)))

        # putting it together
        def Psi(r, theta, phi):
            # radial component
            def R(r):
                rho = (2 * r) / (n * a_0)
                return np.exp(-rho/2) * (rho ** l) * genlaguerre(n-l-1, 2*l+1)(rho)
            print(theta, phi)
            return R(r)*sph_harm(m, l, theta, phi)

        psi = CoordinateField3D(self.L, self.L, self.L, self.N, self.N, self.N)
        psi.fillContainer(Psi, (), coordinate_system="SPHERICAL")
        self.psi = psi
        return


"""
A function to compute the square root of large integers
"""


def isqrt(x):
    if x < 0:
        raise ValueError('square root not defined for negative numbers')
    n = int(x)
    if n == 0:
        return 0
    a, b = divmod(n.bit_length(), 2)
    x = 2**(a+b)
    while True:
        y = (x + n//x)//2
        if y >= x:
            return x
        x = y


if __name__ == "__main__":

    L = 20 * (4.0 * np.pi * epsilon_0 * hbar**2) / (m_e * e**2)
    # Initialize Particle
    particle = HydrogenAtom(L)
    # Initialize Eigenstate
    particle.set_wavefunction(1, 0, 0)
    print(particle.psi.data)

    # Compute J
    # particle.find_probability_current()
    # Plot
    print(sph_harm(1, 1, particle.psi.theta, particle.psi.phi))
    """   # plot wavefunction in xz-plane
    x = np.linspace(-5, 5)
    z = np.linspace(-5, 5)

    X, Z = np.meshgrid(x, z)

    R = np.sqrt(X**2 + Z**2)
    theta = np.arctan2(Z, X)

    phi = np.zeros(theta.shape)
    P = np.abs(g(R, theta, phi))**2"""

    plt.imshow(np.abs(particle.psi.data[:, 0, :])**2)
    plt.show()
