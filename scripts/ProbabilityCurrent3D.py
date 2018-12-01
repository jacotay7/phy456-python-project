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

    def __init__(self, L, M, N=250, name=''):
        self.coords = CoordinateField3D(L, np.pi, 2*np.pi, N, N, N)

        self.name = name
        return

    def normalize(self, psi):
        norm = np.trapz(np.abs(psi)**2, x=self.x)
        return psi/sqrt(norm)

    def psi_conj(self, t):
        return np.conj(self.psi(t))

    def psi_squared(self, t):
        return np.abs(self.psi(t))**2

    def psi(self, t):
        psi_t = np.zeros_like(self.x, dtype=np.complex128)
        for i in range(self.cn.size):
            psi_t += self.cn[i]*np.exp(-1j*self.En[i]
                                       * t/hbar)*self.eigenstates[i, :]
        return self.normalize(psi_t)

    def derivative(self, psi_t):
        dx = self.x[1] - self.x[0]
        psi_x = np.empty_like(psi_t)
        psi_x[0] = (psi_t[1] - psi_t[0])/(dx)
        psi_x[-1] = (psi_t[-1] - psi_t[-2])/(dx)
        psi_x[1:-1] = (psi_t[2:] - psi_t[:-2])/(2*dx)
        return psi_x

    def probability_current(self, t):
        psi_t = self.psi(t)
        psi_t_conj = np.conj(psi_t)
        A = hbar/(2*self.M*1j)
        term1 = psi_t_conj*self.derivative(psi_t)
        term2 = psi_t*self.derivative(psi_t_conj)
        J = A*(term1-term2)
        return J.real

    @abstractmethod
    def get_wavefunction(self, n, l, m, realdim=False):
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

    def __init__(self, L, M, N=250):
        QuantumSystem3D.__init__(self, L, M, N=N, name="HydrogenAtom")
        return

    def generate_eigenstates(self, n):

        self.eigenstates = np.zeros((n, self.x.size), dtype=np.complex128)
        self.En = (np.arange(1, n+1)*np.pi*hbar/self.L)**2/(2*self.M)
        for i in range(n):
            pre_factor = sqrt(2/self.L)
            self.eigenstates[i, :] = pre_factor*np.sin((i+1)*np.pi/L*self.x)
        self.generate_initial_cn(n)

    def get_wavefunction(self, n, l, m, realdim=False):
        # real dimensions for the reduced Bohr radius
        if realdim:
            a_0 = (4.0 * np.pi * epsilon_0 * hbar**2) / (m_e * e**2)
        else:
            a_0 = 1

        # normalization
        C = np.sqrt((2)/(n * a_0)**3 * factorial(n-l-1) /
                    (2 * n * factorial(n+l)))

        # radial component
        def R(r):
            rho = (2 * r) / (n * a_0)
            return np.exp(-rho/2) * (rho ** l) * genlaguerre(n-l-1, 2*l+1)(rho)

        # putting it together
        def psi(r, theta, phi):
            return sph_harm(m, l, theta, phi)

        return psi


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
    h = HydrogenAtom(10, 10)
    g = h.get_wavefunction(3, 2, 1)

    # plot wavefunction in xz-plane
    x = np.linspace(-5, 5)
    z = np.linspace(-5, 5)

    X, Z = np.meshgrid(x, z)

    R = np.sqrt(X**2 + Z**2)
    theta = np.arctan2(Z, X)

    phi = np.zeros(theta.shape)
    P = np.abs(g(R, theta, phi))**2

    plt.imshow(np.abs(sph_harm(1, 2, theta, phi))**2, extent=[0, 1, 0, 1])
    plt.show()
