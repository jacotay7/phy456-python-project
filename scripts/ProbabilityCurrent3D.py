from abc import ABC, abstractmethod
from math import sqrt

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from numpy.polynomial.hermite import hermval
from scipy.constants import e, epsilon_0, hbar, m_e
from scipy.fftpack import dst, idst
from scipy.misc import factorial
from scipy.special import genlaguerre, sph_harm

from CoordinateField3D import CoordinateField3D

plt.rcParams["animation.html"] = "jshtml"

"""
Class QuantumSystem3D
Purpose: A generalized 3D quantum system
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
    rotate_coords: rotates coordinates around 

"""


class QuantumSystem3D(ABC):

    def __init__(self, L, M, N=250, name='', realdim=True):
        self.M = M
        self.L = L
        self.N = N
        self.name = name
        self.realdim = realdim
        return

    def normalize(self):
        norm = np.sum(self.psi.data)*psi.volume_element
        self.psi.data /= sqrt(norm)
        return psi/sqrt(norm)

    def psi_conj(self):
        psi_conj = self.psi.copy()
        psi_conj.data = np.conj(self.psi.data)
        return psi_conj

    def psi_squared(self):
        psi_squared = self.psi.copy()
        psi_squared.data = np.abs(self.psi.data)**2
        return psi_squared

    def gradient(self, psi):
        def grad_func(x, y, z, psi):
            psi_grad = np.empty(
                (x.shape[0], x.shape[1], x.shape[2], 3), np.complex128)
            volume_element = [x[1, 0, 0] - x[0, 0, 0],
                              y[0, 1, 0] - y[0, 0, 0], z[0, 0, 1] - z[0, 0, 0]]

            i = 0
            dv = volume_element[i]
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
        psi_conj = self.psi_conj()
        psi_grad = self.gradient(self.psi)
        psi_conj_grad = self.gradient(psi_conj)
        A = hbar/(2*self.M*1j)
        term1, term2 = np.empty_like(
            psi_grad.data), np.empty_like(psi_grad.data)
        for i in range(3):
            term1[:, :, :, i] = psi_conj.data*psi_grad.data[:, :, :, i]
            term2[:, :, :, i] = self.psi.data*psi_conj_grad.data[:, :, :, i]
        J = A*(term1-term2)
        self.J = J.real
        self.J_mag = np.sqrt(
            self.J[:, :, :, 0]**2 + self.J[:, :, :, 1]**2 + self.J[:, :, :, 2]**2)
            
    @abstractmethod
    def set_wavefunction(self, n, l, m, realdim=False):
        pass


"""
Class: HydrogenAtom
Parent: QuantumSystem3D
Purpose: A Hydrogen atom quantum system
Parameters:
    - eigenstates: an array holding the first n eigenstates of the system
    - En: an array holding the energy of each eigenstate
Methods:
    -set_wavefunction: specify eigenfunction that you want to plot
        - n, l, m: quantum numbers specifying the eigenfunction
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
            return C*R(r)*sph_harm(m, l, phi,  theta)

        psi = CoordinateField3D(self.L, self.L, self.L, self.N, self.N, self.N)
        psi.fillContainer(Psi, (), coordinate_system="SPHERICAL")
        self.psi = psi
        return
    
    def rotate(self, k, theta):
        self.psi.x, self.psi.y, self.psi.z = self.psi.rotate_coords(k, -theta)
        self.psi.r , self.psi.theta, self.psi.phi = cart2sphere()
        psi.fillContainer(Psi, (), coordinate_system="SPHERICAL")
        return


if __name__ == "__main__":
    a_0 = (4.0 * np.pi * epsilon_0 * hbar**2) / (m_e * e**2)
    L = 20 * a_0
    # Initialize Particle
    particle = HydrogenAtom(L)
    particle.rotate([0,1,0], np.pi/2)
    # Initialize Eigenstate
    particle.set_wavefunction(2, 1, 1)
    # Compute J
    particle.find_probability_current()
    # Plot
    # print(particle.J)
    # fig, ax = plt.subplots()
    im = particle.J_mag
    # im2 = ax.imshow(im[:, :, im.shape[2]//2])
    # plt.show()
    # plt.imshow(im[:, im.shape[1]//2, :])
    # plt.show()
    # plt.imshow(im[im.shape[0]//2, :, :])
    # plt.show()

    def animate(i):
        #im = im2.get_array()
        # print(im.shape)
        k = int(i*im.shape[2]/20)
        im2.set_array(im[:, :, k])
        # Update animation text
        return im2,

    def plot_quiver():
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        s = 15  # down-sample a bit
        ax.quiver(particle.psi.x[::s, ::s, ::s],
                  particle.psi.y[::s, ::s, ::s],
                  particle.psi.z[::s, ::s, ::s],
                  particle.J[::s, ::s, ::s, 0],
                  particle.J[::s, ::s, ::s, 1],
                  particle.J[::s, ::s, ::s, 2],
                  length=2*a_0/np.max(particle.J[:]),
                  normalize=False)

        # crop image a bit, might want to change this
        ax.set_xlim3d(-L/4, L/4)
        ax.set_ylim3d(-L/4, L/4)
        ax.set_xlim3d(-L/4, L/4)

        plt.show()

    # anim = animation.FuncAnimation(fig, animate, frames=20, blit=True)
    # plt.show()

    plot_quiver()
