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
        '''
        Student must complete
        
        Requirements: Write a function to normalize the 1D array, psi_t,
        representing the wavefunction at time t. This function should return
        the normalized array. Hint: Integrate the absolute square
        '''
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
            """
            Student Must Complete
            
            Requirements: Write a function which takes a 3D gradient using a generalized 
            central differences method. Note that x, y, and z are meshgrid arrays 
            specifying the coordinate inputs.  The function should return a 3D meshgrid
            populated with vector field array instead of numbers (i.e. a 4D meshgrid).
            Be careful with boundary points!
            """
            return psi_grad
        psi_grad = CoordinateField3D(
            self.L, self.L, self.L, self.N, self.N, self.N)
        psi_grad.fillContainer(grad_func, (psi.data,))
        return psi_grad

    def find_probability_current(self):
        """
        Student must complete
        
        Requirements: Write a function to compute the probability current
        of the psi. Hint: re-use code whenever possible, if the above functions
        have been written corectly, this should not take long to complete. Save
        the array as a property of the system (self.J).  Save the magnitude as a 
        3D meshgrid array populated with scalar (self.J_mag).
        Hint: J should be real. Complex number may be due to rounding error and can be neglected
        """

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


if __name__ == "__main__":
    a_0 = (4.0 * np.pi * epsilon_0 * hbar**2) / (m_e * e**2)
    L = 20 * a_0
    # Initialize Particle
    particle = HydrogenAtom(L)
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
        """
        Student must complete
        
        Requirements: Write a function to plot particl.J against its coordinates.
        You can reuse your plotting code from the analytical plots.
        Hint:  you will need to use python's slice notation to undersample,
        or else the program run time will be too long
        """

    # anim = animation.FuncAnimation(fig, animate, frames=20, blit=True)
    # plt.show()

    plot_quiver()
