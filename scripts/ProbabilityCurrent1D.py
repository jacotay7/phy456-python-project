#Import neccessary modules
import numpy as np
import matplotlib.pyplot as plt
#Animation module
import matplotlib.animation as animation
#Embeds animations in jupyter notebooks
plt.rcParams["animation.html"] = "jshtml"
from scipy.fftpack import dst, idst
from scipy.constants import hbar
from numpy.polynomial.hermite import hermval
from math import sqrt

"""
Class QuantumSystem1D

Purpose: A generalized 1D quantum system

Parameters:
    - L: Length of the 1D system
    - M: Mass of the particle
    - N: Number of sample points
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
class QuantumSystem1D:

    def __init__(self, L, M, N=1000, name=''):

        self.x = np.linspace(0,L,N)
        self.N = N
        self.M = M
        self.L = L
        self.name = name
        return

    def set_psi0(self, psi0_func, args):

        self.psi0 = psi0_func(self.x, *args)
        self.psi0 = self.normalize(self.psi0)

    def generate_initial_cn(self,n):
        self.cn = np.zeros(n, dtype = np.complex128)
        for i in range(n):
            self.cn[i] = np.trapz(np.conj(self.eigenstates[i,:])*self.psi0, x=self.x)

    def normalize(self, psi_t):
        norm = np.trapz(np.abs(psi_t)**2, x = self.x)
        return psi_t/sqrt(norm)

    def psi_conj(self, t):
        return np.conj(self.psi(t))

    def psi_squared(self,t):
        return np.abs(self.psi(t))**2

    def psi(self, t):

        psi_t = np.zeros_like(self.x, dtype=np.complex128)
        for i in range(self.cn.size):
            psi_t += self.cn[i]*np.exp(-1j*self.En[i]*t/hbar)*self.eigenstates[i,:]
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

    def create_animation(self, start, end, frames = 100):

        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=[10,8])
        im, = ax[0].plot(self.x*1e9, self.psi_squared(start), color = 'black')
        im_real, = ax[1].plot(self.x*1e9, self.psi(start).real, color = 'blue', label = "real")
        im_imag, = ax[1].plot(self.x*1e9, self.psi(start).imag, color = 'orange', label = "imag")
        im_j, = ax[2].plot(self.x*1e9, self.probability_current(start), color = 'black')
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        ax[2].set_xlabel("X [nm]")
        ax[1].legend()
        if(self.name == 'SHO'):
            ax[0].set_title("Wave Packet in SHO Potential Well")
        elif(self.name == 'PIB'):
            ax[0].set_title("Wave Packet in Inifinite Potential Well")
        else:
            ax[0].set_title("Wave Packet in Unknown Potential Well")
        ax[0].set_ylabel("Probability Density")
        ax[0].set_ylabel("$|\psi(x)|^2$")
        ax[1].set_ylabel("$\psi(x)$")
        ax[2].set_ylabel("J [$m^{-1}s^{-1}$]")
        plt.xlim(0,self.L*1e9)
        ax[2].set_ylim(-1*np.nanmax(im_j.get_ydata()),np.nanmax(im_j.get_ydata()))
        plt.tight_layout()
        #Updateable Text box for time
        ttl = ax[0].text(.78, 0.9, '', transform = ax[0].transAxes, va='center')

        def animate(i):

            t = start + i*(end-start)/(frames-1)
            psi_t = self.psi(t)
            im.set_data(self.x*1e9, np.abs(psi_t)**2)
            im_real.set_data(self.x*1e9, psi_t.real)
            im_imag.set_data(self.x*1e9, psi_t.imag)
            im_j.set_data(self.x*1e9, self.probability_current(t))
            #Update animation text
            ttl.set_text("t = {:.3f} fs".format(t*1e15))
            return im, im_real, im_imag, im_j,  ttl,

        return animation.FuncAnimation(fig, animate,\
                     frames=frames, blit=True)

"""
Class: PIB
Parent: QuantumSystem1D

Purpose: A particle in a box quantum system

Parameters:
    - eigenstates: an array holding the first n eigenstates of the system
    - En: an array holding the energy of each eigenstate
Methods:
    generate_eigenstates: Find the first n eigenstates of the system
        - n: The number of eigenstates to generate
"""
class PIB(QuantumSystem1D):

    def __init__(self, L, M, N=1000):
        QuantumSystem1D.__init__(self, L, M, N=N, name="PIB")
        return

    def generate_eigenstates(self, n):

        self.eigenstates = np.zeros((n,self.x.size), dtype=np.complex128)
        self.En = (np.arange(1,n+1)*np.pi*hbar/self.L)**2/(2*self.M)
        for i in range(n):
            pre_factor = sqrt(2/self.L)
            self.eigenstates[i,:] = pre_factor*np.sin((i+1)*np.pi/L*self.x)
        self.generate_initial_cn(n)

"""
Class: SHO
Parent: QuantumSystem1D

Purpose: A particle in a simple harmonic osccilator potential

Parameters:
    - eigenstates: an array holding the first n eigenstates of the system
    - En: an array holding the energy of each eigenstate
Methods:
    generate_eigenstates: Find the first n eigenstates of the system
        - n: The number of eigenstates to generate
"""
class SHO(QuantumSystem1D):

    def __init__(self, L, M, omega, x0, N=1000):

        QuantumSystem1D.__init__(self, L, M, N=N, name='SHO')
        self.w = omega
        self.x0 = x0
        return

    def generate_eigenstates(self, n):

        self.eigenstates = np.zeros((n,self.x.size), dtype=np.complex128)
        self.En = hbar*self.w*(np.arange(n) + 0.5)
        tmp = self.M*self.w/hbar
        gauss = (tmp/np.pi)**0.25*np.exp(-1/2*tmp*(self.x-self.x0)**2)
        for i in range(n):
            pre_factor = sqrt(1/(2**i * np.math.factorial(i)))
            self.eigenstates[i,:] = pre_factor*gauss*hermval(sqrt(tmp)*(self.x-self.x0), [int(i == j) for j in range(i+1)])
        self.generate_initial_cn(n)

"""
A function to initialize a particle as a wave packet
"""
def psi0_func(x, x0, sigma, kappa):
    return np.exp(-1*(x-x0)**2/(2*sigma**2))*np.exp(1j*kappa*x)

#Constants
L = 1e-8 #m
x0 = L/2
sigma = 2e-10 #m
kappa = 5e10 #m^-1
M = 9.109e-31
n = 500

#Create Particle
particle = PIB(L, M, N=1000)
#Set inital conditions
particle.set_psi0(psi0_func, (x0, sigma, kappa))
particle.generate_eigenstates(n)
#Plot a few times
particle.create_animation(0, 2e-15, frames = 100)
ani = particle.create_animation(0, 2e-15, frames = 100)
#Uncomment to save as a video file
#ani.save('PIB.gif', dpi=80, writer='imagemagick')
plt.show()

#Number of eigensates used (any higher and the hermite polynomials have problems)
n = 200
omega = 2e15
#Create Particle
particle = SHO(L, M, omega, x0, N=1000)
#Set inital conditions
particle.set_psi0(psi0_func, (x0, sigma, kappa))
particle.generate_eigenstates(n)
#Plot a few times
ani = particle.create_animation(0, 2e-15, frames = 100)
#Uncomment to save as a video file
#ani.save('SHO.gif', dpi=80, writer='imagemagick')
plt.show()
