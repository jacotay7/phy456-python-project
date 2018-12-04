#Import neccessary modules
import numpy as np
import matplotlib.pyplot as plt

#Animation module
import matplotlib.animation as animation

#Embeds animations in jupyter notebooks
plt.rcParams["animation.html"] = "jshtml"

#Physical Constants
from scipy.constants import hbar

#Helpful functions
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

        """
        Student must complete
        
        Requirements: Fill a complex, 1D array of size n with 
        the initial coefficients of the eigen decomposition of Psi(x,0).
        Save the resulting array as a attribute of the system (self.cn)
        """   
            
    def normalize(self, psi_t):

        """
        Student must complete

        Requirements: Write a function to normalize the 1D array, psi_t,
        representing the wavefunction at time t. This function should return
        the normalized array. Hint: Integrate the absolute square
        """
    
    def psi_conj(self, t):
        return np.conj(self.psi(t))
    
    def psi(self, t):
        
        """
        Student must complete

        Requirements: Write a function to which comput Psi(x,t) at a given
        time t. This is where you apply the spectral method. 
        """  

   def psi_squared(self,t):

        """
        Student must complete

        Requirements: Write a function which returns the absolute square
        of psi(t). Hint: Re-use code whenever possible
        """

    def derivative(self, psi_t):

        """
        Student must complete

        Requirements: Write a function which returns the derivative
        of the array psi_t. The return value should be an array of
        the same shape as psi_t. You should apply the central differences
        method on the interior points. At the boundary apply the 
        forward/backward difference methods.
        """
    
    def probability_current(self, t):

        """
        Student must complete

        Requirements: Write a function to compute the probability current
        at time t. Hint: re-use code whenever possible, if the above functions
        have been written corectly, this should not take long to complete.
        """
    
    def create_animation(self, start, end, frames = 100):
        
        #Set-Up the figure for the animation
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
        
        #Iterable function which produces each frame of the animation
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
        
        """
        Student can complete for extra credit

        Requirements: Should be analogous to the Particle in
        a box implementation. 
        """
        
"""
A function to initialize a particle as a wave packet
"""        
def psi0_func(x, x0, sigma, kappa):
    return np.exp(-1*(x-x0)**2/(2*sigma**2))*np.exp(1j*kappa*x)


#Main body of the program
if __name__ == "__main__":

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

    #Uncomment to save as a video file (might not work on non-linux)
    #ani.save('PIB.mp4', dpi=80, writer='imagemagick')
    plt.show()

    #Number of eigensates used (any higher and the hermite polynomials have problems)
    n = 172
    omega = 2e15

    """
    Student can add quantum harmonic oscillator main program here for extra credit
    """
