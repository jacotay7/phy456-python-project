import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
"""
Class VectorField3D

Purpose: Stores 3D Co-ordinate Grid and an associated Vector Field.

Parameters: 
    - L1, L2, L3: lengths of co-ordinate arrays
    - N1, N2, N3: Number of sample points along coodinate directions
    - initCoord: Unique name of co-ordinates defined by Li and Ni (ex: "CARTESIAN", "CYLINDRICAL", "SPHERICAL")
    
Properties:
    -L = L1, L2, L3
    -N = N1, N2, N3
    -coordCart: 3tuple of cartesian meshgrid coordinates
    -coordSphere: 3tuple of spherical meshgrid coordinates
    -vectorField: 3tuple of vector values in meshgrid-compatible form
    
Methods:

    cart2sphere: converts cartesian meshgrid co-ordinates into spherical meshgrid coordinates
    sphere2cart: converts spherical meshgrid co-ordinates into cartesian meshgrid coordinates
    plotField3D: makes a 3D plot of the vector field
"""       

class CoordinateField3D:

    def __init__(self, L1, L2, L3, N1, N2, N3, coordInit = "SPHERICAL"):
        
        self.L = L1, L2, L3
        self.N = N1, N2, N3
        self.coordInit = coordInit
        if self.coordInit == "SPHERICAL":
            r = np.linspace(0, L1, N1)          #radial array: omit 0 to avoid blowup at origin
            theta = np.linspace(0., L2, N2)     #polar array
            phi = np.linspace(0, L3, N3)        #azimuthal array
            self.coordSphere = np.meshgrid(r, theta, phi, indexing = 'ij')
            self.coordCart = self.sphere2cart()
        elif self.coordInit == "CARTESIAN":
            x = np.linspace(0 + L1/N1, L1, N1)  # bump up starting index by an intervalto avoid dividing by zero             
            y = np.linspace(0., L2/N2, N2)
            z = np.linspace(0, L3/N3, N3)
            self.coordCart = np.meshgrid(x, y, z, indexing = 'ij')
            self.coordSphere = self.cart2sphere()
        
    def sphere2cart(self):
        
        r, theta, phi = self.coordSphere
        x, y, z = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
        return x, y, z
        
    def cart2sphere(self):
        x, y, z = self.coordCart
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)
        return r, theta, phi