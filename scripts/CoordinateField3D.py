import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
from mpl_toolkits.mplot3d import axes3d


"""
Class VectorField3D

Purpose: Stores 3D Co-ordinate Grid and an associated Vector Field.

Parameters:
    - L1, L2, L3: lengths of co-ordinate arrays
    - N1, N2, N3: Number of sample points along coodinate directions
    - initCoord: Unique name of co-ordinates defined by Li and Ni (ex: "RECTANGULAR", "CYLINDRICAL", "SPHERICAL")

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

    def __init__(self, L1, L2, L3, N1, N2, N3, gridType="RECTANGULAR"):

        self.L = L1, L2, L3

        self.N = N1, N2, N3
        self.gridType = gridType
        if self.gridType == "SPHERICAL":
            # radial array: omit 0 to avoid blowup at origin
            r = np.linspace(0, L1, N1)
            theta = np.linspace(0., L2, N2)  # polar array
            phi = np.linspace(0, L3, N3)  # azimuthal array
            self.r, self.theta, self.phi = np.meshgrid(
                r, theta, phi, indexing='ij')
            self.x, self.y, self.z = self.sphere2cart()
        elif self.gridType == "RECTANGULAR":
            # bump up starting index by an intervalto avoid dividing by zero
            x = np.linspace(-1*L1/2, L1/2, N1)
            y = np.linspace(-1*L2/2, L2/2, N2)
            z = np.linspace(-1*L3/2, L3/2, N3)
            self.volume_element = (x[1]-x[0])**3
            self.x, self.y, self.z = np.meshgrid(x, y, z, indexing='ij')
            self.r, self.theta, self.phi = self.cart2sphere()

    def sphere2cart(self):
        x, y, z = self.r*np.sin(self.theta)*np.cos(self.phi), self.r * \
            np.sin(self.theta)*np.sin(self.phi), self.r*np.cos(self.theta)
        return x, y, z

    def cart2sphere(self):
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        theta = np.arccos(self.z/r)
        phi = np.arctan2(self.y, self.x)
        return r, theta, phi

    def fillContainer(self, func, func_args, coordinate_system="RECTANGULAR"):
        if(coordinate_system == "RECTANGULAR"):
            self.data = func(self.x, self.y, self.z, *func_args)
        elif(coordinate_system == "SPHERICAL"):
            self.data = func(self.r, self.theta, self.phi, *func_args)
        else:
            print("Unrecognised Co-ordinate System")
    

    def copy(self):
        copy = CoordinateField3D(
            self.L[0], self.L[1], self.L[2], self.N[0], self.N[1], self.N[2], gridType=self.gridType)
        copy.data = self.data
        return copy


#This code only runs if you run this script directly
if __name__ == "__main__":
    test = CoordinateField3D(1,1,1,2,2,2, gridType = "RECTANGULAR")