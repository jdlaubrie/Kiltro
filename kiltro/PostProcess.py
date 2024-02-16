import numpy as np
import os, sys
from warnings import warn
from copy import deepcopy

class PostProcess(object):
    """Post-process class for finite element solvers"""

    def __init__(self,ndim,nvar):

        self.ndim = ndim
        self.nvar = nvar
        self.mesh = None
        self.sol = None

#=============================================================================#
    def SetMesh(self,mesh):
        """Set initial (undeformed) mesh"""
        self.mesh = mesh

#=============================================================================#
    def SetSolution(self,sol):
        self.sol = sol

#=============================================================================#
    def WriteVTK(self,filename=None, quantity=0):
        """Writes results to a VTK file for Paraview"""

        try:
            from pyevtk.hl import pointsToVTK, linesToVTK, gridToVTK, unstructuredGridToVTK
            from pyevtk.vtk import VtkVertex, VtkLine, VtkTriangle, VtkQuad, VtkTetra, VtkPyramid, VtkHexahedron
        except ImportError:
            raise ImportError("Could not import evtk. Install it using 'pip install pyevtk'")

        iterator = range(quantity,quantity+1)

        if filename is None:
            warn("file name not specified. I am going to write in the current directory")
            filename = os.path.dirname(os.path.realpath(sys.argv[0])) + "/output.vtu"

        # GET LINEAR MESH & SOLUTION
        lmesh = deepcopy(self.mesh)
        sol = deepcopy(self.sol)

        if lmesh.element_type =='line':
            cellflag = 3
            offset = 2
            actual_ndim = 2
        elif lmesh.element_type =='quad':
            cellflag = 9
            offset = 4
            actual_ndim = 2

        q_names = ["T"]

        LoadIncrement = sol.shape[2]
        increments = range(LoadIncrement)

        for Increment in increments:
            if lmesh.InferSpatialDimension() == 1:
                points = np.zeros((lmesh.points.shape[0],3))
                points[:,:1] = lmesh.points
            elif lmesh.InferSpatialDimension() == 2:
                points = np.zeros((lmesh.points.shape[0],3))
                points[:,:2] = lmesh.points
            else:
                points = lmesh.points

            for counter, quant in enumerate(iterator):
                unstructuredGridToVTK(filename.split('.')[0]+'_quantity_'+str(quant)+'_increment_'+str(Increment),
                        np.ascontiguousarray(points[:,0]), np.ascontiguousarray(points[:,1]), np.ascontiguousarray(points[:,2]),
                        np.ascontiguousarray(lmesh.elements.ravel()), np.arange(0,offset*lmesh.nelem,offset)+offset,
                        np.ones(lmesh.nelem)*cellflag,
                        pointData={q_names[counter]: np.ascontiguousarray(sol[:,quant,Increment])})

        return

