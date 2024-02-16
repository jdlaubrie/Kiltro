# python3
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn
import os, sys
sys.path.append("/media/jdlaubrie/c57cebce-8099-4a2e-9731-0bf5f6e163a5/Kiltro/")
#sys.path.append(os.path.expanduser("~"))
from kiltro import *

#=============================================================================#
def Output(mesh, sol, formula):

    npoints = mesh.points.shape[0]

    fig,ax = plt.subplots()

    if len(sol.shape)==1:
        ax.plot(mesh.points,sol, 'P-')
    else:
        for i in range(sol.shape[1]):
            ax.plot(mesh.points,sol[:,i])

    fig.tight_layout()
    plt.show
    FIGURENAME = 'steady_1d_'+formula+'.pdf'
    plt.savefig(FIGURENAME)
    plt.close('all')

    return

#=============================================================================#
# mesh
mesh = Mesh(element_type="line")
mesh.Line(0.0,0.5,5)
ndim = mesh.ndim

# establish material for the problem
material = FourierConduction(ndim, k=1.0e3, area=10.0e3, density=1.0,
                             heat_capacity_v=1.0, q=0.0)

# Dirichlet boundary conditions
def DirichletFunction(mesh):
    # (v,T): given velocity and unknown temperature
    boundary_data = np.zeros((mesh.nnodes,1), dtype=np.float64)
    boundary_data[:,0] += np.NAN
    boundary_data[0,0] = 100.0
    boundary_data[-1,0] = 500.0
    return boundary_data

# Set boundary conditions in the problem
boundary_condition = BoundaryCondition()
boundary_condition.SetDirichletCriteria(DirichletFunction,mesh)

u = np.zeros((mesh.nnodes,1), dtype=np.float64) + 4.0e7
#-----------------------------------------------------------------------------#
print('\n==================  STANDARD-GALERKIN FORMULATION  ==================')
# establish problem formulation
formulation = Temperature(mesh, velocity=u)

# solve the thermal problem
fem_solver = FEMSolver(analysis_type="steady")
solution = fem_solver.Solve(formulation=formulation, mesh=mesh, material=material,
                            boundary_condition=boundary_condition)

# export results to vtk file
solution.WriteVTK('steady1d_diff')

# make an output for the data
Output(mesh, solution.sol, 'diff')

#-----------------------------------------------------------------------------#
print('\n===================  PETROV-GELERKING FORMULATION  ==================')
# establish problem formulation
formulation = TemperaturePG(mesh, velocity=u)

# solve the thermal problem
fem_solver = FEMSolver(analysis_type="steady")
solution = fem_solver.Solve(formulation=formulation, mesh=mesh, material=material,
                            boundary_condition=boundary_condition)

# export results to vtk file
solution.WriteVTK('steady1d_addi')

# make an output for the data
Output(mesh, solution.sol, 'addi')

