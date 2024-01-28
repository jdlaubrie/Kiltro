# python3
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/jdlaubrie/Coding/Kiltro/")
#sys.path.append(os.path.expanduser("~"))
from kiltro import *

#=============================================================================#
def Output(points, sol, formula):

    npoints = points.shape[0]

    fig,ax = plt.subplots()

    if len(sol.shape)==1:
        ax.plot(points,sol, 'P-')
    else:
        for i in range(sol.shape[1]):
            ax.plot(points,sol[:,i])

    fig.tight_layout()
    plt.show
    FIGURENAME = 'transient_1d_'+formula+'.pdf'
    plt.savefig(FIGURENAME)
    plt.close('all')

#=============================================================================#
# mesh
mesh = Mesh()
mesh.Line(0.0,0.02,5)
ndim = mesh.ndim

# establish material for the problem
material = FourierConduction(ndim, k=10.0, area=1.0, density=10.0e6, heat_capacity_v=1.0)

# Initial boundary conditions
def InitialFunction(mesh):
    boundary_data = np.zeros((mesh.nnodes,1), dtype=np.float64)
    boundary_data[:,0] = 200.0
    return boundary_data

# Dirichlet boundary conditions
def DirichletFunction(mesh):
    boundary_data = np.zeros((mesh.nnodes,1), dtype=np.float64) + np.NAN
    boundary_data[-1,0] = 0.0
    return boundary_data

# Neumann boundary conditions
def NeumannFunction(mesh):
    boundary_data = np.zeros((mesh.nnodes,1), dtype=np.float64) + np.NAN
    boundary_data[0,0] = 0.0
    return boundary_data

boundary_condition = BoundaryCondition()
boundary_condition.SetInitialConditions(InitialFunction,mesh)
boundary_condition.SetDirichletCriteria(DirichletFunction,mesh)
boundary_condition.SetNeumannCriteria(NeumannFunction,mesh)

#-----------------------------------------------------------------------------#
print('\n======================  PURE-DIFFUSION PROBLEM  ======================')
# establish problem formulation
formulation = HeatDiffusion(mesh)

# solve the thermal problem
fem_solver = FEMSolver(analysis_type="transient",
                       number_of_increments=61)
TotalSol = fem_solver.Solve(formulation=formulation, mesh=mesh, material=material,
                            boundary_condition=boundary_condition)

# make an output for the data
Output(mesh.points, TotalSol, 'diff')

#-----------------------------------------------------------------------------#
print('\n===================  ADVECTION-DIFFUSION PROBLEM  ===================')
u = np.zeros((mesh.nnodes,1), dtype=np.float64) + 0.0
# establish problem formulation
formulation = HeatAdvectionDiffusion(mesh, velocity=u)

# solve the thermal problem
fem_solver = FEMSolver(analysis_type="transient",
                       number_of_increments=61)
TotalSol = fem_solver.Solve(formulation=formulation, mesh=mesh, material=material,
                            boundary_condition=boundary_condition)

# make an output for the data
Output(mesh.points, TotalSol, 'addi')

