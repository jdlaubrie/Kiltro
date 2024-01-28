# python3
import numpy as np
import sys
sys.path.append("/home/jdlaubrie/Coding/Kiltro/")
#sys.path.append(os.path.expanduser("~"))
from kiltro import *

#=============================================================================#
def Output(points, sol, formula, nx, ny):

    import matplotlib.pyplot as plt
    X = np.reshape(points[:,0], (ny+1,nx+1))
    Y = np.reshape(points[:,1], (ny+1,nx+1))

    fig,ax = plt.subplots(2,1)

    for i in range(sol.shape[1]):
        Temp = np.reshape(sol[:,i], (ny+1,nx+1))
        ax[0].plot(X[1,:], Temp[1,:])

    cs = ax[1].contourf(X, Y, Temp, cmap='RdBu_r')
    cbar = fig.colorbar(cs)

    fig.tight_layout()
    plt.show
    FIGURENAME = 'transient_2d_'+formula+'.pdf'
    plt.savefig(FIGURENAME)
    plt.close('all')

#=============================================================================#
# mesh
x_elems = 5
y_elems = 3
mesh = Mesh()
mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(0.02,1.0),
        nx=x_elems, ny=y_elems)
ndim = mesh.ndim

# establish material for the problem
material = FourierConduction(ndim, k=10.0, area=1.0, density=10.0e6, heat_capacity_v=1.0)

left_border = np.where(mesh.points[:,0]==0.0)[0]
right_border = np.where(mesh.points[:,0]==0.02)[0]
# Initial boundary conditions
def InitialFunction(mesh):
    boundary_data = np.zeros((mesh.nnodes,1), dtype=np.float64)
    boundary_data[:,0] = 200.0
    return boundary_data

# Dirichlet boundary conditions
def DirichletFunction(mesh,right_border):
    boundary_data = np.zeros((mesh.nnodes,1), dtype=np.float64) + np.NAN
    boundary_data[right_border,0] = 0.0
    return boundary_data

# Neumann boundary conditions
def NeumannFunction(mesh,left_border):
    boundary_data = np.zeros((mesh.nnodes,1), dtype=np.float64) + np.NAN
    boundary_data[left_border,0] = 0.0
    return boundary_data

boundary_condition = BoundaryCondition()
boundary_condition.SetInitialConditions(InitialFunction,mesh)
boundary_condition.SetDirichletCriteria(DirichletFunction,mesh,right_border)
boundary_condition.SetNeumannCriteria(NeumannFunction,mesh,left_border)

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
Output(mesh.points, TotalSol, 'diff', x_elems, y_elems)

#-----------------------------------------------------------------------------#
print('\n===================  ADVECTION-DIFFUSION PROBLEM  ===================')
u = np.zeros((mesh.nnodes,2), dtype=np.float64) + 0.0
# establish problem formulation
formulation = HeatAdvectionDiffusion(mesh, velocity=u)

# solve the thermal problem
fem_solver = FEMSolver(analysis_type="transient",
                       number_of_increments=61)
TotalSol = fem_solver.Solve(formulation=formulation, mesh=mesh, material=material,
                            boundary_condition=boundary_condition)

# make an output for the data
Output(mesh.points, TotalSol, 'addi', x_elems, y_elems)

