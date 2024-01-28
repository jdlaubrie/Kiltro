# python3
import numpy as np
import sys
sys.path.append("/home/jdlaubrie/Coding/Kiltro/")
#sys.path.append(os.path.expanduser("~"))
from kiltro import *

# properties and conditions
u = np.array([0.0, 0.0]) #m/s

#=============================================================================#
def Output(points, sol, formula, nx, ny):

    import matplotlib.pyplot as plt
    X = np.reshape(points[:,0], (ny+1,nx+1))
    Y = np.reshape(points[:,1], (ny+1,nx+1))
    Temp = np.reshape(sol, (ny+1,nx+1))

    fig,ax = plt.subplots(2,1)

    ax[0].plot(X[1,:], Temp[1,:])
    
    cs = ax[1].contourf(X, Y, Temp, cmap='RdBu_r')
    cbar = fig.colorbar(cs)

    fig.tight_layout()
    plt.show
    FIGURENAME = 'steady_2d_'+formula+'.pdf'
    plt.savefig(FIGURENAME)
    plt.close('all')

#=============================================================================#
# mesh
x_elems = 5
y_elems = 3
mesh = Mesh()
mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(0.5,0.01),
        nx=x_elems, ny=y_elems)
ndim = mesh.ndim

# establish material for the problem
material = FourierConduction(ndim, k=1.0e3, area=1.0, density=1.0, c_v=1.0, q=0.0)

# Dirichlet boundary conditions
left_border = np.where(mesh.points[:,0]==0.0)[0]
right_border = np.where(mesh.points[:,0]==0.5)[0]
def DirichletFunction(mesh,left_border,right_border):
    boundary_data = np.zeros((mesh.nnodes,1), dtype=np.float64) + np.NAN
    boundary_data[left_border,0] = 100.0
    boundary_data[right_border,0] = 500.0
    return boundary_data

# Set boundary conditions in problem
boundary_condition = BoundaryCondition()
boundary_condition.SetDirichletCriteria(DirichletFunction,mesh,left_border,right_border)

#-----------------------------------------------------------------------------#
print('\n======================  PURE-DIFFUSION PROBLEM  ======================')
# establish problem formulation
formulation = HeatDiffusion(mesh)

# solve the thermal problem
fem_solver = FEMSolver(analysis_type="steady")
TotalSol = fem_solver.Solve(formulation=formulation, mesh=mesh, material=material,
                            boundary_condition=boundary_condition)

# solve temperature problem
print(TotalSol.reshape(y_elems+1,x_elems+1))

# make an output for the data
Output(mesh.points, TotalSol, 'diff', x_elems, y_elems)

#-----------------------------------------------------------------------------#
print('\n===================  ADVECTION-DIFFUSION PROBLEM  ===================')
u = np.zeros((mesh.nnodes,2), dtype=np.float64)
u[:,0] += 9.0e3
# establish problem formulation
formulation = HeatAdvectionDiffusion(mesh, velocity=u)

# solve the thermal problem
fem_solver = FEMSolver(analysis_type="steady")
TotalSol = fem_solver.Solve(formulation=formulation, mesh=mesh, material=material,
                            boundary_condition=boundary_condition)

# solve temperature problem
print(TotalSol.reshape(y_elems+1,x_elems+1))

# make an output for the data
Output(mesh.points, TotalSol, 'addi', x_elems, y_elems)

