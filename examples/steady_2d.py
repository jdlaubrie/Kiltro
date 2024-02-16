# python3
import numpy as np
import os, sys
from warnings import warn
sys.path.append("/media/jdlaubrie/c57cebce-8099-4a2e-9731-0bf5f6e163a5/Kiltro/")
#sys.path.append(os.path.expanduser("~"))
from kiltro import *

# properties and conditions
u = np.array([0.0, 0.0]) #m/s

#=============================================================================#
def Output(mesh, sol, formula, nx, ny):

    import matplotlib.pyplot as plt
    X = np.reshape(mesh.points[:,0], (ny+1,nx+1))
    Y = np.reshape(mesh.points[:,1], (ny+1,nx+1))
    Temp = np.reshape(sol, (ny+1,nx+1))

    fig,ax = plt.subplots()

    ax.plot(X[1,:], Temp[1,:])

    fig.tight_layout()
    plt.show
    FIGURENAME = 'steady_2d_'+formula+'.pdf'
    plt.savefig(FIGURENAME)
    plt.close('all')

    return
#=============================================================================#
# mesh
x_elems = 5
y_elems = 3
mesh = Mesh(element_type="quad")
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
solution = fem_solver.Solve(formulation=formulation, mesh=mesh, material=material,
                            boundary_condition=boundary_condition)

# export results to vtk file
solution.WriteVTK('stady2d_diff')

# solve temperature problem
print(solution.sol.reshape(y_elems+1,x_elems+1))

# make an output for the data
Output(mesh, solution.sol, 'diff', x_elems, y_elems)

#-----------------------------------------------------------------------------#
print('\n===================  ADVECTION-DIFFUSION PROBLEM  ===================')
u = np.zeros((mesh.nnodes,2), dtype=np.float64)
u[:,0] += 9.0e3
# establish problem formulation
formulation = HeatAdvectionDiffusion(mesh, velocity=u)

# solve the thermal problem
fem_solver = FEMSolver(analysis_type="steady")
solution = fem_solver.Solve(formulation=formulation, mesh=mesh, material=material,
                            boundary_condition=boundary_condition)

# export results to vtk file
solution.WriteVTK('stady2d_addi')

# solve temperature problem
print(solution.sol.reshape(y_elems+1,x_elems+1))

# make an output for the data
Output(mesh, solution.sol, 'addi', x_elems, y_elems)

