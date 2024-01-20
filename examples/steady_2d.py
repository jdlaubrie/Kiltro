# python3
import numpy as np
import sys
sys.path.append("/media/jdlaubrie/c57cebce-8099-4a2e-9731-0bf5f6e163a5/Kiltro/")
#sys.path.append(os.path.expanduser("~"))
from kiltro import *

# properties and conditions
u = np.array([0.0, 0.0]) #m/s

#=============================================================================#
def Output(points, sol, nx, ny):

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
    FIGURENAME = 'steady_2d.pdf'
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
material = FourierConduction(ndim, k=1.0e3, area=1.0, rhoc=0.0, q=0.0)

# establish problem formulation
formulation = HeatTransfer(mesh)

# Dirichlet boundary conditions
left_border = np.where(mesh.points[:,0]==0.0)[0]
right_border = np.where(mesh.points[:,0]==0.5)[0]
def DirichletFunction(mesh,left_border,right_border):
    boundary_data = np.zeros((mesh.nnodes), dtype=np.float64) + np.NAN
    boundary_data[left_border] = 100.0
    boundary_data[right_border] = 500.0
    return boundary_data

# Set boundary conditions in problem
boundary_condition = BoundaryCondition()
boundary_condition.SetDirichletCriteria(DirichletFunction,mesh,left_border,right_border)

# solve the thermal problem
fem_solver = FEMSolver(analysis_type="steady")
TotalSol = fem_solver.Solve(formulation, mesh, material, boundary_condition, u)

# solve temperature problem
print(TotalSol.reshape(y_elems+1,x_elems+1))

# make an output for the data
Output(mesh.points, TotalSol, x_elems, y_elems)

