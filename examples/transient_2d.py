# python3
import numpy as np
import sys
sys.path.append("/media/jdlaubrie/c57cebce-8099-4a2e-9731-0bf5f6e163a5/Kiltro/")
#sys.path.append(os.path.expanduser("~"))
from kiltro import *

# properties and conditions
#k = 10.0 #W/(m*K)
h = 0.0 #W/(m2*K)
q = 0.0 #W/m3
T_inf = 0.0 #°C
u = np.array([0.0, 0.0]) #m/s
#rhoc = 10.0e6
#A = 1.0

#=============================================================================#
def Output(points, sol, nx, ny):

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
    FIGURENAME = 'transient_2d.pdf'
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
material = FourierConduction(ndim, k=10.0, area=1.0, rhoc=10.0e6)

# establish problem formulation
formulation = HeatTransfer(mesh)

left_border = np.where(mesh.points[:,0]==0.0)[0]
right_border = np.where(mesh.points[:,0]==0.02)[0]
# Initial boundary conditions
def InitialFunction(mesh):
    boundary_data = np.zeros((mesh.nnodes), dtype=np.float64)
    boundary_data[:] = 200.0
    return boundary_data

# Dirichlet boundary conditions
def DirichletFunction(mesh,right_border):
    boundary_data = np.zeros((mesh.nnodes), dtype=np.float64) + np.NAN
    boundary_data[right_border] = 0.0
    return boundary_data

# Neumann boundary conditions
def NeumannFunction(mesh,left_border):
    boundary_data = np.zeros((mesh.nnodes), dtype=np.float64) + np.NAN
    boundary_data[left_border] = 0.0
    return boundary_data

boundary_condition = BoundaryCondition()
boundary_condition.SetInitialConditions(InitialFunction,mesh)
boundary_condition.SetDirichletCriteria(DirichletFunction,mesh,right_border)
boundary_condition.SetNeumannCriteria(NeumannFunction,mesh,left_border)

# solve the thermal problem
fem_solver = FEMSolver(analysis_type="transient",
                       number_of_increments=61)
TotalSol = fem_solver.Solve(formulation, mesh, material, boundary_condition, u)

# make an output for the data
Output(mesh.points, TotalSol, x_elems, y_elems)

