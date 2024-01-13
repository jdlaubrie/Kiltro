# python3
import numpy as np
import sys
sys.path.append("/media/jdlaubrie/c57cebce-8099-4a2e-9731-0bf5f6e163a5/Kiltro/")
#sys.path.append(os.path.expanduser("~"))
from kiltro import *

# properties and conditions
k = 1.0e3 #W/(m*K)
h = 0.0 #W/(m2*K)
q = 0.0 #W/m3
T_inf = 0.0 #°C
u = np.array([0.0, 0.0]) #m/s
rhoc = 0.0
A = 1.0

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

# Dirichlet boundary conditions
left_border = np.where(mesh.points[:,0]==0.0)[0]
right_border = np.where(mesh.points[:,0]==0.5)[0]
dirichlet_flags = np.zeros((mesh.nnodes), dtype=np.float64) + np.NAN
dirichlet_flags[left_border] = 100.0
dirichlet_flags[right_border] = 500.0
# Neumann boundary conditions
neumann_flags = np.zeros((mesh.nnodes), dtype=np.float64) + np.NAN

boundary_condition = BoundaryCondition()
boundary_condition.dirichlet_flags = dirichlet_flags
boundary_condition.neumann_flags = neumann_flags

# solve the thermal problem
fem_solver = FEMSolver(analysis_type="steady")
TotalSol = fem_solver.Solve(mesh, boundary_condition, u, k, q, h, rhoc, A, T_inf)

# solve temperature problem
print(TotalSol.reshape(y_elems+1,x_elems+1))

# make an output for the data
Output(mesh.points, TotalSol, x_elems, y_elems)

