# python3
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/media/jdlaubrie/c57cebce-8099-4a2e-9731-0bf5f6e163a5/Kiltro/")
#sys.path.append(os.path.expanduser("~"))
from src import *

# properties and conditions
k = 10.0 #W/(m*K)
h = 0.0 #W/(m2*K)
Per = 0.0 #m
A = 1.0 #m2
q = 0.0 #W/m3
T_inf = 0.0 #°C
u = 0.0 #m/s
rhoc = 10.0e6

#=============================================================================#
def Output(points, sol):

    npoints = points.shape[0]

    fig,ax = plt.subplots()

    if len(sol.shape)==1:
        ax.plot(points,sol, 'P-')
    else:
        for i in range(sol.shape[1]):
            ax.plot(points,sol[:,i])

    fig.tight_layout()
    plt.show
    FIGURENAME = 'transient_1d.pdf'
    plt.savefig(FIGURENAME)
    plt.close('all')

#=============================================================================#
# mesh
mesh = Mesh()
mesh.Line(0.0,0.02,5)

# boundary conditions
dirichlet_flags = np.zeros((mesh.nnodes), dtype=np.float64) + np.NAN
#dirichlet_flags[0] = 100.0
dirichlet_flags[-1] = 0.0
neumann_flags = np.zeros((mesh.nnodes), dtype=np.float64) + np.NAN
neumann_flags[0] = 0.0
boundary_condition = BoundaryCondition()
boundary_condition.dirichlet_flags = dirichlet_flags
boundary_condition.neumann_flags = neumann_flags

# solve the thermal problem
fem_solver = FEMSolver(analysis_type="transient",
                       number_of_increments=61)
TotalSol = fem_solver.Solve(mesh, boundary_condition, u, k, q, h*Per, rhoc, A, T_inf)

# make an output for the data
Output(mesh.points, TotalSol)

