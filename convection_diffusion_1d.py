# python3
import numpy as np
import matplotlib.pyplot as plt
import sys

# properties and conditions
k = 1.0 #W/(m*K)
h = 0.0 #W/(m2*K)
Per = 0.0 #m
A = 1.0 #m2
q = 0.0 #W/m3
T_inf = 0.0 #°C
u = 50.0 #m/s

#=============================================================================#
def function_space():
    # define the function space for quadrature and iterpolation
    def Line(xi):
        # simple linear function. maybe Legendre
        N = np.array([0.5*(1-xi), 0.5*(1+xi)])
        dN = np.array([-0.5, 0.5])
        return N, dN

    ns = 2
    #ngauss = 2
    z = np.array([-0.57735027, 0.57735027])
    w = np.array([1.0,1.0])
    Bases = np.zeros((ns,z.shape[0]), dtype=np.float64)
    gBases = np.zeros((ns,z.shape[0]), dtype=np.float64)
    for i in range(z.shape[0]):
        Bases[:,i], gBases[:,i] = Line(z[i])

    return Bases, gBases, w

#=============================================================================#
def Temperature1D(points, elements, dirichlet_flags, neumann_flags):
    # d/dx(k*dT/dx) + q = 0

    npoints = points.shape[0]
    nelem = elements.shape[0]
    
    # system of equations to solve
    K = np.zeros((npoints,npoints), dtype=np.float64)
    Flux = np.zeros((npoints,1), dtype=np.float64)
    Temperature = np.zeros((npoints,1), dtype=np.float64)
    # point P
    for elem in range(nelem):
        # capture element coordinates
        elem_coords = points[elements[elem]]
        l_e = np.abs(elem_coords[1] - elem_coords[0])
        Peclet = u*l_e/(2.0*k)
        alpha = 1.0/np.tanh(Peclet) - 1.0/np.abs(Peclet)
        # invoke the function space
        Bases, gBases, AllGauss = function_space()
        Bases_up = Bases + alpha*gBases*(u/np.abs(u))
        gBases_up = gBases #+ alpha*dgBases*(u/np.abs(u))
        # compute diffusive matrix
        Gradient1 = np.einsum('ik,jk->ijk',gBases_up,gBases)
        K1_elem = A*k*2.0/l_e*np.einsum('ijk,k->ij',Gradient1,AllGauss)
        # compute diffusive matrix
        Gradient2 = np.einsum('ik,jk->ijk',Bases_up,gBases)
        K2_elem = u*np.einsum('ijk,k->ij',Gradient2,AllGauss)
        # stiffness matrix resulting from boundary conditions
        Gradient3 = np.einsum('ik,jk->ijk',Bases_up,Bases)
        K3_elem = 0.5*l_e*h*Per*np.einsum('ijk,k->ij',Gradient3,AllGauss)
        # source terms due to heat generation and convection
        f1_e = 0.5*l_e*q*A*np.einsum('ij,j->i',Bases,AllGauss)
        f3_e = 0.5*l_e*h*Per*T_inf*np.einsum('ij,j->i',Bases,AllGauss)
        
        for i in range(2):
            for j in range(2):
                K[elements[elem,i],elements[elem,j]] += K1_elem[i,j] + \
                        K2_elem[i,j] + K3_elem[i,j]

        Flux[elements[elem,0]] += f1_e[0] + f3_e[0]
        Flux[elements[elem,1]] += f1_e[1] + f3_e[1]
  
    # stock information about dirichlet conditions
    columns_out = np.arange(dirichlet_flags.size)[~np.isnan(dirichlet_flags)]
    applied_dirichlet = dirichlet_flags[~np.isnan(dirichlet_flags)]
    columns_in = np.delete(np.arange(0,npoints),columns_out)

    # Apply dirichlet conditions in the problem
    nnz_cols = ~np.isclose(applied_dirichlet,0.0)
    Flux[columns_in] = Flux[columns_in] - np.dot(K[columns_in,:]\
            [:,columns_out[nnz_cols]],applied_dirichlet[nnz_cols])[:,None]
    Temperature[columns_out,0] = applied_dirichlet
 
    # solve the reduced linear system
    F_b = Flux[columns_in]
    K_b = K[columns_in,:][:,columns_in]
    K_inv = np.linalg.inv(K_b)
    sol = np.dot(K_inv,F_b)
    Temperature[columns_in] = sol

    return Temperature

#=============================================================================#
def Output(points, sol):

    npoints = points.shape[0]

    fig,ax = plt.subplots()

    ax.plot(points,sol, 'P-')

    fig.tight_layout()
    plt.show
    FIGURENAME = 'output.pdf'
    plt.savefig(FIGURENAME)
    plt.close('all')

#=============================================================================#
# mesh
points = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], dtype=np.float64)
elements = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10]], dtype=np.int64)
npoints = points.shape[0]
nelem = elements.shape[0]
#elem_center, NeighborElem = ElementNeighbourhood(points, elements)

# boundary conditions
dirichlet_flags = np.zeros((npoints), dtype=np.float64) + np.NAN
dirichlet_flags[0] = 1.0
dirichlet_flags[-1] = 0.0
neumann_flags = np.zeros((npoints), dtype=np.float64) + np.NAN
neumann_flags[-1] = 0.0

# solve temperature problem
sol = Temperature1D(points, elements, dirichlet_flags, neumann_flags)
print(sol)
# make an output for the data
Output(points, sol)

