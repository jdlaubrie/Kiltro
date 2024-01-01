# python3
import numpy as np
import sys

# properties and conditions
k = 1.0 #W/(m*K)
h = 0.0 #W/(m2*K)
q = 0.0 #W/m3
T_inf = 0.0 #°C
velocity = np.array([100.0, 100.0]) #m/s
speed = np.linalg.norm(velocity)

#=============================================================================#
def Rectangle(lower_left_point=(0,0), upper_right_point=(1,1), nx=5, ny=5):
    # creates the mesh of a rectangle
    nx, ny = int(nx), int(ny)
    
    x=np.linspace(lower_left_point[0],upper_right_point[0],nx+1)
    y=np.linspace(lower_left_point[1],upper_right_point[1],ny+1)

    X,Y = np.meshgrid(x,y)
    coordinates = np.dstack((X.ravel(),Y.ravel()))[0,:,:]
    
    nelem = int(nx*ny)
    elements = np.zeros((nelem,4),dtype=np.int64)

    dum_0 = np.arange((nx+1)*ny)
    dum_1 = np.array([(nx+1)*i+nx for i in range(ny)])
    col0 = np.delete(dum_0,dum_1)
    elements[:,0] = col0
    elements[:,1] = col0 + 1
    elements[:,2] = col0 +  nx + 2
    elements[:,3] = col0 +  nx + 1

    return coordinates, elements

#=============================================================================#
def function_space():
    # define the function space for quadrature and iterpolation
    def Line(eta):
        # simple linear function. maybe Legendre
        N = np.array([0.5*(1-eta), 0.5*(1+eta)])
        dN = np.array([-0.5, 0.5])
        return N, dN

    node_arranger = np.array([0,1,3,2])
    ns = 4  #(C+2)**2, C=0
    #ngauss = 2
    z = np.array([-0.57735027, 0.57735027])
    w = np.array([1.0,1.0])
    Bases = np.zeros((ns,z.shape[0]**2), dtype=np.float64)
    gBases = np.zeros((2,ns,z.shape[0]**2), dtype=np.float64)
    AllGauss = np.zeros((w.shape[0]**2), dtype=np.float64)
    counter = 0
    for i in range(z.shape[0]):
        for j in range(z.shape[0]):
            Nzeta, dNzeta = Line(z[i])
            Neta, dNeta = Line(z[j])
            N = np.outer(Nzeta,Neta).flatten()
            gx = np.outer(Nzeta,dNeta).flatten()
            gy = np.outer(dNzeta,Neta).flatten()
            Bases[:,counter] = N[node_arranger]
            gBases[0,:,counter] = gx[node_arranger]
            gBases[1,:,counter] = gy[node_arranger]
            AllGauss[counter] = w[i]*w[j]
            counter += 1

    return Bases, gBases, AllGauss

#=============================================================================#
def Temperature2D(points, elements, dirichlet_flags, neumann_flags):
    # d/dx(k*dT/dx) + q = 0

    npoints = points.shape[0]
    nelem = elements.shape[0]
    
    # system of equations to solve
    K = np.zeros((npoints,npoints), dtype=np.float64)
    Source = np.zeros((npoints,1), dtype=np.float64)
    Temperature = np.zeros((npoints,1), dtype=np.float64)
    # point P
    for elem in range(nelem):
        # compute element parameters
        ElemCoords = points[elements[elem]]
        ElemLength = np.linalg.norm(ElemCoords[2,:] - ElemCoords[0,:])
        # invoke the function space
        Bases, gBases, AllGauss = function_space()
        # dX/deta (ndim*nepoin*ngauss,nepoin*ndim->ngauss*ndim*ndim)
        etaGradientX = np.einsum('ijk,jl->kil',gBases,ElemCoords)
        # dN/dX (ngauss*ndim*ndim,ndim*nepoin*ngauss->ngauss*ndim*nepoin)
        XGradientN = np.einsum('ijk,kli->ijl',np.linalg.inv(etaGradientX),gBases)
        
        # evaluates the effects of convection. activates upwinding
        Peclet = speed*ElemLength/(2.0*k)
        if ~np.isclose(Peclet,0.0):
            alpha = 1.0/np.tanh(Peclet) - 1.0/np.abs(Peclet)
            Weight = Bases + \
                0.5*alpha*(ElemLength/speed)*np.einsum('ijk,j->ki',XGradientN,velocity)
            gWeight = gBases #+ alpha*ggBases*(u/np.abs(u))
        else:
            alpha = 0.0
            Weight = Bases
            gWeight = gBases
        
        print("Element={}.".format(elem) +\
              " Peclet={0:>10.5g}.".format(Peclet) +\
              " alpha={0:>10.5g}".format(alpha))
        # compute differential for integral
        dV = np.einsum('i,i->i',AllGauss,np.abs(np.linalg.det(etaGradientX)))
        # compute diffusive matrix
        diff_mat = np.einsum('ijk,ijl->ikl',XGradientN,XGradientN)
        ElemDiffusion = k*np.einsum('ijk,i->jk',diff_mat,dV)
        # compute convective matrix
        vel_XgN = np.einsum('j,ijk->ik',velocity,XGradientN)
        conv_mat = np.einsum('ij,jk->ikj',Weight,vel_XgN)
        ElemConvection = np.einsum('ijk,k->ij',conv_mat,dV)
        # stiffness matrix resulting from boundary conditions
        bound_mat = np.einsum('ik,jk->ijk',Weight,Bases)
        ElemBoundary = h*np.einsum('ijk,k->ij',bound_mat,dV)
        # source terms due to heat generation and convection
        ElemSourceQ = q*np.einsum('ij,j->i',Weight,dV)
        ElemSourceH = h*T_inf*np.einsum('ij,j->i',Weight,dV)
        
        for i in range(ElemCoords.shape[0]):
            Source[elements[elem,i]] += ElemSourceQ[i] + ElemSourceH[i]
            for j in range(ElemCoords.shape[0]):
                K[elements[elem,i],elements[elem,j]] += ElemDiffusion[i,j] + \
                        ElemConvection[i,j] + ElemBoundary[i,j]

    # stock information about dirichlet conditions
    columns_out = np.arange(dirichlet_flags.size)[~np.isnan(dirichlet_flags)]
    applied_dirichlet = dirichlet_flags[~np.isnan(dirichlet_flags)]
    columns_in = np.delete(np.arange(0,npoints),columns_out)

    # Apply dirichlet conditions in the problem
    nnz_cols = ~np.isclose(applied_dirichlet,0.0)
    Source[columns_in] = Source[columns_in] - np.dot(K[columns_in,:]\
            [:,columns_out[nnz_cols]],applied_dirichlet[nnz_cols])[:,None]
    Temperature[columns_out,0] = applied_dirichlet
 
    # solve the reduced linear system
    F_b = Source[columns_in]
    K_b = K[columns_in,:][:,columns_in]
    K_inv = np.linalg.inv(K_b)
    sol = np.dot(K_inv,F_b)
    Temperature[columns_in] = sol

    return Temperature

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
    FIGURENAME = 'output.pdf'
    plt.savefig(FIGURENAME)
    plt.close('all')

#=============================================================================#
# mesh
x_elems = 9
y_elems = 2
points, elements = Rectangle(lower_left_point=(0,0),upper_right_point=(1.0,1.0),
        nx=x_elems, ny=y_elems)
npoints = points.shape[0]
nelem = elements.shape[0]
ndim = points.shape[1]

# boundary conditions
# Neumann flags are not actually used by the code by now
left_border = np.where(points[:,0]==0.0)[0]
right_border = np.where(points[:,0]==1.0)[0]
dirichlet_flags = np.zeros((npoints), dtype=np.float64) + np.NAN
dirichlet_flags[left_border] = 1.0
dirichlet_flags[right_border] = 0.0
neumann_flags = np.zeros((npoints), dtype=np.float64) + np.NAN
#neumann_flags[right_border] = 0.0

# solve temperature problem
sol = Temperature2D(points, elements, dirichlet_flags, neumann_flags)
print(sol.reshape(y_elems+1,x_elems+1))

# make an output for the data
Output(points, sol, x_elems, y_elems)

