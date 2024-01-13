import numpy as np

#=============================================================================#
def Assembly(mesh, material, formulation, function_space, u, q, h, T_inf):

    #h = h*Perimeter for 1D
    npoints = mesh.nnodes
    nelem = mesh.nelem
    nodeperelem = mesh.elements.shape[1]

    M = np.zeros((npoints,npoints), dtype=np.float64)
    K = np.zeros((npoints,npoints), dtype=np.float64)
    Source = np.zeros((npoints,1), dtype=np.float64)
    for elem in range(nelem):

        ElemMass, ElemDiffusion, ElemConvection, ElemBoundary, \
        ElemSourceQ, ElemSourceH = formulation.GetElementalMatrices(elem, mesh, 
        material, function_space, u, q, h, T_inf)

        for i in range(nodeperelem):
            Source[mesh.elements[elem,i]] += ElemSourceQ[i] + ElemSourceH[i]
            for j in range(nodeperelem):
                K[mesh.elements[elem,i],mesh.elements[elem,j]] += ElemDiffusion[i,j] + \
                        ElemConvection[i,j] + ElemBoundary[i,j]
                M[mesh.elements[elem,i],mesh.elements[elem,j]] += ElemMass[i,j]

    return K, Source, M

#=============================================================================#
def AssemblyCharacteristicGalerkin(mesh, material, formulation, function_space, u, q, h, T_inf):

    npoints = mesh.nnodes
    nelem = mesh.nelem

    k = material.k
    A = material.area
    rhoc = material.rhoc

    K = np.zeros((npoints,npoints), dtype=np.float64)
    Source = np.zeros((npoints,1), dtype=np.float64)
    for elem in range(nelem):
        # capture element coordinates
        ElemCoords = mesh.points[mesh.elements[elem]]
        ElemLength = np.linalg.norm(ElemCoords[1,:] - ElemCoords[0,:])
        # invoke the function space
        Bases = function_space.Bases
        gBases = function_space.gBases
        AllGauss = function_space.AllGauss
        #nepoin
        nodeperelem = function_space.Bases.shape[0]
        # dX/deta (ndim*nepoin*ngauss,nepoin*ndim->ngauss*ndim*ndim)
        etaGradientX = np.einsum('ijk,jl->kil',gBases,ElemCoords)
        inv_etaGradientX = np.linalg.inv(etaGradientX)
        # dN/dX (ngauss*ndim*ndim,ndim*nepoin*ngauss->ngauss*ndim*nepoin)
        XGradientN = np.einsum('ijk,kli->ijl',inv_etaGradientX,gBases)

        # evaluates the effects of convection. activates upwinding
        Peclet = np.linalg.norm(u)*ElemLength/(2.0*k)
        if ~np.isclose(Peclet,0.0):
            alpha = 1.0/np.tanh(Peclet) - 1.0/np.abs(Peclet)
            Weight = Bases + (alpha/np.linalg.norm(u))*np.einsum('i,ijk->jk',u,gBases)
            gWeight = gBases #+ alpha*ggBases*(u/np.abs(u))
        else:
            alpha = 0.0
            Weight = Bases #+ alpha*gBases*(u/np.abs(u))
            gWeight = gBases #+ alpha*ggBases*(u/np.abs(u))

        # compute differential for integral
        dV = np.einsum('i,i->i',AllGauss,np.abs(np.linalg.det(etaGradientX)))
        # compute diffusive matrix (ngauss*ndim*nepoin,ngauss*ndim*nepoin->ngauss*nepoin*nepoint)
        UXGradientN = np.einsum('j,ijk->ik',u,XGradientN)
        diff_mat = np.einsum('ij,ik->ijk',UXGradientN,UXGradientN)
        ElemDiffusion = np.einsum('ijk,i->jk',diff_mat,dV)
        # source terms due to heat generation and convection
        ElemSourceQ = q*A*np.einsum('ijk,j,i->i',XGradientN,u,dV)
        ElemSourceH = h*T_inf*np.einsum('ijk,j,i->i',XGradientN,u,dV)

        for i in range(nodeperelem):
            Source[mesh.elements[elem,i]] += ElemSourceQ[i] + ElemSourceH[i]
            for j in range(nodeperelem):
                K[mesh.elements[elem,i],mesh.elements[elem,j]] += ElemDiffusion[i,j]

    return K, Source


