import numpy as np

#=============================================================================#
def Assembly(mesh, function_space, u, k, q, h, rhoc, A, T_inf):
    
    #h = h*Perimeter for 1D
    npoints = mesh.nnodes
    nelem = mesh.nelem
    
    M = np.zeros((npoints,npoints), dtype=np.float64)
    K = np.zeros((npoints,npoints), dtype=np.float64)
    Source = np.zeros((npoints,1), dtype=np.float64)
    for elem in range(nelem):
        # capture element coordinates
        ElemCoords = mesh.points[mesh.elements[elem]]
        ElemLength = np.abs(ElemCoords[1] - ElemCoords[0])
        # invoke the function space
        Bases = function_space.Bases
        gBases = function_space.gBases
        AllGauss = function_space.AllGauss
        # dX/deta (nepoin*ngauss,nepoin->ngauss)
        etaGradientX = np.einsum('ij,i->j',gBases,ElemCoords)
        inv_etaGradientX = 1.0/etaGradientX
        # dN/dX (ngauss,nepoin*ngauss->ngauss*nepoin)
        XGradientN = np.einsum('j,ij->ji',inv_etaGradientX,gBases)
        
        # evaluates the effects of convection. activates upwinding
        Peclet = u*ElemLength/(2.0*k)
        if ~np.isclose(Peclet,0.0):
            alpha = 1.0/np.tanh(Peclet) - 1.0/np.abs(Peclet)
            Weight = Bases + alpha*gBases*(u/np.abs(u))
            gWeight = gBases #+ alpha*ggBases*(u/np.abs(u))
        else:
            alpha = 0.0
            Weight = Bases #+ alpha*gBases*(u/np.abs(u))
            gWeight = gBases #+ alpha*ggBases*(u/np.abs(u))

        print("Element={}.".format(elem) +\
              " Peclet={0:>10.5g}.".format(Peclet) +\
              " alpha={0:>10.5g}".format(alpha))
        # compute differential for integral
        dV = np.einsum('i,i->i',AllGauss,np.abs(etaGradientX))
        # compute diffusive matrix
        diff_mat = np.einsum('ij,ik->ijk',XGradientN,XGradientN)
        ElemDiffusion = A*k*np.einsum('ijk,i->jk',diff_mat,dV)
        # compute convection matrix
        conv_mat = np.einsum('ij,jk->ikj',Weight,XGradientN)
        ElemConvection = u*np.einsum('ijk,k->ij',conv_mat,dV)
        # mass matrix
        mass_mat = np.einsum('ik,jk->ijk',Weight,Bases)
        ElemMass = rhoc*np.einsum('ijk,k->ij',mass_mat,dV)
        # stiffness matrix resulting from boundary conditions
        ElemBoundary = h*ElemMass
        # source terms due to heat generation and convection
        ElemSourceQ = q*A*np.einsum('ij,j->i',Weight,dV)
        ElemSourceH = h*T_inf*np.einsum('ij,j->i',Weight,dV)
       
        for i in range(2):
            Source[mesh.elements[elem,i]] += ElemSourceQ[i] + ElemSourceH[i]
            for j in range(2):
                K[mesh.elements[elem,i],mesh.elements[elem,j]] += ElemDiffusion[i,j] + \
                        ElemConvection[i,j] + ElemBoundary[i,j]
                M[mesh.elements[elem,i],mesh.elements[elem,j]] += ElemMass[i,j]

    return K, Source, M

#=============================================================================#
def AssemblyU(mesh, function_space, u, k, q, h, A, T_inf):
    
    npoints = mesh.nnodes
    nelem = mesh.nelem
    
    K = np.zeros((npoints,npoints), dtype=np.float64)
    Source = np.zeros((npoints,1), dtype=np.float64)
    for elem in range(nelem):
        # capture element coordinates
        ElemCoords = mesh.points[mesh.elements[elem]]
        ElemLength = np.abs(ElemCoords[1] - ElemCoords[0])
        # invoke the function space
        Bases = function_space.Bases
        gBases = function_space.gBases
        AllGauss = function_space.AllGauss
        # dX/deta (nepoin*ngauss,nepoin->ngauss)
        etaGradientX = np.einsum('ij,i->j',gBases,ElemCoords)
        inv_etaGradientX = 1.0/etaGradientX
        # dN/dX (ngauss,nepoin*ngauss->ngauss*nepoin)
        XGradientN = np.einsum('j,ij->ji',inv_etaGradientX,gBases)
        
        # evaluates the effects of convection. activates upwinding
        Peclet = u*ElemLength/(2.0*k)
        if ~np.isclose(Peclet,0.0):
            alpha = 1.0/np.tanh(Peclet) - 1.0/np.abs(Peclet)
            Weight = Bases + alpha*gBases*(u/np.abs(u))
            gWeight = gBases #+ alpha*ggBases*(u/np.abs(u))
        else:
            alpha = 0.0
            Weight = Bases #+ alpha*gBases*(u/np.abs(u))
            gWeight = gBases #+ alpha*ggBases*(u/np.abs(u))

        # compute differential for integral
        dV = np.einsum('i,i->i',AllGauss,np.abs(etaGradientX))
        # compute diffusive matrix
        diff_mat = np.einsum('ij,ik->ijk',XGradientN,XGradientN)
        ElemDiffusion = (u**2)*np.einsum('ijk,i->jk',diff_mat,dV)
        # source terms due to heat generation and convection
        ElemSourceQ = u*q*A*np.einsum('ij,j->i',XGradientN,dV)
        ElemSourceH = u*h*T_inf*np.einsum('ij,j->i',XGradientN,dV)
       
        for i in range(2):
            Source[mesh.elements[elem,i]] += ElemSourceQ[i] + ElemSourceH[i]
            for j in range(2):
                K[mesh.elements[elem,i],mesh.elements[elem,j]] += ElemDiffusion[i,j]

    return K, Source


