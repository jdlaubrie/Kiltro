import numpy as np

#=============================================================================#
class VariationalPrinciple(object):

    def __init__(self, mesh):
        self.ndim = mesh.points.shape[1]

#=============================================================================#
class HeatTransfer(VariationalPrinciple):

    def __init__(self, mesh):
        super(HeatTransfer, self).__init__(mesh)

    def GetElementalMatrices(self, elem, mesh, material, function_space, u, q, h, T_inf):

        k = material.k
        A = material.area
        rhoc = material.rhoc

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

        print("Element={}.".format(elem) +\
              " Peclet={0:>10.5g}.".format(Peclet) +\
              " alpha={0:>10.5g}".format(alpha))
        # compute differential for integral
        dV = np.einsum('i,i->i',AllGauss,np.abs(np.linalg.det(etaGradientX)))
        # compute diffusive matrix (ngauss*ndim*nepoin,ngauss*ndim*nepoin->ngauss*nepoin*nepoint)
        diff_mat = np.einsum('ijk,ijl->ikl',XGradientN,XGradientN)
        ElemDiffusion = A*k*np.einsum('ijk,i->jk',diff_mat,dV)
        # compute convection matrix (nepoin*ngauss,ndim,ngauss*ndim*nepoin->ngauss*nepoin*nepoin)
        conv_mat = np.einsum('ij,k,jkl->jil',Weight,u,XGradientN)
        ElemConvection = np.einsum('ijk,i->jk',conv_mat,dV)
        # mass matrix (nepoin*ngauss,nepoin*ngauss->ngauss*nepoin*nepoin)
        mass_mat = np.einsum('ik,jk->kij',Weight,Bases)
        ElemMass = rhoc*np.einsum('ijk,i->jk',mass_mat,dV)
        # stiffness matrix resulting from boundary conditions
        ElemBoundary = h*ElemMass
        # source terms due to heat generation and convection
        ElemSourceQ = q*A*np.einsum('ij,j->i',Weight,dV)
        ElemSourceH = h*T_inf*np.einsum('ij,j->i',Weight,dV)

        return ElemMass, ElemDiffusion, ElemConvection, ElemBoundary, ElemSourceQ, ElemSourceH

