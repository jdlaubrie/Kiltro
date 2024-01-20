# python3
import numpy as np

#=============================================================================#
class VariationalPrinciple(object):

    def __init__(self, mesh):
        self.ndim = mesh.points.shape[1]

    #-------------------------------------------------------------------------#
    def UpWindBases(self, Bases, gBases, ElemLength, k, u):
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

#        print("Element={}.".format(elem) +\
#              " Peclet={0:>10.5g}.".format(Peclet) +\
#              " alpha={0:>10.5g}".format(alpha))

        return Weight, gWeight
#=============================================================================#
class HeatTransfer(VariationalPrinciple):

    def __init__(self, mesh):
        super(HeatTransfer, self).__init__(mesh)

    #-------------------------------------------------------------------------#
    def GetElementalMatrices(self, elem, mesh, material, function_space, boundary_condition, u):

        k = material.k
        rhoc = material.rhoc
        h = boundary_condition.applied_robin_convection
        T_inf = boundary_condition.ground_robin_convection

        # capture element coordinates
        ElemCoords = mesh.points[mesh.elements[elem]]
        ElemLength = np.linalg.norm(ElemCoords[1,:] - ElemCoords[0,:])
        # invoke the function space
        Bases = function_space.Bases
        gBases = function_space.gBases
        AllGauss = function_space.AllGauss
        #nepoin
        nodeperelem = function_space.Bases.shape[0]

        Weight, gWeight = self.UpWindBases(Bases,gBases,ElemLength,k,u)

        # definition of gradients
        # dX/deta (ndim*nepoin*ngauss,nepoin*ndim->ngauss*ndim*ndim)
        etaGradientX = np.einsum('ijk,jl->kil',gBases,ElemCoords)
        inv_etaGradientX = np.linalg.inv(etaGradientX)
        # dN/dX (ngauss*ndim*ndim,ndim*nepoin*ngauss->ngauss*ndim*nepoin)
        XGradientN = np.einsum('ijk,kli->ijl',inv_etaGradientX,gBases)

        # compute differential for integral
        dV = np.einsum('i,i->i',AllGauss,np.abs(np.linalg.det(etaGradientX)))

        # loop on guass points
        Diffusion = np.zeros((nodeperelem,nodeperelem), dtype=np.float64)
        Convection = np.zeros((nodeperelem,nodeperelem), dtype=np.float64)
        Mass = np.zeros((nodeperelem,nodeperelem), dtype=np.float64)
        SourceQ = np.zeros((nodeperelem), dtype=np.float64)
        BoundaryH = np.zeros((nodeperelem,nodeperelem), dtype=np.float64)
        SourceH = np.zeros((nodeperelem), dtype=np.float64)
        for counter in range(dV.shape[0]):
            # compute diffusion matrix (ngauss*ndim*nepoin,ngauss*ndim*nepoin->ngauss*nepoin*nepoin)
            gdiffusion = material.HeatDiffusion(XGradientN,elem,counter)
            Diffusion += gdiffusion*dV[counter]
            # compute convection matrix (nepoin*ngauss,ndim,ngauss*ndim*nepoin->ngauss*nepoin*nepoin)
            gconvection = self.LocalConvection(Weight,XGradientN,u,elem,counter)
            Convection += gconvection*dV[counter]
            # mass matrix (nepoin*ngauss,nepoin*ngauss->ngauss*nepoin*nepoin)
            gmass = self.LocalMass(Weight,Bases,elem,counter)
            Mass += rhoc*gmass*dV[counter]
            # source flux due to heat generation
            ggeneration = material.HeatGeneration(Weight,elem,counter)
            SourceQ += ggeneration*dV[counter]

            # stiffness matrix resulting from boundary conditions
            gk,gf = boundary_condition.LocalFlux(Weight,Bases,elem,counter)
            BoundaryH += gk*dV[counter]
            SourceH += gf*dV[counter]

        return Mass, Diffusion, Convection, BoundaryH, SourceQ, SourceH

    #-------------------------------------------------------------------------#
    def GetElementalCharacteristicGalerkin(self, elem, mesh, material, function_space, boundary_condition, u):

        k = material.k
        q = material.q
        A = material.area
        h = boundary_condition.applied_robin_convection
        T_inf = boundary_condition.ground_robin_convection

        # capture element coordinates
        ElemCoords = mesh.points[mesh.elements[elem]]
        ElemLength = np.linalg.norm(ElemCoords[1,:] - ElemCoords[0,:])
        # invoke the function space
        Bases = function_space.Bases
        gBases = function_space.gBases
        AllGauss = function_space.AllGauss
        #nepoin
        nodeperelem = function_space.Bases.shape[0]

        Weight, gWeight = self.UpWindBases(Bases,gBases,ElemLength,k,u)

        # definition of gradients
        # dX/deta (ndim*nepoin*ngauss,nepoin*ndim->ngauss*ndim*ndim)
        etaGradientX = np.einsum('ijk,jl->kil',gBases,ElemCoords)
        inv_etaGradientX = np.linalg.inv(etaGradientX)
        # dN/dX (ngauss*ndim*ndim,ndim*nepoin*ngauss->ngauss*ndim*nepoin)
        XGradientN = np.einsum('ijk,kli->ijl',inv_etaGradientX,gBases)

        # compute differential for integral
        dV = np.einsum('i,i->i',AllGauss,np.abs(np.linalg.det(etaGradientX)))

        # compute diffusive matrix (ngauss*ndim*nepoin,ngauss*ndim*nepoin->ngauss*nepoin*nepoint)
        UXGradientN = np.einsum('j,ijk->ik',u,XGradientN)
        diff_mat = np.einsum('ij,ik->ijk',UXGradientN,UXGradientN)
        Diffusion = np.einsum('ijk,i->jk',diff_mat,dV)
        # source terms due to heat generation and convection
        SourceQ = q*A*np.einsum('ijk,j,i->i',XGradientN,u,dV)
        SourceH = h*T_inf*np.einsum('ijk,j,i->i',XGradientN,u,dV)

        return Diffusion, SourceQ, SourceH

    #-------------------------------------------------------------------------#
    def LocalConvection(self, Weight, XGradientN, u, elem=0, gcounter=0):
        Weight0 = Weight[:,gcounter]
        Gradient = XGradientN[gcounter]
        # compute convection matrix (nepoin,ndim,ndim*nepoin->nepoin*nepoin)
        convection = np.einsum('i,j,jl->il',Weight0,u,Gradient)
        return convection

    #-------------------------------------------------------------------------#
    def LocalMass(self, Weight, Bases, elem=0, gcounter=0):
        Weight0 = Weight[:,gcounter]
        Bases0 = Weight[:,gcounter]
        # mass matrix (nepoin,nepoin->nepoin*nepoin)
        mass = np.einsum('i,j->ij',Weight0,Bases0)
        return mass


