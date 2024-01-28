# python3
import numpy as np

from kiltro import FunctionSpace

#=============================================================================#
class VariationalPrinciple(object):

    def __init__(self, mesh, function_spaces=None):

        self.nvar = None
        self.ndim = mesh.points.shape[1]
        self.function_spaces = function_spaces

    #-------------------------------------------------------------------------#
    def GetFunctionSpaces(self, mesh, function_spaces=None):

        if function_spaces == None and self.function_spaces == None:
            # CREATE FUNCTIONAL SPACES
            function_space = FunctionSpace(mesh)
            # CREATE BOUNDARY FUNCTIONAL SPACES
#            bfunction_space = FunctionSpace(mesh.CreateDummyLowerDimensionalMesh(),
#                self.quadrature_rules[2], p=C+1, equally_spaced=equally_spaced_bases)

            self.function_spaces = (function_space,) #post_function_space,bfunction_space)
        else:
            self.function_spaces = function_spaces

        # FOR CONVECTION
        local_size = self.function_spaces[0].Bases.shape[0]*self.nvar
        self.local_rows = np.repeat(np.arange(0,local_size),local_size,axis=0)
        self.local_columns = np.tile(np.arange(0,local_size),local_size)
        self.local_size = local_size

        # FOR MASS
        local_size_m = self.function_spaces[0].Bases.shape[0]*self.ndim
        self.local_rows_mass = np.repeat(np.arange(0,local_size_m),local_size_m,axis=0)
        self.local_columns_mass = np.tile(np.arange(0,local_size_m),local_size_m)
        self.local_size_m = local_size_m

    #-------------------------------------------------------------------------#
    def GetLocalMass(self, elem, function_space, mesh, material, ElemCoords):

        nvar = self.nvar
        ndim = self.ndim

        # invoke the function space
        Bases = function_space.Bases
        gBases = function_space.gBases
        AllGauss = function_space.AllGauss

        #nepoin
        nodeperelem = function_space.Bases.shape[0]

        # ALLOCATE
        mass = np.zeros((nodeperelem*nvar,nodeperelem*nvar), dtype=np.float64)

        # definition of gradients
        # dX/deta (ndim*nepoin*ngauss,nepoin*ndim->ngauss*ndim*ndim)
        ParentGradientX = np.einsum('ijk,jl->kil',gBases,ElemCoords)

        # compute differential for integral
        dV = np.einsum('i,i->i',AllGauss,np.abs(np.linalg.det(ParentGradientX)))

        # mass matrix (nepoin*ngauss,nepoin*ngauss->ngauss*nepoin*nepoin)
        gmass = material.rho*material.c_v*np.einsum('ik,jk->kij',Bases,Bases)
        mass = np.einsum('ijk,i->jk',gmass,dV)

        return mass

    #-------------------------------------------------------------------------#
    def FindIndices(self,A):
        return self.local_rows, self.local_columns, A.ravel()

#=============================================================================#
class HeatDiffusion(VariationalPrinciple):

    def __init__(self, mesh, function_spaces=None):
        super(HeatDiffusion, self).__init__(mesh, function_spaces=function_spaces)

        self.fields = "diffusion"
        self.nvar = 1

        self.GetFunctionSpaces(mesh, function_spaces=function_spaces)

    #-------------------------------------------------------------------------#
    def GetElementalMatrices(self, elem, function_space, mesh, material,
        fem_solver):

        # capture element coordinates
        ElemCoords = mesh.points[mesh.elements[elem]]

        # COMPUTE THE CONVECTION MATRIX
        convection, flux = self.GetLocalConvection(elem, function_space, mesh, material, ElemCoords)

        # COMPUTE THE MASS MATRIX
        mass = []
        if fem_solver.analysis_type != 'steady':
            mass = self.GetLocalMass(elem, function_space, mesh, material, ElemCoords)

#        I_conve_elem, J_conve_elem, V_conve_elem = self.FindIndices(convection)
#        if fem_solver.analysis_type != 'steady':
#            I_mass_elem, J_mass_elem, V_mass_elem = self.FindIndices(mass)
#        print(I_conve_elem, J_conve_elem, V_conve_elem)

        return mass, convection, flux

    #-------------------------------------------------------------------------#
    def GetLocalConvection(self, elem, function_space, mesh, material, ElemCoords):

        nvar = self.nvar
        ndim = self.ndim

        # invoke the function space
        Bases = function_space.Bases
        gBases = function_space.gBases
        AllGauss = function_space.AllGauss
        nodeperelem = function_space.Bases.shape[0]

        # ALLOCATE
        convection = np.zeros((nodeperelem*nvar,nodeperelem*nvar), dtype=np.float64)
        flux = np.zeros((nodeperelem*nvar,1), dtype=np.float64)

        # definition of gradients
        # dX/deta (ndim*nepoin*ngauss,nepoin*ndim->ngauss*ndim*ndim)
        ParentGradientX = np.einsum('ijk,jl->kil',gBases,ElemCoords)
        inv_ParentGradientX = np.linalg.inv(ParentGradientX)
        # dN/dX (ngauss*ndim*ndim,ndim*nepoin*ngauss->ngauss*ndim*nepoin)
        SpatialGradient = np.einsum('ijk,kli->ijl',inv_ParentGradientX,gBases)

        # compute differential for integral
        dV = np.einsum('i,i->i',AllGauss,np.abs(np.linalg.det(ParentGradientX)))

        # loop on guass points
        for counter in range(dV.shape[0]):
            # compute diffusion matrix (ndim*nepoin)
            diffusion = material.area*material.k*SpatialGradient[counter]
            # source flux due to heat generation
            flux_q = material.area*material.q*Bases[:,counter]

            DB, q = self.ConstitutiveConvectionIntegrand(SpatialGradient[counter], diffusion.T, flux_q)

            convection += DB*dV[counter]
            flux += q*dV[counter]

        return convection, flux

    #-------------------------------------------------------------------------#
    def ConstitutiveConvectionIntegrand(self, SpatialGradient, diffusion, flux):
        """Applies to displacement based formulation"""

        B = SpatialGradient.copy()

        DB = np.dot(diffusion,B)
        q = flux[:,None]

        return DB, q

#=============================================================================#
class HeatAdvectionDiffusion(VariationalPrinciple):

    def __init__(self, mesh, velocity=None, function_spaces=None):
        super(HeatAdvectionDiffusion, self).__init__(mesh, function_spaces=function_spaces)

        self.fields = "advection_diffusion"
        self.nvar = 1 

        if velocity is None:
            velocity = np.zeros((mesh.nnodes,self.ndim), dtype=np.float64)
        else:
            if not velocity.shape[1] == self.ndim:
                raise ValueError("Dimension of velocity does not match with problem dimension")
        self.velocity = velocity

        self.GetFunctionSpaces(mesh, function_spaces=function_spaces)

    #-------------------------------------------------------------------------#
    def GetElementalMatrices(self, elem, function_space, mesh, material,
        fem_solver):

        mass = []
        # capture element coordinates
        ElemCoords = mesh.points[mesh.elements[elem]]

        # COMPUTE THE CONVECTION MATRIX
        convection, flux = self.GetLocalConvection(elem, function_space, mesh, material, ElemCoords)

        # COMPUTE THE MASS MATRIX
        if fem_solver.analysis_type != 'steady':
            mass = self.GetLocalMass(elem, function_space, mesh, material, ElemCoords)

#        I_conve_elem, J_conve_elem, V_conve_elem = self.FindIndices(convection)
#        if fem_solver.analysis_type != 'steady':
#            I_mass_elem, J_mass_elem, V_mass_elem = self.FindIndices(mass)
#        print(I_conve_elem, J_conve_elem, V_conve_elem)

        return mass, convection, flux

    #-------------------------------------------------------------------------#
    def GetLocalConvection(self, elem, function_space, mesh, material, ElemCoords):

        nvar = self.nvar
        ndim = self.ndim

        ElemLength = np.linalg.norm(ElemCoords[1,:] - ElemCoords[0,:])
        ElemVelocity = np.mean(self.velocity[mesh.elements[elem]], axis=0)

        # invoke the function space
        Bases = function_space.Bases
        gBases = function_space.gBases
        AllGauss = function_space.AllGauss
        nodeperelem = function_space.Bases.shape[0]
        Weight, gWeight = self.PetrovGalerkinBases(function_space, material, ElemVelocity, ElemLength)

        # ALLOCATE
        convection = np.zeros((nodeperelem*nvar,nodeperelem*nvar), dtype=np.float64)
        flux = np.zeros((nodeperelem*nvar,1), dtype=np.float64)

        # definition of gradients
        # dX/deta (ndim*nepoin*ngauss,nepoin*ndim->ngauss*ndim*ndim)
        ParentGradientX = np.einsum('ijk,jl->kil',gBases,ElemCoords)
        inv_ParentGradientX = np.linalg.inv(ParentGradientX)
        # dN/dX (ngauss*ndim*ndim,ndim*nepoin*ngauss->ngauss*ndim*nepoin)
        SpatialGradient = np.einsum('ijk,kli->ijl',inv_ParentGradientX,gBases)

        # compute differential for integral
        dV = np.einsum('i,i->i',AllGauss,np.abs(np.linalg.det(ParentGradientX)))

        # loop on guass points
        for counter in range(dV.shape[0]):
            # compute diffusion matrix (ndim*nepoin)
            diffusion = material.area*material.k*SpatialGradient[counter]
            # compute advection matrix (nepoin,ndim->nepoin*ndim)
            advection = material.rho*material.c_v*np.outer(Weight[counter],ElemVelocity)
            # source flux due to heat generation
            flux_q = material.area*material.q*Bases[:,counter]

            DB, q = self.ConstitutiveConvectionIntegrand(SpatialGradient[counter], diffusion.T, advection, flux_q)

            convection += DB*dV[counter]
            flux += q*dV[counter]

        return convection, flux

    #-------------------------------------------------------------------------#
    def GetElementalCharacteristicGalerkin(self, elem, mesh, material, function_space, boundary_condition):

        nvar = self.nvar
        ndim = self.ndim

        # capture element coordinates
        ElemCoords = mesh.points[mesh.elements[elem]]
        ElemLength = np.linalg.norm(ElemCoords[1,:] - ElemCoords[0,:])
        ElemVelocity = np.mean(self.velocity[mesh.elements[elem]], axis=0)

        # invoke the function space
        Bases = function_space.Bases
        gBases = function_space.gBases
        AllGauss = function_space.AllGauss
        nodeperelem = function_space.Bases.shape[0]
        Weight, gWeight = self.PetrovGalerkinBases(function_space, material, ElemVelocity, ElemLength)

        # ALLOCATE
        convection = np.zeros((nodeperelem*nvar,nodeperelem*nvar), dtype=np.float64)
        flux = np.zeros((nodeperelem*nvar,1), dtype=np.float64)

        # definition of gradients
        # dX/deta (ndim*nepoin*ngauss,nepoin*ndim->ngauss*ndim*ndim)
        ParentGradientX = np.einsum('ijk,jl->kil',gBases,ElemCoords)
        inv_ParentGradientX = np.linalg.inv(ParentGradientX)
        # dN/dX (ngauss*ndim*ndim,ndim*nepoin*ngauss->ngauss*ndim*nepoin)
        SpatialGradient = np.einsum('ijk,kli->ijl',inv_ParentGradientX,gBases)

        # compute differential for integral
        dV = np.einsum('i,i->i',AllGauss,np.abs(np.linalg.det(ParentGradientX)))

        # convective bases matrix (ndim,ngauss*ndim*nepoin->ngauss*nepoin)
        USpatialGradient = np.einsum('j,ijk->ik',ElemVelocity,SpatialGradient)

        for counter in range(dV.shape[0]):
            # compute characteristic-advection matrix (nepoin,nepoin->nepoin*nepoin)
            advection = np.outer(USpatialGradient[counter],USpatialGradient[counter])
            # source terms due to heat generation
            flux_q = material.q*material.area*USpatialGradient[counter]
            q = flux_q[:,None]

            convection += advection*dV[counter]
            flux += q*dV[counter]

        return convection, flux

    #-------------------------------------------------------------------------#
    def PetrovGalerkinBases(self, function_space, material, ElemVelocity, ElemLength):

        # invoke the function space
        Bases = function_space.Bases
        gBases = function_space.gBases

        # evaluates the effects of convection. activates upwinding
        Peclet = np.linalg.norm(ElemVelocity)*ElemLength/(2.0*material.k)
        if ~np.isclose(Peclet,0.0):
            alpha = 1.0/np.tanh(Peclet) - 1.0/np.abs(Peclet)
            Weight = Bases + (alpha/np.linalg.norm(ElemVelocity))*np.einsum('i,ijk->jk',ElemVelocity,gBases)
            gWeight = gBases #+ alpha*ggBases*(u/np.abs(u))
        else:
            alpha = 0.0
            Weight = Bases
            gWeight = gBases

        print(" Peclet={0:>10.5g}.".format(Peclet) +\
              " alpha={0:>10.5g}".format(alpha))

        return Weight, gWeight

    #-------------------------------------------------------------------------#
    def ConstitutiveConvectionIntegrand(self, SpatialGradient, diffusion, advection, flux):
        """Applies to displacement based formulation"""

        B = SpatialGradient.copy()

        DB = np.dot(diffusion,B)
        DB += np.dot(advection,B)
        q = flux[:,None]

        return DB, q


