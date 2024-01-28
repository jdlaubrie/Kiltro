import numpy as np

#=============================================================================#
class Material(object):

    def __init__(self, mtype, ndim, density=None, conductivity=None, heat_capacity_v=None, **kwargs):

        # SAFETY CHECKS
        if not isinstance(mtype, str):
            raise TypeError("Type of material model should be given as a string")

        # MATERIAL CONSTANTS
        self.k = conductivity
        self.rho = density
        self.c_v = heat_capacity_v

        # SET ALL THE OPTIONAL KEYWORD ARGUMENTS
        for i in kwargs.keys():
            if "__" not in i:
                setattr(self,i,kwargs[i])

        self.mtype = mtype
        self.ndim = ndim

#=============================================================================#
class FourierConduction(Material):

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(FourierConduction, self).__init__(mtype, ndim, **kwargs)

        if (not 'k' in self.__dict__.keys()) and \
        (not 'conductivity' in self.__dict__.keys()):
            raise ValueError("Thermal conductivity not defined.")

        if not 'q' in self.__dict__.keys():
            self.q = 0.0

        self.nvar = 1 #self.ndim + 1

    #-------------------------------------------------------------------------#
    def HeatDiffusion(self, XGradientN, elem=0, gcounter=0):
        A = self.area
        k = self.k
        Gradient = XGradientN[gcounter]
        # compute diffusion matrix (ndim*nepoin,ndim*nepoin->nepoin*nepoin)
        diffusion = A*k*np.einsum('ij,ik->jk',Gradient,Gradient)
        return diffusion

    #-------------------------------------------------------------------------#
    def HeatConvection(self, Weight, XGradientN, u, elem=0, gcounter=0):
        rho = self.rho
        c_v = self.c_v
        Weight0 = Weight[:,gcounter]
        Gradient = XGradientN[gcounter]
        # compute convection matrix (nepoin,ndim,ndim*nepoin->nepoin*nepoin)
        convection = rho*c_v*np.einsum('i,j,jl->il',Weight0,u,Gradient)
        return convection

    #-------------------------------------------------------------------------#
    def HeatMass(self, Weight, Bases, elem=0, gcounter=0):
        rho = self.rho
        c_v = self.c_v
        Weight0 = Weight[:,gcounter]
        Bases0 = Weight[:,gcounter]
        # mass matrix (nepoin,nepoin->nepoin*nepoin)
        mass = rho*c_v*np.einsum('i,j->ij',Weight0,Bases0)
        return mass

    #-------------------------------------------------------------------------#
    def HeatGeneration(self, Weight, elem=0, gcounter=0):
        A = self.area
        q = self.q
        Weight0 = Weight[gcounter]
        # compute generation vector (nepoin->nepoin)
        generation = q*A*Weight0
        return generation

