import numpy as np

#=============================================================================#
class Material(object):

    def __init__(self, mtype, ndim, **kwargs):

        # SAFETY CHECKS
        if not isinstance(mtype, str):
            raise TypeError("Type of material model should be given as a string")

        # SET ALL THE OPTIONAL KEYWORD ARGUMENTS
        for i in kwargs.keys():
            if "__" not in i:
                setattr(self,i,kwargs[i])

#=============================================================================#
class FourierConduction(Material):

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(FourierConduction, self).__init__(mtype, ndim, **kwargs)

        if not 'k' in self.__dict__.keys():
            raise ValueError("Thermal conductivity not defined.")

        if not 'q' in self.__dict__.keys():
            self.q = 0.0

    def HeatDiffusion(self, XGradientN, elem=0, gcounter=0):
        A = self.area
        k = self.k
        Gradient = XGradientN[gcounter]
        # compute diffusion matrix (ndim*nepoin,ndim*nepoin->nepoin*nepoin)
        diffusion = A*k*np.einsum('ij,ik->jk',Gradient,Gradient)
        return diffusion

    def HeatGeneration(self, Weight, elem=0, gcounter=0):
        A = self.area
        q = self.q
        Weight0 = Weight[gcounter]
        # compute generation vector (nepoin->nepoin)
        generation = q*A*Weight0
        return generation



