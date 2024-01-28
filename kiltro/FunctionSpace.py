import numpy as np

class FunctionSpace(object):

    def __init__(self, mesh):

        ndim = mesh.ndim

        if ndim == 1:
            Bases, gBases, w = self.GetBases1D()
        else:
            Bases, gBases, w = self.GetBases2D()

        self.Bases = Bases
        self.gBases = gBases
        self.AllGauss = w

#=============================================================================#
    # define the function space for quadrature and iterpolation
    def Lagrange(self, eta):
        # simple linear function. maybe Legrange
        N = np.array([0.5*(1-eta), 0.5*(1+eta)])
        dN = np.array([-0.5, 0.5])
        return N, dN

#=============================================================================#
    def GetBases1D(self,):
        ns = 2
        #ngauss = 2
        z = np.array([-0.57735027, 0.57735027])
        w = np.array([1.0,1.0])
        Bases = np.zeros((ns,z.shape[0]), dtype=np.float64)
        gBases = np.zeros((1,ns,z.shape[0]), dtype=np.float64)
        for i in range(z.shape[0]):
            Bases[:,i], gBases[0,:,i] = self.Lagrange(z[i])

        return Bases, gBases, w

#=============================================================================#
    def GetBases2D(self,):
        node_arranger = np.array([0,1,3,2])
        ns = 4  #(C+2)**2, C=0
        z = np.array([-0.57735027, 0.57735027])
        w = np.array([1.0,1.0])
        Bases = np.zeros((ns,z.shape[0]**2), dtype=np.float64)
        gBases = np.zeros((2,ns,z.shape[0]**2), dtype=np.float64)
        AllGauss = np.zeros((w.shape[0]**2), dtype=np.float64)
        counter = 0
        for i in range(z.shape[0]):
            for j in range(z.shape[0]):
                Neta, dNeta = self.Lagrange(z[i])
                Nzeta, dNzeta = self.Lagrange(z[j])
                N = np.outer(Neta,Nzeta).flatten()
                gx = np.outer(Neta,dNzeta).flatten()
                gy = np.outer(dNeta,Nzeta).flatten()
                Bases[:,counter] = N[node_arranger]
                gBases[0,:,counter] = gx[node_arranger]
                gBases[1,:,counter] = gy[node_arranger]
                AllGauss[counter] = w[i]*w[j]
                counter += 1

        return Bases, gBases, AllGauss


