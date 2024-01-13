import numpy as np

class Mesh(object):

    def __init__(self,):
        self.points = None
        self.elements = None
        self.nnodes = None
        self.nelem = None

#=============================================================================#
    def Line(self, left_point=0.0, right_point=1.0, n=10):

        left_point = float(left_point)
        right_point = float(right_point)

        n = int(n)

        points = np.linspace(left_point,right_point,n+1)[:,None]
        elements = np.zeros((n,2), dtype=np.int64)
        for i in range(2):
            elements[:,i] = np.arange(0,n)+i

        self.points = points
        self.elements = elements
        self.nnodes = points.shape[0]
        self.nelem = elements.shape[0]
        self.ndim = 1

#=============================================================================#
    def Rectangle(self, lower_left_point=(0,0), upper_right_point=(1,1), nx=5, ny=5):
        # creates the mesh of a rectangle
        nx, ny = int(nx), int(ny)

        x=np.linspace(lower_left_point[0],upper_right_point[0],nx+1)
        y=np.linspace(lower_left_point[1],upper_right_point[1],ny+1)

        X,Y = np.meshgrid(x,y)
        points = np.dstack((X.ravel(),Y.ravel()))[0,:,:]

        nelem = int(nx*ny)
        elements = np.zeros((nelem,4),dtype=np.int64)

        dum_0 = np.arange((nx+1)*ny)
        dum_1 = np.array([(nx+1)*i+nx for i in range(ny)])
        col0 = np.delete(dum_0,dum_1)
        elements[:,0] = col0
        elements[:,1] = col0 + 1
        elements[:,2] = col0 +  nx + 2
        elements[:,3] = col0 +  nx + 1

        self.points = points
        self.elements = elements
        self.nnodes = points.shape[0]
        self.nelem = elements.shape[0]
        self.ndim = 2

