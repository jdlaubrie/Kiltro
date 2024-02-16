import numpy as np

class Mesh(object):

    def __init__(self, element_type=None):
        self.points = None
        self.elements = None
        self.nnodes = None
        self.nelem = None

        self.element_type = element_type

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

#=============================================================================#
    def InferSpatialDimension(self):
        """Infer the spatial dimension of the mesh"""

        assert self.points is not None
        # if self.points.shape[1] == 3:
        #     if self.element_type == "tri" or self.element_type == "quad":
        #         print("3D surface mesh of ", self.element_type)

        return self.points.shape[1]

#=============================================================================#
    def InferNumberOfNodesPerElement(self, p=None, element_type=None):
        """Infers number of nodes per element. If p and element_type are
            not None then returns the number of nodes required for the given
            element type with the given polynomial degree"""

        if p is not None and element_type is not None:
            if element_type=="line":
                return int(p+1)
            elif element_type=="tri":
                return int((p+1)*(p+2)/2)
            elif element_type=="quad":
                return int((p+1)**2)
            elif element_type=="tet":
                return int((p+1)*(p+2)*(p+3)/6)
            elif element_type=="hex":
                return int((p+1)**3)
            else:
                raise ValueError("Did not understand element type")

        assert self.elements.shape[0] is not None
        return self.elements.shape[1]

