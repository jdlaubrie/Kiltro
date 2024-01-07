import numpy as np

class Mesh(object):

    def __init__(self,):
        self.points = None
        self.elements = None
        self.nnodes = None
        self.nelem = None

    def Line(self, left_point=0.0, right_point=1.0, n=10):

        left_point = float(left_point)
        right_point = float(right_point)

        n = int(n)

        points = np.linspace(left_point,right_point,n+1) #[:,None]
        elements = np.zeros((n,2), dtype=np.int64)
        for i in range(2):
            elements[:,i] = np.arange(0,n)+i

        self.points = points
        self.elements = elements
        self.nnodes = points.shape[0]
        self.nelem = elements.shape[0]


