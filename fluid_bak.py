# python3
import numpy as np
import matplotlib.pyplot as plt
import sys

# properties and conditions
k = 1.0 #W/(m*K)
h = 25.0 #W/(m2*K)
Per = 1.0 #m
A = 1.0 #m2
q = 0.0 #W/m3
T_inf = 20.0 #°C

#=============================================================================#
def ElementNeighbourhood(points, elements):

    npoints = points.shape[0]
    nelem = elements.shape[0]

    ElemCenter = np.mean(points[elements], axis=1)
    ElemNeighbour = np.zeros((nelem,2), dtype=np.int64)
    for elem in range(nelem):
        WestElem = -1
        EastElem = -1
        # capture element coordinates
        ElemCoords = points[elements[elem]]
        # local west and east nodes
        west_node = np.argmin(ElemCoords)
        east_node = np.argmax(ElemCoords)
        # global west and east nodes
        WestNode = elements[elem][west_node]
        EastNode = elements[elem][east_node]
        # assess west element
        elm,pos = np.where(WestNode==elements)
        for i in range(elm.shape[0]):
            if elm[i] != elem: WestElem = elm[i]
        # assess east element
        elm,pos = np.where(EastNode==elements)
        for i in range(elm.shape[0]):
            if elm[i] != elem: EastElem = elm[i]

        # element neighbourhood
        ElemNeighbour[elem,0] = WestElem
        ElemNeighbour[elem,1] = EastElem

    return ElemCenter, ElemNeighbour

#=============================================================================#
def Temperature1D(points, elements, dirichlet_flags, neumann_flags):
    # d/dx(k*dT/dx) + q = 0

    npoints = points.shape[0]
    nelem = elements.shape[0]
    
    # system of equations to solve
    K = np.zeros((nelem,nelem), dtype=np.float64)
    F = np.zeros((nelem), dtype=np.float64)
    # point P
    for elem in range(nelem):
        # capture element coordinates
        elem_coords = points[elements[elem]]
        # local west and east nodes
        west_node = np.argmin(elem_coords)
        east_node = np.argmax(elem_coords)
        # element properties
        l_e = elem_coords[east_node] - elem_coords[west_node]
        S_P = -h*Per*l_e
        S_u = q*A*l_e + h*Per*l_e*T_inf
        if ElemNeighbor[elem,0] == -1:
            # element at the west border
            l_WP = ElemCenter[elem] - elem_coords[west_node]
            l_PE = ElemCenter[ElemNeighbor[elem,1]] - ElemCenter[elem]
            a_W = 0.0
            a_E = k*A/l_PE
            if not np.isnan(dirichlet_flags[elements[elem]][west_node]):
                S_u += k*A*dirichlet_flags[elements[elem]][west_node]/l_WP
            if np.isnan(neumann_flags[elements[elem]][west_node]):
                S_P += -k*A/l_WP
            K[elem,ElemNeighbor[elem,1]] = -a_E
            K[elem,elem] = a_W + a_E - S_P
            F[elem] = S_u
        elif ElemNeighbor[elem,1] == -1:
            # element at the east border
            l_WP = ElemCenter[elem] - ElemCenter[ElemNeighbor[elem,0]]
            l_PE = elem_coords[east_node] - ElemCenter[elem]
            a_W = k*A/l_WP
            a_E = 0.0
            if not np.isnan(dirichlet_flags[elements[elem]][east_node]):
                S_u += k*A*dirichlet_flags[elements[elem]][east_node]/l_PE
            if np.isnan(neumann_flags[elements[elem]][east_node]):
                S_P += -k*A/l_PE
            K[elem,ElemNeighbor[elem,0]] = -a_W
            K[elem,elem] = a_W + a_E - S_P
            F[elem] = S_u
        else:
            l_WP = ElemCenter[elem] - ElemCenter[ElemNeighbor[elem,0]]
            l_PE = ElemCenter[ElemNeighbor[elem,1]] - ElemCenter[elem]
            a_W = k*A/l_WP
            a_E = k*A/l_PE
            # element without border
            K[elem,ElemNeighbor[elem,0]] = -a_W
            K[elem,ElemNeighbor[elem,1]] = -a_E
            K[elem,elem] = a_W + a_E - S_P
            F[elem] = S_u

    K_inv = np.linalg.inv(K)
    sol = np.dot(K_inv,F)

    return sol

#=============================================================================#
def Output(points, elements, dirichlet_flags, sol, elem_center, NeighborElem):

    npoints = points.shape[0]

    T = dirichlet_flags
    for node in range(npoints):
        if np.isnan(dirichlet_flags[node]):
            elem,pos = np.where(elements==node)
            if elem.shape[0]==1:
                elem = elem[0]
                if NeighborElem[elem][0] == -1:
                    T[node] = (points[node]-elem_center[elem])*\
                        (sol[NeighborElem[elem][1]]-sol[elem])/\
                        (elem_center[NeighborElem[elem][1]]-elem_center[elem]) + \
                        sol[elem]
                elif NeighborElem[elem][1] == -1:
                    T[node] = (points[node]-elem_center[elem])*\
                        (sol[NeighborElem[elem][0]]-sol[elem])/\
                        (elem_center[NeighborElem[elem][0]]-elem_center[elem]) + \
                        sol[elem]
            else:
                T[node] = (points[node]-elem_center[elem[0]])*\
                        (sol[elem[1]]-sol[elem[0]])/\
                        (elem_center[elem[1]]-elem_center[elem[0]]) + sol[elem[0]]
        else:
            T[node] = dirichlet_flags[node]

    fig,ax = plt.subplots()

    ax.plot(points,T, 'P-')
    ax.plot(elem_center,sol, 'P-')

    fig.tight_layout()
    plt.show
    FIGURENAME = 'output.pdf'
    plt.savefig(FIGURENAME)
    plt.close('all')

#=============================================================================#
# mesh
points = np.array([0.0,0.2,0.4,0.6,0.8,1.0], dtype=np.float64)
elements = np.array([[0,1],[1,2],[2,3],[3,4],[4,5]], dtype=np.int64)
npoints = points.shape[0]
nelem = elements.shape[0]
#elem_center, NeighborElem = ElementNeighbourhood(points, elements)

# boundary conditions
dirichlet_flags = np.zeros((npoints), dtype=np.float64) + np.NAN
dirichlet_flags[0] = 100.0
dirichlet_flags[-1] = 500.0
neumann_flags = np.zeros((npoints), dtype=np.float64) + np.NAN
#neumann_flags[-1] = 0.0

# solve temperature problem
sol = Temperature1D(points, elements, dirichlet_flags, neumann_flags)

# make an output for the data
#Output(points, elements, dirichlet_flags, sol, elem_center, NeighborElem)

