import numpy as np

#=============================================================================#
def Assembly(mesh, material, formulation, function_space, boundary_condition, u):

    #h = h*Perimeter for 1D
    npoints = mesh.nnodes
    nelem = mesh.nelem
    nodeperelem = mesh.elements.shape[1]

    M = np.zeros((npoints,npoints), dtype=np.float64)
    K = np.zeros((npoints,npoints), dtype=np.float64)
    Source = np.zeros((npoints,1), dtype=np.float64)
    for elem in range(nelem):

        ElemMass, ElemDiffusion, ElemConvection, ElemBoundary, \
        ElemSourceQ, ElemSourceH = formulation.GetElementalMatrices(elem, mesh, 
        material, function_space, boundary_condition, u)

        for i in range(nodeperelem):
            Source[mesh.elements[elem,i]] += ElemSourceQ[i] + ElemSourceH[i]
            for j in range(nodeperelem):
                K[mesh.elements[elem,i],mesh.elements[elem,j]] += ElemDiffusion[i,j] + \
                        ElemConvection[i,j] + ElemBoundary[i,j]
                M[mesh.elements[elem,i],mesh.elements[elem,j]] += ElemMass[i,j]

    return K, Source, M

#=============================================================================#
def AssemblyCharacteristicGalerkin(mesh, material, formulation, function_space, boundary_condition, u):

    npoints = mesh.nnodes
    nelem = mesh.nelem
    nodeperelem = mesh.elements.shape[1]

    k = material.k
    A = material.area
    rhoc = material.rhoc

    K = np.zeros((npoints,npoints), dtype=np.float64)
    Source = np.zeros((npoints,1), dtype=np.float64)
    for elem in range(nelem):

        ElemDiffusion, ElemSourceQ, ElemSourceH = formulation.GetElementalCharacteristicGalerkin(elem, mesh, 
        material, function_space, boundary_condition, u)

        for i in range(nodeperelem):
            Source[mesh.elements[elem,i]] += ElemSourceQ[i] + ElemSourceH[i]
            for j in range(nodeperelem):
                K[mesh.elements[elem,i],mesh.elements[elem,j]] += ElemDiffusion[i,j]

    return K, Source


