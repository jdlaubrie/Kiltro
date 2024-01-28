import numpy as np

__all__ = ['Assemble', 'AssembleCharacteristicGalerkin'] #, 'AssembleForces', 'AssembleExplicit', 'AssembleMass', 'AssembleForm', 'AssembleFollowerForces']

#=============================================================================#
def Assemble(fem_solver, function_spaces, formulation, mesh, material, boundary_condition):

    nvar = formulation.nvar
    ndim = formulation.ndim
    nelem = mesh.nelem
    nodeperelem = mesh.elements.shape[1]
    ndof = nodeperelem*nvar
    local_capacity = ndof*ndof
    npoints = mesh.nnodes

    # SPARSE INDICES
#    indices, indptr = fem_solver.indices, fem_solver.indptr
#    data_global_indices = fem_solver.data_global_indices
#    data_local_indices = fem_solver.data_local_indices

#    V_convection=np.zeros(indices.shape[0],dtype=np.float64)
#    if fem_solver.analysis_type !='steady':
#        V_mass=np.zeros(indices.shape[0],dtype=np.float64)

    M = np.zeros((npoints*nvar,npoints*nvar), dtype=np.float64)
    K = np.zeros((npoints*nvar,npoints*nvar), dtype=np.float64)
    Flux = np.zeros((npoints*nvar,1), dtype=np.float64)
    for elem in range(nelem):

        massel, convel, f = formulation.GetElementalMatrices(elem, function_spaces[0],
                        mesh, material, fem_solver)

        # ASSEMBLY - DIFFUSION MATRIX
        for i in range(nodeperelem):
            for j in range(nodeperelem):
                K[mesh.elements[elem,i],mesh.elements[elem,j]] += convel[i,j]

        # ASSEMBLY - MASS MATRIX
        if fem_solver.analysis_type != 'steady':
            for i in range(nodeperelem):
                for j in range(nodeperelem):
                    M[mesh.elements[elem,i],mesh.elements[elem,j]] += massel[i,j]

        # ASSEMBLY - INTERNAL FLUX
        for i in range(nodeperelem):
            Flux[mesh.elements[elem,i]] += f[i]

    # ASSEMBLY - EXTERNAL FLUX
    AssemblyRobinForces(fem_solver, function_spaces[0], mesh, material, boundary_condition)

    return K, Flux, M

#=============================================================================#
def AssembleCharacteristicGalerkin(function_spaces, formulation, mesh, material, boundary_condition):

    npoints = mesh.nnodes
    nelem = mesh.nelem
    nodeperelem = mesh.elements.shape[1]

    K = np.zeros((npoints,npoints), dtype=np.float64)
    Flux = np.zeros((npoints,1), dtype=np.float64)
    for elem in range(nelem):

        convel, f = formulation.GetElementalCharacteristicGalerkin(elem, mesh, 
        material, function_spaces[0], boundary_condition)

        # ASSEMBLY - DIFFUSION MATRIX
        for i in range(nodeperelem):
            for j in range(nodeperelem):
                K[mesh.elements[elem,i],mesh.elements[elem,j]] += convel[i,j]

        # ASSEMBLY - INTERNAL FLUX
        for i in range(nodeperelem):
            Flux[mesh.elements[elem,i]] += f[i]

    return K, Flux

#=============================================================================#
#--------------- ASSEMBLY ROUTINE FOR EXTERNAL FLUXES ---------------#
def AssemblyRobinForces(fem_solver, function_space, mesh, material, boundary_condition):

    nvar = material.nvar
    ndim = material.ndim
    npoints = mesh.nnodes

    K = np.zeros((npoints*nvar,npoints*nvar), dtype=np.float64)
    Flux = np.zeros((npoints*nvar,1), dtype=np.float64)
    for elem in range(mesh.nelem):

        h = boundary_condition.applied_robin_convection
        T_inf = boundary_condition.ground_robin_convection

        # capture element coordinates
        ElemCoords = mesh.points[mesh.elements[elem]]

        # invoke the function space
        Bases = function_space.Bases
        gBases = function_space.gBases
        AllGauss = function_space.AllGauss
        nodeperelem = function_space.Bases.shape[0]
        Weight = Bases

        # ALLOCATE
        diffusion = np.zeros((nodeperelem*nvar,nodeperelem*nvar), dtype=np.float64)
        flux = np.zeros((nodeperelem*nvar), dtype=np.float64)

        # definition of gradients
        # dX/deta (ndim*nepoin*ngauss,nepoin*ndim->ngauss*ndim*ndim)
        ParentGradientX = np.einsum('ijk,jl->kil',gBases,ElemCoords)

        # compute differential for integral
        dV = np.einsum('i,i->i',AllGauss,np.abs(np.linalg.det(ParentGradientX)))

        # stiffness matrix resulting from boundary conditions
        # resistance (nepoin*gauss,nepoin*gauss->gauss*nepoin*nepoin)
        gdiff = h*np.einsum('ik,jk->kij',Weight,Bases)
        # source terms due to heat convection (nepoin)
        gf = h*T_inf*Weight

        # loop on guass points
        for counter in range(dV.shape[0]):
            diffusion += gdiff[counter]*dV[counter]
            flux += gf[counter]*dV[counter]

        # ASSEMBLY - DIFFUSION MATRIX
        for i in range(nodeperelem):
            for j in range(nodeperelem):
                K[mesh.elements[elem,i],mesh.elements[elem,j]] += diffusion[i,j]

        # ASSEMBLY - EXTERNAL FLUX
        for i in range(nodeperelem):
            Flux[mesh.elements[elem,i]] += flux[i]

    return K, Flux

