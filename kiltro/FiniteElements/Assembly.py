import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from .ComputeSparsityPattern import ComputeSparsityPattern

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
    if fem_solver.recompute_sparsity_pattern is False:
        indices, indptr = fem_solver.indices, fem_solver.indptr
        if fem_solver.squeeze_sparsity_pattern is False:
            data_global_indices = fem_solver.data_global_indices
            data_local_indices = fem_solver.data_local_indices

    if fem_solver.recompute_sparsity_pattern:
        # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF CONVECTION MATRIX - CHANGE TYPES TO INT64 FOR DoF > 1e09
        I_convection=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
        J_convection=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
        V_convection=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.float64)

        I_mass=[]; J_mass=[]; V_mass=[]
        if fem_solver.analysis_type !='steady':
            # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF MASS MATRIX - CHANGE TYPES TO INT64 FOR DoF > 1e09
            I_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
            J_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
            V_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.float64)
    else:
        V_convection=np.zeros(indices.shape[0],dtype=np.float64)
        if fem_solver.analysis_type !='steady':
            V_mass=np.zeros(indices.shape[0],dtype=np.float64)

    Flux = np.zeros((npoints*nvar,1), dtype=np.float64)

    mass = []

    for elem in range(nelem):

        # COMPUATE ALL LOCAL ELEMENTAL MATRICES (CONVECTION, MASS, INTERNAL TRACTION FORCES )
        I_conve_elem, J_conve_elem, V_conve_elem, f, \
        I_mass_elem, J_mass_elem, V_mass_elem = formulation.GetElementalMatrices(elem,
            function_spaces[0], mesh, material, fem_solver)

        if fem_solver.recompute_sparsity_pattern:
            raise ValueError("not implemented yet")
        else:
            if fem_solver.squeeze_sparsity_pattern:
                raise ValueError("not implemented yet")
            else:
                # SPARSE ASSEMBLY - CONVECTION MATRIX
                V_convection[data_global_indices[elem*local_capacity:(elem+1)*local_capacity]] \
                += V_conve_elem[data_local_indices[elem*local_capacity:(elem+1)*local_capacity]]

                if fem_solver.analysis_type != 'steady':
                    # SPARSE ASSEMBLY - MASS MATRIX
                    V_mass[data_global_indices[elem*local_capacity:(elem+1)*local_capacity]] \
                    += V_mass_elem[data_local_indices[elem*local_capacity:(elem+1)*local_capacity]]

        # INTERNAL FLUX ASSEMBLY
        for i in range(nodeperelem):
            F_idx = mesh.elements[elem,i]*nvar
            for iterator in range(nvar):
                Flux[F_idx+iterator] += f[i*nvar+iterator]
        #RHSAssemblyNative(Flux,f,elem,nvar,nodeperelem,mesh.elements)

    if fem_solver.recompute_sparsity_pattern:
        raise ValueError("not yet")
    else:
        convection = csr_matrix((V_convection,indices,indptr),
            shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])))

    # GET STORAGE/MEMORY DETAILS
    fem_solver.spmat = convection.data.nbytes/1024./1024.
    fem_solver.ijv = (indptr.nbytes + indices.nbytes + V_convection.nbytes)/1024./1024.

    if fem_solver.analysis_type != 'steady':
        mass = csr_matrix((V_mass,indices,indptr),
            shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])))

    # ASSEMBLY - EXTERNAL FLUX
    K_e, Flux_e = AssemblyRobinForces(fem_solver, function_spaces[0], mesh, material, boundary_condition)

    return convection, Flux, mass

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

#=============================================================================#

