import gc
from time import time
import numpy as np

from kiltro import Mesh, Assemble #, AssembleCharacteristicGalerkin
from kiltro.PostProcess import *
#from src import BoundaryCondition

#=============================================================================#
#==========================  FINITE-ELEMENT SOLVER  ==========================#
#=============================================================================#
class FEMSolver(object):

    def __init__(self, analysis_type="steady", analysis_nature="linear",
        number_of_load_increments=1, print_incremental_log=False, 
        memory_store_frequency=1, recompute_sparsity_pattern=False,
        squeeze_sparsity_pattern=False):

        self.analysis_type = analysis_type
        self.analysis_nature = analysis_nature
        self.number_of_load_increments = number_of_load_increments
        self.save_frequency = int(memory_store_frequency)

        self.print_incremental_log = print_incremental_log

        self.recompute_sparsity_pattern = recompute_sparsity_pattern
        self.squeeze_sparsity_pattern = squeeze_sparsity_pattern
        self.is_sparsity_pattern_computed = False

#=============================================================================#
    def __checkdata__(self, material, boundary_condition, formulation, mesh,
        function_spaces, solver):
        """Checks the state of data for FEMSolver"""

        # INITIAL CHECKS
        #---------------------------------------------------------------------#
        if mesh is None:
            raise ValueError("No mesh detected for the analysis")
        elif not isinstance(mesh,Mesh):
            raise ValueError("mesh has to be an instance of Kiltro.Mesh")
        if boundary_condition is None:
            raise ValueError("No boundary conditions detected for the analysis")
        if material is None:
            raise ValueError("No material model chosen for the analysis")
        if formulation is None:
            raise ValueError("No variational formulation specified")

        # GET FUNCTION SPACES FROM THE FORMULATION
        if function_spaces is None:
            if formulation.function_spaces is None:
                raise ValueError("No interpolation functions specified")
            else:
                function_spaces = formulation.function_spaces

        # CHECK IF A SOLVER IS SPECIFIED
        if solver is None:
            solver = LinearSolver(linear_solver="direct", linear_solver_type="lapack")

        if boundary_condition.initial_field is None and self.analysis_type == "transient":
            raise ValueError("The problem is TRANSIENT but there are not initial conditions.")

        return function_spaces, solver

#=============================================================================#
    def __makeoutput__(self, mesh, TotalSol, formulation):

        post_process = PostProcess(formulation.ndim,formulation.nvar)
        post_process.SetMesh(mesh)
        post_process.SetSolution(TotalSol)
        return post_process

#=============================================================================#
    def Solve(self, formulation=None, mesh=None, material=None,
        boundary_condition=None, function_spaces=None, solver=None):
        """Main solution routine for FEMSolver """

        # CHECK DATA CONSISTENCY
        #---------------------------------------------------------------------#
        function_spaces, solver = self.__checkdata__(material, boundary_condition,
            formulation, mesh, function_spaces, solver)
        #---------------------------------------------------------------------#

        # PRINT INFO
        #---------------------------------------------------------------------#
        self.PrintPreAnalysisInfo(mesh, formulation)
        #---------------------------------------------------------------------#

        # COMPUTE SPARSITY PATTERN
        #---------------------------------------------------------------------------#
        self.ComputeSparsityFEM(mesh, formulation)
        #---------------------------------------------------------------------------#

        # solve temperature problem
        NodalFluxes = np.zeros((mesh.nnodes*formulation.nvar,1), dtype=np.float64)
        #Residual = np.zeros((mesh.nnodes*formulation.nvar,1), dtype=np.float64)

        # ALLOCATE FOR SOLUTION FIELDS
        if self.save_frequency == 1:
            TotalSol = np.zeros((mesh.nnodes,formulation.nvar,self.number_of_load_increments),dtype=np.float64)
        else:
            TotalSol = np.zeros((mesh.nnodes,formulation.nvar,
                int(self.number_of_increments/self.save_frequency)),dtype=np.float64)

        # PRE-ASSEMBLY
        print('Assembling the system and acquiring neccessary information for the analysis...')
        tAssembly=time()

        # apply dirichlet boundary conditions
        boundary_condition.GetDirichletBoundaryConditions(formulation, mesh, material, self)
        # find pure neumann nodel forces
        NeumannFluxes = boundary_condition.ComputeNeumannFluxes(mesh, material)
        # initial conditions for transient problems
        if self.analysis_type == "transient":
            TotalSol[:,:,0] = boundary_condition.initial_field

        # ADOPT A DIFFERENT PATH FOR INCREMENTAL LINEAR DIFFUSION
        if self.analysis_nature != "nonlinear":
            if self.analysis_type == "steady":
                # DISPATCH INCREMENTAL LINEAR DIFFUSION SOLVER
                TotalSol = self.SteadyLinearDiffusionSolver(function_spaces, formulation, mesh, material,
                    boundary_condition, solver, TotalSol, NeumannFluxes)
                return self.__makeoutput__(mesh, TotalSol, formulation)

        # ASSEMBLE DIFFUSION MATRIX AND FLUXES FOR THE FIRST TIME
        if self.analysis_type == "steady":
            K, Fluxes, _ = Assemble(self, function_spaces, formulation, mesh, material, boundary_condition)
        else:
            K, Fluxes, M = Assemble(self, function_spaces, formulation, mesh, material, boundary_condition)

        #if self.analysis_nature == 'nonlinear':
        #print('Finished all pre-processing stage. Time elapsed was', time()-tAssembly, 'seconds')
        #else:
        print('Finished the assembly stage. Time elapsed was', time()-tAssembly, 'seconds')

        if self.analysis_type != "steady":
            #verify time step-size
            raise ValueError("working on this")
            delta_t = np.zeros((mesh.nelem), dtype=np.float64)
            for elem in range(mesh.nelem):
                # capture element coordinates
                ElemCoords = mesh.points[mesh.elements[elem]]
                ElemLength = np.linalg.norm(ElemCoords[1,:] - ElemCoords[0,:])
                delta_t[elem] = material.rho*material.c_v*ElemLength**2/(2.0*material.k)
            TimeIncrement = 0.1*np.min(delta_t)
            print("Time Increment={0:>10.5g}.".format(TimeIncrement))
            TotalSol = self.TransientSolver(function_spaces, formulation, solver,
                    K, M, NeumannFluxes, NodalFluxes, mesh, TotalSol, material,
                    boundary_condition, TimeIncrement)
        else:
            TotalSol = self.SteadySolver(formulation, solver, K, NeumannFluxes,
                    NodalFluxes, mesh, TotalSol, boundary_condition)

        return self.__makeoutput__(mesh, TotalSol, formulation)

#=============================================================================#
    def SteadyLinearDiffusionSolver(self, function_spaces, formulation, omesh, material,
                boundary_condition, solver, TotalSol, NeumannFluxes):
        """An icremental linear diffusion solver.
            In this approach instead of solving the problem inside a non-linear routine,
            a somewhat explicit and more efficient way is adopted to avoid pre-assembly of the system
            of equations needed for non-linear analysis
        """

        # CREATE A COPY OF ORIGINAL MESH AS WE MODIFY IT
        mesh = deepcopy(omesh)

        LoadIncrement = self.number_of_load_increments
        LoadFactor = 1./LoadIncrement

        # COMPUTE INCREMENTAL FORCES - FOR ADAPTIVE LOAD STEPPING THIS NEEDS TO BE INSIDE THE LOOP
        NodalFluxes = -LoadFactor*NeumannFluxes
        AppliedDirichletInc = LoadFactor*boundary_condition.applied_dirichlet

        for Increment in range(LoadIncrement):

            t_assembly = time()

            # RESET EVERY TIME
            Residual = np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float64)

            # IF STRESSES ARE NOT TO BE CALCULATED - FOR LinearElastic and IncrementalLinearElastic
            # ASSEMBLE
            K = Assemble(self, function_spaces, formulation, mesh, material,
                    boundary_condition)[0]
            # NO NEED FOR LoadFactor HERE AS mesh.points IS NOT UPDATED
            Residual += NodalFluxes

            print('Finished assembling the system of equations. Time elapsed is', time() - t_assembly, 'seconds')
            # APPLY DIRICHLET BOUNDARY CONDITIONS & GET REDUCED MATRICES
            K_b, F_b = boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,AppliedDirichletInc)[:2]
            # SOLVE THE SYSTEM
            t_solver=time()
            sol = solver.Solve(K_b,F_b)
            t_solver = time()-t_solver

            dphi = boundary_condition.TotalComponentSol(sol, AppliedDirichletInc, 0, K.shape[0], formulation.nvar)

            # STORE TOTAL SOLUTION DATA
            TotalSol[:,:,Increment] += dphi

            if LoadIncrement > 1:
                print("Finished load increment "+str(Increment)+" for incrementally linear problem. Solver time is", t_solver)
            else:
                print("Finished load increment "+str(Increment)+" for linear problem. Solver time is", t_solver)
            gc.collect()

            # LOG REQUESTS
            if self.print_incremental_log:
                dumT = np.copy(TotalSol)
                dumT[:,:,Increment] = np.sum(dumT[:,:,:Increment+1],axis=2)
                self.LogSave(formulation, dumT, Increment)

            # STORE THE INFORMATION IF THE SOLVER BLOWS UP
            if Increment > 0:
                phi0 = TotalSol[:,:,Increment-1].ravel()
                phi = TotalSol[:,:,Increment].ravel()
                tol = 1e200 if Increment < 5 else 10.
                if np.isnan(norm(phi)) or np.abs(phi.max()/(phi0.max()+1e-14)) > tol:
                    print("Solver blew up! Norm of incremental solution is too large")
                    TotalSol = TotalSol[:,:,:Increment]
                    self.number_of_load_increments = Increment
                    break

        # ADD EACH INCREMENTAL CONTRIBUTION TO MAKE IT CONSISTENT WITH THE NONLINEAR ANALYSIS
        for i in range(TotalSol.shape[2]-1,0,-1):
            TotalSol[:,:,i] = np.sum(TotalSol[:,:,:i+1],axis=2)

        return TotalSol

#=============================================================================#
    def SteadySolver(self, formulation, solver, K, NeumannFluxes, NodalFluxes,
        mesh, TotalSol, boundary_condition):
        # d/dx(k*dT/dx) + q = 0

        npoints = mesh.nnodes
        nelem = mesh.nelem

        # Apply dirichlet conditions in the problem
        applied_dirichlet = boundary_condition.applied_dirichlet
        NodalFluxes = -boundary_condition.ApplyDirichletGetReducedMatrices(K, 
                    np.zeros_like(NodalFluxes), applied_dirichlet)
        NodalFluxes += -NeumannFluxes
        TotalSol[:,:,0] += boundary_condition.UpdateFixDoFs(applied_dirichlet, K.shape[0], formulation.nvar)

        # solve the reduced linear system
        K_b, F_b, _ = boundary_condition.GetReducedMatrices(K, NodalFluxes)
        sol = solver.Solve(K_b,F_b)
        TotalSol[:,:,0] += boundary_condition.UpdateFreeDoFs(sol, K.shape[0], formulation.nvar)

        return TotalSol

#=============================================================================#
    def TransientSolver(self, function_spaces, formulation, solver, K, M,
        NeumannFluxes, NodalFluxes, mesh, TotalSol, material, boundary_condition,
        TimeStep):
        # dT/dt + U*dT/dx - d/dx(k*dT/dx) + q = 0

        TimeIncrements = self.number_of_increments
        npoints = mesh.nnodes
        nelem = mesh.nelem

        invM = np.linalg.inv(M)

        # ASSEMBLE CONVECTION AND FLUX MATRICES FOR CHARACTERISTIC-GALERKIN
        #if formulation.fields == "advection_diffusion":
        #    K_u,Flux_u = AssembleCharacteristicGalerkin(function_spaces, formulation, mesh, material, boundary_condition)

        # apply boundary condition
        applied_dirichlet = boundary_condition.applied_dirichlet
        NodalFluxes = -boundary_condition.ApplyDirichletGetReducedMatrices(K, 
                    np.zeros_like(NodalFluxes), applied_dirichlet)
        NodalFluxes += -NeumannFluxes
        # explicit time integration. characteristic-Galerkin
        TotalSol[:,:,0] += boundary_condition.UpdateFixDoFs(applied_dirichlet, K.shape[0], formulation.nvar)
        TotalTime = 0.0
        for Increment in range(TimeIncrements-1):
            TotalSol[:,:,Increment+1] += boundary_condition.UpdateFixDoFs(applied_dirichlet, K.shape[0], formulation.nvar)

            # time-integration
            K_b,F_b,invM_b = boundary_condition.GetReducedMatrices(K, NodalFluxes, invM)
            dTdt = np.dot(K_b,TotalSol[boundary_condition.columns_in,0,Increment]) + F_b

            TotalSol[boundary_condition.columns_in,0,Increment+1] += \
                    TotalSol[boundary_condition.columns_in,0,Increment] - \
                    TimeStep*np.dot(invM_b,dTdt)

            # time-integration including characteristic-Galerkin
            #if formulation.fields == "advection_diffusion":
            #    Ku_b,Fu_b,_ = boundary_condition.GetReducedMatrices(K_u, Flux_u)
            #    dTdt2 = np.dot(Ku_b,TotalSol[boundary_condition.columns_in,0,Increment]) + Fu_b

            #    TotalSol[boundary_condition.columns_in,0,Increment+1] += 0.5*TimeStep*TimeStep*np.dot(invM_b,dTdt2)

            TotalTime += TimeStep

        print(TotalTime)
        return TotalSol[:,0,[0,1,int(TimeIncrements/3),int(2*TimeIncrements/3),-1]]

#=============================================================================#
    def PrintPreAnalysisInfo(self, mesh, formulation):

        print('Pre-processing the information. Getting paths, solution parameters, mesh info, interpolation info etc...')
        print('Number of nodes is',mesh.points.shape[0], 'number of DoFs is', mesh.points.shape[0]*formulation.nvar)
        print('Number of elements is', mesh.elements.shape[0])
#        if formulation.ndim==2:
#            print('Number of elements is', mesh.elements.shape[0], \
#                 'and number of boundary nodes is', np.unique(mesh.edges).shape[0])
#        elif formulation.ndim==3:
#            print('Number of elements is', mesh.elements.shape[0], \
#                 'and number of boundary nodes is', np.unique(mesh.faces).shape[0])

#=============================================================================#
    def ComputeSparsityFEM(self, mesh, formulation):

        if self.is_sparsity_pattern_computed is False:
            if self.recompute_sparsity_pattern is False:
                t_sp = time()
                from kiltro.FiniteElements.Assembly import ComputeSparsityPattern
                if self.squeeze_sparsity_pattern:
                    raise ValueError("Not implemented yet")
                else:
                    self.indices, self.indptr, self.data_local_indices, \
                        self.data_global_indices = ComputeSparsityPattern(mesh, formulation.nvar)
                self.is_sparsity_pattern_computed = True
                print("Computed sparsity pattern for the mesh. Time elapsed is {} seconds".format(time()-t_sp))
            else:
                self.indices, self.indptr, self.data_local_indices,\
                self.data_global_indices = np.array([0],dtype=np.int32), \
                np.array([0],dtype=np.int32), np.array([0],dtype=np.int32), \
                np.array([0],dtype=np.int32)

#=============================================================================#
#==================  SOLVER FOR LINEAR SYSTEM OF EQUATIONS  ==================#
#=============================================================================#
class LinearSolver(object):
    """Base class for all linear sparse direct and iterative solvers"""

    def __init__(self, linear_solver="direct", linear_solver_type="lapack"):

        self.is_sparse = True
        self.solver_type = linear_solver
        self.solver_subtype = linear_solver_type

        self.has_umfpack = True
        try:
            from scikits.umfpack import spsolve
        except (ImportError, AttributeError) as umfpack_error:
            self.has_umfpack = False

    def Solve(self, A,b):

        from scipy.sparse import issparse
        from scipy.sparse.linalg import spsolve

        if not issparse(A):
            raise ValueError("Linear system is not of sparse type")

        if A.shape == (0,0) and b.shape[0] == 0:
            warn("Empty linear system!!! Nothing to solve!!!")
            return np.copy(b)

        sol = spsolve(A,b)

        return sol

