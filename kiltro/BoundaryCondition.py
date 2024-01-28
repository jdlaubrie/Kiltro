import numpy as np
from warnings import warn

class BoundaryCondition(object):

    def __init__(self,):

        self.initial_field = None
        self.dirichlet_data_applied_at = 'node' # or 'faces'
        self.neumann_data_applied_at = 'node' # or 'faces'

        self.dirichlet_flags = None
        self.applied_dirichlet = None
        self.columns_in = None
        self.columns_out = None

        self.neumann_flags = None
        self.applied_neumann = None

        self.robin_convection_flags = None
        self.applied_robin_convection = 0.0
        self.ground_robin_convection = 0.0

#=============================================================================#
    def SetInitialConditions(self, func, *args, **kwargs):
        """
        Includes user defined Dirichlet data in this object
        """

        self.initial_field = func(*args, **kwargs)
        return self.initial_field

#=============================================================================#
    def SetDirichletCriteria(self, func, *args, **kwargs):
        """
        Includes user defined Dirichlet data in this object
        """

        self.dirichlet_flags = func(*args, **kwargs)
        return self.dirichlet_flags

#=============================================================================#
    def SetNeumannCriteria(self, func, *args, **kwargs):
        """
        Applies user defined Neumann data to self
        """

        tups = func(*args, **kwargs)

        # if "tups" is not a tuple and Neumann is applied at nodes
        if not isinstance(tups,tuple) and self.neumann_data_applied_at == "node":
            self.neumann_flags = tups
            return self.neumann_flags
        else:
            self.neumann_data_applied_at == "face"
            if len(tups) !=2:
                raise ValueError("User-defined Neumann criterion function {} "
                    "should return one flag and one data array".format(func.__name__))
            self.neumann_flags = tups[0]
            self.applied_neumann = tups[1]
            return tups

#=============================================================================#
    def SetRobinCriteria(self, func, *args, **kwargs):
        """Applies user defined Robin data to self, just working on surfaces
        """

        dics = func(*args, **kwargs)

        if isinstance(dics,dict):
            self.RobinSelector(dics)
        elif isinstance(dics,tuple):
            for idic in range(len(dics)):
                if isinstance(dics[idic],dict):
                    self.RobinSelector(dics[idic])
                else:
                    raise ValueError("User-defined Robin criterion function {} "
                        "should return dictionary or tuple(dict,dict,...)".format(func.__name__))
        else:
            raise ValueError("User-defined Robin criterion function {} "
                "should return dictionary or tuple".format(func.__name__))

        return dics

#=============================================================================#
    def RobinSelector(self, tups):
        if tups['type'] == 'Convection':
            self.robin_convection_flags = tups['flags']
            self.applied_robin_convection = tups['data']
        #elif tups['type'] == 'Resistance':
        #    self.robin_resistance_flags = tups['flags']
        #    self.applied_robin_resistance = tups['data']
        else:
            raise ValueError("Type force {} not understood or not available. "
                "Types are Pressure, Spring, SpringJoint and Dashpot.".format(tups['type']))

#=============================================================================#
    def GetDirichletBoundaryConditions(self, formulation, mesh, material=None, fem_solver=None):
        # stock information about dirichlet conditions

        nvar = formulation.nvar
        ndim = formulation.ndim
        self.columns_in, self.applied_dirichlet = [], []

        # IF DIRICHLET BOUNDARY CONDITIONS ARE APPLIED DIRECTLY AT NODES
        if self.dirichlet_flags is None:
            raise RuntimeError("Dirichlet boundary conditions are not set for the analysis")

        flat_dirich = self.dirichlet_flags.ravel()
        self.columns_out = np.arange(self.dirichlet_flags.size)[~np.isnan(flat_dirich)]
        self.applied_dirichlet = flat_dirich[~np.isnan(flat_dirich)]

        # GENERAL PROCEDURE - GET REDUCED MATRICES FOR FINAL SOLUTION
        self.columns_out = self.columns_out.astype(np.int64)
        self.columns_in = np.delete(np.arange(0,nvar*mesh.points.shape[0]),self.columns_out)

        if self.columns_in.shape[0] == 0:
            warn("Dirichlet boundary conditions have been applied on the entire mesh")
        if self.columns_in.shape[0] == 0:
            warn("No Dirichlet boundary conditions have been applied. The system is unconstrained")

#=============================================================================#
    def ComputeNeumannFluxes(self, mesh, material):
        #def ComputeNeumannFluxes(self, mesh, material, function_spaces, compute_boundary_fluxes=True, compute_body_fluxes=False):
        """Compute/assemble traction and body forces"""

        if self.neumann_flags is None:
            return np.zeros((mesh.points.shape[0]*material.nvar,1),dtype=np.float64)

        nvar = material.nvar

        F = np.zeros((mesh.points.shape[0]*nvar,1))
        flat_neu = self.neumann_flags.ravel()
        to_apply = np.arange(self.neumann_flags.size)[~np.isnan(flat_neu)]
        applied_neumann = flat_neu[~np.isnan(flat_neu)]
        F[to_apply,0] = applied_neumann

        return F

#=============================================================================#
    def ComputeRobinForces(self, mesh, materials, function_spaces, fem_solver, K, F):
        """Compute/assemble traction and body forces"""

        from kiltro.Assembly import AssembleRobinForces
        if not self.robin_convection_flags is None:
            K_convection, F_convection = AssembleRobinForces(self, mesh,
                materials[0], function_spaces, fem_solver, Eulerx, 'pressure')
            K -= K_convection
            F -= F_convection[:,None]

        return K, F

#=============================================================================#
    def GetReducedMatrices(self, K, F, M=None):

        F_b = F[self.columns_in,0]
        K_b = K[self.columns_in,:][:,self.columns_in]

        if not M is None:
            M_b = M[self.columns_in,:][:,self.columns_in]
        else:
            M_b = np.array([])

        return K_b, F_b, M_b

#=============================================================================#
    def ApplyDirichletGetReducedMatrices(self, K, F, AppliedDirichlet):

        # Apply dirichlet conditions in the problem
        nnz_cols = ~np.isclose(AppliedDirichlet,0.0)
        F[self.columns_in] = F[self.columns_in] + np.dot(K[self.columns_in,:]\
            [:,self.columns_out[nnz_cols]],AppliedDirichlet[nnz_cols])[:,None]

        return F

#=============================================================================#
    def UpdateFixDoFs(self, AppliedDirichletInc, fsize, nvar):
        """Updates the geometry (DoFs) with incremental Dirichlet boundary conditions
            for fixed/constrained degrees of freedom only. Needs to be applied per time steps"""

        # GET TOTAL SOLUTION
        TotalSol = np.zeros((fsize,1))
        TotalSol[self.columns_out,0] = AppliedDirichletInc

        # RE-ORDER SOLUTION COMPONENTS
        dphi = TotalSol.reshape(int(TotalSol.shape[0]/nvar),nvar)

        return dphi

#=============================================================================#
    def UpdateFreeDoFs(self, sol, fsize, nvar):
        """Updates the geometry with iterative solutions of Newton-Raphson
            for free degrees of freedom only. Needs to be applied per time NR iteration"""

        # GET TOTAL SOLUTION
        TotalSol = np.zeros((fsize,1))
        TotalSol[self.columns_in,0] = sol

        # RE-ORDER SOLUTION COMPONENTS
        dphi = TotalSol.reshape(int(TotalSol.shape[0]/nvar),nvar)

        return dphi

