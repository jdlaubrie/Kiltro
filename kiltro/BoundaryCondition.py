import numpy as np

class BoundaryCondition(object):

    def __init__(self,):
        self.initial_field = None
        self.columns_in = None
        self.columns_out = None
        self.applied_dirichlet = None
        self.dirichlet_flags = None
        self.neumann_flags = None

    def GetDirichletBoundaryConditions(self, npoints):
        # stock information about dirichlet conditions
        flat_dirich = self.dirichlet_flags.ravel()
        self.applied_dirichlet = self.dirichlet_flags[~np.isnan(flat_dirich)]
        self.columns_out = np.arange(self.dirichlet_flags.size)[~np.isnan(self.dirichlet_flags)]
        self.columns_in = np.delete(np.arange(0,npoints),self.columns_out)

    def ComputeNeumannFlux(self, npoints):
        # compute Neumann forces
        F = np.zeros((npoints,1))
        flat_neu = self.neumann_flags.ravel()
        to_apply = np.arange(self.neumann_flags.size)[~np.isnan(flat_neu)]
        applied_neumann = flat_neu[~np.isnan(flat_neu)]
        F[to_apply,0] = applied_neumann

        return F

    def GetReducedMatrices(self, K, F, M=None):

        F_b = F[self.columns_in,0]

        K_b = K[self.columns_in,:][:,self.columns_in]
        
        if not M is None:
            M_b = M[self.columns_in,:][:,self.columns_in]
        else:
            M_b = np.array([])

        return K_b, F_b, M_b

    def ApplyDirichletGetReducedMatrices(self, K, F, AppliedDirichlet):

        # Apply dirichlet conditions in the problem
        nnz_cols = ~np.isclose(AppliedDirichlet,0.0)
        F[self.columns_in] = F[self.columns_in] + np.dot(K[self.columns_in,:]\
            [:,self.columns_out[nnz_cols]],AppliedDirichlet[nnz_cols])[:,None]

        return F
 
    def UpdateFixDoFs(self, AppliedDirichletInc, TotalSol):

        #TotalSol = np.zeros((fsize,1))
        TotalSol[self.columns_out] = AppliedDirichletInc

        return TotalSol

    def UpdateFreeDoFs(self, sol, TotalSol):

        TotalSol[self.columns_in] = sol
        
        return TotalSol

