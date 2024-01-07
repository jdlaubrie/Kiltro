import numpy as np
from src import FunctionSpace, Assembly, AssemblyU
#from src import BoundaryCondition

class FEMSolver(object):

    def __init__(self, analysis_type="steady", number_of_increments=1):
        self.analysis_type = analysis_type
        self.number_of_increments = number_of_increments

#=============================================================================#
    def Solve(self, mesh, boundary_condition, u, k, q, h, rhoc, A, T_inf):

        # solve temperature problem
        function_space = FunctionSpace()
        Residual = np.zeros((mesh.nnodes,1), dtype=np.float64)
        TotalSol = np.zeros((mesh.nnodes,self.number_of_increments))
        # apply dirichlet boundary conditions
        boundary_condition.GetDirichletBoundaryConditions(mesh.nnodes)
        # find pure neumann nodel forces
        NeumannFlux = boundary_condition.ComputeNeumannFlux(mesh.nnodes)
        # initial conditions
        TotalSol[:,0] = 200.0

        # Assemble Convective-Diffusion matrix and Source
        if self.analysis_type == "steady":
            K, Source, _ = Assembly(mesh, function_space, u, k, q, h, rhoc, A, T_inf)
        else:
            K, Source, M = Assembly(mesh, function_space, u, k, q, h, rhoc, A, T_inf)

        if self.analysis_type != "steady":
            #verify time step-size
            delta_t = np.zeros((mesh.nelem), dtype=np.float64)
            for elem in range(mesh.nelem):
                # capture element coordinates
                ElemCoords = mesh.points[mesh.elements[elem]]
                ElemLength = np.abs(ElemCoords[1] - ElemCoords[0])
                delta_t[elem] = rhoc*ElemLength**2/(2.0*k)
            TimeIncrement = 0.25*np.min(delta_t)
            print("Time Increment={0:>10.5g}.".format(TimeIncrement))
            TotalSol = self.TransientSolver(K, M, NeumannFlux, Residual, TotalSol, 
                    mesh, boundary_condition, TimeIncrement, 
                    function_space, u, k, q, h, A, T_inf)
        else:
            TotalSol = self.SteadySolver(K, NeumannFlux, Residual, TotalSol, 
                    mesh, boundary_condition)

        return TotalSol

#=============================================================================#
    def TransientSolver(self, K, M, NeumannFlux, Residual, TotalSol, 
        mesh, boundary_condition, TimeStep, function_space,
        u, k, q, h, A, T_inf):
        # dT/dt + U*dT/dx - d/dx(k*dT/dx) + q = 0

        TimeIncrements = self.number_of_increments
        npoints = mesh.nnodes
        nelem = mesh.nelem

        invM = np.linalg.inv(M)
        K_u,Source_u = AssemblyU(mesh, function_space, u, k, q, h, A, T_inf)

        # apply boundary condition
        applied_dirichlet = boundary_condition.applied_dirichlet
        Residual = -boundary_condition.ApplyDirichletGetReducedMatrices(K, 
                    np.zeros_like(Residual), applied_dirichlet)
        Residual += -NeumannFlux
        # explicit time integration. characteristic-Galerkin
        TotalSol[:,0] = boundary_condition.UpdateFixDoFs(applied_dirichlet, TotalSol[:,0])
        TotalTime = 0.0
        for Increment in range(TimeIncrements-1):
            TotalSol[:,Increment+1] = boundary_condition.UpdateFixDoFs(applied_dirichlet, 
                    TotalSol[:,Increment+1])

            # time-integration with chracteristic-Galerkin
            K_b,F_b,invM_b = boundary_condition.GetReducedMatrices(K, Residual, invM)
            dTdt = np.dot(K_b,TotalSol[boundary_condition.columns_in,Increment]) + F_b
            TotalSol[boundary_condition.columns_in,Increment+1] += \
                    TotalSol[boundary_condition.columns_in,Increment] - \
                    TimeStep*np.dot(invM_b,dTdt)
            TotalTime += TimeStep

        print(TotalTime)
        return TotalSol[:,[0,1,int(TimeIncrements/3),int(2*TimeIncrements/3),-1]]

#=============================================================================#
    def SteadySolver(self, K, NeumannFlux, Residual, TotalSol, 
            mesh, boundary_condition):
        # d/dx(k*dT/dx) + q = 0

        npoints = mesh.nnodes
        nelem = mesh.nelem

        # Apply dirichlet conditions in the problem
        applied_dirichlet = boundary_condition.applied_dirichlet
        Residual = -boundary_condition.ApplyDirichletGetReducedMatrices(K, 
                    np.zeros_like(Residual), applied_dirichlet)
        Residual += -NeumannFlux
        TotalSol[:,0] = boundary_condition.UpdateFixDoFs(applied_dirichlet, TotalSol[:,0])

        # solve the reduced linear system
        K_b, F_b, _ = boundary_condition.GetReducedMatrices(K, Residual)
        K_inv = np.linalg.inv(K_b)
        sol = np.dot(K_inv,F_b)
        TotalSol[:,0] = boundary_condition.UpdateFreeDoFs(sol, TotalSol[:,0])

        return TotalSol


