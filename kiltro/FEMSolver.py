import numpy as np
from kiltro import FunctionSpace, Assembly, AssemblyCharacteristicGalerkin
#from src import BoundaryCondition

class FEMSolver(object):

    def __init__(self, analysis_type="steady", number_of_increments=1):
        self.analysis_type = analysis_type
        self.number_of_increments = number_of_increments

#=============================================================================#
    def __checkdata__(self, mesh, material, boundary_condition):

        if boundary_condition.initial_field is None and self.analysis_type == "transient":
            raise ValueError("The problem is TRANSIENT but there are not initial conditions.")

        return

#=============================================================================#
    def Solve(self, formulation, mesh, material, boundary_condition, u):

        self.__checkdata__(mesh, material, boundary_condition)

        # solve temperature problem
        function_space = FunctionSpace(mesh.ndim)
        Residual = np.zeros((mesh.nnodes,1), dtype=np.float64)
        TotalSol = np.zeros((mesh.nnodes,self.number_of_increments))

        # apply dirichlet boundary conditions
        boundary_condition.GetDirichletBoundaryConditions(mesh.nnodes)
        # find pure neumann nodel forces
        NeumannFlux = boundary_condition.ComputeNeumannFlux(mesh.nnodes)
        # initial conditions for transient problems
        if self.analysis_type == "transient":
            TotalSol[:,0] = boundary_condition.initial_field

        # Assemble Convective-Diffusion matrix and Source
        if self.analysis_type == "steady":
            K, Source, _ = Assembly(mesh, material, formulation, function_space, boundary_condition, u)
        else:
            K, Source, M = Assembly(mesh, material, formulation, function_space, boundary_condition, u)

        if self.analysis_type != "steady":
            #verify time step-size
            delta_t = np.zeros((mesh.nelem), dtype=np.float64)
            for elem in range(mesh.nelem):
                # capture element coordinates
                ElemCoords = mesh.points[mesh.elements[elem]]
                ElemLength = np.linalg.norm(ElemCoords[1,:] - ElemCoords[0,:])
                delta_t[elem] = material.rhoc*ElemLength**2/(2.0*material.k)
            TimeIncrement = 0.25*np.min(delta_t)
            print("Time Increment={0:>10.5g}.".format(TimeIncrement))
            TotalSol = self.TransientSolver(K, M, NeumannFlux, Residual, TotalSol, 
                    mesh, material, formulation, boundary_condition, TimeIncrement, 
                    function_space, u)
        else:
            TotalSol = self.SteadySolver(K, NeumannFlux, Residual, TotalSol, 
                    mesh, boundary_condition)

        return TotalSol

#=============================================================================#
    def TransientSolver(self, K, M, NeumannFlux, Residual, TotalSol, 
        mesh, material, formulation, boundary_condition, TimeStep,
        function_space, u):
        # dT/dt + U*dT/dx - d/dx(k*dT/dx) + q = 0

        TimeIncrements = self.number_of_increments
        npoints = mesh.nnodes
        nelem = mesh.nelem

        invM = np.linalg.inv(M)
        K_u,Source_u = AssemblyCharacteristicGalerkin(mesh, material, formulation, function_space, boundary_condition, u)

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

            Ku_b,Fu_b,_ = boundary_condition.GetReducedMatrices(K_u, Source_u)
            dTdt2 = np.dot(Ku_b,TotalSol[boundary_condition.columns_in,Increment]) + Fu_b

            TotalSol[boundary_condition.columns_in,Increment+1] += \
                    TotalSol[boundary_condition.columns_in,Increment] - \
                    TimeStep*np.dot(invM_b,dTdt) + 0.5*TimeStep*TimeStep*np.dot(invM_b,dTdt2)
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


