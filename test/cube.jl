include("../src/Kiltro.jl")
using .Kiltro

#======================== MESH ===========================#
# Mesh input to the problem, this case a hexaedra of l=1.0.
points = [0.0 0.0 0.0; 1.0 0.0 0.0; 1.0 1.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.5; 1.0 0.0 0.5; 1.0 1.0 0.5; 0.0 1.0 0.5; 0.0 0.0 1.0; 1.0 0.0 1.0; 1.0 1.0 1.0; 0.0 1.0 1.0]
elements = [1 2 3 4 5 6 7 8; 5 6 7 8 9 10 11 12]
faces = [1 2 3 4; 1 5 6 2; 2 6 7 3; 3 7 8 4; 1 4 8 5; 5 9 10 6; 6 10 11 7; 7 11 12 8; 5 8 12 9; 9 12 11 10]

nnode = size(points,1)
ndim = size(points,2)

DirichletBoundary1 = falses(nnode)
DirichletBoundary2 = falses(nnode)
DirichletBoundary3 = falses(nnode)
DirichletBoundary4 = falses(nnode)
DirichletBoundary5 = falses(nnode)
DirichletBoundary1[faces[1,:]] .= true
DirichletBoundary2[faces[2,:]] .= true
DirichletBoundary3[faces[5,:]] .= true
DirichletBoundary4[faces[6,:]] .= true
DirichletBoundary5[faces[9,:]] .= true
NeumannBoundary = falses(nnode)
NeumannBoundary[faces[10,:]] .= true

#====================== MATERIAL =========================#
#material parameters for neo-Hooke hyperelastic model
mu = 15.0e-3
kappa = 100.0*mu

#===================== FORMULATION =======================#
#sets the function space for the displacement problem
nvar = 3
x, w = QuadratureRule(2,-1.0,1.0)
AllGauss, Basis, Jm = FunctionSpace(x,w)

#================= BOUNDARY CONDITIONS ===================#
#sets the boundary conditions, with symmetries (dirichlet) and the external force (neumann)
dirichlet_flags = zeros(nnode,3).+NaN
dirichlet_flags[DirichletBoundary1,3] .= 0.0
dirichlet_flags[DirichletBoundary2,2] .= 0.0
dirichlet_flags[DirichletBoundary3,1] .= 0.0
dirichlet_flags[DirichletBoundary4,2] .= 0.0
dirichlet_flags[DirichletBoundary5,1] .= 0.0

neumann_flags = zeros(nnode,3).+NaN
neumann_flags[NeumannBoundary,3] .= 0.005

#======================= SOLVER ==========================#
TotalDisp = Solve(AllGauss,Basis,Jm,points,elements,mu,kappa,dirichlet_flags,neumann_flags,ndim,nvar)

#====================== SOLUTION =========================#

println(TotalDisp)

