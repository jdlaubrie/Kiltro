include("../src/Kiltro.jl")
using .Kiltro
using WriteVTK

#======================== MESH ===========================#
# Mesh input to the problem, this case a hexaedra of l=1.0.
points = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
elements = [1 2 4; 2 3 4]
edges = [1 2; 2 3; 3 4; 4 1]

nnode = size(points,1)
ndim = size(points,2)

DirichletBoundary1 = falses(nnode)
DirichletBoundary2 = falses(nnode)
DirichletBoundary1[edges[1,:]] .= true
DirichletBoundary2[edges[4,:]] .= true
NeumannBoundary1 = falses(nnode)
NeumannBoundary2 = falses(nnode)
NeumannBoundary1[edges[2,:]] .= true
NeumannBoundary2[edges[3,:]] .= true

#====================== MATERIAL =========================#
#material parameters for neo-Hooke hyperelastic model
mu = 15.0e-3
kappa = 50.0*mu

#===================== FORMULATION =======================#
#sets the function space for the displacement problem
nvar = 2
x, w = QuadratureRule(2,-1.0,1.0)
AllGauss, Basis, Jm = FunctionSpace(x,w)

#================= BOUNDARY CONDITIONS ===================#
#sets the boundary conditions, with symmetries (dirichlet) and the external force (neumann)
dirichlet_flags = zeros(nnode,2).+NaN
dirichlet_flags[DirichletBoundary1,2] .= 0.0
dirichlet_flags[DirichletBoundary2,1] .= 0.0

neumann_flags = zeros(nnode,2).+NaN
neumann_flags[NeumannBoundary1,1] .= 0.01
#neumann_flags[NeumannBoundary2,2] .= 0.01

#======================= SOLVER ==========================#
TotalDisp = Solve(AllGauss,Basis,Jm,points,elements,mu,kappa,dirichlet_flags,neumann_flags,ndim,nvar)

#====================== SOLUTION =========================#

x_points = points[:,1]
y_points = points[:,2]

cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, [1, 2, 4]),
         MeshCell(VTKCellTypes.VTK_TRIANGLE, [2, 3, 4])]

disp_x = zeros(length(x_points))
disp_y = zeros(length(y_points))

disp_x = TotalDisp[:,1,end]
disp_y = TotalDisp[:,2,end]
disp = (disp_x, disp_y)

vtkfile = vtk_grid("my_dataset", x_points, y_points, cells)
vtkfile["displacement", VTKPointData()] = disp
outfiles = vtk_save(vtkfile)

