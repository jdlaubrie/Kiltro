module Kiltro

using LinearAlgebra
using Printf

export QuadratureRule, FunctionSpace, Solve

include("Utils.jl")

#======================================================================#

function NodeArrangementHex(C)
  """It manages the order of nodes and faces in an Hexaedral element."""
  linear_bases_idx = [1,(C+2),(C+2)^2,(C+2)^2-(C+1)]
  quad_aranger = [linear_bases_idx; delete1d(collect(range(1,stop=(C+2)^2)),linear_bases_idx)]
  element_numbering = copy(quad_aranger)
  for i in 0:C
    faces_z = quad_aranger.+maximum(element_numbering)
    element_numbering = [element_numbering; faces_z]
    if i==C
      element_numbering = element_numbering[.~in1d(element_numbering,faces_z[1:4])]
      element_numbering = [element_numbering[1:4];faces_z[1:4];element_numbering[5:end]]
    end
  end
  traversed_edge_numbering_hex = nothing

  # GET FACE NUMBERING ORDER FROM TETRAHEDRAL ELEMENT
  face_0,face_1,face_2,face_3 = [],[],[],[]
  if C==0
    face_0 = [0,1,2,3]  # constant Z =-1 plane
    face_1 = [4,5,6,7]  # constant Z = 1 plane
    face_2 = [0,1,5,4]  # constant Y =-1 plane
    face_3 = [3,2,6,7]  # constant Y = 1 plane
    face_4 = [0,3,7,4]  # constant X =-1 plane
    face_5 = [1,2,6,5]  # constant X = 1 plane

    face_numbering = [face_0; face_1; face_2; face_3; face_4; face_5] .+ 1
    face_numbering = transpose(reshape(face_numbering,4,6))
  else
    # THIS IS A FLOATING POINT BASED ALGORITHM
    error("Polynomial degreee not implemented yet")
  end

  return face_numbering, traversed_edge_numbering_hex, element_numbering
end

function Material(mu,kappa,F)
  """It computes the stress and elasticity of the material from the deformation gradient"""
  I = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
  J = det(F)
  b = F*transpose(F)
  trb = tr(b)

  energy = 0.5*mu*(trb -3.0)

  stress = mu*J^(-5.0/3.0)*(b - 1.0/3.0*trb*I) + kappa*(J-1.0)*I

  II_ijkl = einsum("ijkl",I,I)
  II_ikjl = einsum("ikjl",I,I)
  II_iljk = einsum("iljk",I,I)
  bI_ijkl = einsum("ijkl",b,I)
  Ib_ijkl = einsum("ijkl",I,b)
  elasticity = 2*mu*J^(-5.0/3.0)*(1.0/9.0*trb*II_ijkl - 1.0/3.0*(bI_ijkl + Ib_ijkl) + 1.0/6.0*trb*(II_ikjl + II_iljk) )
  elasticity += kappa*((2.0*J-1.0)*II_ijkl - (J-1.0)*(II_ikjl + II_iljk))
  H_Voigt = Voigt(elasticity)

  return stress, H_Voigt
end

function QuadratureRule(N,a,b)
  """Return integration points and weights from Gauss-Legendre polynomials.
     This is a gaussian quadrature rule."""
  deps = eps(1.0)
  x = zeros(N)
  w = zeros(N)
  M = div((N+1),2)
  xm = 0.5*(b+a)
  xl = 0.5*(b-a)
  for i in 1:M
    z = cos(pi*(i-0.25)/(N+0.5))
    z1 = 2.0
    while abs(z-z1) > deps
      global pp
      p1 = 1.0
      p2 = 0.0
      for j in 1:N
        p3 = p2
        p2 = p1
        p1 = ((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j
      end
      pp = N*(z*p1-p2)/(z*z-1.0)
      z1 = z
      z = z1-p1/pp
    end
    x[i] = xm-xl*z
    x[N+1-i] = xm+xl*z
    w[i] = 2.0*xl/((1.0-z*z)*pp*pp)
    w[N+1-i] = w[i]
  end
  if size(x,1) == size(w,1)
    N = size(x,1)
    x_flatten = zeros(Int(N*N*N),3)
    w_flatten = zeros(Int(N*N*N))
    counter = 1
    for i in 1:N
      for j in 1:N
        for k in 1:N
          w_flatten[counter] = w[i]*w[j]*w[k]
          x_flatten[counter,1] = x[i]
          x_flatten[counter,2] = x[j]
          x_flatten[counter,3] = x[k]
          counter += 1
        end
      end
    end
  end
  return x_flatten, w_flatten
end

function OneDLagrange(C,xi)
  """Returns a Lagrange interpolation (Base) and its derivatives given the 
     interpolation order C and the point xi."""
  n = C+2
  ranger = collect(range(0,stop=(n-1)))
  eps = collect(range(-1.0,1.0,length=n))

  A = zeros(n,n)
  A[:,1] .= 1.0
  for i in 2:n
    A[:,i] = eps.^(i-1)
  end

  RHS = zeros(n,n)
  for i in 1:n
    RHS[i,i] = 1
  end
  coeff = inv(A)*RHS
  xis = ones(n).*xi.^ranger
  N = transpose(coeff)*xis
  dN = transpose(coeff[2:end,:]) * xis[1:end-1].*(1 .+ ranger[1:end-1])

  return N, dN, eps
end

function HexLagrange(C,zeta,eta,beta)
  """Computes C order Lagrangian bases with equally spaced points
     from 1D Lagrange interpolation"""

  Neta = zeros(C+2); Nzeta = zeros(C+2); Nbeta = zeros(C+2)

  Nzeta .= OneDLagrange(C,zeta)[1]
  Neta .=  OneDLagrange(C,eta)[1]
  Nbeta .=  OneDLagrange(C,beta)[1]

  # Ternsorial product
  node_arranger = NodeArrangementHex(C)[3]
  Bases = zeros((C+2)*(C+2)*(C+2))
  for i in 1:(C+2)
    for j in 1:(C+2)
      for k in 1:(C+2)
        Bases[(i-1)*(C+2)*(C+2)+(j-1)*(C+2)+k] = Nbeta[i]*Neta[j]*Nzeta[k]
      end
    end
  end
  Bases = Bases[(node_arranger)]

  return Bases
end


function HexGradLagrange(C,zeta,eta,beta)
  """Computes gradient of C order Lagrangian bases with equally spaced points
     from 1D Lagrange interpolation"""

  gBases = zeros((C+2)^3,3)
  Nzeta = zeros(C+2); Neta = zeros(C+2); Nbeta = zeros(C+2)
  gNzeta = zeros(C+2); gNeta = zeros(C+2);  gNbeta = zeros(C+2)
  # Compute each from one-dimensional bases
  Nzeta = OneDLagrange(C,zeta)[1]
  Neta = OneDLagrange(C,eta)[1]
  Nbeta = OneDLagrange(C,beta)[1]
  gNzeta = OneDLagrange(C,zeta)[2]
  gNeta = OneDLagrange(C,eta)[2]
  gNbeta = OneDLagrange(C,beta)[2]

  # Ternsorial product
  node_arranger = NodeArrangementHex(C)[3]
  g0 = zeros((C+2)*(C+2)*(C+2))
  g1 = zeros((C+2)*(C+2)*(C+2))
  g2 = zeros((C+2)*(C+2)*(C+2))
  for i in 1:(C+2)
    for j in 1:(C+2)
      for k in 1:(C+2)
        g0[(i-1)*(C+2)*(C+2)+(j-1)*(C+2)+k] = Nbeta[i]*Neta[j]*gNzeta[k]
        g1[(i-1)*(C+2)*(C+2)+(j-1)*(C+2)+k] = Nbeta[i]*gNeta[j]*Nzeta[k]
        g2[(i-1)*(C+2)*(C+2)+(j-1)*(C+2)+k] = gNbeta[i]*Neta[j]*Nzeta[k]
      end
    end
  end
  gBases[:,1] = g0[node_arranger]
  gBases[:,2] = g1[node_arranger]
  gBases[:,3] = g2[node_arranger]

  return gBases
end

function FunctionSpace(z,w)
  """Collects the Bases and its gradient from quadrature rule to build
     the function space for the element."""
  C = 0
  ndim = 3
  ns = Int((C+2)^ndim)
  Basis = zeros(ns,size(w,1))
  gBasisx = zeros(ns,size(w,1))
  gBasisy = zeros(ns,size(w,1))
  gBasisz = zeros(ns,size(w,1))

  for i in 1:size(w,1)
    ndummy = HexLagrange(C,z[i,1],z[i,2],z[i,3])
    dummy = HexGradLagrange(C,z[i,1],z[i,2],z[i,3])
    Basis[:,i] = ndummy
    gBasisx[:,i] = dummy[:,1]
    gBasisy[:,i] = dummy[:,2]
    gBasisz[:,i] = dummy[:,3]
  end
  Jm = zeros(ndim,size(Basis,1),size(w,1))
  AllGauss = zeros(size(w,1))
  for counter in 1:size(w,1)
    # GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
    Jm[1,:,counter] = gBasisx[:,counter]
    Jm[2,:,counter] = gBasisy[:,counter]
    Jm[3,:,counter] = gBasisz[:,counter]

    AllGauss[counter] = w[counter]
  end
  return AllGauss, Basis, Jm
end

function KinematicMeasures(AllGauss,Jm,LagrangeElemCoords,EulerElemCoords)
  """Given the function space, Lagrangian and Eulerian coordinates returns
     the kinematics measures (deformation measures) such as the deformation gradient."""

  #ParentGradientX = np.einsum('ijk,jl->kil', Jm, LagrangeElemCoords)
  ParentGradientX = zeros(size(Jm,3),size(Jm,1),size(LagrangeElemCoords,2))
  for i in 1:size(Jm,1)
    for j in 1:size(Jm,2)
      for k in 1:size(Jm,3)
        for l in 1:size(LagrangeElemCoords,2)
          ParentGradientX[k,i,l] += Jm[i,j,k]*LagrangeElemCoords[j,l]
        end
      end
    end
  end
  #MaterialGradient = np.einsum('ijk,kli->ijl', inv(ParentGradientX), Jm)
  invParentGradientX = zeros(size(ParentGradientX,1),size(ParentGradientX,2),size(ParentGradientX,3))
  for i in 1:size(ParentGradientX,1)
    invParentGradientX[i,:,:] = inv(ParentGradientX[i,:,:])
  end
  MaterialGradient = zeros(size(ParentGradientX,1),size(ParentGradientX,2),size(Jm,2))
  for i in 1:size(invParentGradientX,1)
    for j in 1:size(invParentGradientX,2)
      for k in 1:size(Jm,1)
        for l in 1:size(Jm,2)
          MaterialGradient[i,j,l] += invParentGradientX[i,j,k]*Jm[k,l,i]
        end
      end
    end
  end
  #F = np.einsum('ij,kli->kjl', EulerElemCoords, MaterialGradient)
  F = zeros(size(MaterialGradient,1),size(EulerElemCoords,2),size(MaterialGradient,2))
  for i in 1:size(EulerElemCoords,1)
    for j in 1:size(EulerElemCoords,2)
      for k in 1:size(MaterialGradient,1)
        for l in 1:size(MaterialGradient,2)
          F[k,j,l] += EulerElemCoords[i,j]*MaterialGradient[k,l,i]
        end
      end
    end
  end

  #ParentGradientx = np.einsum('ijk,jl->kil',Jm, EulerElemCoords)
  ParentGradientx = zeros(size(Jm,3),size(Jm,1),size(EulerElemCoords,2))
  for i in 1:size(Jm,1)
    for j in 1:size(Jm,2)
      for k in 1:size(Jm,3)
        for l in 1:size(EulerElemCoords,2)
          ParentGradientx[k,i,l] += Jm[i,j,k]*EulerElemCoords[j,l]
        end
      end
    end
  end
  #SpatialGradient = np.einsum('ijk,kli->ilj',inv(ParentGradientx), Jm)
  invParentGradientx = zeros(size(ParentGradientx,1),size(ParentGradientx,2),size(ParentGradientx,3))
  for i in 1:size(ParentGradientx,1)
    invParentGradientx[i,:,:] = inv(ParentGradientx[i,:,:])
  end
  SpatialGradient = zeros(size(ParentGradientx,1),size(Jm,2),size(ParentGradientx,2))
  for i in 1:size(invParentGradientx,1)
    for j in 1:size(invParentGradientx,2)
      for k in 1:size(Jm,1)
        for l in 1:size(Jm,2)
          SpatialGradient[i,l,j] += invParentGradientx[i,j,k]*Jm[k,l,i]
        end
      end
    end
  end
  #detJ = np.einsum('i,i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)),np.abs(StrainTensors['J']))
  detJ = zeros(size(AllGauss,1))
  for i in 1:size(AllGauss,1)
    detParentGradientX = abs(det(ParentGradientX[i,:,:]))
    J = det(F[i,:,:])
    detJ[i] = AllGauss[i]*detParentGradientX*J
  end
  return SpatialGradient,F,detJ
end

function FillConstitutiveB(SpatialGradient,ndim,nvar,VoigtSize)
  """It fills the B-matrix (or N-gradient) to multiply the the elasticity tensor for
     its integration in the element."""

  nodeperelem = size(SpatialGradient,2)
  B = zeros(nodeperelem*nvar,VoigtSize)

  for node in 1:nodeperelem
    B[nvar*node-2,1] = SpatialGradient[1,node]
    B[nvar*node-1,2] = SpatialGradient[2,node]
    B[nvar*node,3] = SpatialGradient[3,node]

    B[nvar*node-1,6] = SpatialGradient[3,node]
    B[nvar*node,6] = SpatialGradient[2,node]

    B[nvar*node-2,5] = SpatialGradient[3,node]
    B[nvar*node,5] = SpatialGradient[1,node]

    B[nvar*node-2,4] = SpatialGradient[2,node]
    B[nvar*node-1,4] = SpatialGradient[1,node]
  end
  return B
end

function FillGeometricB(SpatialGradient,CauchyStress,ndim,nvar)
  """It fills the geometric B-matrix to be multiply with the stress
     to get the geometric stiffness matrix from integration."""

  nodeperelem = size(SpatialGradient,2)
  B = zeros(nvar*nodeperelem,ndim*ndim)
  S = zeros(ndim*ndim,ndim*ndim)

  for node in 1:nodeperelem
    B[nvar*node-2,1] = SpatialGradient[1,node]
    B[nvar*node-2,2] = SpatialGradient[2,node]
    B[nvar*node-2,3] = SpatialGradient[3,node]
    B[nvar*node-1,4] = SpatialGradient[1,node]
    B[nvar*node-1,5] = SpatialGradient[2,node]
    B[nvar*node-1,6] = SpatialGradient[3,node]
    B[nvar*node,7] = SpatialGradient[1,node]
    B[nvar*node,8] = SpatialGradient[2,node]
    B[nvar*node,9] = SpatialGradient[3,node]
  end

  S[1,1] = CauchyStress[1,1]
  S[1,2] = CauchyStress[1,2]
  S[1,3] = CauchyStress[1,3]
  S[2,1] = CauchyStress[2,1]
  S[2,2] = CauchyStress[2,2]
  S[2,3] = CauchyStress[2,3]
  S[3,1] = CauchyStress[3,1]
  S[3,2] = CauchyStress[3,2]
  S[3,3] = CauchyStress[3,3]

  S[4,4] = CauchyStress[1,1]
  S[4,5] = CauchyStress[1,2]
  S[4,6] = CauchyStress[1,3]
  S[5,4] = CauchyStress[2,1]
  S[5,5] = CauchyStress[2,2]
  S[5,6] = CauchyStress[2,3]
  S[6,4] = CauchyStress[3,1]
  S[6,5] = CauchyStress[3,2]
  S[6,6] = CauchyStress[3,3]

  S[7,7] = CauchyStress[1,1]
  S[7,8] = CauchyStress[1,2]
  S[7,9] = CauchyStress[1,3]
  S[8,7] = CauchyStress[2,1]
  S[8,8] = CauchyStress[2,2]
  S[8,9] = CauchyStress[2,3]
  S[9,7] = CauchyStress[3,1]
  S[9,8] = CauchyStress[3,2]
  S[9,9] = CauchyStress[3,3]

  return B,S
end

function GetTotalTraction(CauchyStress)
  """Returns the stress as a vector into Voigt notation."""
  TotalTraction = zeros(6)
  TotalTraction[1] = CauchyStress[1,1]
  TotalTraction[2] = CauchyStress[2,2]
  TotalTraction[3] = CauchyStress[3,3]
  TotalTraction[4] = CauchyStress[1,2]
  TotalTraction[5] = CauchyStress[1,3]
  TotalTraction[6] = CauchyStress[2,3]
  return TotalTraction
end

function ConstitutiveStiffnessIntegrand(SpatialGradient, CauchyStress, H_Voigt)
  """Applies to displacement based formulation, to integrate the constitutive stiffness"""
  ndim = 3
  nvar = 3
  VoigtSize = size(H_Voigt,1)
  # SpatialGradient(ndim x nodesperelem) and B(nodesperelem*nvar,H_VoigtSize)
  SpatialGradient = transpose(SpatialGradient)
  B = FillConstitutiveB(SpatialGradient,ndim,nvar,VoigtSize)

  BDB = B*(H_Voigt*(transpose(B)))

  t=zeros(size(B,1))
  TotalTraction = GetTotalTraction(CauchyStress)
  t = B*TotalTraction
  
  return BDB, t
end

function GeometricStiffnessIntegrand(SpatialGradient, CauchyStress,ndim,nvar)
    """Applies to displacement based, displacement potential based and all mixed
    formulations that involve static condensation"""

  SpatialGradient = transpose(SpatialGradient)

  B,S = FillGeometricB(SpatialGradient,CauchyStress,ndim,nvar)
  BDB = (B*S)*transpose(B)

  return BDB
end

function Assemble(AllGauss,Basis,Jm,points,elements,mu,kappa,Eulerx,ndim,nvar)
  """It assembles the stiffness and internal forces by element."""

  nodeperelem = size(Basis,1)
  ranger = collect(1:1:nodeperelem)

  stiffness_global = zeros(size(points,1)*nvar,size(points,1)*nvar)
  traction = zeros(size(points,1)*nvar)

  # loop on elements
  for elem in 1:size(elements,1)
    LagrangeElemCoords = points[elements[elem,:],:]
    EulerElemCoords = Eulerx[elements[elem,:],:]

    stiffness = zeros(nodeperelem*nvar,nodeperelem*nvar)
    tractionforce = zeros(nodeperelem*nvar)
    SpatialGradient,F,detJ = KinematicMeasures(AllGauss,Jm,LagrangeElemCoords,EulerElemCoords)

    for counter in 1:size(AllGauss,1)
      # COMPUTE CAUCHY STRESS AND HESSIAN FROM MATERIAL AT GAUSS POINT
      CauchyStress,H_Voigt = Material(mu,kappa,F[counter,:,:])
      # COMPUTE THE TANGENT STIFFNESS MATRIX
      BDB_1, t = ConstitutiveStiffnessIntegrand(SpatialGradient[counter,:,:], CauchyStress, H_Voigt)
      # COMPUTE GEOMETRIC STIFFNESS MATRIX
      BDB_1 += GeometricStiffnessIntegrand(SpatialGradient[counter,:,:],CauchyStress,ndim,nvar)
      # INTEGRATE TRACTION FORCE
      tractionforce += t*detJ[counter]
      # INTEGRATE STIFFNESS
      stiffness += BDB_1*detJ[counter]
    end

    for i in 0:(nvar-1)
      for j in 0:(nvar-1)
        stiffness_global[nvar.*elements[elem,:].-i,nvar.*elements[elem,:].-j] += stiffness[nvar.*ranger.-i,nvar.*ranger.-j]
      end
      traction[nvar.*elements[elem,:].-i] += tractionforce[nvar.*ranger.-i]
    end
  end

  return stiffness_global,traction
end

function NewtonRaphson(Increment,AllGauss,Basis,Jm,points,elements,mu,kappa,
                       norm_residual,NormForces,K,Residual,NodalForces,Eulerx,columns_in,ndim,nvar)
  """This is the Newton-Raphson loop of iterations to find a solution"""

  Tolerance = 1.0e-6
  Iter = 1

  while norm_residual > Tolerance

    # GET REDUCED FORCE VECTOR
    F_b = Residual[columns_in]
    # GET REDUCED STIFFNESS MATRIX
    K_b = K[columns_in,columns_in]

    sol = -inv(K_b)*F_b

    # UPDATE THE FREE DOFs
    # GET TOTAL SOLUTION
    TotalSol = zeros(size(K,1),1)
    TotalSol[columns_in] = sol
    # RE-ORDER SOLUTION COMPONENTS
    dU = transpose(reshape(TotalSol,nvar,Int(size(TotalSol,1)/nvar)))

    # UPDATE THE EULERIAN COMPONENTS, UPDATE THE GEOMETRY
    Eulerx += dU[:,1:ndim]

    # RE-ASSEMBLE - COMPUTE STIFFNESS AND INTERNAL TRACTION FORCES
    K,TractionForces = Assemble(AllGauss,Basis,Jm,points,elements,mu,kappa,Eulerx,ndim,nvar)

    # FIND THE RESIDUAL
    Residual[columns_in] = TractionForces[columns_in] - NodalForces[columns_in]

    # SAVE THE NORM
    abs_norm_residual = norm(Residual[columns_in])
    if Iter==1
      NormForces = norm(Residual[columns_in])
    end
    norm_residual = abs(norm(Residual[columns_in])/NormForces)

    @printf("Iteration %d for increment %d. Residual (abs) %e\t Residual (rel) %e\n",Iter,Increment,abs_norm_residual,norm_residual)
    #{0:>16.7g}

    # BREAK BASED ON RELATIVE NORM
    if abs(abs_norm_residual) < Tolerance
      break
    end

    # BREAK BASED ON INCREMENTAL SOLUTION - KEEP IT AFTER UPDATE
    if norm(dU) <=  Tolerance*0.1
      @printf("Incremental solution within tolerance i.e. norm(dU): %f\n",norm(dU))
      break
    end

    # UPDATE ITERATION NUMBER
    Iter += 1

    # BREAK BASED ON ITERATION LIMIT NUMBER
    if Iter==10
      break
    end

  end
  return Eulerx, K, Residual
end

function StaticSolver(LoadIncrements,AllGauss,Basis,Jm,points,elements,mu,kappa,
                      applied_dirichlet,NeumannForces,
                      K,Residual,NodalForces,TotalDisp,Eulerx,
                      columns_in,columns_out,ndim,nvar)
  """Static solver"""

  NormForces = 0.0
  LoadFactor = 1.0/LoadIncrements
  AppliedDirichletInc = zeros(size(applied_dirichlet,1))
  LoadFactorInc = 0.0

  for Increment in 1:LoadIncrements

    LoadFactorInc += LoadFactor

    DeltaF = LoadFactor*NeumannForces
    NodalForces += DeltaF
    # OBRTAIN INCREMENTAL RESIDUAL - CONTRIBUTION FROM BOTH NEUMANN AND DIRICHLET
    F = zeros(size(Residual))
    nnz_cols = .~isapprox.(applied_dirichlet,0.0)
    if size(columns_out[nnz_cols],1)==0
      F[columns_in] = F[columns_in]
    else
      F[columns_in] = F[columns_in] - (K[columns_in,columns_out[nnz_cols]]*applied_dirichlet[nnz_cols])
    end
    Residual = -F
    Residual -= DeltaF
    AppliedDirichletInc = LoadFactor*applied_dirichlet

    # LET NORM OF THE FIRST RESIDUAL BE THE NORM WITH RESPECT TO WHICH WE
    # HAVE TO CHECK THE CONVERGENCE OF NEWTON RAPHSON. TYPICALLY THIS IS
    # NORM OF NODAL FORCES
    if Increment==1
      NormForces = norm(Residual)
      # AVOID DIVISION BY ZERO
      if isapprox(NormForces,0.0)
        NormForces = 1e-14
      end
    end

    norm_residual = norm(Residual)/NormForces

    # APPLY INCREMENTAL DIRICHLET PER LOAD STEP (THIS IS INCREMENTAL NOT ACCUMULATIVE)
    # GET TOTAL SOLUTION
    TotalSol = zeros(size(K,1),1)
    TotalSol[columns_out] = AppliedDirichletInc
    # RE-ORDER SOLUTION COMPONENTS
    IncDirichlet = transpose(reshape(TotalSol,nvar,Int(size(TotalSol,1)/nvar)))
    # UPDATE EULERIAN COORDINATE
    Eulerx += IncDirichlet[:,1:ndim]

    #Newton-Raphson loop of iterations
    Eulerx,K,Residual = NewtonRaphson(Increment,AllGauss,Basis,Jm,points,elements,mu,kappa,norm_residual,NormForces,
                                      K,Residual,NodalForces,Eulerx,columns_in,ndim,nvar)

    # UPDATE DISPLACEMENTS FOR THE CURRENT LOAD INCREMENT
    TotalDisp[:,1:ndim,Increment] = Eulerx - points

    @printf("\nFinished Load increment %d\n",Increment)

    MaxDisp = [maximum(TotalDisp[:,1,Increment]),maximum(TotalDisp[:,2,Increment]),maximum(TotalDisp[:,3,Increment])]
    MinDisp = [minimum(TotalDisp[:,1,Increment]),minimum(TotalDisp[:,2,Increment]),minimum(TotalDisp[:,3,Increment])]
    #@printf("\nMinimum and maximum incremental solution values at increment are %e %e %e \n",MaxDisp)
    println("Minimum incremental solution values ",MinDisp)
    println("Maximum incremental solution values ",MaxDisp)
    println()

  end
  return TotalDisp
end

function Solve(AllGauss,Basis,Jm,points,elements,mu,kappa,dirichlet_flags,neumann_flags,ndim,nvar)
  """The heart of the solver."""

  # It runs the solver to seek the solution under a Newton-Raphson scheme of iteration.
  println("Pre-processing the information. Getting paths, solution parameters, mesh info, interpolation info etc...")
  println("Number of nodes is ",size(points,1)," and number of DoFs is ",size(points,1)*nvar)

  LoadIncrements = 3
  # INITIATE DATA FOR THE ANALYSIS
  NodalForces = zeros(size(points,1)*nvar)
  Residual = zeros(size(points,1)*nvar)
  TotalDisp = zeros(size(points,1),nvar,LoadIncrements)

  println("Assembling the system and acquiring neccessary information for the analysis...")

  # APPLY DIRICHELT BOUNDARY CONDITIONS AND GET DIRICHLET RELATED FORCES
  flat_dirich = vec(transpose(dirichlet_flags))
  columns_out = collect(1:size(flat_dirich,1))[.~isnan.(flat_dirich)] #Int64
  applied_dirichlet = flat_dirich[.~isnan.(flat_dirich)]
  # GENERAL PROCEDURE - GET REDUCED MATRICES FOR FINAL SOLUTION
  columns_in = delete1d(collect(1:nvar*size(points,1)),columns_out)

  # FIND PURE NEUMANN (EXTERNAL) NODAL FORCE VECTOR
  NeumannForces = zeros(size(points,1)*nvar)
  flat_neu = vec(transpose(neumann_flags))
  to_apply = collect(1:size(flat_neu,1))[.~isnan.(flat_neu)]
  applied_neumann = flat_neu[.~isnan.(flat_neu)]
  NeumannForces[to_apply] = applied_neumann

  # initialize euler coordinates
  Eulerx = points

  # assembly
  K,TractionForces = Assemble(AllGauss,Basis,Jm,points,elements,mu,kappa,Eulerx,ndim,nvar)

  TotalDsip = StaticSolver(LoadIncrements,AllGauss,Basis,Jm,points,elements,mu,kappa,
                           applied_dirichlet,NeumannForces,
                           K,Residual,NodalForces,TotalDisp,Eulerx,
                           columns_in,columns_out,ndim,nvar)

  return TotalDisp
end

end
