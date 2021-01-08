function einsum(index,A,B)
  """ Einstein sumation. Limited to produce fourth order tensor from
      two order tensors."""
  # tensor product of two second order tensor A_ij B_kl
  if index == "ijkl"
    C=zeros(Float64,3,3,3,3)
    for i in 1:3
      for j in 1:3
        for k in 1:3
          for l in 1:3
            C[i,j,k,l] += A[i,j]*B[k,l]
          end
        end
      end
    end
  # tensor product of two second order tensor A_ik B_jl with permutation
  elseif index == "ikjl"
    C=zeros(Float64,3,3,3,3)
    for i in 1:3
      for j in 1:3
        for k in 1:3
          for l in 1:3
            C[i,j,k,l] += A[i,k]*B[j,l]
          end
        end
      end
    end
  # tensor product of two second order tensor A_il B_jk with permutation
  elseif index == "iljk"
    C=zeros(Float64,3,3,3,3)
    for i in 1:3
      for j in 1:3
        for k in 1:3
          for l in 1:3
            C[i,j,k,l] += A[i,l]*B[j,k]
          end
        end
      end
    end
  else
    println("Operation not understood.")
    C = Nothing
  end
  return C
end

function Voigt(C)
  """Transform of a fourth-order tensor into a square matrix under Voigt notation.
     Just working for 3dimension."""
  VoigtA = zeros(Float64,6,6)
  VoigtA[1,1] = C[1,1,1,1]
  VoigtA[1,2] = C[1,1,2,2]
  VoigtA[1,3] = C[1,1,3,3]
  VoigtA[1,4] = 0.5*(C[1,1,1,2]+C[1,1,2,1])
  VoigtA[1,5] = 0.5*(C[1,1,1,3]+C[1,1,3,1])
  VoigtA[1,6] = 0.5*(C[1,1,2,3]+C[1,1,3,2])
  VoigtA[2,1] = VoigtA[1,2]
  VoigtA[2,2] = C[2,2,2,2]
  VoigtA[2,3] = C[2,2,3,3]
  VoigtA[2,4] = 0.5*(C[2,2,1,2]+C[2,2,2,1])
  VoigtA[2,5] = 0.5*(C[2,2,1,3]+C[2,2,3,1])
  VoigtA[2,6] = 0.5*(C[2,2,2,3]+C[2,2,3,2])
  VoigtA[3,1] = VoigtA[1,3]
  VoigtA[3,2] = VoigtA[2,3]
  VoigtA[3,3] = C[3,3,3,3]
  VoigtA[3,4] = 0.5*(C[3,3,1,2]+C[3,3,2,1])
  VoigtA[3,5] = 0.5*(C[3,3,1,3]+C[3,3,3,1])
  VoigtA[3,6] = 0.5*(C[3,3,2,3]+C[3,3,3,2])
  VoigtA[4,1] = VoigtA[1,4]
  VoigtA[4,2] = VoigtA[2,4]
  VoigtA[4,3] = VoigtA[3,4]
  VoigtA[4,4] = 0.5*(C[1,2,1,2]+C[1,2,2,1])
  VoigtA[4,5] = 0.5*(C[1,2,1,3]+C[1,2,3,1])
  VoigtA[4,6] = 0.5*(C[1,2,2,3]+C[1,2,3,2])
  VoigtA[5,1] = VoigtA[1,5]
  VoigtA[5,2] = VoigtA[2,5]
  VoigtA[5,3] = VoigtA[3,5]
  VoigtA[5,4] = VoigtA[4,5]
  VoigtA[5,5] = 0.5*(C[1,3,1,3]+C[1,3,3,1])
  VoigtA[5,6] = 0.5*(C[1,3,2,3]+C[1,3,3,2])
  VoigtA[6,1] = VoigtA[1,6]
  VoigtA[6,2] = VoigtA[2,6]
  VoigtA[6,3] = VoigtA[3,6]
  VoigtA[6,4] = VoigtA[4,6]
  VoigtA[6,5] = VoigtA[5,6]
  VoigtA[6,6] = 0.5*(C[2,3,2,3]+C[2,3,3,2])
  return VoigtA
end

function in1d(ar1,ar2)
  """Return a boolean 1D-array with trues where an element of ar2 is presented."""
  ar3 = falses(size(ar1,1))
  for i in 1:size(ar1,1)
    for j in 1:size(ar2,1)
      if ar1[i]==ar2[j]
        ar3[i] = true
      end
    end
  end
  return ar3
end

function delete1d(ar1,ar2)
  """Remove elements in ar1 from de inices in ar2"""
  ar3 = trues(size(ar1,1))
  for i in ar2
    ar3[i] = false
  end
  ar1 = ar1[ar3]
  return ar1
end

