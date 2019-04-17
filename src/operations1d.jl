
# The LinearAlgebra package contains some useful functions on arrays
# We import some of its operations to extend them to our data here
using LinearAlgebra
import LinearAlgebra:dot, norm

#============  INNER PRODUCTS AND NORMS ===============#

# To compute inner products, we will extend the Julia function `dot`. Note that
# we exclude the ghost cells from the dot product.
"""
    dot(p1::CellData1D,p2::CellData1D) -> Real

Computes the inner product between two sets of cell-centered data on the same grid.
"""
function dot(p1::CellData1D{N},p2::CellData1D{N}) where {N}
  return dot(p1.data[2:N+1],p2.data[2:N+1])/N
end

"""
    dot(p1::EdgeData1D,p2::EdgeData1D) -> Real

Computes the inner product between two sets of edge data on the same grid.
"""
function dot(p1::EdgeData1D{N},p2::EdgeData1D{N}) where {N}

  # interior
  tmp = dot(p1.data[2:N],p2.data[2:N])

  # boundaries
  tmp += 0.5*p1.data[1]*p2.data[1]
  tmp += 0.5*p1.data[N+1]*p2.data[N+1]

  return tmp/N
end

"""
    norm(p::GridData1D) -> Real

Computes the L2 norm of data on a grid.
"""
function norm(p::GridData1D{N}) where {N}
  return sqrt(dot(p,p))
end

# This function computes an integral by just taking the inner product with
# another set of cell data uniformly equal to 1
"""
    integrate(p::CellData1D) -> Real

Computes a numerical quadrature of the cell-centered data.
"""
function integrate(p::CellData1D{N}) where {N}
  p2 = CellData1D(p)
  fill!(p2.data,1) # fill it with ones
  return dot(p,p2)
end

"""
    integrate(p::EdgeData1D) -> Real

Computes a numerical quadrature of the edge data.
"""
function integrate(p::EdgeData1D{N}) where {N}
  p2 = EdgeData1D(p)
  fill!(p2.data,1) # fill it with ones
  return dot(p,p2)
end

#=============== DIFFERENCING OPERATIONS ==================#

"""
    divergence(q::EdgeData1D) -> CellData1D

Compute the discrete divergence of edge data `q`, returning cell-centered
data on the same grid.
"""
function divergence(q::EdgeData1D{N}) where {N}
   p = CellData1D(q)
   # Loop over interior cells
   for i in 2:N+1
     p.data[i] = q.data[i] - q.data[i-1]
   end
   return p

end

"""
    gradient(p::CellData1D) -> EdgeData1D

Compute the discrete gradient of cell-centered data `p`, returning edge
data on the same grid.
"""
function gradient(p::CellData1D{N}) where {N}
    q = EdgeData1D(p)
    # Loop over all edges, including ghosts
    for i in 1:N+1
      q.data[i] = p.data[i+1] - p.data[i]
    end
    return q
end

"""
    laplacian(p::CellData1D) -> CellData1D

Compute the discrete Laplacian of the cell-centered data `p`, using
ghost cells where needed. Set ghost values in `p` to 0 to ignore these ghost
contributions.
"""
function laplacian(p::CellData1D{N}) where {N}
  lap = CellData1D(p)
  # Loop through all interior cells, making use of ghost values
  for i in 2:N+1
    lap.data[i] = -2*p.data[i] + p.data[i-1] + p.data[i+1]
  end
  return lap
end

"""
    laplacian(q::EdgeData1D) -> EdgeData1D

Compute the discrete Laplacian of both components of the edge data
`q` at its interior edges.
"""
function laplacian(q::EdgeData1D{N}) where {N}
  lap = EdgeData1D(q)
  for i in 2:N
    lap.data[i] = -2*q.data[i] + q.data[i-1] + q.data[i+1]
  end
  return lap
end

#=============== TRANSLATING OPERATIONS ==================#
#=
Note that we call this translate!, because the first argument
is changed by the function.
=#
"""
    translate!(q::EdgeData1D,s::CellData1D)

Translate (by simply averaging) cell data `s` into edge data `q`
on the same grid.
"""
function translate!(q::EdgeData1D{N},s::CellData1D{N}) where {N}
    # Loop over all interior and boundary edges
    for i in 1:N+1
      q.data[i] = 0.5*(s.data[i+1] + s.data[i])
    end
    return q
end

"""
    translate!(s::CellData1D,q::EdgeData1D)

Translate (by simply averaging) edge data `q` into cell data `s`
on the same grid.
"""
function translate!(s::CellData1D{N},q::EdgeData1D{N}) where {N}
    # Loop over all interior cells
    for i in 2:N+1
      s.data[i] = 0.5*(q.data[i] + q.data[i-1])
    end
    return s
end
