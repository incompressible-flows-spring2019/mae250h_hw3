#=
A key for remembering the dimensions of each data type for a grid with
N interior cells:

CellData: N interior, N+2 total
          interior indexing 2:N+1 (and no boundary)
EdgeData: N+1 interior/boundary, so interior indexing 2:N
                       and boundary at 1, N+1

Also, the index spaces of each of these variables are a little different from
each other. Treating i_c as a continuous variable in the cell center index space, and
i_e as a continuous variable in edge index space, then our convention here is that

        i_c = i_e + 1/2

In other words, an edge with index i_e = 1 corresponds to a location in the cell
center space at i_c = 3/2.
=#

# We import the + and - operations from Julia so that we can extend them to
# our new data types here
import Base:+, -,*,/

#================= THE DATA TYPES =================#

# This is the parent of all of the grid data types. Note that we have declared
# it to be a subtype of AbstractVector. This will allow us to do array-like
# things on grid data.
abstract type GridData1D{N} <: AbstractVector{Float64} end

#= Here we are constructing a data type for data that live at the cell
centers on a grid.

The N written inside {} is called the parameter for the type.
These parameters will hold the number of cells of the underlying grid (interior).
 We can then use this parameter to ensure that all operations
are carried out only on data that correspond to the same size grid.

We are declaring this as a subtype of `GridData1D{N}`. GridData1D{N} will
be the `parent`type of all data on the grid of size N.
=#
struct CellData1D{N} <: GridData1D{N}
  data :: Array{Float64,1}
end

#=
Here we are constructing a data type for cell edges. Notice that we use
N again as parameter. This still corresponds to the number of interior
cells, so that edge data associated with other data types on the same grid
share the same parameter values. Again, it is a subtype of GridData1D{N}.
=#
struct EdgeData1D{N} <: GridData1D{N}
  data :: Array{Float64,1}
end


#================  CELL-CENTERED DATA CONSTRUCTORS ====================#

# This is called a constructor function for the data type. Here, it
# simply fills in the parameter N and returns the data in the new type.
"""
    CellData1D(data)

Set up a type of data that sit at cell centers. The `data` include the
interior cells and the ghost cells, so the resulting grid will be smaller
by 2.

Example:
```
julia> w = ones(5);

julia> HW3.CellData1D(w)
5-element HW3.CellData1D{3}:
 1.0
 1.0
 1.0
 1.0
 1.0
 ```
"""
function CellData1D(data::Array)
  n_pad, = size(data)
  N = n_pad-2

  return CellData1D{N}(data)
end

#== Some other constructors for cell data ==#

# This constructor function allows us to just specify the interior grid size.
# It initializes with zeros
"""
    CellData1D(nx)

Set up cell centered data equal to zero on a grid with `nx` interior cells.
Pads with a layer of ghost cells.

Example:
```
julia> HW3.CellData1D(5)
7-element HW3.CellData1D{5}:
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
```
"""
CellData1D(nx::Int) = CellData1D{nx}(zeros(nx+2))

# This constructor function is useful when we already have some grid data
# It initializes with zeros
"""
    CellData1D(p::GridData1D)

Set up cell centered data equal to zero on a grid corresponding to supplied
1d grid data `p`. Pads with a layer of ghost cells.
"""
CellData1D(p::GridData1D{N}) where {N} = CellData1D{N}(zeros(N+2))



#================  EDGE DATA CONSTRUCTORS ====================#

"""
    EdgeData1D(nx)

Set up edge data equal to zero on a grid with `nx` interior cells. Does
not pad with ghosts.
"""
EdgeData1D(nx::Int) = EdgeData1D{nx}(zeros(nx+1))

"""
    EdgeData1D(p::GridData1D)

Set up edge data equal to zero on a grid of a size corresponding to the
given grid data `p`.

Example:
```
julia> p = HW3.CellData1D(5)
7-element HW3.CellData1D{5}:
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0

julia> q = HW3.EdgeData1D(p)
6-element HW3.EdgeData1D{5}:
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 ```
"""
function EdgeData1D(p::GridData1D{N}) where {N}
  n_data = N+1

  return EdgeData1D{N}(zeros(n_data))
end

# If the array for data is provided:
function EdgeData1D(data::Array{Float64,1})
  n_data, = size(data)
  N = n_data-1
  return EdgeData1D{N}(data)
end


#====== SOME BASIC THINGS WE SHOULD BE ABLE TO DO ON THIS DATA =======#

#=
We need to explicitly tell Julia how to do some things on our new data types
=#

# Here we extend the functions `size` and `length` that is part of Julia to work on our new data type.
# The `where {N}` addition at the end just allows N to be any value.
# In other words, this function will work on CellData1D on any size of grid.
# Note that this returns the total size, including ghost cells.
Base.size(p::GridData1D{N}) where {N} = size(p.data)
Base.length(p::GridData1D{N}) where {N} = length(p.data)

# The following two lines allow us to index grid data just as though it were an array
Base.getindex(p::GridData1D, i::Int) = p.data[i]
Base.setindex!(p::GridData1D, v, i::Int) = p.data[i] = convert(Float64, v)

# Set it to negative of itself
function (-)(p::GridData1D)
  pnew = deepcopy(p)
  pnew.data .= -pnew.data
  return pnew
end

# Add and subtract the same type
function (-)(p1::T,p2::T) where {T<:GridData1D}
  return T(p1.data .- p2.data)
end

function (+)(p1::T,p2::T) where {T<:GridData1D}
  return T(p1.data .+ p2.data)
end


# Multiply and divide by a constant
function (*)(p::T,c::Number) where {T<:GridData1D}
  return T(c*p.data)
end


function (/)(p::T,c::Number) where {T<:GridData1D}
  return T(p.data ./ c)
end

(*)(c::Number,p::T) where {T<:GridData1D} = *(p,c)
