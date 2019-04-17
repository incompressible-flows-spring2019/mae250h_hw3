# A package of useful testing commands
using Test

# We need to tell this test to load the module we are testing
using HW3

# The Random package is useful for making tests as arbitrary as possible
using Random

# The LinearAlgebra package is useful for some norms
using LinearAlgebra

@testset "1d Data Types" begin

p = HW3.CellData1D(ones(7))

# This tests that our use of p as an array just accesses its own data
@test all(p .== p.data)

# Test that the negative of p gives what we expect
p2 = -p
@test all(p2 .== -1)
@test all(p .== 1)

# Set up edge data on the same grid
q = HW3.EdgeData1D(p)
q[4] = 1
@test all(q .== q.data)

q2 = 2*q
@test q2[4] == 2
@test q[4] == 1
@test dot(q,q2) == 0.4

q2 = q/2
@test q2[4] == 0.5
@test q[4] == 1

n = 5
p = HW3.CellData1D(n)
p .= 1:n+2

q2 = HW3.gradient(p)
@test all(q2 .== 1)

q = HW3.EdgeData1D(p)
HW3.translate!(q,p)
@test q[3] == 3.5

p2 = HW3.divergence(q)
@test all(p2[2:n+1] .== 1)

end

@testset "TimeMarching" begin

# Set up the state vector's initial condition
u = [1.0]

# set up right-hand side of du/dt = -t^2
function f(u,t)
  du = deepcopy(u)
  du[1] = -t^2
  return du
end

# exact solution
uex(t) = 1 - t^3/3

# set up the integrator
Δt = 0.01
rk = HW3.RK(u,Δt,f;rk=HW3.RK4)

t = 0.0
uarray = [u[1]]
tarray = [t]
for j = 1:100
    t, u = rk(t,u)
    push!(uarray,u[1])
    push!(tarray,t)
end

@test norm(uarray-uex.(tarray)) < 1e-14


end
