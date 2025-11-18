module InverseTest

using SecondOrderTopOpt
using Test, Gridap, GridapTopOpt
using FiniteDifferences
using Zygote
using ForwardDiff

# FE setup
order = 1 
xmax = ymax = 1.0
dom = (0,xmax,0,ymax)
el_size = (2,2)
model = CartesianDiscreteModel(dom,el_size)
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe_scalar;dirichlet_tags=[1,2,3,4,5,6,7])
U = TrialFESpace(V,0.0)
V_p = TestFESpace(model,reffe_scalar;dirichlet_tags=["boundary"])
ṗ = rand(num_free_dofs(V_p))
p = rand(num_free_dofs(V_p))
ph = FEFunction(V_p,p)

# Unit tests for the pushforward rules 
f(x) = 1.0
res(u,v,p) = ∫( (u+1)*(p+cos∘(p))*∇(u)⋅∇(v) - f*v )dΩ
J(u,p) = ∫( f*(1.0(sin∘(2π*u))+1)*(1.0(cos∘(2π*p))+1)*p)dΩ 
state_map = NonlinearFEStateMap(res,U,V,V_p)
objective = GridapTopOpt.StateParamMap(J,state_map)
u = copy(state_map(p))
uh = FEFunction(U,u)
u̇ = incremental_state_pushforward(state_map,ṗ)#res,uh,ph,ṗ,spaces)
du̇, dṗ = incremental_objective_pushforward(objective,u̇,ṗ)#J,uh,ph,u̇,ṗ,spaces)
Zygote.gradient(p->objective(state_map(p),p),p) # update λ
λ = state_map.cache.adj_cache[3]
λh = FEFunction(V,λ)
spaces = (U,V,V_p)

# # incremental objective (and pullback) (u̇->du̇)
# objective(u,p)
# du̇, dṗ = incremental_objective_pushforward(objective,u̇,ṗ)
# N = num_free_dofs(V)
# function up_to_j(up)
#     u = up[1:N]
#     p = up[N+1:end]
#     j = objective(u,p)
# end
# up = vcat(u,p)
# u̇ṗ_FD =FiniteDifferences.jacobian(central_fdm(5,1),up->Zygote.gradient(up_to_j,up)[1],up)[1]*vcat(u̇,ṗ)
# @test u̇ṗ_FD[1:N] ≈ du̇ atol = 1e-11
# @test u̇ṗ_FD[N+1:end] ≈ dṗ

function p_to_j2(p)
    u2 = u.*p[1]
    objective(u2,p)
end
ForwardDiff.gradient(p->p_to_j2(p),p)

using ChainRulesCore

function p_to_j2(p)
    u2 = u.*p[1]
    objective(u2,p)
end

∇f = p->Zygote.gradient(p_to_j2,p)[1]
FOR = ForwardDiff.derivative(α -> ∇f(p + α*ṗ), 0)

FD = FiniteDifferences.jacobian(central_fdm(5,1),p->Zygote.gradient(p_to_j2,p)[1],p)[1]*ṗ

@test FOR ≈ FD 







end 