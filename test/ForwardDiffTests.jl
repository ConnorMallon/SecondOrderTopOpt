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
# ṗ = rand(num_free_dofs(V_p))
# p = rand(num_free_dofs(V_p))
# ṗ = [0.7618804203895805]
# p = [0.4873885516037635]
ṗ = [0.16337618888610783]
p = [0.3253596201459815]

ph = FEFunction(V_p,p)

# Unit tests for the pushforward rules 
f(x) = 1.0
res(u,v,p) = ∫( (u+1)*(p)*∇(u)⋅∇(v) - f*v )dΩ
J(u,p) = ∫( f*(1.0(sin∘(2π*u))+1)*(1.0(cos∘(2π*p))+1)*p)dΩ 


state_map = NonlinearFEStateMap(res,U,V,V_p)
objective = GridapTopOpt.StateParamMap(J,state_map)
u = copy(state_map(p))
uh = FEFunction(U,u)
u̇ = incremental_state_pushforward(state_map,ṗ,ph)
Zygote.gradient(p->objective(state_map(p),p),p) # update λ



# incremental objective (and pullback) (u̇->du̇)
objective(u,p)
du̇, dṗ = incremental_objective_pushforward(objective,u̇,ṗ)
N = num_free_dofs(V)
function up_to_j(up)
    u = up[1:N]
    p = up[N+1:end]
    j = objective(u,p)
end
up = vcat(u,p)
u̇ṗ_FD = FiniteDifferences.jacobian(central_fdm(5,1),up->Zygote.gradient(up_to_j,up)[1],up)[1]*vcat(u̇,ṗ)
@test u̇ṗ_FD[1:N] ≈ du̇ atol = 1e-11
@test u̇ṗ_FD[N+1:end] ≈ dṗ

∇f = up->Zygote.gradient(up_to_j,up)[1]
∇f(up)

u̇ṗ = vcat(u̇,ṗ)
du̇dṗ =  ForwardDiff.derivative(α -> ∇f(up + α*u̇ṗ), 0)
@test du̇dṗ ≈ u̇ṗ_FD

function p_to_j3(p)
    u = state_map(p)
    objective(u,p)
end

@show state_map(p)
# FD = FiniteDifferences.jacobian(central_fdm(5,1),p->Zygote.gradient(p_to_j3,p)[1],p)[1]*ṗ

function p_to_j4(p)
    ph = FEFunction(V_p,p)
    op = FEOperator((u,v)->res(u,v,ph),U,V)
    uh = solve(op)
    sum(J(uh,ph))
end
p_to_j4(p)
FD = FiniteDifferences.jacobian(central_fdm(5,2),p_to_j4,p)[1]*ṗ
∇f = p->Zygote.gradient(p_to_j3,p)[1]
FOR = ForwardDiff.derivative(α -> ∇f(p + α*ṗ), 0)

@test FOR ≈ FD rtol = 1e-6


# Hṗ = dṗ + dṗ_adj

# dṗ+dṗ_adj

# FOR-u̇ṗ_FD[N+1:end]

# dṗ_adj

# Hṗ

# FOR

end 