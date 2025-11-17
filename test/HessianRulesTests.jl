module InverseTest

using SecondOrderTopOpt
using Test, Gridap, GridapTopOpt
using FiniteDifferences
using Zygote

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
V_φ = TestFESpace(model,reffe_scalar;dirichlet_tags=["boundary"])

# Self-adjoint tests 
f(x) = 1.0
res(u,v,p) = ∫( p*∇(u)⋅∇(v)-f*v )dΩ   
J(u,p) = ∫( f*u + 0*p )dΩ
state_map = NonlinearFEStateMap(res,U,V,V_φ)
objective = GridapTopOpt.StateParamMap(J,state_map)
ṗ = rand(num_free_dofs(V_φ))
φ = rand(num_free_dofs(V_φ))
p = φ
φh = FEFunction(V_φ,φ)
u = copy(state_map(φ))
uh = FEFunction(U,u)
Zygote.gradient(p->objective(state_map(p),p),φ) # update λ
λ = state_map.cache.adj_cache[3]
@test u ≈ λ

λh = FEFunction(V,λ)
#spaces = U,V,V_φ
u̇ = incremental_state_pushforward(state_map,ṗ)#res,uh,φh,ṗ,spaces)
du̇, dṗ = incremental_objective_pushforward(objective,u̇,ṗ)
λ = state_map.cache.adj_cache[3]
λh = FEFunction(V,λ)
spaces = (U,V,V_φ)
∂2R∂u2_mat, ∂2R∂u∂φ_mat, ∂2R∂φ2_mat, ∂2R∂φ∂u_mat = incremental_adjoint_partials(res,uh,φh,λh,spaces)
du̇_R = ∂2R∂u2_mat*u̇ + ∂2R∂u∂φ_mat*ṗ
dφ̇_R = ∂2R∂φ2_mat*ṗ + ∂2R∂φ∂u_mat*u̇
# λ⁻ = solve_incremental_adjoint(res,J,uh,λh,φh,u̇,ṗ,du̇,du̇_R,spaces,state_map).free_values
# @test u̇ ≈ λ⁻
# add this back later when \lambda⁻ is available in a cache

# Second order partial derivative tests 
J(u,p) = ∫(u*u*p*p)dΩ # keep p term otherwise dual error

spaces = (U,V_φ)
∂2J∂u2_mat, ∂2J∂u∂φ_mat, ∂2J∂φ2_mat, ∂2J∂φ∂u_mat = SecondOrderTopOpt.incremental_objective_partials(J,uh,φh,spaces)

# ∂²J / ∂u² * u̇
dv = get_fe_basis(V)
du = get_trial_fe_basis(U)
dφ = get_fe_basis(V_φ)
dφ_ = get_trial_fe_basis(V_φ)

∂2∂u2_analytical(uh) = ∫( 2*φh*φh*du⋅dv )dΩ
∂2∂u2_matrix_analytical = assemble_matrix(∂2∂u2_analytical(uh),U,U)
@test ∂2∂u2_matrix_analytical ≈ ∂2J∂u2_mat

# ∂/∂p (∂J/∂u ) * ṗ
∂2J∂u∂φ_analytical(uh,φh) = ∫( 4*φh*uh*dφ_⋅dv )dΩ
∂2J∂u∂φ_matrix_analytical = assemble_matrix(∂2J∂u∂φ_analytical(uh,φh),V_φ,U)
@test ∂2J∂u∂φ_matrix_analytical  ≈ ∂2J∂u∂φ_mat

# ∂²J / ∂p² * ṗ
∂2J∂φ2_analytical(uh) = ∫( 2*uh*uh*dφ⋅dφ_ )dΩ
∂2J∂φ2_matrix_analytical = assemble_matrix(∂2J∂φ2_analytical(uh),V_φ,V_φ)
@test ∂2J∂φ2_matrix_analytical  ≈ ∂2J∂φ2_mat

# ∂/∂u (∂J / ∂p) * u̇
∂2J∂φ∂u_analytical(uh,φh) = ∫( 4*uh*φh*du⋅dφ )dΩ
∂2J∂φ∂u_matrix_analytical = assemble_matrix(∂2J∂φ∂u_analytical(uh,φh),U,V_φ)
@test ∂2J∂φ∂u_matrix_analytical  ≈ ∂2J∂φ∂u_mat

f(x) = 1.0
res(u,v,p) = ∫( p*∇(u)⋅∇(v) - f*v )dΩ
spaces = (U,V,V_φ)
∂2R∂u2_mat, ∂2R∂u∂φ_mat, ∂2R∂φ2_mat, ∂2R∂φ∂u_mat = incremental_adjoint_partials(res,uh,φh,λh,spaces)

# ∂²R / ∂u² * u̇ * λ
∂2∂u2R_analytical(uh,λh,φh) = ∫( 0*du*dv )dΩ
∂2∂u2R_matrix_analytical = assemble_matrix(∂2∂u2R_analytical(uh,λh,φh),U,U)
@test ∂2∂u2R_matrix_analytical ≈ ∂2R∂u2_mat

# ∂/∂p (∂R/∂u * λ) * ṗ
∂2R∂u∂φ_analytical(uh,λh,φh) = ∫( dφ_* ∇(dv) ⋅ ∇(λh)  )dΩ
∂2R∂u∂φ_matrix_analytical = assemble_matrix(∂2R∂u∂φ_analytical(uh,λh,φh),V_φ,U)
@test ∂2R∂u∂φ_matrix_analytical ≈ ∂2R∂u∂φ_mat
# ∂²R / ∂p² * ṗ * λ
∂2R∂φ2_analytical(uh,λh) = ∫( 0*dφ⋅dφ_ )dΩ
∂2R∂φ2_matrix_analytical = assemble_matrix(∂2R∂φ2_analytical(uh,λh),V_φ,V_φ)
@test ∂2R∂φ2_matrix_analytical ≈ ∂2R∂φ2_mat   

# ∂/∂u (∂R/∂p * λ) * ṗ
∂2R∂φ∂u_analytical(uh,λh,φh) = ∫( dφ * ∇(du) ⋅ ∇(λh) )dΩ   
∂2R∂φ∂u_matrix_analytical = assemble_matrix(∂2R∂φ∂u_analytical(uh,λh,φh),U,V_φ)
@test ∂2R∂φ∂u_matrix_analytical ≈ ∂2R∂φ∂u_mat

# Unit tests for the pushforward rules 
res(u,v,p) = ∫( (u+1)*(p+cos∘(p))*∇(u)⋅∇(v) - f*v )dΩ
J(u,p) = ∫( f*(1.0(sin∘(2π*u))+1)*(1.0(cos∘(2π*p))+1)*p)dΩ 
u = copy(state_map(φ))
uh = FEFunction(U,u)
u̇ = incremental_state_pushforward(state_map,ṗ)#res,uh,φh,ṗ,spaces)
du̇, dṗ = incremental_objective_pushforward(objective,u̇,ṗ)#J,uh,φh,u̇,ṗ,spaces)
Zygote.gradient(p->objective(state_map(p),p),φ) # update λ
λ = state_map.cache.adj_cache[3]
λh = FEFunction(V,λ)
spaces = (U,V,V_φ)
∂2R∂u2_mat_u̇, ∂2R∂u∂φ_mat_ṗ, ∂2R∂φ2_mat_ṗ, ∂2R∂φ∂u_mat_u̇ = incremental_adjoint_partials(res,uh,φh,λh,spaces)
#λ⁻ = incremental_adjoint_value(res,J,uh,λh,φh,u̇,ṗ,du̇,du̇_R,spaces).free_values

# incremental state test (ṗ->u̇)
function p_to_u(p)
    ph = FEFunction(V_φ,[p])
    op = FEOperator((u,v)->res(u,v,ph),U,V)
    uh = solve(op)
    return uh.free_values
end
u̇ = incremental_state_pushforward(state_map,ṗ)
∂u_∂p_FD = FiniteDifferences.central_fdm(5,1)(p_to_u,φ[1])
∂u_∂p_FD_ṗ = ∂u_∂p_FD .* ṗ
@test u̇ ≈ ∂u_∂p_FD_ṗ 
 
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
u̇ṗ_FD =FiniteDifferences.jacobian(central_fdm(5,1),up->Zygote.gradient(up_to_j,up)[1],up)[1]*vcat(u̇,ṗ)
@test u̇ṗ_FD[1:N] ≈ du̇ atol = 1e-11
@test u̇ṗ_FD[N+1:end] ≈ dṗ

# entire incremental map (including the adjoint part) (ṗ->dṗ)
φ_to_u = state_map
dṗ_adj = incremental_adjoint_pushforward(φ_to_u,u̇,ṗ,du̇) 
Hṗ = dṗ + dṗ_adj

p_to_j(p) = objective(state_map(p),p)
H_fd = central_fdm(5,1)(p->Zygote.gradient(p_to_j,[p])[1][1],p[1])
Hṗ_fd = H_fd * ṗ
@test Hṗ ≈ Hṗ_fd 

end