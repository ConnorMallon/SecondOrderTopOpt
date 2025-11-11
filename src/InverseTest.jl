module InverseTest

using Test, Gridap, GridapTopOpt
using GridapTopOpt: assemble_adjoint_matrix
using Gridap: FESpaces
using Gridap
using Gridap.Helpers, Gridap.Algebra, Gridap.TensorValues
using Gridap.Geometry, Gridap.CellData, Gridap.Fields, Gridap.Arrays
using Gridap.ReferenceFEs, Gridap.FESpaces,  Gridap.MultiField, Gridap.Polynomials

using Gridap.Geometry: get_faces, num_nodes, TriangulationView
using Gridap.FESpaces: get_assembly_strategy
using Gridap.ODEs: ODESolver
using Gridap: writevtk

using FiniteDifferences
using ChainRulesCore

order = 1 
xmax = ymax = 1.0
dom = (0,xmax,0,ymax)
el_size = (2,2)

## FE Setup
model = CartesianDiscreteModel(dom,el_size)

## Triangulations and measures
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)

## Spaces
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe_scalar;dirichlet_tags=[1,2,3,4,5,6,7])
U = TrialFESpace(V,0.0)
V_φ = TestFESpace(model,reffe_scalar;dirichlet_tags=["boundary"])

f(x) = 1.0
a(u,v,p) = ∫(p*∇(u)⋅∇(v))dΩ
l(v,p) = ∫(f⋅v)dΩ

state_map = AffineFEStateMap(a,l,U,V,V_φ)

res(u,v,p) = a(u,v,p) - l(v,p)

#u_data(x) = sin(pi*x[1])*sin(pi*x[2])
#α= 1

#misfit(u) = (u-u_da ta)^2

#J(u,p) = ∫( (u-u_data)*(u-u_data) + 0*p*p )dΩ # keep p term otherwise dual error

J(u,p) = ∫( f*u + 0*p )dΩ # keep p term otherwise dual error

objective = GridapTopOpt.StateParamMap(J,state_map)

function p_to_j(p)
    u = state_map(p)
    j = objective(u,p)
end

p = interpolate(20,V_φ).free_values
u = copy(state_map(p))

using Zygote

djdp = Zygote.gradient(p_to_j, p)[1]
λ = state_map.cache.adj_cache[3]
ṗ = djdp * 1e4# take as the first guess for the direction

uh = FEFunction(U,u)
ph = FEFunction(V_φ,p)
φh = ph 
λh = FEFunction(V,λ)
ṗh = FEFunction(V_φ,ṗ)

function jacobian_fdm_4th(func, x::AbstractVector; h = 1e-4)
    # Work on a floating-point copy of x
    x_work = float.(copy(x))

    y0 = func(x_work)
    y0 isa AbstractVector || error("jacobian_fdm_4th: func(x) must return a vector, got $(typeof(y0))")

    m = length(y0)
    n = length(x_work)

    T = promote_type(eltype(x_work), eltype(y0))
    J = Matrix{T}(undef, m, n)

    for j in 1:n
        xj = x_work[j]

        # f(x + 2h e_j)
        x_work[j] = xj + 2h
        f_pp = copy(func(x_work))

        # f(x + h e_j)
        x_work[j] = xj + h
        f_p = copy(func(x_work))

        # f(x - h e_j)
        x_work[j] = xj - h
        f_m = copy(func(x_work))

        # f(x - 2h e_j)
        x_work[j] = xj - 2h
        f_mm = copy(func(x_work))

        # restore x
        x_work[j] = xj

        @inbounds J[:, j] .= (-f_pp .+ 8f_p .- 8f_m .+ f_mm) ./ (12h)
    end

    return J
end


######
###### incremental equation (ṗ-u̇)
######

# RHS: 
dv = get_fe_basis(V)
∂R∂φ = Gridap.jacobian(φ->res(uh,dv,φ),φh)
∂R∂φ_mat = assemble_matrix(∂R∂φ,V_φ,V)
∂R∂u_mat_ṗ = ∂R∂φ_mat * ṗ

# LHS:
∂R∂u = Gridap.jacobian(uh->res(uh,dv,φh),uh) 
∂R∂u_mat = assemble_matrix(∂R∂u,U,V)  

# solving  
u̇ = ∂R∂u_mat \ (-∂R∂u_mat_ṗ)

###### TESTING
function p_to_u(p)
    ph = FEFunction(V_φ,[p])
    op = AffineFEOperator((u,v)->a(u,v,ph),v->l(v,ph),U,V)
    uh = solve(op)
    return uh.free_values
end

∂u_∂p_FD = FiniteDifferences.central_fdm(5,1)(p_to_u,p[1])
∂u_∂p_FD_ṗ = ∂u_∂p_FD .* ṗ

@test u̇ ≈ ∂u_∂p_FD_ṗ rtol = 1e-4

######
###### u̇ -> J̇
######
# dont need

######
###### directional derivate of the map dj-> du,dp in the direction (u̇,ṗ)
######
# we need 

# ∂²J / ∂u² * u̇
∂2J∂u2 = Gridap.hessian(uh->J(uh,φh),uh)
∂2J∂u2_mat = assemble_matrix(∂2J∂u2,V,V)
∂2J∂u2_mat_u̇ = ∂2J∂u2_mat * u̇

# ∂/∂p (∂J/∂u ) * ṗ
∂J∂u(uh,φh) = Gridap.gradient(uh->J(uh,φh),uh)
∂2J∂u∂φ = Gridap.jacobian(φ->∂J∂u(uh,φ),φh)
∂2J∂u∂φ_mat = assemble_matrix(∂2J∂u∂φ,V_φ,V)
∂2J∂u∂φ_mat_ṗ = ∂2J∂u∂φ_mat * ṗ

# ∂²J / ∂p² * ṗ
∂2J∂φ2 = Gridap.hessian(φ->J(uh,φ),φh)
∂2J∂φ2_mat = assemble_matrix(∂2J∂φ2,V_φ,V_φ)
∂2J∂φ2_mat_ṗ = ∂2J∂φ2_mat * ṗ

# ∂/∂u (∂J / ∂p) * u̇
∂J∂φ(uh,φh) = Gridap.gradient(φ->J(uh,φ),φh)
∂2J∂φ∂u = Gridap.jacobian(uh->∂J∂φ(uh,φh),uh)
∂2J∂φ∂u_mat = assemble_matrix(∂2J∂φ∂u,U,V_φ)
∂2J∂φ∂u_mat_u̇ = ∂2J∂φ∂u_mat * u̇

dṗ = ∂2J∂φ2_mat_ṗ + ∂2J∂φ∂u_mat_u̇
du̇ = ∂2J∂u2_mat_u̇ + ∂2J∂u∂φ_mat_ṗ

# ## Testing
# # ∂²J / ∂u² * u̇
# dv = get_fe_basis(V)
# du = get_trial_fe_basis(U)
# dφ = get_fe_basis(V_φ)
# dφ_ = get_trial_fe_basis(V_φ)

# ∂2∂u2_analytical(uh) = ∫( 2*φh*φh*du⋅dv )dΩ
# ∂2∂u2_matrix_analytical = assemble_matrix(∂2∂u2_analytical(uh),U,U)
# @test ∂2∂u2_matrix_analytical ≈ ∂2J∂u2_mat
# @test ∂2∂u2_matrix_analytical * u̇ ≈ ∂2J∂u2_mat_u̇

# # ∂/∂p (∂J/∂u ) * ṗ
# ∂2J∂u∂φ_analytical(uh,φh) = ∫( 4*φh*uh*dφ_⋅dv )dΩ
# ∂2J∂u∂φ_matrix_analytical = assemble_matrix(∂2J∂u∂φ_analytical(uh,φh),V_φ,U)
# @test ∂2J∂u∂φ_matrix_analytical ≈ ∂2J∂u∂φ_mat
# @test ∂2J∂u∂φ_matrix_analytical * ṗ ≈ ∂2J∂u∂φ_mat_ṗ

# # ∂²J / ∂p² * ṗ
# ∂2J∂φ2_analytical(uh) = ∫( 2*uh*uh*dφ⋅dφ_ )dΩ
# ∂2J∂φ2_matrix_analytical = assemble_matrix(∂2J∂φ2_analytical(uh),V_φ,V_φ)
# @test ∂2J∂φ2_matrix_analytical ≈ ∂2J∂φ2_mat
# @test ∂2J∂φ2_matrix_analytical * ṗ ≈ ∂2J∂φ2_mat_ṗ

# # ∂/∂u (∂J / ∂p) * u̇
# ∂2J∂φ∂u_analytical(uh,φh) = ∫( 4*uh*φh*du⋅dφ )dΩ
# ∂2J∂φ∂u_matrix_analytical = assemble_matrix(∂2J∂φ∂u_analytical(uh,φh),U,V_φ)
# @test ∂2J∂φ∂u_matrix_analytical ≈ ∂2J∂φ∂u_mat
# @test ∂2J∂φ∂u_matrix_analytical * u̇ ≈ ∂2J∂φ∂u_mat_u̇

dj = 1.0
function dj_to_du(u)
    j_val, j_pullback = rrule(objective,u,φ)   # Compute functional and pull back
    _, dFdu, dFdφ     = j_pullback(dj)    # Compute dFdu, dFdφ
    dFdu,dFdφ
    dFdu
end
dJ̇ = 1.0
@test jacobian_fdm_4th( dj_to_du, u) * u̇ * dJ̇  ≈ du̇ atol = 1e-6

function dj_to_dφ(φ)
    j_val, j_pullback = rrule(objective,u,φ)   # Compute functional and pull back
    _, dFdu, dFdφ     = j_pullback(dj)    # Compute dFdu, dFdφ
    dFdφ
end
dJ̇ = 1.0
@test jacobian_fdm_4th( dj_to_dφ, p) * ṗ * dJ̇ ≈ dṗ

######
###### partials related to the state map
######

# differentiating the lhs of the adjoint equation: (for the partials we need for the incremental adjoint)

# ∂²R / ∂u² * u̇ * λ
∂2R∂u2 = Gridap.hessian(uh->res(uh,λh,φh),uh) 
∂2R∂u2_mat = assemble_matrix(∂2R∂u2,U,V)  
∂2R∂u2_mat_u̇ = ∂2R∂u2_mat * u̇

# ∂/∂p (∂R/∂u * λ) * ṗ
∂R∂u_λ(uh,φh) = Gridap.gradient(uh->res(uh,λh,φh),uh)
∂2R∂u∂φ = Gridap.jacobian(φ->∂R∂u_λ(uh,φ),φh) 
∂2R∂u∂φ_mat = assemble_matrix(∂2R∂u∂φ,V_φ,V)
∂2R∂u∂φ_mat_ṗ = ∂2R∂u∂φ_mat * ṗ

# ∂²R / ∂p² * ṗ * λ
∂2R∂φ2 = Gridap.hessian(φh->res(uh,λh,φh),φh)
∂2R∂φ2_mat = assemble_matrix(∂2R∂φ2,V_φ,V_φ)
∂2R∂φ2_mat_ṗ = ∂2R∂φ2_mat * ṗ

# ∂/∂u (∂R/∂p * λ) * ṗ
∂R∂φ_λ(uh,φh) = Gridap.gradient(φh->res(uh,λh,φh),φh)
∂2R∂φ∂u = Gridap.jacobian(uh->∂R∂φ_λ(uh,φh),uh)
∂2R∂φ∂u_mat = assemble_matrix(∂2R∂φ∂u,U,V_φ)
∂2R∂φ∂u_mat_u̇ = ∂2R∂φ∂u_mat * u̇

# ## TESTING
# # ∂²R / ∂u² * u̇ * λ
# ∂2∂u2R_analytical(uh,λh,φh) = ∫( 0*du*dv )dΩ
# ∂2∂u2R_matrix_analytical = assemble_matrix(∂2∂u2R_analytical(uh,λh,φh),U,U)
# @test ∂2∂u2R_matrix_analytical ≈ ∂2R∂u2_mat
# @test ∂2∂u2R_matrix_analytical * u̇ ≈ ∂2R∂u2_mat_u̇

# # ∂/∂p (∂R/∂u * λ) * ṗ
# ∂2R∂u∂φ_analytical(uh,λh,φh) = ∫( dφ_* ∇(dv) ⋅ ∇(λh)  )dΩ
# ∂2R∂u∂φ_matrix_analytical = assemble_matrix(∂2R∂u∂φ_analytical(uh,λh,φh),V_φ,U)
# @test ∂2R∂u∂φ_matrix_analytical ≈ ∂2R∂u∂φ_mat
# @test ∂2R∂u∂φ_matrix_analytical * ṗ ≈ ∂2R∂u∂φ_mat_ṗ

# # ∂²R / ∂p² * ṗ * λ
# ∂2R∂φ2_analytical(uh,λh) = ∫( 0*dφ⋅dφ_ )dΩ
# ∂2R∂φ2_matrix_analytical = assemble_matrix(∂2R∂φ2_analytical(uh,λh),V_φ,V_φ)
# @test ∂2R∂φ2_matrix_analytical ≈ ∂2R∂φ2_mat
# @test ∂2R∂φ2_matrix_analytical * ṗ ≈ ∂2R∂φ2_mat_ṗ   

# # ∂/∂u (∂R/∂p * λ) * ṗ
# ∂2R∂φ∂u_analytical(uh,λh,φh) = ∫( dφ * ∇(du) ⋅ ∇(λh) )dΩ   
# ∂2R∂φ∂u_matrix_analytical = assemble_matrix(∂2R∂φ∂u_analytical(uh,λh,φh),U,V_φ)
# @test ∂2R∂φ∂u_matrix_analytical ≈ ∂2R∂φ∂u_mat
# @test ∂2R∂φ∂u_matrix_analytical * u̇ ≈ ∂2R∂φ∂u_mat_u̇

######
###### incremental adjoint equation
######

# RHS 
inc_adjoint_rhs = - ∂2J∂u2_mat_u̇ - ∂2J∂u∂φ_mat_ṗ - ∂2R∂u2_mat_u̇ - ∂2R∂u∂φ_mat_ṗ






# LHS 2w
assem_adjoint = SparseMatrixAssembler(V,U)
∂R∂u_adjoint = (du,v) -> Gridap.jacobian(res,[uh,v,φh],1)
∂R∂u_adjoint_mat = assemble_adjoint_matrix(∂R∂u_adjoint,assem_adjoint,U,V)

λ⁻ = ∂R∂u_adjoint_mat \ inc_adjoint_rhs

λ⁻ 
u̇




λ⁻h = FEFunction(V,λ⁻)
∂R∂p_λ⁻ = Gridap.gradient(φh->res(uh,λ⁻h,φh),φh)
∂R∂p_mat_λ⁻ = assemble_vector(∂R∂p_λ⁻,V_φ)

dṗ_adj = ∂R∂p_mat_λ⁻ + ∂2R∂φ2_mat_ṗ + ∂2R∂φ∂u_mat_u̇

#### testing

u_val, u_pullback = rrule(state_map,φh)   # Compute functional and pull back
function du_to_dφ(du)
    _, dφ_adj         = u_pullback(du) # Compute -dFdu*dudφ via adjoint
    dφ_adj[1]
end
dφdu_fd = grad(central_fdm(5,1),du_to_dφ,u̇)[1]
#u̇_adj = ∂2J∂u2_mat_u̇ + ∂2J∂u∂φ_mat_ṗ 
dṗ_adj_fd = dφdu_fd' * du̇

@test dṗ_adj[1] ≈ dṗ_adj_fd





function dj_to_dφ_adj(φ)
    u, u_pullback = rrule(state_map,φ)
    j_val, j_pullback = rrule(objective,u,φ)   # Compute functional and pull back
    _, dFdu, dFdφ     = j_pullback(dj)    # Compute dFdu, dFdφ
    _, dφ_adj         = u_pullback(dFdu) # Compute -dFdu*dudφ via adjoint
    dφ_adj 
end

dj_to_dφ_adj(φ)
jacobian_fdm_4th( dj_to_dφ_adj, p)* ṗ




# self adjoint problem? 





###### 

# Finally, the hessian action can then be computed as:

∂2J∂φ2_mat_ṗ
∂2J∂φ∂u_mat_u̇
∂R∂p_mat_λ⁻
∂2R∂φ2_mat_ṗ
∂2R∂φ∂u_mat_u̇

Hṗ = dṗ + dṗ_adj

# ive tested all the partials.... 






function p_to_dp(φ)
    u, u_pullback = rrule(state_map,φ)
    j_val, j_pullback = rrule(objective,u,φ)   # Compute functional and pull back
    _, dFdu, dFdφ     = j_pullback(1)    # Compute dFdu, dFdφ
    _, dφ_adj         = u_pullback(dFdu) # Compute -dFdu*dudφ via adjoint
    dφ_adj + dFdφ
end

grad(central_fdm(5,1),p_to_dp,p)
jacobian_fdm_4th(p_to_dp, p)* ṗ







j_val, j_pullback = rrule(objective,uh,φh)   # Compute functional and pull back

function dj_to_du(dj)
    _, dFdu, dFdφ     = j_pullback(dj)    # Compute dFdu, dFdφ
    dFdu,dFdφ
    dFdu
end

dj_to_du(1.0)
central_fdm(5,1)(dj_to_du,1.0)



H_fd = central_fdm(5,1)(p->Zygote.gradient(p_to_j,[p])[1][1],p[1])
Hṗ_fd = H_fd * ṗ
@test Hṗ ≈ Hṗ_fd 









end