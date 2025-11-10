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
U = TrialFESpace(V,1.0)
V_φ = TestFESpace(model,reffe_scalar;dirichlet_tags=["boundary"])

f(x) = 1.0
a(u,v,p) = ∫(p*∇(u)⋅∇(v))dΩ
l(v,p) = ∫(f⋅v)dΩ
state_map = AffineFEStateMap(a,l,U,V,V_φ)

res(u,v,p) = a(u,v,p) - l(v,p)
u_data(x) = sin(pi*x[1])*sin(pi*x[2])
α= 1

misfit(u) = (u-u_data)^2

#J(u,p) = ∫( (u-u_data)*(u-u_data) + 0*p*p )dΩ # keep p term otherwise dual error

J(u,p) = ∫( u*u*p*p  )dΩ # keep p term otherwise dual error

objective = GridapTopOpt.StateParamMap(J,state_map)

function p_to_j(p)
    u = state_map(p)
    j = objective(u,p)
end

p = interpolate(0.5,V_φ).free_values
u = copy(state_map(p))

using Zygote

djdp = Zygote.gradient(p_to_j, p)[1]
λ = state_map.cache.adj_cache[3]
ṗ = djdp # take as the first guess for the direction

uh = FEFunction(U,u)
ph = FEFunction(V_φ,p)
φh = ph 
λh = FEFunction(V,λ)
ṗh = FEFunction(V_φ,ṗ)

######
###### incremental equation (ṗ-u̇)
######

@show uh.free_values

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

# wewwww test passes


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

## Testing
# ∂²J / ∂u² * u̇
dv = get_fe_basis(V)
du = get_trial_fe_basis(U)
dφ = get_fe_basis(V_φ)
dφ_ = get_trial_fe_basis(V_φ)

∂2∂u2_analytical(uh) = ∫( 2*φh*φh*du⋅dv )dΩ
∂2∂u2_matrix_analytical = assemble_matrix(∂2∂u2_analytical(uh),U,U)
@test ∂2∂u2_matrix_analytical ≈ ∂2J∂u2_mat
@test ∂2∂u2_matrix_analytical * u̇ ≈ ∂2J∂u2_mat_u̇

# ∂/∂p (∂J/∂u ) * ṗ
∂2J∂u∂φ_analytical(uh,φh) = ∫( 4*φh*uh*dφ_⋅dv )dΩ
∂2J∂u∂φ_matrix_analytical = assemble_matrix(∂2J∂u∂φ_analytical(uh,φh),V_φ,U)
@test ∂2J∂u∂φ_matrix_analytical ≈ ∂2J∂u∂φ_mat
@test ∂2J∂u∂φ_matrix_analytical * ṗ ≈ ∂2J∂u∂φ_mat_ṗ

# ∂²J / ∂p² * ṗ
∂2J∂φ2_analytical(uh) = ∫( 2*uh*uh*dφ⋅dφ_ )dΩ
∂2J∂φ2_matrix_analytical = assemble_matrix(∂2J∂φ2_analytical(uh),V_φ,V_φ)
@test ∂2J∂φ2_matrix_analytical ≈ ∂2J∂φ2_mat
@test ∂2J∂φ2_matrix_analytical * ṗ ≈ ∂2J∂φ2_mat_ṗ

# ∂/∂u (∂J / ∂p) * u̇
∂2J∂φ∂u_analytical(uh,φh) = ∫( 4*uh*φh*du⋅dφ )dΩ
∂2J∂φ∂u_matrix_analytical = assemble_matrix(∂2J∂φ∂u_analytical(uh,φh),U,V_φ)
@test ∂2J∂φ∂u_matrix_analytical ≈ ∂2J∂φ∂u_mat
@test ∂2J∂φ∂u_matrix_analytical * u̇ ≈ ∂2J∂φ∂u_mat_u̇

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


## TESTING
# ∂²R / ∂u² * u̇ * λ
∂2∂u2R_analytical(uh,λh,φh) = ∫( 0*du*dv )dΩ
∂2∂u2R_matrix_analytical = assemble_matrix(∂2∂u2R_analytical(uh,λh,φh),U,U)
@test ∂2∂u2R_matrix_analytical ≈ ∂2R∂u2_mat
@test ∂2∂u2R_matrix_analytical * u̇ ≈ ∂2R∂u2_mat_u̇

# ∂/∂p (∂R/∂u * λ) * ṗ
∂2R∂u∂φ_analytical(uh,λh,φh) = ∫( dφ_* ∇(dv) ⋅ ∇(λh)  )dΩ
∂2R∂u∂φ_matrix_analytical = assemble_matrix(∂2R∂u∂φ_analytical(uh,λh,φh),V_φ,U)
@test ∂2R∂u∂φ_matrix_analytical ≈ ∂2R∂u∂φ_mat
@test ∂2R∂u∂φ_matrix_analytical * ṗ ≈ ∂2R∂u∂φ_mat_ṗ

# ∂²R / ∂p² * ṗ * λ
∂2R∂φ2_analytical(uh,λh) = ∫( 0*dφ⋅dφ_ )dΩ
∂2R∂φ2_matrix_analytical = assemble_matrix(∂2R∂φ2_analytical(uh,λh),V_φ,V_φ)
@test ∂2R∂φ2_matrix_analytical ≈ ∂2R∂φ2_mat
@test ∂2R∂φ2_matrix_analytical * ṗ ≈ ∂2R∂φ2_mat_ṗ   

# ∂/∂u (∂R/∂p * λ) * ṗ
∂2R∂φ∂u_analytical(uh,λh,φh) = ∫( dφ * ∇(du) ⋅ ∇(λh) )dΩ   
∂2R∂φ∂u_matrix_analytical = assemble_matrix(∂2R∂φ∂u_analytical(uh,λh,φh),U,V_φ)
@test ∂2R∂φ∂u_matrix_analytical ≈ ∂2R∂φ∂u_mat
@test ∂2R∂φ∂u_matrix_analytical * u̇ ≈ ∂2R∂φ∂u_mat_u̇

######
###### incremental adjoint equation
######

# RHS 
inc_adjoint_rhs = - ∂2J∂u2_mat_u̇ - ∂2J∂u∂φ_mat_ṗ - ∂2R∂u2_mat_u̇ - ∂2R∂u∂φ_mat_ṗ

# LHS 2
assem_adjoint = SparseMatrixAssembler(V,U)
∂R∂u_adjoint= (du,v) -> Gridap.jacobian(res,[uh,v,φh],1)
∂R∂u_adjoint_mat = assemble_adjoint_matrix(∂R∂u_adjoint,assem_adjoint,U,V)

λ⁻ = ∂R∂u_adjoint_mat \ inc_adjoint_rhs
λ⁻h = FEFunction(V,λ⁻)
∂R∂p_λ⁻ = Gridap.gradient(φh->res(uh,λ⁻h,φh),φh)
∂R∂p_mat_λ⁻ = assemble_vector(∂R∂p_λ⁻,V_φ)


∂R∂p_mat_λ⁻ - ∂2R∂φ2_mat_ṗ - ∂2R∂φ∂u_mat_u̇



# testing.... 
# how do we test this... 
# well we can 





# what else can we do to debug this...






u_val, u_pullback = rrule(state_map,φh)   # Compute functional and pull back
function du_to_dφ(du)
    _, dφ_adj         = u_pullback(du) # Compute -dFdu*dudφ via adjoint
    dφ_adj[1]
end
dφdu_fd = grad(central_fdm(5,1),du_to_dφ,u̇)[1]
u̇_adj = ∂2J∂u2_mat_u̇ + ∂2J∂u∂φ_mat_ṗ 
ṗ_adj = dφdu_fd' * u̇_adj 






















# the other way would be to finite difference the incremental state ? 












# Finally, the hessian action can then be computed as:


∂2J∂φ2_mat_ṗ
∂2J∂φ∂u_mat_u̇
∂R∂p_mat_λ⁻
∂2R∂φ2_mat_ṗ
∂2R∂φ∂u_mat_u̇

Hṗ = ∂2J∂φ2_mat_ṗ + ∂2J∂φ∂u_mat_u̇ + ∂R∂p_mat_λ⁻ + ∂2R∂φ2_mat_ṗ + ∂2R∂φ∂u_mat_u̇

#end

#Hṗ = hessian_action(u,p,λ,ṗ)
H_fd = central_fdm(5,1)(p->Zygote.gradient(p_to_j,[p])[1][1],p[1])
Hṗ_fd = H_fd * ṗ
@test Hṗ ≈ Hṗ_fd 


end