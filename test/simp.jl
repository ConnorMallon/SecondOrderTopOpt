using Gridap, Gridap.Adaptivity, Gridap.Geometry
using GridapTopOpt
using NLopt

"""
  SIMP-based minimum thermal compliance

  Optimisation problem:
      Min J(ρ) = ∫ κ(ρ)*∇(u)⋅∇(u) dΩ
     ρ∈[0,1]
    s.t., V(ρ) = vf,
          ⎡u∈V=H¹(Ω;u(Γ_D)=0),
          ⎣∫ κ(ρ)*∇(u)⋅∇(v) dΩ = ∫ v dΓ_N, ∀v∈V.
"""
path="./results/Thermal_Compliance_SIMP/"
rm(path, recursive=true, force=true)
mkpath(path)
order = 1
n = 200
vf = 0.4
α_coeff = 2
iter_mod = 10

## FE Setup
model = CartesianDiscreteModel((0,1,0,1),(n,n))
h = 1/n
f_Γ_D(x) = (x[1]-0.5)^2 + (x[2]-0.5)^2 <= 0.05^2-eps()
f_Γ_N(x) = ((x[1] ≈ 0 || x[1] ≈ 1) && (0.2 <= x[2] <= 0.3 + eps() || 0.7 - eps() <= x[2] <= 0.8)) ||
  ((x[2] ≈ 0 || x[2] ≈ 1) && (0.2 <= x[1] <= 0.3 + eps() || 0.7 - eps() <= x[1] <= 0.8))
update_labels!(1,model,f_Γ_D,"Omega_D")
update_labels!(2,model,f_Γ_N,"Gamma_N")
writevtk(model,path*"mesh")

## Triangulations and measures
Ω = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ = Measure(Ω,2*order)
dΓ_N = Measure(Γ_N,2*order)
vol_D = sum(∫(1)dΩ)

## Spaces
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe_scalar;dirichlet_tags=["Omega_D"])
U = TrialFESpace(V,0.0)
V_ρ = TestFESpace(model,ReferenceFE(lagrangian,Float64,0);dirichlet_tags=["Omega_D"])
U_ρ = TrialFESpace(V_ρ,1.0)

## Create FE functions
ρh = interpolate(0.5,U_ρ)

## SIMP
κ_0 = 1e-6; κ_1 = 1
κ(ρ) = ρ^5*(κ_1-κ_0) + κ_0

a(u,v,ρ) = ∫((κ ∘ ρ)*∇(u)⋅∇(v))dΩ
l(v,ρ) = ∫(100v)dΓ_N

## Optimisation functionals
J(u,ρ) = ∫((κ ∘ ρ)*∇(u)⋅∇(u))dΩ
Vol(u,ρ) = ∫(ρ - vf)dΩ;
# Because the above should really be Vol(ρ) = ∑(ρ)/n^2 - vf, we multiple the sensitivity by h^2
#   In future, we should really have a domain contribution that is just the sum of values?
dVol(q,u,ρ) = ∫(q*n^2)dΩ

## Setup solver and FE operators
state_map = AffineFEStateMap(a,l,U,V,U_ρ)
pcfs = PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dC=[dVol])

## Hilbertian extension-regularisation problems
α = α_coeff*h
a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
vel_ext = VelocityExtension(a_hilb,V_ρ,V_ρ)

_it = [0]
function my_objective_fn(ρ::Vector, grad::Vector)
  ρh = FEFunction(U_ρ,ρ)
  J, _, dJ, dC = evaluate!(pcfs,ρh)
  uh = get_state(pcfs)
  #if length(grad) > 0
    #project!(vel_ext,-dJ)
    copy!(grad,dJ)
  #end
  println("Iteration: ", _it[1], " Objective: ", J, " Volume: ", sum(Vol(uh,ρh)))
  writevtk(Ω,path*"out$(_it[1])",cellfields=["ρ"=>ρh,"uh"=>uh,"dJ"=>FEFunction(V_ρ,dJ),"dC"=>FEFunction(V_ρ,dC[1])])
  _it[1] = _it[1] + 1
  return J
end
function my_constraint_fn(ρ::Vector, grad::Vector)
  ρh = FEFunction(U_ρ,ρ)
  _, C, _, dC = evaluate!(pcfs,ρh)
  if length(grad) > 0
    copy!(grad,first(dC))
  end
  return first(C)
end

# Optimser
opt = NLopt.Opt(:LD_MMA, num_free_dofs(V_ρ))
NLopt.lower_bounds!(opt, 0)
NLopt.upper_bounds!(opt, 1)
NLopt.xtol_rel!(opt, 1e-4)
NLopt.min_objective!(opt, my_objective_fn)
NLopt.inequality_constraint!(opt, my_constraint_fn, 1e-6)
min_f, min_x, ret = NLopt.optimize(opt, get_free_dof_values(ρh))
