module NewtonTO

using Gridap
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt
using Optim#, NLSolversBase
using LineSearches

# Requires at least GridapTopOpt#dev-v0.4.2

path = "./results/Thermal_Compliance/"
# rm(path, recursive=true, force=true)
mkpath(path)
order = 1
n = 40
ymax = xmax = 1.0
prop_Γ_N = prop_Γ_D = 0.1

# FE Setup
model = simplexify(CartesianDiscreteModel((0, xmax, 0, ymax), (n, n)))
h = minimum(get_element_diameters(model))
f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax * prop_Γ_D + eps() ||
                           x[2] >= ymax - ymax * prop_Γ_D - eps()))
f_Γ_N(x) = (x[1] ≈ xmax && ymax / 2 - ymax * prop_Γ_N / 2 - eps() <= x[2] <=
                           ymax / 2 + ymax * prop_Γ_N / 2 + eps())
update_labels!(1, model, f_Γ_D, "Gamma_D")
update_labels!(2, model, f_Γ_N, "Gamma_N")
writevtk(model, path * "mesh")

# Triangulations and measures
Ω_bg = Triangulation(model)
Γ_N = BoundaryTriangulation(model, tags="Gamma_N")
dΩ_bg = Measure(Ω_bg, 2 * order)
dΓ_N = Measure(Γ_N, 2 * order)
vol_D = sum(∫(1)dΩ_bg)

# Initial level-set function
ϵ = Ref(1.0) # Tune!
f(x) = (-cos(4π * x[1]) * cos(4π * x[2]) - 0.5)/100
Δf(x) = -Δ(f)(x) + ϵ[]*f(x)

# Reference element
reffe_scalar = ReferenceFE(lagrangian, Float64, order)

# Initial level-set function
V_φhat = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:L2)
φh_hat = interpolate(Δf, V_φhat)

# ̂φ ↦ φ
V_φ = TestFESpace(model, reffe_scalar; dirichlet_tags=["Gamma_N"])
U_φ = TrialFESpace(V_φ, f)
φhat_to_φ = AffineFEStateMap((φ, z, _) -> ∫(∇(φ) ⋅ ∇(z) + ϵ[]*φ*z)dΩ_bg, (z, φ_hat) -> ∫(z * φ_hat)dΩ_bg, U_φ, V_φ, V_φhat,diff_order=2)
φ_0 = φhat_to_φ(φh_hat);
φh_0 = FEFunction(U_φ, φ_0)

ϵ_tikhonov = 1e-9
Jeps(u, φ_hat) = 1/2*ϵ_tikhonov*∫(abs2 ∘ φ_hat+0*u⋅u*φ_hat)dΩ_bg 
Jeps_spm = GridapTopOpt.StateParamMap(Jeps, φhat_to_φ,diff_order=2)

# φ ↦ φ_u
V_φ_u = TestFESpace(model, reffe_scalar)
φ_to_φ_u = AffineFEStateMap((φ_u, z, _) -> ∫(φ_u * z)dΩ_bg, (z, φ) -> ∫(z * φ)dΩ_bg, V_φ_u, V_φ_u, U_φ,diff_order=2)
φ_u_0 = φ_to_φ_u(φh_0);
φh_u_0 = FEFunction(V_φ_u, φ_u_0)
# writevtk(Ω_bg, path * "phi", cellfields=["phi_hat" => φh_hat, "phi" => φh_u_0, "f" => f])

# φ_u ↦ Ω
Ωs = EmbeddedCollection(model, φh_u_0) do cutgeo, cutgeo_facets, _φh_u
  Ω = DifferentiableTriangulation(Triangulation(cutgeo, PHYSICAL), V_φ_u)
  Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo), V_φ_u)
  Γg = GhostSkeleton(cutgeo)
  Ωact = Triangulation(cutgeo, ACTIVE)
  (;
    :Ω => Ω,
    :dΩ => Measure(Ω, 2 * order),
    :Γg => Γg,
    :dΓg => Measure(Γg, 2 * order),
    :n_Γg => get_normal_vector(Γg),
    :Γ => Γ,
    :dΓ => Measure(Γ, 2 * order),
    :n_Γ => get_normal_vector(Γ),
    :Ωact => Ωact,
    # :χ => χ
  )
end

# Ω ↦ u and u ↦ J
γg = 0.1
Λ = Skeleton(model) # global GP
dΛ = Measure(Λ, 2 * order)
n_Λ = get_normal_vector(Λ)

a(u,v,φ_u) = ∫(∇(v)⋅∇(u) + ∇(u)⋅∇(v)*φ_u*0 )Ωs.dΩ +
  ∫(((γg+0mean(φ_u))*mean(h))*jump(n_Λ⋅∇(v))*jump(n_Λ⋅∇(u)))dΛ
l(v, φ_u) = ∫(v)dΓ_N

λ = 0.1
J(u,φ) = ∫(∇(u)⋅∇(u)+0*φ)Ωs.dΩ

V = TestFESpace(model, reffe_scalar; dirichlet_tags=["Gamma_D"])
U = TrialFESpace(V, 0.0)
φ_u_to_u = AffineFEStateMap(a, l, U, V, V_φ_u,diff_order=2)
J_spm = GridapTopOpt.StateParamMap(J, φ_u_to_u,diff_order=2)

vf = 0.3
Vol(u,φ) = ∫(1/vol_D+0*φ+0*u⋅u)Ωs.dΩ - ∫(vf/vol_D)dΩ_bg
C = GridapTopOpt.StateParamMap(Vol, φ_u_to_u,diff_order=2)

# Not neccessary anymore, remove in future
state_collection = EmbeddedCollection(model, φh_u_0; compute_cut=false) do _φh_u
  update_collection!(Ωs, _φh_u)
  (;
    :φ_u_to_u => φ_u_to_u,
    :J => J_spm,
    :C => C
  )
end

# φ_hat ↦ J
function φ_hat_to_J(φ_hat)
  φ = φhat_to_φ(φ_hat);
  φ_u = φ_to_φ_u(φ)

  if eltype(φ_u) == Float64
    GridapTopOpt.ignore_derivatives() do
      GridapTopOpt.correct_ls!(φ_u)
      update_collection!(state_collection,FEFunction(V_φ_u,φ_u))
    end
  end

  u = state_collection.φ_u_to_u(φ_u)
  j = state_collection.J(u,φ_u)
  c = state_collection.C(u,φ_u)

  j_eps = Jeps_spm(φ, φ_hat)

  return j + j_eps #+ 0.1c[1]
end

# Optim
#j0 = 1/10
function fg!(F, G, φ_hat)
  J, dJ = GridapTopOpt.val_and_gradient(φ_hat_to_J, φ_hat)
  if G !== nothing
    dJ_vec = first(dJ)#/j0
    copy!(G,dJ_vec)
  end
  return J#J/j0
end

path_name = "/data_L2_eps$(ϵ[])_n=$(n)_tikenov=$(ϵ_tikhonov)/"
rm(path * "/data/", recursive=true, force=true)
mkpath(path * path_name)
function callback(state)
  x = state[end].metadata["x"]
 #g = state[end].metadata["g(x)"]
  it = state[end].iteration
  val = state[end].value
  g_norm = state[end].g_norm
  println("Iteration: ", it, " Objective: ", val, " g_norm: ", g_norm, " ϵ: ", ϵ[])
  φh_hat = FEFunction(V_φhat,x)
  #gh = FEFunction(V_φhat,g)
  writevtk(Ωs.Ω, path*path_name*"/out$(it)", cellfields=["uh"=>get_state(state_collection.φ_u_to_u)])
  writevtk(Ω_bg, path*path_name*"/out_phis$(it)", cellfields=["phi_hat" => φh_hat, "phi" => get_state(φhat_to_φ), "phi_u" => get_state(φ_to_φ_u)])
  it % 10 == 0 && GC.gc()
  return false
end

φh_hat = interpolate(Δf, V_φhat)
x = get_free_dof_values(φh_hat)
GridapTopOpt.val_and_gradient(φ_hat_to_J, x)

function hv!(Hv, φ, v)
  hv = Hvp(φ_hat_to_J, φ, v) 
  println("Hv running")
  copyto!(Hv, hv)
  Hv
end

d = Optim.TwiceDifferentiableHV(φ_hat_to_J,fg!,hv!,x)
opt_options = Optim.Options(;store_trace=true, extended_trace=true, callback, f_reltol=1e-7, iterations = 1)

optim_result = Optim.optimize(d, x, 
                        Optim.KrylovTrustRegion(
                                      #initial_radius = initial_radius,
                                      #cg_tol = cg_tol,
                                      #rho_upper = rho_upper,
                                      #eta = 0.2
                                      #callback
                                      ),
                                      opt_options)
                        # Optim.Options(g_tol = 1e-14,
                        #               iterations = 10,#max_iters,
                        #               store_trace = true,
                        #               show_trace = true,
                        #               #extended_trace = true
                        #               callback
                        #              ))
φ = optim_result.minimizer


val(optim_result) = optim_result.value
trace = val.(optim_result.trace)
#return Result(state_map, trace, φ)

#write(path*path_name*"hist.txt",string(opt))
end 