using Gridap, Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt
using Optim 
using LineSearches

# Requires at least GridapTopOpt#dev-v0.4.2

path = "./results/Thermal_Compliance/"
# rm(path, recursive=true, force=true)
mkpath(path)
order = 1
n = 15
ymax = xmax = 1.0
prop_Γ_N = prop_Γ_D = 0.1

# FE Setup
base_model = UnstructuredDiscreteModel(CartesianDiscreteModel((0, xmax, 0, ymax), (n, n)))
ref_model = refine(base_model, refinement_method="barycentric")
ref_model = refine(ref_model)
ref_model = refine(ref_model)
model = get_model(ref_model)
h = minimum(get_element_diameters(model))
f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax * prop_Γ_D + eps() ||
                          x[2] >= ymax - ymax * prop_Γ_D - eps()))
f_Γ_N(x) = (x[1] ≈ xmax && ymax / 2 - ymax * prop_Γ_N / 2 - eps() <= x[2] <=
                          ymax / 2 + ymax * prop_Γ_N / 2 + eps())
update_labels!(1, model, f_Γ_D, "Gamma_D")
update_labels!(2, model, f_Γ_N, "Gamma_N")
# writevtk(model, path * "mesh")

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

# writevtk(Ω_bg, path * "phi", cellfields=["f" => f, "-Δ(f)(x)" => Δf])

# Reference element
reffe_scalar = ReferenceFE(lagrangian, Float64, order)

# Initial level-set function
V_φhat = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:L2)
φh_hat = interpolate(Δf, V_φhat)

# ̂φ ↦ φ
V_φ = TestFESpace(model, reffe_scalar; dirichlet_tags=["Gamma_N"])
U_φ = TrialFESpace(V_φ, f)
φhat_to_φ = AffineFEStateMap((φ, z, _) -> ∫(∇(φ) ⋅ ∇(z) + ϵ[]*φ*z)dΩ_bg, (z, φ_hat) -> ∫(z * φ_hat)dΩ_bg, U_φ, V_φ, V_φhat)
φ_0 = φhat_to_φ(φh_hat);
φh_0 = FEFunction(U_φ, φ_0)

ϵ_tikhonov = 1e-9
Jeps(_, φ_hat) = 1/2*ϵ_tikhonov*∫(abs2 ∘ φ_hat)dΩ_bg
Jeps_spm = GridapTopOpt.StateParamMap(Jeps, φhat_to_φ)

# φ ↦ φ_u
V_φ_u = TestFESpace(model, reffe_scalar)
φ_to_φ_u = AffineFEStateMap((φ_u, z, _) -> ∫(φ_u * z)dΩ_bg, (z, φ) -> ∫(z * φ)dΩ_bg, V_φ_u, V_φ_u, U_φ)
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
Λ = Skeleton(Ω_bg) # global GP
dΛ = Measure(Λ, 2 * order)
n_Λ = get_normal_vector(Λ)
a(u,v,φ_u) = ∫(∇(v)⋅∇(u))Ωs.dΩ +
  ∫((γg*mean(h))*jump(n_Λ⋅∇(v))*jump(n_Λ⋅∇(u)))dΛ
l(v, φ_u) = ∫(v)dΓ_N

λ = 0.1
J(u, φ_u) = ∫(u)dΓ_N + ∫(λ)Ωs.dΩ

V = TestFESpace(model, reffe_scalar; dirichlet_tags=["Gamma_D"])
U = TrialFESpace(V, 0.0)
φ_u_to_u = AffineFEStateMap(a, l, U, V, V_φ_u)
J_spm = GridapTopOpt.StateParamMap(J, φ_u_to_u)

# Not neccessary anymore, remove in future
state_collection = EmbeddedCollection(model, φh_u_0; compute_cut=false) do _φh_u
  update_collection!(Ωs, _φh_u)
  (;
    :φ_u_to_u => φ_u_to_u,
    :J => J_spm
  )
end

# φ_hat ↦ J
function φ_hat_to_J(φ_hat)
  φ = φhat_to_φ(φ_hat);
  φ_u = φ_to_φ_u(φ)
  GridapTopOpt.ignore_derivatives() do
    GridapTopOpt.correct_ls!(φ_u)
    update_collection!(state_collection,FEFunction(V_φ_u,φ_u))
  end
  u = state_collection.φ_u_to_u(φ_u)
  j = state_collection.J(u,φ_u)
  j_eps = Jeps_spm(φ, φ_hat)
  return j + j_eps
end

# Optim
j0 = 1/10
function fg!(F, G, φ_hat)
  J, dJ = GridapTopOpt.val_and_gradient(φ_hat_to_J, φ_hat)
  if G !== nothing
    dJ_vec = first(dJ)/j0
    copy!(G,dJ_vec)
  end
  return J/j0
end

path_name = "/data_L2_eps$(ϵ[])_n=$(n)_tikenov=$(ϵ_tikhonov)/"
rm(path * "/data/", recursive=true, force=true)
mkpath(path * path_name)
function callback(state)
  x = state[end].metadata["x"]
  g = state[end].metadata["g(x)"]
  it = state[end].iteration
  val = state[end].value
  g_norm = state[end].g_norm
  println("Iteration: ", it, " Objective: ", val, " g_norm: ", g_norm, " ϵ: ", ϵ[])
  φh_hat = FEFunction(V_φhat,x)
  gh = FEFunction(V_φhat,g)
  writevtk(Ωs.Ω, path*path_name*"/out$(it)", cellfields=["uh"=>get_state(state_collection.φ_u_to_u)])
  writevtk(Ω_bg, path*path_name*"/out_phis$(it)", cellfields=["g"=>gh,"phi_hat" => φh_hat, "phi" => get_state(φhat_to_φ), "phi_u" => get_state(φ_to_φ_u)])
  it % 10 == 0 && GC.gc()
  return false
end

φh_hat = interpolate(Δf, V_φhat)
x = get_free_dof_values(φh_hat)
opt_method = Optim.LBFGS()
opt_options = Optim.Options(;store_trace=true, extended_trace=true, callback, f_reltol=1e-7,iterations = 10)
opt = Optim.optimize(NLSolversBase.only_fg!(fg!), x, opt_method, opt_options)

write(path*path_name*"hist.txt",string(opt))