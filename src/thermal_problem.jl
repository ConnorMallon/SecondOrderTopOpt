function problem_from_physics(θ, ::Val{:thermal}) 
  n = θ["n"]
  η_coeff = θ["η_coeff"]
  α_factor = θ["α_coeff"]
  ξ_ls = θ["ξ_ls"]
  γ = θ["γ"]

  ## Parameters
  order = 1
  xmax=ymax=1.0
  prop_Γ_N = 0.2
  prop_Γ_D = 0.2
  dom = (0,xmax,0,ymax)
  el_size = (n,n)
  γ_reinit = 0.5
  max_steps = floor(Int,order*minimum(el_size)/10)
  tol = 1/(5order^2)/minimum(el_size)
  α_coeff = α_factor*4max_steps*γ
  vf = 0.4
  iter_mod = 10

  ## FE Setup
  model = CartesianDiscreteModel(dom,el_size);
  el_Δ = get_el_Δ(model)
  f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax*prop_Γ_D + eps() ||
      x[2] >= ymax-ymax*prop_Γ_D - eps()))
  f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/2 - eps() <= x[2] <=
      ymax/2+ymax*prop_Γ_N/2 + eps())
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ = Measure(Ω,2*order)
  dΓ_N = Measure(Γ_N,2*order)
  vol_D = sum(∫(1)dΩ)

  ## Spaces
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,0.0)
  V_φ = TestFESpace(model,reffe_scalar)

  ## Create FE functions
  #φh = interpolate(initial_lsf(4,0.2),V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ),ϵ=1e-3)
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  κ =1.0
  a(u,v,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ
  l(v,φ) = ∫(v)dΓ_N
  
  ## Setup solver and FE operators
  state_map = AffineFEStateMap(a,l,U,V,V_φ)

  ## Optimisation functionals
  J(u,φ) = ∫((I ∘ φ)*(κ)*∇(u)⋅∇(u))dΩ  + ∫(1e-3(DH ∘ φ))dΩ;
  Vol(u,φ) = ∫(((ρ ∘ φ) - vf + 0*u)/vol_D)dΩ;
  dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  evo = FiniteDifferenceEvolver(FirstOrderStencil(2,Float64),model,V_φ;max_steps)
  reinit = FiniteDifferenceReinitialiser(FirstOrderStencil(2,Float64),model,V_φ;tol,γ_reinit)
  ls_evo = LevelSetEvolution(evo,reinit)

  ## Setup solver and FE operators
  pcfs = PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dC=[dVol])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q,φ) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ
  l_hilb(q,φ) = ∫(q*φ)dΩ
  filter = AffineFEStateMap(a_hilb,l_hilb,V_φ,V_φ,V_φ,diff_order=2)

  φh = interpolate(initial_lsf(ξ_ls,0.2),V_φ)
  writevtk(Ω,"data/initial",cellfields=["φh"=>φh])

  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)
  vel_ext = VelocityExtension((p,q)->a_hilb(p,q,φh),U_reg,V_reg)

  p0 = φh.free_values
  optimisation_problem = OptimisationProblem(pcfs,filter,vel_ext,ls_evo,interp,p0)

  #   ## Optimiser
  #   optimiser = AugmentedLagr4angian(pcfs,ls_evo,vel_ext,φh;
  #     γ,verbose=true,constraint_names=[:Vol])
  #   for (it,uh,φh) in optimiser
  #     data = ["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh]
  #     iszero(it % iter_mod) && writevtk(Ω,"data/out$it",cellfields=data)
  #     write_history(path*"/history.txt",optimiser.history)
  #   end
  #   it = get_history(optimiser).niter; uh = get_state(pcfs)
  #   writevtk(Ω,"data/tmpout$it",cellfields=["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
  # end

  # optimiser = Optim_KrylovTrustRegion(pcfs,filter,ls_evo)
  # result = optimise(optimiser,φh.free_values)
  # return result

end