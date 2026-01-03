module prof


using SecondOrderTopOpt
using Gridap, GridapTopOpt, Zygote, Optim, ForwardDiff


  ## Parameters
  order = 1
  xmax=ymax=1.0
  prop_Γ_N = 0.2
  prop_Γ_D = 0.2
  dom = (0,xmax,0,ymax)
  el_size = (50,50)
  γ = 0.1
  γ_reinit = 0.5
  max_steps = floor(Int,order*minimum(el_size)/10)
  tol = 1/(5*order^2)/minimum(el_size)
  κ = 1
  vf = 0.4
  η_coeff = 2
  α_coeff = 4max_steps*γ
  iter_mod = 10
  #mkpath(path)

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
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)

  ## Create FE functions
  φh = interpolate(initial_lsf(4,0.2),V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  a(u,v,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ
  l(v,φ) = ∫(v)dΓ_N

  ## Optimisation functionals
  J(u,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(u))dΩ
  dJ(q,u,φ) = ∫(κ*∇(u)⋅∇(u)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
  Vol(u,φ) = ∫(((ρ ∘ φ) - vf+0*u)*((ρ ∘ φ) - vf))dΩ;
  dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  evo = FiniteDifferenceEvolver(FirstOrderStencil(2,Float64),model,V_φ;max_steps)
  reinit = FiniteDifferenceReinitialiser(FirstOrderStencil(2,Float64),model,V_φ;tol,γ_reinit)
  ls_evo = LevelSetEvolution(evo,reinit)

  ## Setup solver and FE operators
  state_map = AffineFEStateMap(a,l,U,V,V_φ,diff_order = 2 )
  objective = GridapTopOpt.StateParamMap(J,state_map,diff_order=2)
  constraint = GridapTopOpt.StateParamMap(Vol,state_map,diff_order=2)
  #pcfs = PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dJ=dJ,analytic_dC=[dVol])

  ## Hilbertian extension-regularisation problems
  α = 1*α_coeff*maximum(el_Δ)
  a_hilb1(p̃,q,p) =∫(α^2*∇(p̃)⋅∇(q) + p̃*q)dΩ
  l_hilb1(q,p) = ∫(q*p)dΩ
  hilb_filter = AffineFEStateMap(a_hilb1,l_hilb1,V_φ,V_φ,V_φ,diff_order=2)

  js=[]
  cs=[]
  jcs = []
  function p_to_j(p)
    #p̃ = hilb_filter(p)
    p̃ = p 
    Zygote.ignore() do
      if isa(p̃,Vector{Float64})
        # @show sum(p̃)
        # φh = FEFunction(V_φ,p̃) 
        # reinit!(ls_evo,φh)
        # @show sum(p̃)
      end
    end

    u = state_map(p̃)
    j = objective(u,p̃)
    c = constraint(u,p̃) 
    Zygote.ignore() do
      if isa(p̃,Vector{Float64})
        φh = FEFunction(V_φ,p̃) 
        @show sum(∫(ρ∘φh)dΩ)
        println("volume violation is $c, objective is $j")
      push!(js,j)
      push!(cs,c)
      push!(jcs,j+c)
      end
    end
    [j+c]
  end

  reinit!(ls_evo,φh)

  p0= φh.free_values

  # pcfs = CustomPDEConstrainedFunctionals(p_to_j,0)

  # #function NewtonCG(pcfs::CustomPDEConstrainedFunctionals,p0)
  
    #p_to_j = pcfs.φ_to_jc
    ∇f = p->Zygote.gradient(p->p_to_j(p)[1],p)[1]
    Hṗ(p,ṗ) =  ForwardDiff.derivative(α -> ∇f(p + α*ṗ), 0)


    ∇f(p0)
    Hṗ(p0,p0)

  #   function f(x::Vector)
  #     p_to_j(x)[1]
  #   end
  #   function fg!(G,x)
  #     F,Gs = Zygote.withgradient(p->p_to_j(p)[1], x)
  #     copyto!(G, Gs[1])
  #     F[1]
  #   end
  #   function hv!(Hv, x, v)
  #     copyto!(Hv, Hṗ(x,v))
  #     Hv
  #   end

  #   p0= φh.free_values

  #   d = Optim.TwiceDifferentiableHV(f,fg!,hv!,p0)
  #   result = Optim.optimize(d, p0, Optim.KrylovTrustRegion(),
  #               Optim.Options(#g_tol = 1e-12,
  #                             iterations = 0,
  #                             show_trace = true,
  #                             store_trace = true,
  #               ))


# function g!(G,x)
#   F,Gs = Zygote.withgradient(p->p_to_j(p)[1], x)
#   copyto!(G, Gs[1])
#   G
# end


# pf = result.minimizer
# writevtk(Ω,"2ndO",cellfields=["ph"=>I∘(FEFunction(V_φ,pf))])

# uh = get_state(state_map)
# writevtk(Ω,"outs2",cellfields=["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])

# using ProfileView
# @profview profile_test(1)  # run once to trigger compilation (ignore this one)
# @profview profile_test(10)

using BenchmarkTools
#@btime f(p0)
#@btime fg!(similar(p0),p0)
#@btime hv!(similar(p0),p0,3*p0)

using LinearMaps, IterativeSolvers
A = LinearMap((x)->Hṗ(p0,x),length(p0),length(p0))
K = assemble_matrix((u,v)->a_hilb1(u,v,p0),V_φ,V_φ)

using Krylov

#x = Krylov.cg_lanczos_shift(K,p0,[1e-4],M=K,verbose=2,check_curvature=true)

x_sol = Krylov.cg_lanczos(A,p0,verbose=1,check_curvature=true)

x_sol[2].status == "negative curvature" 

sum(x_sol[1])

# Ok lets first just see.... 
# use exactly the optimiser as is.... 
# where do we hook in ???


pcfs = CustomPDEConstrainedFunctionals(p_to_j,0,diff_order=2)

## Hilbertian extension-regularisation problems
α = α_coeff*maximum(el_Δ)
a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

## Optimiser
optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
  γ,verbose=true,constraint_names=[])
for (it,uh,φh) in optimiser
  data = ["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh]
  #iszero(it % iter_mod) && writevtk("out$it",cellfields=data)
  write_history("history.txt",optimiser.history)
end
it = get_history(optimiser).niter; uh = get_state(pcfs)
writevtk(Ω,path*"out$it",cellfields=["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])




# p0 = interpolate(initial_lsf(4,0.2),V_φ).free_values
# x_sol = Krylov.cg(K+1e-3A,M=K,p0,verbose=2)#,rtol=0.2)#,check_curvature=true)
x = x_sol[1]
norm(p0.-K*x.-5e-3A*x)

end