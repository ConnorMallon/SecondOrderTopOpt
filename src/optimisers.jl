struct OptimisationProblem 
  pcfs           #:: AbstractPDEConstrainedFunctionals
  filter            #:: Function
  vel_ext
  ls_evo            #:: AbstractLevelSetEvolution
  interp
  φ
end

struct Result
  state_map
  trace
  φ
end

function construct_second_order_map(state_map::NonlinearFEStateMap)
  res = state_map.res
  U = get_trial_space(state_map)
  V = get_test_space(state_map)
  V_φ = get_aux_space(state_map)
  return NonlinearFEStateMap(res,U,V,V_φ,diff_order=2)
end

function construct_second_order_map(state_map::AffineFEStateMap)
  a = state_map.biform
  l = state_map.liform
  U = get_trial_space(state_map)
  V = get_test_space(state_map)
  V_φ = get_aux_space(state_map)
  return AffineFEStateMap(a,l,U,V,V_φ, diff_order=2)
end

function construct_second_order_objective(objective::StateParamMap)
  F = objective.F
  U,V_φ = objective.spaces
  assem_U, assem_deriv = objective.assems
  return StateParamMap(F,U,V_φ,assem_U,assem_deriv,diff_order=2)
end

function optimise(θ, optimisation_problem::OptimisationProblem, ::Val{2})
  pcfs, filter , vel_ext, ls_evo, φ  = optimisation_problem.pcfs, optimisation_problem.filter, optimisation_problem.vel_ext, optimisation_problem.ls_evo, optimisation_problem.φ

  cg_tol = θ["cg_tol"]
  rho_upper = θ["rho_upper"]
  initial_radius = θ["initial_radius"]
  max_iters = θ["max_iters"]

  state_map = construct_second_order_map(pcfs.state_map)
  objective = construct_second_order_objective(pcfs.J)
  constraint = construct_second_order_objective(pcfs.C[1])
  @assert length(pcfs.C) == 1 "Only one constraint is currently supported in Optim_KrylovTrustRegion optimiser."


  

  # Trust region Newton-CG with Optim.jl
  i=0
  
  φ_ = filter(φ)
  u = state_map(φ_)
  j = objective(u,φ_)
  c = constraint(u,φ_)
  @show j+c
  trace0 = [j+c]
  writevtk(get_triangulation(get_aux_space(state_map)),"/scratch/ek63/cm8825/SecondOrderTopOpt/data/φ2hf_0",cellfields=["Iφh_unfiltered"=>optimisation_problem.interp.I∘FEFunction(get_aux_space(state_map),φ),"Iφh"=>optimisation_problem.interp.I∘FEFunction(get_aux_space(state_map),φ_),"uh"=>FEFunction(get_trial_space(state_map),u)])
  dadad
  T = typeof(φ)
  function f(φ)
    # Zygote.ignore() do
    #   if typeof(φ) == T # avoiding trying to reinit when φ is a dual
    #     i+=1
    #     if i % 10 == 0
    #       println("reinitialising")
    #       φh = FEFunction(get_aux_space(state_map),φ) 
    #       #reinit!(ls_evo,φh)
    #     end
    #   end
    # end
    φ_ = filter(φ)
    u = state_map(φ_)
    Zygote.ignore() do
      if typeof(φ) == T
        i += 1
        φh = FEFunction(get_aux_space(state_map),φ_) 
        uh = FEFunction(get_trial_space(state_map),u)
        writevtk(get_triangulation(get_aux_space(state_map)),"/scratch/ek63/cm8825/SecondOrderTopOpt/data/φhf_$i",cellfields=["Iφh"=>optimisation_problem.interp.I∘φh,"uh"=>uh])
      end
    end
    j = objective(u,φ_) 
    c = constraint(u,φ_)
    @show j+c
    return j+c
  end
  function fg!(G,φ)
    value, grad = val_and_gradient(f,φ)
    copyto!(G, grad[1])
    return value
  end
  function hv!(Hv, φ, v)
    hv = Hvp(f, φ, v) 
    println("Hv running")
    copyto!(Hv, hv)
    Hv
  end
  @show sum(φ)
  d = Optim.TwiceDifferentiableHV(f,fg!,hv!,φ)
  optim_result = Optim.optimize(d, φ, 
                          Optim.KrylovTrustRegion(
                                        initial_radius = initial_radius,
                                        cg_tol = cg_tol,
                                        rho_upper = rho_upper,
                                        #eta = 0.2
                                        ),
                          Optim.Options(g_tol = 1e-12,
                                        iterations = max_iters,
                                        store_trace = true,
                                        show_trace = true,
                                        #extended_trace = true
                                        ))
  φ = optim_result.minimizer
  val(optim_result) = optim_result.value
	trace = vcat(trace0,val.(optim_result.trace))
  return Result(state_map, trace, φ)
end

function optimise(θ, optimisation_problem::OptimisationProblem, ::Val{1})
  pcfs = optimisation_problem.pcfs
  # filter = optimisation_problem.filter
  @show vel_ext = optimisation_problem.vel_ext
  ls_evo = optimisation_problem.ls_evo
  φ = optimisation_problem.φ
  # J = pcfs.J
  # C = pcfs.C
  γ = θ["γ"]
  λ = θ["λ"]
  max_iters = θ["max_iters"]

  function φ_to_jc(φ)
    u = pcfs.state_map(φ)
    j = pcfs.J(u,φ)
    c = pcfs.C[1](u,φ)
    j+λ*c
  end

  function φ_to_jc2(φ)
    u = pcfs.state_map(φ)
    j = pcfs.J(u,φ)
    c = pcfs.C[1](u,φ)
    j+λ*c
  end

  pcfs_L = CustomPDEConstrainedFunctionals(φ_to_jc,0;pcfs.state_map)
  ph = FEFunction(get_aux_space(pcfs.state_map),φ)
  optimiser = AugmentedLagrangian(pcfs_L,ls_evo,vel_ext,ph;
    γ,verbose=true,constraint_names=[],maxiter=max_iters)
  #trace = []
  u0 = pcfs.state_map(φ)
  uh0 = FEFunction(get_trial_space(pcfs.state_map),u0)
  ph0 = FEFunction(get_aux_space(pcfs.state_map),φ)
  writevtk(get_triangulation(get_aux_space(pcfs.state_map)),"/scratch/ek63/cm8825/SecondOrderTopOpt/data/φ1hf_0",cellfields=["Iφh"=>optimisation_problem.interp.I∘ph0,"uh"=>uh0])
  trace = [pcfs.J(uh0,ph0) + pcfs.C[1](uh0,ph0)]
  @show trace

  for (it,uh,φh) in optimiser
    j = pcfs.J(uh,φh)
    c = pcfs.C[1](uh,φh)
    @show j+c
    push!(trace, j+c)
    data = ["φ"=>φh,"H(φ)"=>(optimisation_problem.interp.I ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh]
    iszero(it % 10) && writevtk(get_triangulation(get_aux_space(pcfs.state_map)), "/scratch/ek63/cm8825/SecondOrderTopOpt/data/tmp1/out$γ$it", cellfields=data)
    #write_history(path*"/history.txt",optimiser.history)
  end
  #it = get_history(optimiser).niter; uh = get_state(pcfs)
  #writevtk(Ω,path*"out$it",cellfields=["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])

  @show trace
  return Result(pcfs.state_map, trace, ph.free_values)
end