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
  T = typeof(φ)
  function f(φ)
    Zygote.ignore() do
      if typeof(φ) == T # avoiding trying to reinit when φ is a dual
        println("reinitialising")
        φh = FEFunction(get_aux_space(state_map),φ) 
        #reinit!(ls_evo,φh)
      end
    end
    φ_ = filter(φ)
    u = state_map(φ_)
    # Zygote.ignore() do
    #   if typeof(x) == typeof(p)
    #     i+=1
    #     φh = FEFunction(get_aux_space(state_map),x) 
    #     uh = FEFunction(get_trial_space(state_map),u)
    #     writevtk(get_triangulation(get_aux_space(state_map)),"data/φh_$i",cellfields=["Iφh"=>I∘φh,"uh"=>uh])
    #   end
    # end
    j = objective(u,φ_) 
    c = constraint(u,φ_)
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
	trace = val.(optim_result.trace)
  return Result(state_map, trace, φ)
end

function optimise(θ, optimisation_problem::OptimisationProblem, ::Val{1})
  pcfs, filter , vel_ext, ls_evo, φ  = optimisation_problem.pcfs, optimisation_problem.filter, optimisation_problem.vel_ext, optimisation_problem.ls_evo, optimisation_problem.φ
  γ = θ["γ"]

  vel_ext = 

  ph = FEFunction(get_aux_space(pcfs.state_map),φ)
  optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,ph;
    γ,verbose=true,constraint_names=[:Vol])
  for (it,uh,φh) in optimiser
    #data = ["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh]
    #iszero(it % iter_mod) && writevtk(Ω,path*"out$it",cellfields=data)
    #write_history(path*"/history.txt",optimiser.history)
  end
  #it = get_history(optimiser).niter; uh = get_state(pcfs)
  #writevtk(Ω,path*"out$it",cellfields=["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
  result = ()
  return result
end


