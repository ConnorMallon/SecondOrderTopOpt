"""
    struct AugmentedLagrangian <: Optimiser

An augmented Lagrangian method based on Nocedal and Wright, 2006
([link](https://doi.org/10.1007/978-0-387-40065-5)). Note that
this method will function as a Lagrangian method if no constraints
are defined in `problem::AbstractPDEConstrainedFunctionals`.

# Parameters

- `problem::AbstractPDEConstrainedFunctionals`: The objective and constraint setup.
- `ls_evolver::AbstractLevelSetEvolution`: Solver for the evolution and reinitisation equations.
- `vel_ext::AbstractVelocityExtension`: The velocity-extension method for extending
  shape sensitivities onto the computational domain.
- `history::OptimiserHistory{Float64}`: Historical information for optimisation problem.
- `converged::Function`: A function to check optimiser convergence.
- `has_oscillations::Function`: A function to check for oscillations.
- `params::NamedTuple`: Optimisation parameters.

The `has_oscillations` function has been added to avoid oscillations in the
iteration history. By default this uses a mean zero crossing algorithm as implemented
in ChaosTools. Oscillations checking can be disabled by taking `has_oscillations = (args...) -> false`.
"""
# struct Optim_KrylovTrustRegion 
#   problem           #:: AbstractPDEConstrainedFunctionals
#   a_hilb            #:: Function
#   ls_evo            #:: AbstractLevelSetEvolution
#   φ0
#   # @doc """
#   #     Wrapper for the optim.jl KrylovTrustRegion optimiser 
#   # """
#   function Optim_KrylovTrustRegion(
#     problem           :: AbstractPDEConstrainedFunctionals,
#     a_hilb           :: Function,
#     ls_evo            :: AbstractLevelSetEvolution,
#     φ0;
#     #params = NamedTuple() # where N
#     #Λ_max = 10^10, ζ = 1.1, update_mod = 5, reinit_mod = 1, γ = 0.1,
#     #os_γ_mult = 0.75, Λ_update_tol = 0.01, maxiter = 1000, verbose=false, constraint_names = map(i -> Symbol("C_$i"),1:N),
#     #converged::Function = default_al_converged, debug = false,
#     #has_oscillations::Function = default_has_oscillations,
#     #initial_parameters::Function = default_al_init_params,
#     #γ_reinit = NaN
#   ) #where N

#   #   #@assert isnan(γ_reinit) "γ_reinit has been removed from all optimisers. Please set this
#   #   #  in the corresponding reinitialiser (i.e., FiniteDifferenceReinitialiser)"

#   #   # constraint_names = map(Symbol,constraint_names)
#   #   # λ_names = map(i -> Symbol("λ$i"),1:N)
#   #   # Λ_names = map(i -> Symbol("Λ$i"),1:N)
#   #   # al_keys = [:L,:J,constraint_names...,:γ,λ_names...,Λ_names...]
#   #   # al_bundles = Dict(:C => constraint_names, :λ => λ_names, :Λ => Λ_names)
#   #   # history = OptimiserHistory(Float64,al_keys,al_bundles,maxiter,verbose)

#   #   # params = (;Λ_max,ζ,update_mod,reinit_mod,γ,os_γ_mult,Λ_update_tol,debug,initial_parameters)

#     new(problem,ls_evolver,a_hil,φ0)#,history,converged,has_oscillations,params,φ0)
#   end
# end

# struct Optim_KrylovTrustRegion 
#   problem           #:: AbstractPDEConstrainedFunctionals
#   a_hilb            #:: Function
#   ls_evo            #:: AbstractLevelSetEvolution
#   φ0
# end

struct Optim_KrylovTrustRegion 
  problem           #:: AbstractPDEConstrainedFunctionals
  filter            #:: Function
  ls_evo            #:: AbstractLevelSetEvolution
end

# function Optim_KrylovTrustRegion(
#   problem::AbstractPDEConstrainedFunctionals,
#   a_hilb::Function,
#   ls_evo::AbstractLevelSetEvolution,
#   φ0;
#   params = NamedTuple()
# )
#   return Optim_KrylovTrustRegion(problem, ls_evo, a_hilb, params, φ0)
# end

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

function optimise(optimiser::Optim_KrylovTrustRegion,p,I)
  
pcfs, filter , ls_evo  = optimiser.problem, optimiser.filter, optimiser.ls_evo

state_map = construct_second_order_map(pcfs.state_map)
objective = construct_second_order_objective(pcfs.J)
constraint = construct_second_order_objective(pcfs.C[1])
@assert length(pcfs.C) == 1 "Only one constraint is currently supported in Optim_KrylovTrustRegion optimiser."

# Trust region Newton-CG with Optim.jl
i=0
function f(x)
  Zygote.ignore() do
    if typeof(x) == typeof(p)
      println("reinitialising")
      φh = FEFunction(get_aux_space(state_map),x) 
      reinit!(ls_evo,φh)
    end
  end
  φ = filter(x)
  u = state_map((φ))

  Zygote.ignore() do
    if typeof(x) == typeof(p)
      i+=1
      φh = FEFunction(get_aux_space(state_map),x) 
      uh = FEFunction(get_trial_space(state_map),u)
      writevtk(get_triangulation(get_aux_space(state_map)),"data/φh_$i",cellfields=["Iφh"=>I∘φh,"uh"=>uh])
    end
  end

  j = objective(u,φ) 
  c = constraint(u,φ)
  return j+c
end
function fg!(G,x)
	value, grad = val_and_gradient(f,x)
	copyto!(G, grad[1])
	return value
end
function hv!(Hv, x, v)
	hv = Hvp(f,p,v) 
	println("Hv running")
	copyto!(Hv, hv)
	Hv
end
d = Optim.TwiceDifferentiableHV(f,fg!,hv!,p)
result = Optim.optimize(d, p, 
												Optim.KrylovTrustRegion(
																			initial_radius = 1.0,
																			cg_tol = 0.01,
																			rho_upper = 0.85,
																			#eta = 0.2
																			),
												Optim.Options(g_tol = 1e-12,
																			iterations = 30,
																			store_trace = true,
																			show_trace = true,
																			extended_trace = true
																			))

end
