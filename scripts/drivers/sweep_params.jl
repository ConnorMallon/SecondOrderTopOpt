module sweep
using DrWatson; @quickactivate :SecondOrderTopOpt
include("/scratch/ek63/cm8825/SecondOrderTopOpt/scripts/hpc/hpc_utils.jl")

sweep_params = 	Dict(
	"model_name" => ["Testing"],
	"n" => [30],
	"initial_radius" => [1.0],
  "cg_tol" => [0.01],
  "rho_upper" => [0.85],
  "max_iters" => [1,2],
  "ξ_ls" => [70],
  "physics" => ["thermal"],
  "opt_order" => [2],
	)
fixed_params = Dict(
	"ξ_ls" => 70,
	)
pbs_sweeper(sweep_params,fixed_params)
end