module sweep
using DrWatson; @quickactivate :SecondOrderTopOpt
include("/scratch/ek63/cm8825/SecondOrderTopOpt/scripts/hpc/hpc_utils.jl")
# sweep_params = 	Dict(
# 	"model_name" => ["Testing2"],
# 	"n" => [100],
# 	"initial_radius" => [0.2,0.5],
#   "cg_tol" => [0.0001],
#   "rho_upper" => [0.65],
#   "max_iters" => [10],
#   "physics" => ["thermal"],
# 	"ξ_ls" => [3],
#   "η_coeff" => [5.0],
#   "α_coeff" => [2.0],
# 	)
sweep_params = 	Dict(
	"model_name" => ["Testing4"],
	"n" => [30,100],
	"initial_radius" => [1.0],
  "cg_tol" => [0.001,0.01],
  "rho_upper" => [0.75,0.85],
  "max_iters" => [100],
  "physics" => ["thermal"],#,"elastic"],
	"ξ_ls" => [5,10,20,30],
  "η_coeff" => [5,10],
  "α_coeff" => [2,4],
	)
fixed_params = Dict(
  "opt_order" => 2,
	)
pbs_sweeper(sweep_params,fixed_params)
end