module sweep
using DrWatson; @quickactivate :SecondOrderTopOpt
include("/scratch/ek63/cm8825/SecondOrderTopOpt/scripts/hpc/hpc_utils.jl")
# sweep_params = 	Dict(
# 	"model_name" => ["debug"],
# 	"n" => [100],
# 	"initial_radius" => [0.2,0.5],
#   "cg_tol" => [0.0001],
#   "rho_upper" => [0.65],
#   "maxscripts_iters" => [10],
#   "physics" => ["thermal"],
# 	"ξ_ls" => [3],
#   "η_coeff" => [5.0],
#   "α_coeff" => [2.0],
#   "opt_order" => [1],
#   "γ" => [0.4],
# 	)
# sweep_params = 	Dict(
# 	"model_name" => ["Testing_thermal_large"],
# 	"n" => [1000],
# 	"initial_radius" => [2.0],
#   "cg_tol" => [0.01],
#   "rho_upper" => [0.75],
#   "max_iters" => [200],
#   "physics" => ["thermal"],
# 	"ξ_ls" => [4,5,6,7,8],
#   "η_coeff" => [3,4,5,6],
#   "α_coeff" => [3,4,5,6],
#   "opt_order" => [2],
#   "γ" => [0.1],
#   "λ" => [0.5,1.0,2.0,5.0]
# 	)
sweep_params = 	Dict(
	"model_name" => ["Testing_1st_order_62"],
	"n" => [100],
  "cg_tol" => [0.01],
  "rho_upper" => [0.75],
  "max_iters" => [100],
  "physics" => ["thermal"],
	"ξ_ls" => [2,3,4,5],
  "η_coeff" => [1,2,3,4,5],
  "α_coeff" => [1,2],
  "opt_order" => [2],
  "γ" => [0.1],
  "λ" => [0.5,1.0,2.0],
)
fixed_params = Dict(
  "initial_radius" => 1.0,
	)
pbs_params = Dict(
  "queue" => "express",
  "time" => 3, # hours
  "ncpus" => 1,
  "mem" => 4*32, # GB
)
pbs_sweeper(sweep_params,fixed_params,pbs_params)
end