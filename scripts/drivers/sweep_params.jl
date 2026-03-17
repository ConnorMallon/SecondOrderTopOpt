module sweep
using DrWatson; @quickactivate :SecondOrderTopOpt
include("/scratch/ek63/cm8825/SecondOrderTopOpt/scripts/hpc/hpc_utils.jl")
# sweep_params = 	Dict(
# 	"model_name" => ["debug"],
# 	"n" => [100],
# 	"initial_radius" => [0.2,0.5],
#   "cg_tol" => [0.0001],
#   "rho_upper" => [0.65],
#   "max_iters" => [10],
#   "physics" => ["thermal"],
# 	"ξ_ls" => [3],
#   "η_coeff" => [5.0],
#   "α_coeff" => [2.0],
#   "opt_order" => [1],
#   "γ" => [0.4],
# 	)
sweep_params = 	Dict(
	"model_name" => ["Testing_elastic_3"],
	"n" => [100],
	"initial_radius" => [1.0],
  "cg_tol" => [0.001,0.01],
  "rho_upper" => [0.75],
  "max_iters" => [50],
  "physics" => ["elastic"],
	"ξ_ls" => [4,5,6],
  "η_coeff" => [3,4,5,6],
  "α_coeff" => [3,4,5,6],
  "opt_order" => [2],
  "γ" => [0.1],
  "λ" => [3.0,5.0]
	)
# sweep_params = 	Dict(
# 	"model_name" => ["Testing_1st_order_2"],
# 	"n" => [100],
#   "cg_tol" => [0.01],
#   "rho_upper" => [0.75],
#   "max_iters" => [200],
#   "physics" => ["elastic","thermal"],
# 	"ξ_ls" => [4,5],
#   "η_coeff" => [2],
#   "α_coeff" => [1],
#   "opt_order" => [1],
#   "γ" => [0.05,0.1,0.2,0.3,0.4,0.5],
#   "λ" => [3.0,5.0]
# )
fixed_params = Dict(
  "initial_radius" => 1.0,
	)
pbs_params = Dict(
  "queue" => "express",
  "time" => 6, # hours
  "ncpus" => 1,
  "mem" => 4*8, # GB
)
pbs_sweeper(sweep_params,fixed_params,pbs_params)
end