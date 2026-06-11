module sweep
using Revise
using DrWatson
#using Pkg; Pkg.activate(dirname(dirname(@__DIR__))) # i.e. base project .. 
using DrWatson
using DataFrames
using SecondOrderTopOpt
include("/scratch/ek63/cm8825/SecondOrderTopOpt/scripts/hpc/hpc_utils.jl")
model_square = "testing"
sweep_params = 	Dict{String,Any}(
)
fixed_params = Dict{String,Any}(
	"model_name" => "getting_it_history",
	"n" => 100,
	"η_coeff" => 5.0,#2.0
	"α_coeff" => 2.0,#1.0
	"initial_radius" => 1.0,#2.0,
	"cg_tol" => 0.01,
	"rho_upper" => 0.75,
	"max_iters" => 50,
	"γ" => 0.2,
	"ξ_ls" => 5,
	"physics" => "thermal",
	"opt_order" => 2,
	"λ" => 1.0
	)
# fixed_params = Dict{String,Any}(
# 	"model_name" => "getting_it_history_1st_order",
# 	"n" => 100,
# 	"η_coeff" => 2.0,
# 	"α_coeff" => 1.0,
# 	"initial_radius" => 1.0,#2.0,
# 	"cg_tol" => 0.01,
# 	"rho_upper" => 0.75,
# 	"max_iters" => 100,
# 	"γ" => 0.1,
# 	"ξ_ls" => 5,
# 	"physics" => "thermal",
# 	"opt_order" => 1,
# 	"λ" => 1.0
# 	)
job_id = 0
job_array_id = 1 
save_job_dicts(sweep_params,fixed_params,job_id)
run_case_function(job_id, job_array_id)
end