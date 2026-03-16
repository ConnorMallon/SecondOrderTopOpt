function saverun(θ,problem,result,sname)
	state_map = result.state_map
	φ = result.φ
	trace = result.trace
	H = problem.interp.H
	model_name = θ["model_name"]
	@show model_name

	V_φ = get_aux_space(state_map)
	U = get_trial_space(state_map)
	V = get_test_space(state_map)
	Ω = get_triangulation(V_φ)
	φh = FEFunction(V_φ,φ)
	uh = get_state(state_map)
	vtu_data = [
		"uh" => uh, 
		"φh" => φh,
		"Hφ" => H∘φh,
		]	
	opt_results = Dict{String,Any}(
		"trace" => result.trace,
	)
	# Saving
	plots_directory = DrWatson.datadir("sims_raw/vtu/"*model_name)
	mkpath(plots_directory)
	writevtk(Ω,joinpath(plots_directory,"$(sname).vtu"),cellfields = vtu_data)
	results_directory = joinpath(DrWatson.datadir("sims_raw"), model_name)
	mkpath(results_directory)
	results_dict = Dict{String,Any}(string(k) => v for (k, v) in merge(θ, opt_results))
	@show results_dict
	tagsave(joinpath(results_directory,sname), results_dict, storepatch = true)
	println("finished")
	return true	
end