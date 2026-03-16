function run_case_function(job_id, job_array_id)
  println("loading dict name")
  fvd = DrWatson.load(datadir("results", "dict_vector", "x=dict_vector_$(job_id).jld2"))
  fv = fvd[:"dict_vector"]
  f = fv[job_array_id]

  sweep_dict = DrWatson.load(projectdir("_tmp", f), "params")
  fixed_dict = DrWatson.load(datadir("results", "dict_vector","fixed_dict_vector_$(job_id).jld2"))
  @show θ = merge(sweep_dict, fixed_dict)
  
  sname = savename(sweep_dict,"jld2",allowedtypes=(Real, String, Symbol,Vector,))
  run_problem(θ,sname)
  rm(projectdir("_tmp", f))
end