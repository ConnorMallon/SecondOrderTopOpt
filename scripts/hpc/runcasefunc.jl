function run_case_function(job_id, job_array_id)
  println("loading dict name")
  fvd = DrWatson.load(datadir("results", "dict_vector", "x=dict_vector_$(job_id).jld2"))
  fv = fvd[:"dict_vector"]
  f = fv[job_array_id]

  sweep_dict = Dict{String,Any}(string(k) => v for (k, v) in DrWatson.load(projectdir("_tmp", f), "params"))
  fixed_dict = Dict{String,Any}(string(k) => v for (k, v) in DrWatson.load(datadir("results", "dict_vector","fixed_dict_vector_$(job_id).jld2")))
  θ = merge(sweep_dict, fixed_dict)
  sname = savename(θ,"jld2",allowedtypes=(Real, String, Symbol,Vector,))
  if isempty(sname) || startswith(sname, ".")
    sname = "run_$(job_id)_$(job_array_id).jld2"
  end
  run_problem(θ,sname)
  rm(projectdir("_tmp", f))
end