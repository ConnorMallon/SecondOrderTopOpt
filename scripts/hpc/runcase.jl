module runcase
using Pkg
Pkg.activate(".")
using SecondOrderTopOpt; println("packages activated")
job_id = ARGS[1]
job_array_id = parse(Int,ARGS[2])
run_case_function(job_id, job_array_id)
end