module plotting

using SecondOrderTopOpt
using DrWatson
using DataFrames
using PlotlyLight

results_path = projectdir()*"/data/sims_raw/"
model = "test_case_2"
results  = DrWatson.collect_results(results_path*model)
results2 = filter(row -> row.cg_tol == 0.01 , results)
trace = results2.trace[1]
its = 1:length(trace)
p = plot(x = its, y = trace, type="scatter", mode="lines+markers")  # Make plot

end