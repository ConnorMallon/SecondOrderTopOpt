module Plotting

using SecondOrderTopOpt
using DrWatson
using DataFrames
using PlotlyLight

plot_traces = Config[]

# Second order trace for the thermal problem
results_path = projectdir()*"/data/sims_raw/"
model = "Testing4"
_results  = DrWatson.collect_results(results_path*model)
results = filter(r -> r.max_iters == 50 && r.η_coeff == 5.0 && r.α_coeff == 4.0 && r.cg_tol == 0.01 && r.rho_upper == 0.75 && r.ξ_ls==5 && r.n == 100, _results)
trace = results.trace

push!(plot_traces, Config(
    x = 1:length(trace[1]),
    y = trace[1],
    type = "scatter",
    mode = "lines+markers",
    name = "Newton-CG",
))

model = "Testing_1st_order"
_results  = DrWatson.collect_results(results_path*model)
@show _results.γ
results = filter(r -> r.η_coeff == 5.0 && r.α_coeff == 4.0 && r.ξ_ls==5 && r.n == 200 && r.physics == "thermal", _results)

trace = results.trace
γs = results.γ
for i in 1:min(length(γs), length(trace))
    yi = trace[i]
    push!(plot_traces, Config(
        x = 1:length(yi),
        y = yi,
        type = "scatter",
        mode = "lines+markers",
        name = "$(γs[i])",
    ))
end

p = Plot(
    plot_traces,
    Config(
        title = Config(text = "Two datasets"),
        xaxis = Config(title = Config(text = "x")),
        yaxis = Config(title = Config(text = "y")),
    ),
)


end