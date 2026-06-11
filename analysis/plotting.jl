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

model = "Testing_1st_order_6"#"Testing4_replication_reinit_initial"
_results  = DrWatson.collect_results(results_path*model)
@show _results.γ
results = filter(r -> r.η_coeff == 2.0 && r.α_coeff == 1.0 && r.ξ_ls==5 && r.physics == "thermal" && r.λ==1.0 && r.γ<0.4, _results)

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

    # push!(plot_traces, Config(
    #     x = 1:length(yi),
    #     y = -1 .*(x),
    #     type = "scatter",
    #     mode = "lines+markers",
    #     name = "$(γs[i])",
    # ))



p = Plot(
    plot_traces,
    Config(
        title = Config(text = "Two datasets"),
        xaxis = Config(title = Config(text = "x")),
        yaxis = Config(title = Config(text = "y")),
    ),
)
p.layout.xaxis.range = [0, 50]
# p.layout.yaxis.range = [0, 0.7]
p.layout.yaxis.type = "log"

p

end