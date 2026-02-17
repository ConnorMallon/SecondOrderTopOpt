module ParameterEstimation

using SecondOrderTopOpt
using Gridap, GridapTopOpt
using Optim, LineSearches, Zygote
path = "./results/ParameterEstimation/"; mkpath(path)

## Parameters
order = 1
xmax=ymax=1.0
dom = (0,xmax,0,ymax)
el_size = (30,30)
iterations = 300
γ = 0.1

## FE Setup
model = CartesianDiscreteModel(dom,el_size)
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe_scalar;dirichlet_tags=[1,2,5,7]) #"boundary")
U = TrialFESpace(V,0.0)
V_φ = TestFESpace(model,reffe_scalar)

## initial and target conductivities 
f0(x) = 1 
f(x) = 1 + 0.5 * sin(2*π*x[1]) * sin(2*π*x[2]) 
φh = interpolate(f0,V_φ)
φhf = interpolate(f,V_φ)

# state eq and objective
f(v) = 1.0
a(u,v,φ) = ∫( φ*∇(u)⋅∇(v) )dΩ
l(v,φ) = ∫( f*v )dΩ
opf = AffineFEOperator((u,v)->a(u,v,φhf),v->l(v,φhf),U,V)
uhf = solve(opf)
J(u,φ) = ∫(1e8*(u-uhf)*(u-uhf))dΩ + ∫(0*∇(φ)⋅∇(φ))dΩ
state_map = AffineFEStateMap(a,l,U,V,V_φ,diff_order = 2 )
state_map(φh)
objective = StateParamMap(J,state_map,diff_order=2)
function φ_to_jc(φ)
  u = state_map(φ)
  j = objective(u,φ)
  [j]
end
js=[]
function f(x::Vector)
  #println("calling f")
  j = φ_to_jc(x)[1]
  push!(js,j)
  j
end
function fg!(G,x)
  #println("calling fg!")
  F,Gs = Zygote.withgradient(p->φ_to_jc(p)[1], x)
  copyto!(G, Gs[1])
  F[1]
end
result = Optim.optimize(f,fg!,copy(φh.free_values), 
         LBFGS(linesearch = LineSearches.BackTracking()),
         #GradientDescent(linesearch = LineSearches.BackTracking()),
         Optim.Options(
                          iterations = iterations,
                          show_trace = true,
                          store_trace = true,
                          x_abstol = 0,
                          f_abstol= 0,
                          g_tol = 0
            ))
φf = Optim.minimizer(result)
φfr = FEFunction(V_φ,φf)
writevtk(Ω,path*"result",cellfields=["φ_initial"=>φh,"φ_recovered"=>φfr,"φ_target"=>φhf])
# plot(1:length(js),js)

end