module ParameterEstimation

using SecondOrderTopOpt
using Gridap, GridapTopOpt
using Krylov, LinearMaps, ForwardDiff, Zygote, NLopt,Optim, OptimisationSciPy
path = "./results/ParameterEstimation/"; mkpath(path)

## Parameters
order = 1
xmax=ymax=1.0
dom = (0,xmax,0,ymax)
el_size = (50,50)

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
objective = StateParamMap(J,state_map,diff_order=2)
function φ_to_jc(φ)
  u = state_map(φ.+1e-6)
  j = objective(u,φ)
  [j]
end

# derivatives
∇f = p->Zygote.gradient(p->φ_to_jc(p)[1],p)[1]
Hṗ(p,ṗ) =  ForwardDiff.derivative(α -> ∇f(p + α*ṗ), 0)

# optimisation
# _it = [0]
# function my_objective_fn(x::Vector, grad::Vector)
#     if length(grad) > 0
#         F,Gs = Zygote.withgradient(p->φ_to_jc(p)[1], x)
#         copyto!(grad, Gs[1])
#     end
#     println("Objective: ",F[1]," iteration: ", _it[1])
#     _it[1] = _it[1] + 1
#     return F[1]
# end
# opt = NLopt.Opt(:LD_LBFGS, length(φh.free_values))
# maxeval!(opt, 10)
# NLopt.min_objective!(opt, my_objective_fn)
# min_f, min_x, ret = NLopt.optimize(opt, φh.free_values)

# # results
# num_evals = NLopt.numevals(opt)
# writevtk(Ω,path*"Fields",cellfields=["φ_recovered"=>FEFunction(V_φ,min_x),"φ_initial"=>φh,"φ_target"=>φhf])



function f(x::Vector)
  println("calling f")
  φ_to_jc(x)[1]
end
function fg!(G,x)
  println("calling fg!")
  F,Gs = Zygote.withgradient(p->φ_to_jc(p)[1], x)
  copyto!(G, Gs[1])
  F[1]
end
# function hv!(Hv, x, v)
#   copyto!(Hv, Hṗ(x,v))
#   Hv
# end

p0= φh.free_values

d = Optim.TwiceDifferentiableHV(f,fg!,hv!,p0)

using LineSearches
result = Optim.optimize(d, p0, Optim.GradientDescent(linesearch = LineSearches.BackTracking()),
            Optim.Options(#g_tol = 1e-12,
                          iterations = 4,
                          show_trace = true,
                          store_trace = true,
                          #linesearch = LineSearches.HagerZhang()
            ))

end