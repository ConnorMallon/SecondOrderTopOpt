module tmp
using Pkg
Pkg.activate(".")
using Gridap, GridapTopOpt
using SecondOrderTopOpt
using Krylov, LinearMaps, ForwardDiff, Zygote
using Optim
path = "./results/thermal_compliance_ALM/"

## Parameters
order = 1
xmax=ymax=1.0
prop_Γ_N = 0.2
prop_Γ_D = 0.2
dom = (0,xmax,0,ymax)
el_size = (50,50)
γ = 0.1
γ_reinit = 0.5
max_steps = floor(Int,order*minimum(el_size)/10)
tol = 1/(5*order^2)/minimum(el_size)
κ = 1
vf = 0.4
η_coeff = 2
α_coeff = 4max_steps*γ
iter_mod = 10
mkpath(path)

## FE Setup
model = CartesianDiscreteModel(dom,el_size);
el_Δ = get_el_Δ(model)
f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax*prop_Γ_D + eps() ||
    x[2] >= ymax-ymax*prop_Γ_D - eps()))
f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/2 - eps() <= x[2] <=
    ymax/2+ymax*prop_Γ_N/2 + eps())
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_N,"Gamma_N")

## Triangulations and measures
Ω = Triangulation(model)
#Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ = Measure(Ω,2*order)
#dΓ_N = Measure(Γ_N,2*order)
vol_D = sum(∫(1)dΩ)

## Spaces
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe_scalar;dirichlet_tags="boundary")
U = TrialFESpace(V,0.0)
V_φ = TestFESpace(model,reffe_scalar)
V_reg = TestFESpace(model,reffe_scalar)#;dirichlet_tags=["Gamma_N"])
U_reg = TrialFESpace(V_reg,0)

## Create FE functions
#initial_lsf(1.4,0.4)
ae = 0.3
be = 0.2
f0(x) = 10 -((x[1] - 0.5)^2 / be^2 + (x[2] - 0.5)^2 / ae^2 - 1.0) 
f(x) = 10 -((x[1] - 0.5)^2 / ae^2 + (x[2] - 0.5)^2 / be^2 - 1.0) 
φh = interpolate(f0,V_φ)
φhf = interpolate(f,V_φ)
writevtk(Ω,path*"outS",cellfields=["φ"=>φh,"φf"=>φhf])

## Finite difference solver and level set function
evo = FiniteDifferenceEvolver(FirstOrderStencil(2,Float64),model,V_φ;max_steps)
reinit = FiniteDifferenceReinitialiser(FirstOrderStencil(2,Float64),model,V_φ;tol,γ_reinit)
ls_evo = LevelSetEvolution(evo,reinit)

## Interpolation and weak form
interp = SmoothErsatzMaterialInterpolation(η = 10*η_coeff*maximum(el_Δ))
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

## Hilbertian extension-regularisation problems
α0 = α_coeff*maximum(el_Δ)
α = α0*0
a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

p0 = get_free_dof_values(φh)
a_hilb1(p̃,q,p) =∫(4*α0^2*∇(p̃)⋅∇(q) + p̃*q)dΩ
l_hilb1(q,p) = ∫(q*p)dΩ

opf = AffineFEOperator((p,q)->a_hilb1(p,q,φhf),q->l_hilb1(q,φhf),V_φ,V_φ)
φhf_filtered = solve(opf)

op0 = AffineFEOperator((u,v)->a_hilb1(u,v,φh),v->l_hilb1(v,φh),V_φ,V_φ)
φh_filtered = solve(op0)

hilb_filter = AffineFEStateMap(a_hilb1,l_hilb1,V_φ,V_φ,V_φ)

f(v) = 1.0
a(u,v,φ) = ∫((φ)*κ*∇(u)⋅∇(v))dΩ
l(v,φ) = ∫(f*v)dΩ

opf = AffineFEOperator((u,v)->a(u,v,φhf_filtered),v->l(v,φhf_filtered),U,V)
uhf = solve(opf)

op0 = AffineFEOperator((u,v)->a(u,v,φh_filtered),v->l(v,φh_filtered),U,V)
uh0 = solve(op0)

J(u,φ) = ∫(1e8*(u-uhf)*(u-uhf)+0*φ)dΩ #+ ∫(1e-8*∇(φ)⋅∇(φ))dΩ

@show sum(J(uh0,φh_filtered))

state_map = AffineFEStateMap(a,l,U,V,V_φ,diff_order = 2 )
objective = StateParamMap(J,state_map,diff_order=2)

function φ_to_jc(φ_)
#   φ = hilb_filter(φ_)
φ = φ_
  u = state_map(φ.+1e-6)
  j = objective(u,φ)
  [j]
end

∇f = p->Zygote.gradient(p->φ_to_jc(p)[1],p)[1]
Hṗ(p,ṗ) =  ForwardDiff.derivative(α -> ∇f(p + α*ṗ), 0)

# γ2 = 0.2
# pcfs = CustomPDEConstrainedFunctionals(φ_to_jc,0)#,diff_order=2)
# optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
#   γ=γ2,verbose=true,constraint_names=[],maxiter=400)
# js = []
# for (it,uh,φh) in optimiser
#   push!(js,φ_to_jc(φh.free_values)[1])
#   data = ["φ"=>φh,"I(φ)"=>(I ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh))]
#   iszero(it % iter_mod) && writevtk(Ω,path*"out$it",cellfields=data)
#   write_history(path*"/history.txt",optimiser.history)
# end

# writevtk(Ω,path*"outF21",cellfields=["φ"=>φh,"I(φ)"=>(I ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh))])


# uh0 = solve(op0)
# sum(J(uh0,φh))

# γ2 = 0.1
# pcfs2 = CustomPDEConstrainedFunctionals(φ_to_jc,0,diff_order=2)
# α = 0α_coeff*maximum(el_Δ)
# a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
# vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

# optimiser = AugmentedLagrangian(pcfs2,ls_evo,vel_ext,φh;
#   γ=γ2,verbose=true,constraint_names=[],maxiter=20)
# for (it,uh,φh) in optimiser
#   push!(js,φ_to_jc(φh.free_values)[1])
#   data = ["φ"=>φh,"I(φ)"=>(I ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh))]
#   iszero(it % iter_mod) && writevtk(Ω,path*"out$it",cellfields=data)
#   write_history(path*"/history.txt",optimiser.history)
# end

using Pkg
Pkg.activate("postproc"; shared=true)
using NLopt
# using Plots
# plot()
# plot!(1:length(js),js,title="Objective functional J",xlabel="Iteration",ylabel="J")#,ylims=(0.14,0.145))
# savefig(path*"objective_history.png")







using NLopt
function my_objective_fn(x::Vector, grad::Vector)
    #if length(grad) > 0
        F,Gs = Zygote.withgradient(p->φ_to_jc(p)[1], x)
        copyto!(grad, Gs[1])
    #end
    @show sum(x)
    @show sum(grad)
    return F[1]
end

my_objective_fn(p0,zeros(length(p0)))

F,Gs =  Zygote.withgradient(p->φ_to_jc(p)[1], p0)
sum(Gs[1])
F


p0 = φh.free_values
my_objective_fn(p0,zero(p0))
φ_to_jc(φh.free_values)



φ_to_jc(φh.free_values)

opt = NLopt.Opt(:LD_LBFGS, length(p0))
NLopt.xtol_rel!(opt, 0)
NLopt.ftol_rel!(opt, 0)


NLopt.min_objective!(opt, my_objective_fn)
min_f, min_x, ret = NLopt.optimize(opt, φh.free_values)
num_evals = NLopt.numevals(opt)
@show ret








# ## Hilbertian extension-regularisation problems
# # α = α_coeff*h/
# a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
# vel_ext = VelocityExtension(a_hilb,V_φ,V_φ)
# _it = [0]
# function my_objective_fn(ρ::Vector, grad::Vector)
#   ρh = FEFunction(V_φ,ρ)
#   J, _, dJ, dC = evaluate!(pcfs,ρh)
#   if length(grad) > 0
#     copy!(grad,dJ)
#   end
#   println("Iteration: ", _it[1], " Objective: ", J)#, " Volume: ", sum(Vol(uh,ρh)))
#   #writevtk(Ω,path*"out$(_it[1])",cellfields=["ρ"=>ρh,"uh"=>uh,"dJ"=>FEFunction(V_ρ,dJ),"dC"=>FEFunction(V_ρ,dC[1])])
#   _it[1] = _it[1] + 1
#   return J
# end

# # Optimser
# opt = NLopt.Opt(:LD_LBFGS, num_free_dofs(V_φ))
# NLopt.xtol_rel!(opt, 0)
# NLopt.min_objective!(opt, my_objective_fn)
# #NLopt.inequality_constraint!(opt, my_constraint_fn, 1e-6)
# φh.free_values
# min_f, min_x, ret = NLopt.optimize(opt, φh.free_values)
# num_evals = NLopt.numevals(opt)
# println(
#     """
#     objective value       : $min_f
#     solution status       : $ret
#     # function evaluation : $num_evals
#     """
# )

# φ_to_jc(0φh.free_values)


end