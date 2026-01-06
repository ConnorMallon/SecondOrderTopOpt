module tmp


using Pkg
Pkg.activate(".")
using Gridap, GridapTopOpt
using SecondOrderTopOpt

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
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ = Measure(Ω,2*order)
dΓ_N = Measure(Γ_N,2*order)
vol_D = sum(∫(1)dΩ)

## Spaces
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_D"])
U = TrialFESpace(V,0.0)
V_φ = TestFESpace(model,reffe_scalar)
V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
U_reg = TrialFESpace(V_reg,0)

## Create FE functions
#initial_lsf(1.4,0.4)
ae = 0.3
be = 0.2
f0(x) = -((x[1] - 0.5)^2 / be^2 + (x[2] - 0.5)^2 / ae^2 - 1.0)
f(x) = -((x[1] - 0.5)^2 / ae^2 + (x[2] - 0.5)^2 / be^2 - 1.0)
φh = interpolate(f0,V_φ)
φhf = interpolate(f,V_φ)
#writevtk(Ω,path*"outS",cellfields=["φ"=>φh,"H(φ)"=>(H ∘ φh),"Hφf"=>(H∘φhf)])
  
φh = interpolate(initial_lsf(4,0.2),V_φ)


## Finite difference solver and level set function
evo = FiniteDifferenceEvolver(FirstOrderStencil(2,Float64),model,V_φ;max_steps)
reinit = FiniteDifferenceReinitialiser(FirstOrderStencil(2,Float64),model,V_φ;tol,γ_reinit)
ls_evo = LevelSetEvolution(evo,reinit)

## Interpolation and weak form
interp = SmoothErsatzMaterialInterpolation(η = 1*η_coeff*maximum(el_Δ))
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ
a(u,v,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ
l(v,φ) = ∫(v)dΓ_N

reinit!(ls_evo,φh)
reinit!(ls_evo,φhf)
## Optimisation functionals
J(u,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(u))dΩ
#J(u,φ) = ∫((φ-φhf)*(φ-φhf)+0*u)dΩ
dJ(q,u,φ) = ∫(κ*∇(u)⋅∇(u)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
Vol(u,φ) = ∫(1e-5((ρ ∘ φ)+0*u)/vol_D)dΩ;
#Vol(u,φ) = ∫(((φ)*(φ)+0*u)/vol_D)dΩ;




dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ


## Setup solver and FE operators
state_map = AffineFEStateMap(a,l,U,V,V_φ,diff_order = 2 )
#pcfs = PDEConstrainedFunctionals(J,[Vol],state_map)#,analytic_dJ=dJ,analytic_dC=[dVol])

objective = StateParamMap(J,state_map,diff_order=2)
constraint = StateParamMap(Vol,state_map,diff_order=2)
# function φ_to_jc(φ)
#   u = state_map(φ)
#   j = objective(u,φ)
#   c = constraint(u,φ)
#   [j+c]
# end

u0 = zero(U).free_values
function φ_to_jc(φ)
  u = state_map(φ)
  #u=u0.*φ[1]
  j = objective(u,φ)
  c = constraint(u,φ)
  [j+1e4c]
end
pcfs = CustomPDEConstrainedFunctionals(φ_to_jc,0,diff_order=2)

## Hilbertian extension-regularisation problems
α = 0.0*α_coeff*maximum(el_Δ)
a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

using Krylov, LinearMaps, ForwardDiff, Zygote
reinit!(ls_evo,φh)
p0 = get_free_dof_values(φh)
a_hilb1(p̃,q,p) =∫(α^2*∇(p̃)⋅∇(q) + p̃*q)dΩ
l_hilb1(q,p) = ∫(q*p)dΩ
hilb_filter = AffineFEStateMap(a_hilb1,l_hilb1,V_φ,V_φ,V_φ,diff_order=2)
K = assemble_matrix((u,v)->a_hilb1(u,v,p0),V_φ,V_φ)

∇f = p->Zygote.gradient(p->φ_to_jc(p)[1],p)[1]
Hṗ(p,ṗ) =  ForwardDiff.derivative(α -> ∇f(p + α*ṗ), 0)
#A = LinearMap((x)->Hṗ(p0,x),length(p0),length(p0))
#x = Krylov.cg(A,p0,verbose=2,itmax=100)

# u0h = one(U)
# d2Vdφ2 = Gridap.hessian(φ->Vol(u0h,φ),φh)
# assem_∂2Vdφ2 = SparseMatrixAssembler(V_φ,V_φ)
# d2Vdφ2_mat = assemble_matrix(d2Vdφ2,assem_∂2Vdφ2,V_φ,V_φ)

# dp = ∇f(φh.free_values)
# Hṗ_map = LinearMap((x)->Hṗ(p0,x),length(p0),length(p0)) 
# dJ_newton_results = Krylov.cg(Hṗ_map,dp,verbose=1,itmax=100,radius=0.1)#,λ=1.0)
# dJ_newton = dJ_newton_results[1]

## Optimiser

γ2 = 0.1#γ
optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
  γ=γ2,verbose=true,constraint_names=[],maxiter=100)
js = []
for (it,uh,φh) in optimiser
  push!(js,φ_to_jc(φh.free_values)[1])
  data = ["φ"=>φh,"I(φ)"=>(I ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh))]
  iszero(it % iter_mod) && writevtk(Ω,path*"out$it",cellfields=data)
  write_history(path*"/history.txt",optimiser.history)
end


sum(∫(H∘φh)dΩ)
u = state_map(φh.free_values)
uh = FEFunction(U,u)
 


using Pkg
Pkg.activate("postproc"; shared=true)
using Plots
plot()
plot!(1:length(js),js,title="Objective functional J",xlabel="Iteration",ylabel="J")
it = get_history(optimiser).niter; uh = get_state(pcfs)
writevtk(Ω,path*"outF2",cellfields=["φ"=>φh,"I(φ)"=>(I ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh))])

end