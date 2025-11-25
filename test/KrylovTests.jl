module KrylovTests

using Optim
using Gridap
using GridapTopOpt
using Test

## Parameters
order = 1
xmax=ymax=1.0
prop_Γ_N = 0.2
prop_Γ_D = 0.2
dom = (0,xmax,0,ymax)
el_size = (10,10)
γ = 0.1
γ_reinit = 0.5
max_steps = floor(Int,order*minimum(el_size)/10)
tol = 1/(5*order^2)/minimum(el_size)
κ = 1
vf = 0.4
η_coeff = 2
α_coeff = 4max_steps*γ
iter_mod = 10

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
φh = interpolate(initial_lsf(4,0.2),V_φ)

## Interpolation and weak form
interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

a(u,v,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ
l(v,φ) = ∫(v)dΓ_N

## Finite difference solver and level set function
evo = FiniteDifferenceEvolver(FirstOrderStencil(2,Float64),model,V_φ;max_steps)
reinit = FiniteDifferenceReinitialiser(FirstOrderStencil(2,Float64),model,V_φ;tol,γ_reinit)
ls_evo = LevelSetEvolution(evo,reinit)

## Setup solver and FE operators
state_map = NonlinearFEStateMap((u,v,φ)->a(u,v,φ) - l(v,φ),U,V,V_φ)
# change to linear 


## Optimisation functionals
φhd = interpolate(initial_lsf(6,0.1),V_φ) 
uhd = FEFunction(U,copy(state_map(φhd)))
J(u,φ) = ∫((u-uhd)⋅(u-uhd) + φ*0)dΩ
Vol(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
objective = GridapTopOpt.StateParamMap(J,state_map)

uhd
uh = FEFunction(U,copy(state_map(φh)))
sum(J(uh,φh))

## Hilbertian extension-regularisation problems
α = α_coeff*maximum(el_Δ)
a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

function F(p)
    u = copy(state_map(p))
    j = objective(u,p)
    [j]
end
pcfs = CustomPDEConstrainedFunctionals(F,0)

## Optimiser
optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
γ,verbose=true)

# Do a few iterations
vars, state = iterate(optimiser)

using Zygote
using ForwardDiff

function F(p)
    u = copy(state_map(p))
    j = objective(u,p)
    j
end

p = φh.free_values
G(p) = Zygote.gradient(F,p)[1]
ṗ = G(p)
Hṗ(p,ṗ) =  ForwardDiff.derivative(α -> G(p + α*ṗ), 0)
Hṗ(p,ṗ)

# Test on actual optimization problems

function f(x::Vector)
    F(x)
end

function fg!(G,x)
    copyto!(G, Zygote.gradient(F,x)[1])
    F(x)
end

function hv!(Hv, x, v)
    hv = Hṗ(x,v)
    copyto!(Hv, hv)
    Hv
end

d = Optim.TwiceDifferentiableHV(f,fg!,hv!,p)
result = Optim.optimize(d, p, Optim.KrylovTrustRegion(),
            Optim.Options(g_tol = 1e-12,
                             iterations = 20,
                             show_trace = true,
            ))
sum(p- result.minimizer)

end