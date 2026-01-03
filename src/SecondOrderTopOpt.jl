module SecondOrderTopOpt

using Gridap
using GridapTopOpt
using GridapTopOpt: assemble_adjoint_matrix
using Gridap.Helpers, Gridap.Algebra, Gridap.TensorValues, Gridap.Geometry, Gridap.CellData, Gridap.Fields, Gridap.Arrays, Gridap.ReferenceFEs, Gridap.FESpaces,  Gridap.MultiField, Gridap.Polynomials
using GridapTopOpt: get_state, get_parameter, StateParamMap, AbstractFEStateMap, NonlinearFEStateMap, AffineFEStateMap, get_plb_cache, update_incremental_adjoint_partials!, update_incremental_state_partials!
using Gridap.Algebra: NLSolversCache, NewtonRaphsonCache
using GridapSolvers.NonlinearSolvers: NewtonCache
using LinearAlgebra
using Optim
#using Zygote

using ChainRulesCore
using ForwardDiff

include("HessianRules.jl")
include("SecondOrderOptimisers.jl")

export incremental_state_pushforward
export objective_partials
export incremental_objective_pushforward
export update_incremental_adjoint_partials
export solve_incremental_adjoint
export incremental_adjoint_pushforward
export NewtonCG

end