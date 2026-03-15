module SecondOrderTopOpt

using Gridap
using GridapTopOpt
# using GridapTopOpt: assemble_adjoint_matrix
# using Gridap.Helpers, Gridap.Algebra, Gridap.TensorValues, Gridap.Geometry, Gridap.CellData, Gridap.Fields, Gridap.Arrays, Gridap.ReferenceFEs, Gridap.FESpaces,  Gridap.MultiField, Gridap.Polynomials

using GridapTopOpt: StateParamMap, AbstractFEStateMap, NonlinearFEStateMap, AffineFEStateMap
using GridapTopOpt: get_trial_space, get_test_space, get_aux_space, get_state, get_parameter
# using Gridap.Algebra: NLSolversCache, NewtonRaphsonCache
# using GridapSolvers.NonlinearSolvers: NewtonCache
# using LinearAlgebra
# using Optim
using Zygote
using Optim

#using ChainRulesCore
#using ForwardDiff

import GridapTopOpt: AbstractPDEConstrainedFunctionals, PDEConstrainedFunctionals, get_aux_space, AbstractLevelSetEvolution

include("Newton.jl")

export optimise
export Optim_KrylovTrustRegion

end