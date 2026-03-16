module SecondOrderTopOpt

using Gridap
using GridapTopOpt
# using GridapTopOpt: assemble_adjoint_matrix
# using Gridap.Helpers, Gridap.Algebra, Gridap.TensorValues, Gridap.Geometry, Gridap.CellData, Gridap.Fields, Gridap.Arrays, Gridap.ReferenceFEs, Gridap.FESpaces,  Gridap.MultiField, Gridap.Polynomials

import GridapTopOpt: AbstractPDEConstrainedFunctionals, PDEConstrainedFunctionals, AbstractLevelSetEvolution
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

using DrWatson

include("elastic_problem.jl")
include("optimisers.jl")
include("routine.jl")
include("thermal_problem.jl")
include("saverun.jl")

export optimise
export OptimisationProblem
export run_problem

end