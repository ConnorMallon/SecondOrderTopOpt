module SecondOrderTopOpt

using Gridap
using GridapTopOpt: assemble_adjoint_matrix
using Gridap.Helpers, Gridap.Algebra, Gridap.TensorValues, Gridap.Geometry, Gridap.CellData, Gridap.Fields, Gridap.Arrays, Gridap.ReferenceFEs, Gridap.FESpaces,  Gridap.MultiField, Gridap.Polynomials

include("HessianRules.jl")

export incremental_state_pushforward
export objective_partials
export incremental_objective_pushforward
export incremental_adjoint_partials
export incremental_adjoint_value
export incremental_adjoint_pushforward

end