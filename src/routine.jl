_physics_symbol(physics::AbstractString) = Symbol(lowercase(physics))
problem_from_physics(θ) = problem_from_physics(θ, Val(_physics_symbol(θ["physics"])))

_order_int(order::Integer) = Int(order)
optimise(θ,problem) = optimise(θ,problem,Val(_order_int(θ["opt_order"])))

function run_problem(θ,sname)
  println(sname)
  problem = problem_from_physics(θ)
  result = optimise(θ, problem)
  saverun(θ,problem,result,sname)
end