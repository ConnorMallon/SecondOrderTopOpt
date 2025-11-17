#############################################################################
# ṗ->u̇ : # Solving the "incremental state equation" ∂R/∂u * u̇ = - ∂R/∂p * ṗ #
#############################################################################

function incremental_state_pushforward(φ_to_u,ṗ)
  # retrieve cached data
  U,V,V_φ = φ_to_u.spaces 
  res = φ_to_u.res
  uh = get_state(φ_to_u)
  φh = get_parameter(φ_to_u)
  
  # this should be retrieved from whatever matrix is used in the forward pass -- how can it be done cleanly so we dont have to write seperate methods for Affine and Nonlinear maps
  dv = get_fe_basis(V)
  ∂R∂u = Gridap.jacobian(uh->res(uh,dv,φh),uh) 
  ∂R∂u_mat = assemble_matrix(∂R∂u,U,V)  

  # once per outer iteration
  dv = get_fe_basis(V)
  ∂R∂φ = Gridap.jacobian(φ->res(uh,dv,φ),φh)
  ∂R∂φ_mat = assemble_matrix(∂R∂φ,V_φ,V)

  # once per inner iteration
  u̇ = ∂R∂u_mat \ (-∂R∂φ_mat * ṗ)
end 

######################################################################
# u̇ -> du̇, dṗ: Computing the increments of the objective functional #
######################################################################

function incremental_objective_partials(F,uh,φh,spaces)
  U,V_φ = spaces
  
  # ∂²J / ∂u² * u̇
  ∂2J∂u2 = Gridap.hessian(uh->F(uh,φh),uh)
  ∂2J∂u2_mat = assemble_matrix(∂2J∂u2,U,U)

  # ∂/∂p (∂J/∂u ) * ṗ
  ∂J∂u(uh,φh) = Gridap.gradient(uh->F(uh,φh),uh)
  ∂2J∂u∂φ = Gridap.jacobian(φ->∂J∂u(uh,φ),φh)
  ∂2J∂u∂φ_mat = assemble_matrix(∂2J∂u∂φ,V_φ,U)

  # ∂²J / ∂p² * ṗ
  ∂2J∂φ2 = Gridap.hessian(φ->F(uh,φ),φh)
  ∂2J∂φ2_mat = assemble_matrix(∂2J∂φ2,V_φ,V_φ)

  # ∂/∂u (∂J / ∂p) * u̇
  ∂J∂φ(uh,φh) = Gridap.gradient(φ->F(uh,φ),φh)
  ∂2J∂φ∂u = Gridap.jacobian(uh->∂J∂φ(uh,φh),uh)
  ∂2J∂φ∂u_mat = assemble_matrix(∂2J∂φ∂u,U,V_φ)

  return ∂2J∂u2_mat, ∂2J∂u∂φ_mat, ∂2J∂φ2_mat, ∂2J∂φ∂u_mat
end

function incremental_objective_pushforward(u_to_j,u̇,ṗ)
  # retrieve cached data
  spaces = u_to_j.spaces
  F = u_to_j.F
  uh = get_state(u_to_j)
  φh = get_parameter(u_to_j)

  # once per outer iteration
  ∂2J∂u2_mat, ∂2J∂u∂φ_mat, ∂2J∂φ2_mat, ∂2J∂φ∂u_mat = incremental_objective_partials(F,uh,φh,spaces)

  # once per inner iteration
  dṗ = ∂2J∂φ2_mat*ṗ + ∂2J∂φ∂u_mat*u̇
  du̇ = ∂2J∂u2_mat*u̇ + ∂2J∂u∂φ_mat*ṗ

  return du̇, dṗ
end

################################################################################################################
# du̇->dṗ : Solving the "incremental adjoint equation" ∂R/∂uᵗ * λ⁻ = du̇ - ∂²R/∂u² * u̇ * λ - ∂/∂p(∂R/∂u) * ṗ * λ #
################################################################################################################  

function incremental_adjoint_partials(res,uh,φh,λh,spaces)
  U,V,V_φ = spaces

  # ∂²R / ∂u² * u̇ * λ
  ∂2R∂u2 = Gridap.hessian(uh->res(uh,λh,φh),uh) 
  ∂2R∂u2_mat = assemble_matrix(∂2R∂u2,U,V)  

  # ∂/∂p (∂R/∂u * λ) * ṗ
  ∂R∂u_λ(uh,φh) = Gridap.gradient(uh->res(uh,λh,φh),uh)
  ∂2R∂u∂φ = Gridap.jacobian(φ->∂R∂u_λ(uh,φ),φh) 
  ∂2R∂u∂φ_mat = assemble_matrix(∂2R∂u∂φ,V_φ,V)

  # ∂²R / ∂p² * ṗ * λ
  ∂2R∂φ2 = Gridap.hessian(φh->res(uh,λh,φh),φh)
  ∂2R∂φ2_mat = assemble_matrix(∂2R∂φ2,V_φ,V_φ)

  # ∂/∂u (∂R/∂p * λ) * ṗ
  ∂R∂φ_λ(uh,φh) = Gridap.gradient(φh->res(uh,λh,φh),φh)
  ∂2R∂φ∂u = Gridap.jacobian(uh->∂R∂φ_λ(uh,φh),uh) 
  ∂2R∂φ∂u_mat = assemble_matrix(∂2R∂φ∂u,U,V_φ)

  return ∂2R∂u2_mat, ∂2R∂u∂φ_mat, ∂2R∂φ2_mat, ∂2R∂φ∂u_mat
end

function incremental_adjoint_pushforward(φ_to_u,u̇,ṗ,du̇)
  # retrieve cached data
  spaces = φ_to_u.spaces
  U,V,V_φ = spaces
  res = φ_to_u.res
  adjoint_ns, _, λ = φ_to_u.cache.adj_cache
  λh = FEFunction(V,λ)
  uh = get_state(φ_to_u)
  φh = get_parameter(φ_to_u)

  # new caches - needs work
  λ⁻ = copy(λ)

  # once per outer iteration
  ∂2R∂u2_mat, ∂2R∂u∂φ_mat, ∂2R∂φ2_mat, ∂2R∂φ∂u_mat = incremental_adjoint_partials(res,uh,φh,λh,spaces)

  # once per inner iteration
  du̇_R = ∂2R∂u2_mat*u̇ + ∂2R∂u∂φ_mat*ṗ
  dφ̇_R = ∂2R∂φ2_mat*ṗ + ∂2R∂φ∂u_mat*u̇
  λ⁻ = solve!(λ⁻,adjoint_ns,du̇-du̇_R)
  λ⁻h = FEFunction(V,λ⁻)
  ∂R∂p_λ⁻ = Gridap.gradient(φh->res(uh,λ⁻h,φh),φh)
  ∂R∂p_mat_λ⁻ = assemble_vector(∂R∂p_λ⁻,V_φ)
  dṗ_adj = - ∂R∂p_mat_λ⁻ - dφ̇_R
end