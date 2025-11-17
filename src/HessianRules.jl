#############################################################################
# ṗ->u̇ : # Solving the "incremental state equation" ∂R/∂u * u̇ = - ∂R/∂p * ṗ #
#############################################################################

function incremental_state_pushforward(p_to_u,ṗ)
  # retrieve cached data
  U,V,V_p = p_to_u.spaces 
  res = p_to_u.res
  uh = get_state(p_to_u)
  ph = get_parameter(p_to_u)
  
  # this should be retrieved from whatever matrix is used in the forward pass -- how can it be done cleanly so we dont have to write seperate methods for Affine and Nonlinear maps
  dv = get_fe_basis(V)
  ∂R∂u = Gridap.jacobian(uh->res(uh,dv,ph),uh) 
  ∂R∂u_mat = assemble_matrix(∂R∂u,U,V)  

  # once per outer iteration
  dv = get_fe_basis(V)
  ∂R∂p = Gridap.jacobian(p->res(uh,dv,p),ph)
  ∂R∂p_mat = assemble_matrix(∂R∂p,V_p,V)

  # once per inner iteration
  u̇ = ∂R∂u_mat \ (-∂R∂p_mat * ṗ)
end 

######################################################################
# u̇ -> du̇, dṗ: Computing the increments of the objective functional #
######################################################################

function incremental_objective_partials(F,uh,ph,spaces)
  U,V_p = spaces
  
  # ∂²J / ∂u² * u̇
  ∂2J∂u2 = Gridap.hessian(uh->F(uh,ph),uh)
  ∂2J∂u2_mat = assemble_matrix(∂2J∂u2,U,U)

  # ∂/∂p (∂J/∂u ) * ṗ
  ∂J∂u(uh,ph) = Gridap.gradient(uh->F(uh,ph),uh)
  ∂2J∂u∂p = Gridap.jacobian(p->∂J∂u(uh,p),ph)
  ∂2J∂u∂p_mat = assemble_matrix(∂2J∂u∂p,V_p,U)

  # ∂²J / ∂p² * ṗ
  ∂2J∂p2 = Gridap.hessian(p->F(uh,p),ph)
  ∂2J∂p2_mat = assemble_matrix(∂2J∂p2,V_p,V_p)

  # ∂/∂u (∂J / ∂p) * u̇
  ∂J∂p(uh,ph) = Gridap.gradient(p->F(uh,p),ph)
  ∂2J∂p∂u = Gridap.jacobian(uh->∂J∂p(uh,ph),uh)
  ∂2J∂p∂u_mat = assemble_matrix(∂2J∂p∂u,U,V_p)

  return ∂2J∂u2_mat, ∂2J∂u∂p_mat, ∂2J∂p2_mat, ∂2J∂p∂u_mat
end

function incremental_objective_pushforward(u_to_j,u̇,ṗ)
  # retrieve cached data
  spaces = u_to_j.spaces
  F = u_to_j.F
  uh = get_state(u_to_j)
  ph = get_parameter(u_to_j)

  # once per outer iteration
  ∂2J∂u2_mat, ∂2J∂u∂p_mat, ∂2J∂p2_mat, ∂2J∂p∂u_mat = incremental_objective_partials(F,uh,ph,spaces)

  # once per inner iteration
  dṗ = ∂2J∂p2_mat*ṗ + ∂2J∂p∂u_mat*u̇
  du̇ = ∂2J∂u2_mat*u̇ + ∂2J∂u∂p_mat*ṗ

  return du̇, dṗ
end

################################################################################################################
# du̇->dṗ : Solving the "incremental adjoint equation" ∂R/∂uᵗ * λ⁻ = du̇ - ∂²R/∂u² * u̇ * λ - ∂/∂p(∂R/∂u) * ṗ * λ #
################################################################################################################  

function incremental_adjoint_partials(res,uh,ph,λh,spaces)
  U,V,V_p = spaces

  # ∂²R / ∂u² * u̇ * λ
  ∂2R∂u2 = Gridap.hessian(uh->res(uh,λh,ph),uh) 
  ∂2R∂u2_mat = assemble_matrix(∂2R∂u2,U,V)  

  # ∂/∂p (∂R/∂u * λ) * ṗ
  ∂R∂u_λ(uh,ph) = Gridap.gradient(uh->res(uh,λh,ph),uh)
  ∂2R∂u∂p = Gridap.jacobian(p->∂R∂u_λ(uh,p),ph) 
  ∂2R∂u∂p_mat = assemble_matrix(∂2R∂u∂p,V_p,V)

  # ∂²R / ∂p² * ṗ * λ
  ∂2R∂p2 = Gridap.hessian(ph->res(uh,λh,ph),ph)
  ∂2R∂p2_mat = assemble_matrix(∂2R∂p2,V_p,V_p)

  # ∂/∂u (∂R/∂p * λ) * ṗ
  ∂R∂p_λ(uh,ph) = Gridap.gradient(ph->res(uh,λh,ph),ph)
  ∂2R∂p∂u = Gridap.jacobian(uh->∂R∂p_λ(uh,ph),uh) 
  ∂2R∂p∂u_mat = assemble_matrix(∂2R∂p∂u,U,V_p)

  return ∂2R∂u2_mat, ∂2R∂u∂p_mat, ∂2R∂p2_mat, ∂2R∂p∂u_mat
end

function incremental_adjoint_pushforward(p_to_u,u̇,ṗ,du̇)
  # retrieve cached data
  spaces = p_to_u.spaces
  U,V,V_p = spaces
  res = p_to_u.res
  adjoint_ns, _, λ = p_to_u.cache.adj_cache
  λh = FEFunction(V,λ)
  uh = get_state(p_to_u)
  ph = get_parameter(p_to_u)

  # new caches - needs work
  λ⁻ = copy(λ)

  # once per outer iteration
  ∂2R∂u2_mat, ∂2R∂u∂p_mat, ∂2R∂p2_mat, ∂2R∂p∂u_mat = incremental_adjoint_partials(res,uh,ph,λh,spaces)

  # once per inner iteration
  du̇_R = ∂2R∂u2_mat*u̇ + ∂2R∂u∂p_mat*ṗ
  dṗ_R = ∂2R∂p2_mat*ṗ + ∂2R∂p∂u_mat*u̇
  λ⁻ = solve!(λ⁻,adjoint_ns,du̇-du̇_R)
  λ⁻h = FEFunction(V,λ⁻)
  ∂R∂p_λ⁻ = Gridap.gradient(ph->res(uh,λ⁻h,ph),ph)
  ∂R∂p_mat_λ⁻ = assemble_vector(∂R∂p_λ⁻,V_p)
  dṗ_adj = - ∂R∂p_mat_λ⁻ - dṗ_R
end